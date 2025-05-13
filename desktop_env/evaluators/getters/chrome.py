import json
import logging
import os
import platform
import sqlite3
import time
import asyncio
from urllib.parse import unquote
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, parse_qs

import lxml.etree
import requests
import aiohttp
from lxml.cssselect import CSSSelector
from lxml.etree import _Element
from playwright.async_api import async_playwright, expect
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive, GoogleDriveFileList, GoogleDriveFile

_accessibility_ns_map = {
    "st": "uri:deskat:state.at-spi.gnome.org",
    "attr": "uri:deskat:attributes.at-spi.gnome.org",
    "cp": "uri:deskat:component.at-spi.gnome.org",
    "doc": "uri:deskat:document.at-spi.gnome.org",
    "docattr": "uri:deskat:attributes.document.at-spi.gnome.org",
    "txt": "uri:deskat:text.at-spi.gnome.org",
    "val": "uri:deskat:value.at-spi.gnome.org",
    "act": "uri:deskat:action.at-spi.gnome.org"
}

logger = logging.getLogger("desktopenv.getters.chrome")

"""
WARNING: 
1. Functions from this script assume that no account is registered on Chrome, otherwise the default file path needs to be changed.
2. The functions are not tested on Windows and Mac, but they should work.
"""


async def get_info_from_website(env, config: Dict[Any, Any]) -> Any:
    """Get information from a website asynchronously."""
    try:
        host = env.vm_ip
        port = env.chromium_port
        server_port = env.server_port
        remote_debugging_url = f"http://{host}:{port}"
        backend_url = f"http://{host}:{server_port}"
        
        async with async_playwright() as p:
            try:
                browser = await p.chromium.connect_over_cdp(remote_debugging_url)
            except Exception as e:
                app = 'chromium' if 'arm' in platform.machine() else 'google-chrome'
                payload = json.dumps({
                    "command": [app, "--remote-debugging-port=1337"],
                    "shell": False
                })
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        f"{backend_url}/setup/launch",
                        headers={"Content-Type": "application/json"},
                        data=payload
                    )
                await asyncio.sleep(5)
                browser = await p.chromium.connect_over_cdp(remote_debugging_url)
            page = await browser.contexts[0].new_page()
            await page.goto(config["url"])
            await page.wait_for_load_state('load')
            infos = []
            for info_dict in config.get('infos', []):
                if page.url != config["url"]:
                    await page.goto(config["url"])
                    await page.wait_for_load_state('load')
                action = info_dict.get('action', 'inner_text')
                selector = info_dict['selector']
                if action == "inner_text":
                    element = await page.wait_for_selector(selector, state='attached', timeout=10000)
                    infos.append(await element.inner_text())
                elif action == "attribute":
                    element = await page.wait_for_selector(selector, state='attached', timeout=10000)
                    infos.append(await element.get_attribute(info_dict['attribute']))
                elif action == 'click_and_inner_text':
                    for idx, sel in enumerate(selector):
                        if idx != len(selector) - 1:
                            link = await page.wait_for_selector(sel, state='attached', timeout=10000)
                            await link.click()
                            await page.wait_for_load_state('load')
                        else:
                            element = await page.wait_for_selector(sel, state='attached', timeout=10000)
                            infos.append(await element.inner_text())
                elif action == 'click_and_attribute':
                    for idx, sel in enumerate(selector):
                        if idx != len(selector) - 1:
                            link = await page.wait_for_selector(sel, state='attached', timeout=10000)
                            await link.click()
                            await page.wait_for_load_state('load')
                        else:
                            element = await page.wait_for_selector(sel, state='attached', timeout=10000)
                            infos.append(await element.get_attribute(info_dict['attribute']))
                else:
                    raise NotImplementedError(f'Unsupported action: {action}')        
            return infos
            
    except Exception as e:
        logger.error(f'Failed to get info from {config["url"]}: {e}, using backup')
        return config.get('backups', None)


# The following ones just need to load info from the files of software, no need to connect to the software
def get_default_search_engine(env, config: Dict[str, str]):
    os_type = env.vm_platform
    if os_type == 'Windows':
        preference_file_path = env.controller.execute_python_command("""import os; print(os.path.join(os.getenv('LOCALAPPDATA'),
                                            'Google\\Chrome\\User Data\\Default\\Preferences'))""")['output'].strip()
    elif os_type == 'Darwin':
        preference_file_path = env.controller.execute_python_command(
            "import os; print(os.path.join(os.getenv('HOME'), 'Library/Application Support/Google/Chrome/Default/Preferences'))")[
            'output'].strip()
    elif os_type == 'Linux':
        if "arm" in platform.machine():
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), 'snap/chromium/common/chromium/Default/Preferences'))")[
                'output'].strip()
        else:
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), '.config/google-chrome/Default/Preferences'))")[
                'output'].strip()
    else:
        raise Exception('Unsupported operating system')

    try:
        content = env.controller.get_file(preference_file_path)
        data = json.loads(content)

        # The path within the JSON data to the default search engine might vary
        search_engine = data.get('default_search_provider_data', {}).get('template_url_data', {}).get('short_name',
                                                                                                      'Google')
        return search_engine
    except Exception as e:
        logger.error(f"Error: {e}")
        return "Google"


def get_cookie_data(env, config: Dict[str, str]):
    """
    Get the cookies from the Chrome browser.
    Assume the cookies are stored in the default location, not encrypted and not large in size.
    """
    os_type = env.vm_platform
    if os_type == 'Windows':
        chrome_cookie_file_path = env.controller.execute_python_command("""import os; print(os.path.join(os.getenv('LOCALAPPDATA'),
                                            'Google\\Chrome\\User Data\\Default\\Cookies'))""")['output'].strip()
    elif os_type == 'Darwin':
        chrome_cookie_file_path = env.controller.execute_python_command(
            "import os; print(os.path.join(os.getenv('HOME'), 'Library/Application Support/Google/Chrome/Default/Cookies'))")[
            'output'].strip()
    elif os_type == 'Linux':
        if "arm" in platform.machine():
            chrome_cookie_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), 'snap/chromium/common/chromium/Default/Cookies'))")[
                'output'].strip()
        else:
            chrome_cookie_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), '.config/google-chrome/Default/Cookies'))")[
                'output'].strip()
    else:
        raise Exception('Unsupported operating system')

    try:
        content = env.controller.get_file(chrome_cookie_file_path)
        _path = os.path.join(env.cache_dir, config["dest"])

        with open(_path, "wb") as f:
            f.write(content)

        conn = sqlite3.connect(_path)
        cursor = conn.cursor()

        # Query to check for OpenAI cookies
        cursor.execute("SELECT * FROM cookies")
        cookies = cursor.fetchall()
        return cookies
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def get_history(env, config: Dict[str, str]):
    os_type = env.vm_platform
    if os_type == 'Windows':
        chrome_history_path = env.controller.execute_python_command(
            """import os; print(os.path.join(os.getenv('USERPROFILE'), "AppData", "Local", "Google", "Chrome", "User Data", "Default", "History"))""")[
            'output'].strip()
    elif os_type == 'Darwin':
        chrome_history_path = env.controller.execute_python_command(
            """import os; print(os.path.join(os.getenv('HOME'), "Library", "Application Support", "Google", "Chrome", "Default", "History"))""")[
            'output'].strip()
    elif os_type == 'Linux':
        if "arm" in platform.machine():
            chrome_history_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), 'snap/chromium/common/chromium/Default/History'))")[
                'output'].strip()
        else:
            chrome_history_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), '.config', 'google-chrome', 'Default', 'History'))")[
                'output'].strip()
    else:
        raise Exception('Unsupported operating system')

    try:
        content = env.controller.get_file(chrome_history_path)
        _path = os.path.join(env.cache_dir, config["dest"])

        with open(_path, "wb") as f:
            f.write(content)

        conn = sqlite3.connect(_path)
        cursor = conn.cursor()

        # Query to check for OpenAI cookies
        cursor.execute("SELECT url, title, last_visit_time FROM urls")
        history_items = cursor.fetchall()
        return history_items
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def get_enabled_experiments(env, config: Dict[str, str]):
    os_type = env.vm_platform
    if os_type == 'Windows':
        preference_file_path = env.controller.execute_python_command("""import os; print(os.path.join(os.getenv('LOCALAPPDATA'),
                                                'Google\\Chrome\\User Data\\Local State'))""")[
            'output'].strip()
    elif os_type == 'Darwin':
        preference_file_path = env.controller.execute_python_command(
            "import os; print(os.path.join(os.getenv('HOME'), 'Library/Application Support/Google/Chrome/Local State'))")[
            'output'].strip()
    elif os_type == 'Linux':
        if "arm" in platform.machine():
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), 'snap/chromium/common/chromium/Local State'))")[
                'output'].strip()
        else:
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), '.config/google-chrome/Local State'))")[
                'output'].strip()
    else:
        raise Exception('Unsupported operating system')

    try:
        content = env.controller.get_file(preference_file_path)
        data = json.loads(content)

        # The path within the JSON data to the default search engine might vary
        enabled_labs_experiments = data.get('browser', {}).get('enabled_labs_experiments', [])
        return enabled_labs_experiments
    except Exception as e:
        logger.error(f"Error: {e}")
        return []


def get_profile_name(env, config: Dict[str, str]):
    """
    Get the username from the Chrome browser.
    Assume the cookies are stored in the default location, not encrypted and not large in size.
    """
    os_type = env.vm_platform
    if os_type == 'Windows':
        preference_file_path = env.controller.execute_python_command("""import os; print(os.path.join(os.getenv('LOCALAPPDATA'),
                                            'Google\\Chrome\\User Data\\Default\\Preferences'))""")['output'].strip()
    elif os_type == 'Darwin':
        preference_file_path = env.controller.execute_python_command(
            "import os; print(os.path.join(os.getenv('HOME'), 'Library/Application Support/Google/Chrome/Default/Preferences'))")[
            'output'].strip()
    elif os_type == 'Linux':
        if "arm" in platform.machine():
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), 'snap/chromium/common/chromium/Default/Preferences'))")[
                'output'].strip()
        else:
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), '.config/google-chrome/Default/Preferences'))")[
                'output'].strip()
    else:
        raise Exception('Unsupported operating system')

    try:
        content = env.controller.get_file(preference_file_path)
        data = json.loads(content)

        # The path within the JSON data to the default search engine might vary
        profile_name = data.get('profile', {}).get('name', None)
        return profile_name
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def get_chrome_language(env, config: Dict[str, str]):
    os_type = env.vm_platform
    if os_type == 'Windows':
        preference_file_path = env.controller.execute_python_command("""import os; print(os.path.join(os.getenv('LOCALAPPDATA'),
                                                    'Google\\Chrome\\User Data\\Local State'))""")[
            'output'].strip()
    elif os_type == 'Darwin':
        preference_file_path = env.controller.execute_python_command(
            "import os; print(os.path.join(os.getenv('HOME'), 'Library/Application Support/Google/Chrome/Local State'))")[
            'output'].strip()
    elif os_type == 'Linux':
        if "arm" in platform.machine():
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), 'snap/chromium/common/chromium/Local State'))")[
                'output'].strip()
        else:
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), '.config/google-chrome/Local State'))")[
                'output'].strip()
    else:
        raise Exception('Unsupported operating system')

    try:
        content = env.controller.get_file(preference_file_path)
        data = json.loads(content)

        # The path within the JSON data to the default search engine might vary
        enabled_labs_experiments = data.get('intl', {}).get('app_locale', "en-US")
        return enabled_labs_experiments
    except Exception as e:
        logger.error(f"Error: {e}")
        return "en-US"


def get_chrome_font_size(env, config: Dict[str, str]):
    os_type = env.vm_platform
    if os_type == 'Windows':
        preference_file_path = env.controller.execute_python_command("""import os; print(os.path.join(os.getenv('LOCALAPPDATA'),
                                                'Google\\Chrome\\User Data\\Default\\Preferences'))""")[
            'output'].strip()
    elif os_type == 'Darwin':
        preference_file_path = env.controller.execute_python_command(
            "import os; print(os.path.join(os.getenv('HOME'), 'Library/Application Support/Google/Chrome/Default/Preferences'))")[
            'output'].strip()
    elif os_type == 'Linux':
        if "arm" in platform.machine():
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), 'snap/chromium/common/chromium/Default/Preferences'))")[
                'output'].strip()
        else:
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), '.config/google-chrome/Default/Preferences'))")[
                'output'].strip()
    else:
        raise Exception('Unsupported operating system')

    try:
        content = env.controller.get_file(preference_file_path)
        data = json.loads(content)

        # The path within the JSON data to the default search engine might vary
        search_engine = data.get('webkit', {}).get('webprefs', {
            "default_fixed_font_size": 13,
            "default_font_size": 16,
            "minimum_font_size": 13
        })
        return search_engine
    except Exception as e:
        logger.error(f"Error: {e}")
        return {
            "default_fixed_font_size": 13,
            "default_font_size": 16
        }


def get_bookmarks(env, config: Dict[str, str]):
    os_type = env.vm_platform
    if os_type == 'Windows':
        preference_file_path = env.controller.execute_python_command("""import os; print(os.path.join(os.getenv('LOCALAPPDATA'),
                                            'Google\\Chrome\\User Data\\Default\\Bookmarks'))""")['output'].strip()
    elif os_type == 'Darwin':
        preference_file_path = env.controller.execute_python_command(
            "import os; print(os.path.join(os.getenv('HOME'), 'Library/Application Support/Google/Chrome/Default/Bookmarks'))")[
            'output'].strip()
    elif os_type == 'Linux':
        if "arm" in platform.machine():
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), 'snap/chromium/common/chromium/Default/Bookmarks'))")[
                'output'].strip()
        else:
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), '.config/google-chrome/Default/Bookmarks'))")[
                'output'].strip()
    else:
        raise Exception('Unsupported operating system')

    content = env.controller.get_file(preference_file_path)
    if not content:
        return []
    data = json.loads(content)
    bookmarks = data.get('roots', {})
    return bookmarks


# todo: move this to the main.py
def get_extensions_installed_from_shop(env, config: Dict[str, str]):
    """Find the Chrome extensions directory based on the operating system."""
    os_type = env.vm_platform
    if os_type == 'Windows':
        chrome_extension_dir = env.controller.execute_python_command(
            """os.path.expanduser('~') + '\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Extensions\\'""")[
            'output'].strip()
    elif os_type == 'Darwin':  # macOS
        chrome_extension_dir = env.controller.execute_python_command(
            """os.path.expanduser('~') + '/Library/Application Support/Google/Chrome/Default/Extensions/'""")[
            'output'].strip()
    elif os_type == 'Linux':
        if "arm" in platform.machine():
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), 'snap/chromium/common/chromium/Default/Extensions/'))")[
                'output'].strip()
        else:
            chrome_extension_dir = env.controller.execute_python_command(
                """os.path.expanduser('~') + '/.config/google-chrome/Default/Extensions/'""")['output'].strip()
    else:
        raise Exception('Unsupported operating system')

    manifests = []
    for extension_id in os.listdir(chrome_extension_dir):
        extension_path = os.path.join(chrome_extension_dir, extension_id)
        if os.path.isdir(extension_path):
            # Iterate through version-named subdirectories
            for version_dir in os.listdir(extension_path):
                version_path = os.path.join(extension_path, version_dir)
                manifest_path = os.path.join(version_path, 'manifest.json')
                if os.path.isfile(manifest_path):
                    with open(manifest_path, 'r') as file:
                        try:
                            manifest = json.load(file)
                            manifests.append(manifest)
                        except json.JSONDecodeError:
                            logger.error(f"Error reading {manifest_path}")
    return manifests


# The following ones require Playwright to be installed on the target machine, and the chrome needs to be pre-config on
# port info to allow remote debugging, see README.md for details

async def get_page_info(env, config: Dict[str, str]) -> Dict[str, str]:
    """
    Asynchronously retrieve page information using Playwright with Chrome DevTools Protocol.
    
    Args:
        env: Environment object containing connection parameters
        config: Configuration dictionary with 'url' key
        
    Returns:
        Dictionary containing page title, URL, and content
    """
    host = env.vm_ip
    port = env.chromium_port
    server_port = env.server_port
    target_url = config["url"]

    # Construct remote debugging URL
    remote_debugging_url = f"http://{host}:{port}"
    
    async with async_playwright() as p:
        # Attempt to connect to existing browser instance
        try:
            browser = await p.chromium.connect_over_cdp(remote_debugging_url)
        except Exception as e:
            # If connection fails, launch new browser instance
            machine_arch = platform.machine().lower()
            command = ["chromium", "--remote-debugging-port=1337"] if "arm" in machine_arch else ["google-chrome", "--remote-debugging-port=1337"]
            
            # Send POST request to launch browser
            payload = json.dumps({
                "command": command,
                "shell": False
            })
            
            headers = {"Content-Type": "application/json"}
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"http://{host}:{server_port}/setup/launch",
                    headers=headers,
                    data=payload
                )
            
            # Wait for browser initialization
            await asyncio.sleep(5)
            
            # Retry connection
            browser = await p.chromium.connect_over_cdp(remote_debugging_url)

        # Create new page and navigate to target URL
        context = browser.contexts[0]
        page = await context.new_page()
        await page.goto(target_url)

        page_info = {}
        try:
            # Wait for page load completion
            await page.wait_for_load_state('networkidle', timeout=15000)
            # Extract page information
            page_info = {
                'title': await page.title(),
                'url': page.url,
                'content': await page.content()
            }
            
        except PlaywrightTimeoutError:
            # Handle timeout scenario
            page_info = {
                'title': 'Load Timeout',
                'url': page.url,
                'content': await page.content()
            }
        except Exception as e:
            # Handle other exceptions
            page_info = {
                'title': f'Error: {str(e)}',
                'url': page.url,
                'content': await page.content()
            }
        finally:
            # Clean up resources
            await page.close()
            await browser.close()
        return page_info


async def get_open_tabs_info(env, config: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Asynchronously retrieve information about all open tabs in a remote browser instance.
    
    Args:
        env: Environment object containing connection parameters
        config: Configuration dictionary (unused in current implementation)
        
    Returns:
        List of dictionaries containing tab titles and URLs
    """
    host = env.vm_ip
    port = env.chromium_port
    server_port = env.server_port

    remote_debugging_url = f"http://{host}:{port}"
    
    async with async_playwright() as p:
        # Attempt to connect to existing browser instance
        try:
            browser = await p.chromium.connect_over_cdp(remote_debugging_url)
        except Exception as e:
            # If connection fails, launch new browser instance
            machine_arch = platform.machine().lower()
            command = ["chromium", "--remote-debugging-port=1337"] if "arm" in machine_arch else ["google-chrome", "--remote-debugging-port=1337"]
            # Send async POST request to launch browser
            payload = json.dumps({
                "command": command,
                "shell": False
            })
            headers = {"Content-Type": "application/json"}
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"http://{host}:{server_port}/setup/launch",
                    headers=headers,
                    data=payload
                )
            
            # Wait for browser initialization
            await asyncio.sleep(5)
            
            # Retry connection
            try:
                browser = await p.chromium.connect_over_cdp(remote_debugging_url)
            except Exception as e:
                return []

        tabs_info = []
        # Iterate through all browser contexts
        for context in browser.contexts:
            # Process all pages in each context
            for page in context.pages:
                try:
                    # Wait for network stability
                    await page.wait_for_load_state('networkidle', timeout=10000)
                    
                    # Get page information
                    tab_info = {
                        'title': await page.title(),
                        'url': page.url
                    }
                    tabs_info.append(tab_info)
                    
                except PlaywrightTimeoutError:
                    # Handle timeout with available information
                    tabs_info.append({
                        'title': 'Load Timeout',
                        'url': page.url
                    })
                except Exception as e:
                    # Handle other exceptions
                    tabs_info.append({
                        'title': f'Error: {str(e)}',
                        'url': page.url
                    })

        # Clean up resources
        await browser.close()
        return tabs_info


def get_active_url_from_accessTree(env, config):
    """
        Playwright cannot get the url of active tab directly, 
        so we need to use accessibility tree to get the active tab info.
        This function is used to get the active tab url from the accessibility tree.
        config: 
            Dict[str, str]{
                # we no longer need to specify the xpath or selectors, since we will use defalut value
                # 'xpath': 
                #     the same as in metrics.general.accessibility_tree.
                # 'selectors': 
                #     the same as in metrics.general.accessibility_tree.
                'goto_prefix':
                    the prefix you want to add to the beginning of the url to be opened, default is "https://",
                    (the url we get from accTree does not have prefix)
                ...(other keys, not used in this function)
        }
        Return
            url: str
    """
    # Ensure the controller and its method are accessible and return a valid result
    if hasattr(env, 'controller') and callable(getattr(env.controller, 'get_accessibility_tree', None)):
        accessibility_tree = env.controller.get_accessibility_tree()
        if accessibility_tree is None:
            print("Failed to get the accessibility tree.")
            return None
    else:
        print("Controller or method 'get_accessibility_tree' not found.")
        return None

    logger.debug("AT@eval: %s", accessibility_tree)

    at = None
    try:
        at = lxml.etree.fromstring(accessibility_tree)
    except ValueError as e:
        logger.error(f"Error parsing accessibility tree: {e}")
        return None

    # Determine the correct selector based on system architecture
    selector = None
    arch = platform.machine()
    print(f"Your architecture is: {arch}")

    if "arm" in arch:
        selector_string = "application[name=Chromium] entry[name=Address\\ and\\ search\\ bar]"
    else:
        selector_string = "application[name=Google\\ Chrome] entry[name=Address\\ and\\ search\\ bar]"

    try:
        selector = CSSSelector(selector_string, namespaces=_accessibility_ns_map)
    except Exception as e:
        logger.error(f"Failed to parse the selector for active tab URL: {e}")
        return None

    elements = selector(at) if selector else []
    if not elements:
        print("No elements found.")
        return None
    elif not elements[-1].text:
        print("No text found in the latest element.")
        return None

    # Use a default prefix if 'goto_prefix' is not specified in the config
    goto_prefix = config.get("goto_prefix", "https://")

    active_tab_url = f"{goto_prefix}{elements[0].text}"
    print(f"Active tab url now: {active_tab_url}")
    return active_tab_url


async def get_active_tab_info(env, config: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    Asynchronously retrieves information about the active browser tab.
    
    Warning: This function will reload the target page. Ensure the URL doesn't redirect
    due to cookies or cache before using. Test in incognito mode first.
    
    Args:
        env: Environment configuration object
        config: Dictionary containing configuration parameters
            (uses keys for get_active_url_from_accessTree)
    
    Returns:
        Dictionary with title, URL, and content of active tab, or None on failure
    """
    # Get active URL from external service
    active_tab_url = await get_active_url_from_accessTree(env, config)
    if not active_tab_url:
        logger.error("Failed to retrieve active tab URL")
        return None

    host = env.vm_ip
    port = env.chromium_port  # TODO: Make port configurable
    remote_debugging_url = f"http://{host}:{port}"

    async with async_playwright() as p:
        try:
            # Connect to existing browser instance
            browser = await p.chromium.connect_over_cdp(remote_debugging_url)
        except Exception as e:
            logger.error(f"Browser connection failed: {str(e)}")
            return None

        active_tab_info = {}
        try:
            # Create new page context
            page = await browser.new_page()
            
            # Navigate to target URL with timeout handling
            try:
                await page.goto(active_tab_url, wait_until="domcontentloaded")
                await page.wait_for_load_state('networkidle', timeout=15000)
            except PlaywrightTimeoutError:
                logger.warning("Page load timed out, capturing partial content")
            except Exception as e:
                logger.error(f"Navigation failed: {str(e)}")
                return None

            # Extract page information
            active_tab_info = {
                'title': await page.title(),
                'url': page.url,
                'content': await page.content()
            }

        finally:
            # Cleanup resources
            await page.close()
            await browser.close()

        logger.debug(f"Active tab title: {active_tab_info.get('title', 'N/A')}")
        logger.debug(f"Active tab URL: {active_tab_info.get('url', 'N/A')}")
        return active_tab_info


async def get_pdf_from_url(env, config: Dict[str, str]) -> str:
    """
    Asynchronously download a PDF from a URL using remote Chrome instance.
    
    Args:
        env: Environment configuration object
        config: Dictionary containing:
            - "path": URL to download PDF from
            - "dest": Destination filename in cache directory
            
    Returns:
        Path to downloaded PDF file
    
    Raises:
        ConnectionError: If browser connection fails after retry
        RuntimeError: If PDF generation fails
    """
    # Extract configuration parameters
    target_url = config["path"]
    pdf_filename = os.path.join(env.cache_dir, config["dest"])
    
    # Connection parameters
    host = env.vm_ip
    port = env.chromium_port  # TODO: Make configurable via env
    server_port = env.server_port
    remote_debugging_url = f"http://{host}:{port}"

    async with async_playwright() as p:
        browser = None
        try:
            # Attempt primary connection
            browser = await p.chromium.connect_over_cdp(remote_debugging_url)
        except Exception as e:
            # Fallback to browser launch if connection fails
            machine_arch = platform.machine().lower()
            browser_cmd = ["chromium"] if "arm" in machine_arch else ["google-chrome"]
            
            # Configure launch command
            launch_payload = {
                "command": browser_cmd + ["--remote-debugging-port=1337"],
                "shell": False
            }
            
            # Send async launch request
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"http://{host}:{server_port}/setup/launch",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(launch_payload)
                )
            
            # Wait for browser initialization
            await asyncio.sleep(5)
            
            # Retry connection
            browser = await p.chromium.connect_over_cdp(remote_debugging_url)

        # Create new page context
        page = await browser.new_page()
        try:
            # Navigate with extended timeout
            await page.goto(target_url, wait_until="networkidle", timeout=30000)
            
            # Generate PDF with print parameters
            pdf_params = {
                "path": pdf_filename,
                "format": "A4",
                "print_background": True,
                "margin": {"top": "0.5in", "bottom": "0.5in"}
            }
            await page.pdf(**pdf_params)
            
        except Exception as e:
            # Cleanup failed download
            if os.path.exists(pdf_filename):
                os.remove(pdf_filename)
            raise RuntimeError(f"PDF generation failed: {str(e)}")
            
        finally:
            # Ensure resource cleanup
            await page.close()
            await browser.close()

    return pdf_filename


# fixme: needs to be changed (maybe through post-processing) since it's not working
async def get_chrome_saved_address(env, config: Dict[str, str]) -> str:
    """
    Asynchronously retrieves saved addresses from Chrome's settings using remote debugging.
    
    Args:
        env: Environment configuration object
        config: Configuration dictionary (unused in current implementation)
        
    Returns:
        HTML content of Chrome's address settings page
        
    Raises:
        ConnectionError: If unable to connect to browser instance
        RuntimeError: If navigation to settings page fails
    """
    host = env.vm_ip
    port = env.chromium_port
    server_port = env.server_port
    remote_debugging_url = f"http://{host}:{port}"

    async with async_playwright() as p:
        browser = None
        try:
            # Attempt to connect to existing browser instance
            browser = await p.chromium.connect_over_cdp(remote_debugging_url)
        except Exception as e:
            # Fallback to launching new browser instance
            machine_arch = platform.machine().lower()
            command = ["chromium", "--remote-debugging-port=1337"] if "arm" in machine_arch else ["google-chrome", "--remote-debugging-port=1337"]
            
            # Send async launch request
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"http://{host}:{server_port}/setup/launch",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({
                        "command": command,
                        "shell": False
                    })
                )
            
            # Wait for browser initialization
            await asyncio.sleep(5)
            browser = await p.chromium.connect_over_cdp(remote_debugging_url)

        try:
            # Create new isolated context
            context = await browser.new_context()
            page = await context.new_page()
            
            # Navigate to Chrome's internal addresses page
            try:
                await page.goto("chrome://settings/addresses", timeout=15000)
            except Exception as e:
                raise RuntimeError(f"Failed to navigate to settings: {str(e)}")
            
            # Wait for address list to load
            await page.wait_for_selector('//settings-ui', state='attached')
            
            # Retrieve page content
            content = await page.content()
            return content
            
        finally:
            # Cleanup resources
            await page.close()
            await context.close()
            await browser.close()


def get_shortcuts_on_desktop(env, config: Dict[str, str]):
    # Find out the operating system
    os_name = env.vm_platform

    # Depending on the OS, define the shortcut file extension
    if os_name == 'Windows':
        # Windows shortcuts are typically .url or .lnk files
        shortcut_extension = '.lnk'
    elif os_name == 'Darwin':
        # macOS's shortcuts are .webloc files
        shortcut_extension = '.webloc'
    elif os_name == 'Linux':
        # Linux (Ubuntu, etc.) shortcuts are typically .desktop files
        shortcut_extension = '.desktop'
    else:
        logger.error(f"Unsupported operating system: {os_name}")
        return []

    # Get the path to the desktop folder
    desktop_path = env.controller.get_vm_desktop_path()
    desktop_directory_tree = env.controller.get_vm_directory_tree(desktop_path)

    shortcuts_paths = [file['name'] for file in desktop_directory_tree['children'] if
                       file['name'].endswith(shortcut_extension)]

    short_cuts = {}

    for shortcut_path in shortcuts_paths:
        short_cuts[shortcut_path] = env.controller.get_file(env.controller.execute_python_command(
            f"import os; print(os.path.join(os.path.expanduser('~'), 'Desktop', '{shortcut_path}'))")[
                                                                'output'].strip()).decode('utf-8')

    return short_cuts


async def get_number_of_search_results(env, config: Dict[str, str]) -> int:
    """
    Asynchronously counts search results on a search engine page using remote Chrome instance.
    
    Args:
        env: Environment configuration object
        config: Dictionary containing configuration parameters
            (currently uses hardcoded values, needs migration to config)
            
    Returns:
        Number of visible search results
    
    Raises:
        ConnectionError: If browser connection fails
        TimeoutError: If page navigation times out
    """
    # Configuration parameters (TODO: Move to config)
    search_url = "https://google.com/search?q=query"
    result_selector = '.search-result'  # Verify actual search result selector
    
    # Connection parameters
    host = env.vm_ip
    port = env.chromium_port
    server_port = env.server_port
    remote_debugging_url = f"http://{host}:{port}"

    async with async_playwright() as p:
        try:
            # Connect to existing browser instance
            browser = await p.chromium.connect_over_cdp(remote_debugging_url)
        except Exception as e:
            # Launch new browser if connection fails
            machine_arch = platform.machine().lower()
            command = ["chromium", "--remote-debugging-port=1337"] if "arm" in machine_arch else ["google-chrome", "--remote-debugging-port=1337"]
            
            # Async browser launch request
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"http://{host}:{server_port}/setup/launch",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"command": command, "shell": False})
                )
            
            await asyncio.sleep(5)  # Wait for browser initialization
            browser = await p.chromium.connect_over_cdp(remote_debugging_url)

        try:
            # Create isolated browsing context
            context = await browser.new_context()
            page = await context.new_page()
            
            # Navigate with error handling
            try:
                await page.goto(search_url, wait_until="networkidle", timeout=20000)
            except Exception as e:
                raise TimeoutError(f"Navigation timed out: {str(e)}")
            
            # Wait for results to load
            await page.wait_for_selector(result_selector, state="attached", timeout=10000)
            
            # Count visible results
            search_results = await page.query_selector_all(result_selector)
            return len(search_results)
            
        finally:
            # Cleanup resources
            await page.close()
            await context.close()
            await browser.close()


def get_googledrive_file(env, config: Dict[str, Any]) -> str:
    """ Get the desired file from Google Drive based on config, return the downloaded local filepath.
    @args: keys in config dict
        settings_file(str): target filepath to the settings file for Google Drive authentication, default is 'evaluation_examples/settings/googledrive/settings.yml'
        query/path[_list](Union[str, List[str]]): the query or path [list] to the file(s) on Google Drive. To retrieve the file, we provide multiple key options to specify the filepath on drive in config dict:
            1) query: a list of queries to search the file, each query is a string that follows the format of Google Drive search query. The documentation is available here: (support more complex search but too complicated to use)
                https://developers.google.com/drive/api/guides/search-files?hl=en
            2) path: a str list poingting to file path on googledrive, e.g., 'folder/subfolder/filename.txt' -> 
                config contain one key-value pair "path": ['folder', 'subfolder', 'filename.txt']
            3) query_list: query extends to list to download multiple files
            4) path_list: path extends to list to download multiple files, e.g.,
                "path_list": [['folder', 'subfolder', 'filename1.txt'], ['folder', 'subfolder', 'filename2.txt']]
    @return:
        dest(Union[List[str], str]): target file name or list. If *_list is used in input config, dest should also be a list of the same length. Return the downloaded local filepath.
    """
    settings_file = config.get('settings_file', 'evaluation_examples/settings/googledrive/settings.yml')
    auth = GoogleAuth(settings_file=settings_file)
    drive = GoogleDrive(auth)

    def get_single_file(_query, _path):
        parent_id = 'root'
        try:
            for q in _query:
                search = f'( {q} ) and "{parent_id}" in parents'
                filelist: GoogleDriveFileList = drive.ListFile({'q': search}).GetList()
                if len(filelist) == 0:  # target file not found
                    return None
                file: GoogleDriveFile = filelist[0]  # HACK: if multiple candidates, just use the first one
                parent_id = file['id']

            file.GetContentFile(_path, mimetype=file['mimeType'])
        except Exception as e:
            logger.info('[ERROR]: Failed to download the file from Google Drive', e)
            return None
        return _path

    if 'query' in config:
        return get_single_file(config['query'], os.path.join(env.cache_dir, config['dest']))
    elif 'path' in config:
        query = [f"title = '{fp}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false" if idx < len(
            config['path']) - 1
                 else f"title = '{fp}' and trashed = false" for idx, fp in enumerate(config['path'])]
        return get_single_file(query, os.path.join(env.cache_dir, config['dest']))
    elif 'query_list' in config:
        _path_list = []
        assert len(config['query_list']) == len(config['dest'])
        for idx, query in enumerate(config['query_list']):
            dest = config['dest'][idx]
            _path_list.append(get_single_file(query, os.path.join(env.cache_dir, dest)))
        return _path_list
    else:  # path_list in config
        _path_list = []
        assert len(config['path_list']) == len(config['dest'])
        for idx, path in enumerate(config['path_list']):
            query = [
                f"title = '{fp}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false" if jdx < len(
                    path) - 1
                else f"title = '{fp}' and trashed = false" for jdx, fp in enumerate(path)]
            dest = config['dest'][idx]
            _path_list.append(get_single_file(query, os.path.join(env.cache_dir, dest)))
        return _path_list


def get_enable_do_not_track(env, config: Dict[str, str]):
    os_type = env.vm_platform
    if os_type == 'Windows':
        preference_file_path = env.controller.execute_python_command("""import os; print(os.path.join(os.getenv('LOCALAPPDATA'),
                                            'Google\\Chrome\\User Data\\Default\\Preferences'))""")['output'].strip()
    elif os_type == 'Darwin':
        preference_file_path = env.controller.execute_python_command(
            "import os; print(os.path.join(os.getenv('HOME'), 'Library/Application Support/Google/Chrome/Default/Preferences'))")[
            'output'].strip()
    elif os_type == 'Linux':
        if "arm" in platform.machine():
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), 'snap/chromium/common/chromium/Default/Preferences'))")[
                'output'].strip()
        else:
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), '.config/google-chrome/Default/Preferences'))")[
                'output'].strip()

    else:
        raise Exception('Unsupported operating system')

    try:
        content = env.controller.get_file(preference_file_path)
        data = json.loads(content)

        if_enable_do_not_track = data.get('enable_do_not_track', {})  # bool
        return "true" if if_enable_do_not_track else "false"
    except Exception as e:
        logger.error(f"Error: {e}")
        return "false"


def get_enable_enhanced_safety_browsing(env, config: Dict[str, str]):
    os_type = env.vm_platform
    if os_type == 'Windows':
        preference_file_path = env.controller.execute_python_command("""import os; print(os.path.join(os.getenv('LOCALAPPDATA'),
                                            'Google\\Chrome\\User Data\\Default\\Preferences'))""")['output'].strip()
    elif os_type == 'Darwin':
        preference_file_path = env.controller.execute_python_command(
            "import os; print(os.path.join(os.getenv('HOME'), 'Library/Application Support/Google/Chrome/Default/Preferences'))")[
            'output'].strip()
    elif os_type == 'Linux':
        if "arm" in platform.machine():
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), 'snap/chromium/common/chromium/Default/Preferences'))")[
                'output'].strip()
        else:
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), '.config/google-chrome/Default/Preferences'))")[
                'output'].strip()

    else:
        raise Exception('Unsupported operating system')

    try:
        content = env.controller.get_file(preference_file_path)
        data = json.loads(content)

        if_enable_do_not_track = data.get('safebrowsing', {}).get('enhanced', {})  # bool
        return "true" if if_enable_do_not_track else "false"
    except Exception as e:
        logger.error(f"Error: {e}")
        return "Google"


def get_new_startup_page(env, config: Dict[str, str]):
    os_type = env.vm_platform
    if os_type == 'Windows':
        preference_file_path = env.controller.execute_python_command("""import os; print(os.path.join(os.getenv('LOCALAPPDATA'),
                                            'Google\\Chrome\\User Data\\Default\\Preferences'))""")['output'].strip()
    elif os_type == 'Darwin':
        preference_file_path = env.controller.execute_python_command(
            "import os; print(os.path.join(os.getenv('HOME'), 'Library/Application Support/Google/Chrome/Default/Preferences'))")[
            'output'].strip()
    elif os_type == 'Linux':
        if "arm" in platform.machine():
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), 'snap/chromium/common/chromium/Default/Preferences'))")[
                'output'].strip()
        else:
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), '.config/google-chrome/Default/Preferences'))")[
                'output'].strip()

    else:
        raise Exception('Unsupported operating system')

    try:
        content = env.controller.get_file(preference_file_path)
        data = json.loads(content)

        # if data has no key called 'session', it means the chrome is on a fresh-start mode, which is a true state;
        # otherwise, try to find the code number in 'restored_on_startup' in 'session'
        if "session" not in data.keys():
            return "true"
        else:
            if_enable_do_not_track = data.get('session', {}).get('restore_on_startup', {})  # int, need to be 5
            return "true" if if_enable_do_not_track == 5 else "false"
    except Exception as e:
        logger.error(f"Error: {e}")
        return "Google"


def get_find_unpacked_extension_path(env, config: Dict[str, str]):
    os_type = env.vm_platform
    if os_type == 'Windows':
        preference_file_path = env.controller.execute_python_command("""import os; print(os.path.join(os.getenv('LOCALAPPDATA'),
                                            'Google\\Chrome\\User Data\\Default\\Preferences'))""")['output'].strip()
    elif os_type == 'Darwin':
        preference_file_path = env.controller.execute_python_command(
            "import os; print(os.path.join(os.getenv('HOME'), 'Library/Application Support/Google/Chrome/Default/Preferences'))")[
            'output'].strip()
    elif os_type == 'Linux':
        if "arm" in platform.machine():
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), 'snap/chromium/common/chromium/Default/Preferences'))")[
                'output'].strip()
        else:
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), '.config/google-chrome/Default/Preferences'))")[
                'output'].strip()

    else:
        raise Exception('Unsupported operating system')

    try:
        content = env.controller.get_file(preference_file_path)
        data = json.loads(content)
        # Preferences store all the path of installed extensions, return them all and let metrics try to find one matches the targeted extension path
        all_extensions_path = []
        all_extensions = data.get('extensions', {}).get('settings', {})
        for id in all_extensions.keys():
            path = all_extensions[id]["path"]
            all_extensions_path.append(path)
        return all_extensions_path
    except Exception as e:
        logger.error(f"Error: {e}")
        return "Google"


def get_find_installed_extension_name(env, config: Dict[str, str]):
    os_type = env.vm_platform
    if os_type == 'Windows':
        preference_file_path = env.controller.execute_python_command("""import os; print(os.path.join(os.getenv('LOCALAPPDATA'),
                                            'Google\\Chrome\\User Data\\Default\\Preferences'))""")['output'].strip()
    elif os_type == 'Darwin':
        preference_file_path = env.controller.execute_python_command(
            "import os; print(os.path.join(os.getenv('HOME'), 'Library/Application Support/Google/Chrome/Default/Preferences'))")[
            'output'].strip()
    elif os_type == 'Linux':
        if "arm" in platform.machine():
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), 'snap/chromium/common/chromium/Default/Preferences'))")[
                'output'].strip()
        else:
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), '.config/google-chrome/Default/Preferences'))")[
                'output'].strip()

    else:
        raise Exception('Unsupported operating system')

    try:
        content = env.controller.get_file(preference_file_path)
        data = json.loads(content)
        # Preferences store all the path of installed extensions, return them all and let metrics try to find one matches the targeted extension path
        all_extensions_name = []
        all_extensions = data.get('extensions', {}).get('settings', {})
        for id in all_extensions.keys():
            name = all_extensions[id]["manifest"]["name"]
            all_extensions_name.append(name)
        return all_extensions_name
    except Exception as e:
        logger.error(f"Error: {e}")
        return "Google"


def get_data_delete_automacally(env, config: Dict[str, str]):
    """
    This function is used to open th "auto-delete" mode of chromium
    """
    os_type = env.vm_platform
    if os_type == 'Windows':
        preference_file_path = env.controller.execute_python_command("""import os; print(os.path.join(os.getenv('LOCALAPPDATA'),
                                            'Google\\Chrome\\User Data\\Default\\Preferences'))""")['output'].strip()
    elif os_type == 'Darwin':
        preference_file_path = env.controller.execute_python_command(
            "import os; print(os.path.join(os.getenv('HOME'), 'Library/Application Support/Google/Chrome/Default/Preferences'))")[
            'output'].strip()
    elif os_type == 'Linux':
        if "arm" in platform.machine():
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), 'snap/chromium/common/chromium/Default/Preferences'))")[
                'output'].strip()
        else:
            preference_file_path = env.controller.execute_python_command(
                "import os; print(os.path.join(os.getenv('HOME'), '.config/google-chrome/Default/Preferences'))")[
                'output'].strip()
    else:
        raise Exception('Unsupported operating system')

    try:
        content = env.controller.get_file(preference_file_path)
        data = json.loads(content)
        data_delete_state = data["profile"].get("default_content_setting_values", None)
        return "true" if data_delete_state is not None else "false"
    except Exception as e:
        logger.error(f"Error: {e}")
        return "Google"


async def get_active_tab_html_parse(env, config: Dict[str, Any]) -> Dict[str, str]:
    """
    Asynchronously extracts specific element content from active tab using various selectors.
    
    Args:
        env: Environment configuration object
        config: Dictionary containing parsing instructions with keys:
            - category: One of ["class", "label", "xpath", "input"]
            - Corresponding selector objects based on category
            
    Returns:
        Dictionary containing extracted text content with specified keys
        
    Raises:
        ConnectionError: If browser connection fails
        ValueError: For invalid configurations
    """
    # Get active URL from external service
    active_tab_url = await get_active_url_from_accessTree(env, config)
    if not isinstance(active_tab_url, str):
        logger.error("Invalid active tab URL format")
        return {}

    # Connection parameters
    host = env.vm_ip
    port = env.chromium_port
    server_port = env.server_port
    remote_debugging_url = f"http://{host}:{port}"

    async with async_playwright() as p:
        browser = None
        try:
            # Attempt browser connection
            browser = await p.chromium.connect_over_cdp(remote_debugging_url)
        except Exception as e:
            # Fallback browser launch
            machine_arch = platform.machine().lower()
            command = ["chromium", "--remote-debugging-port=1337"] if "arm" in machine_arch else ["google-chrome", "--remote-debugging-port=1337"]
            
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"http://{host}:{server_port}/setup/launch",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"command": command, "shell": False})
                )
            
            await asyncio.sleep(5)
            browser = await p.chromium.connect_over_cdp(remote_debugging_url)

        target_page = None
        try:
            # Find matching page across contexts
            for context in browser.contexts:
                for page in context.pages:
                    try:
                        await page.wait_for_load_state("networkidle", timeout=10000)
                        if unquote(page.url) == unquote(active_tab_url):
                            target_page = page
                            logger.debug(f"Target page found: {page.url}")
                            break
                    except Exception as e:
                        continue
                
                if target_page:
                    break

            if not target_page:
                logger.error("Target tab not found")
                return {}

            return_json = {}

            async def safely_get_text_content(selector: str) -> list:
                """Async safe element text extraction"""
                elements = await target_page.query_selector_all(selector)
                return [await element.text_content() or "" for element in elements]

            # Class-based extraction
            if config["category"] == "class":
                if "class_multiObject" in config:
                    for class_name, object_dict in config["class_multiObject"].items():
                        texts = await safely_get_text_content(f".{class_name}")
                        for index_str, key in object_dict.items():
                            try:
                                index = int(index_str)
                                if index < len(texts):
                                    return_json[key] = texts[index].strip()
                            except ValueError:
                                continue

                if "class_singleObject" in config:
                    for class_name, key in config["class_singleObject"].items():
                        texts = await safely_get_text_content(f".{class_name}")
                        if texts:
                            return_json[key] = texts[0].strip()

            # Label-based extraction
            elif config['category'] == "label":
                label_data = config.get("labelObject", {})
                for label_selector, key in label_data.items():
                    element = target_page.locator(f"text={label_selector}").first
                    if await element.count():
                        return_json[key] = (await element.text_content() or "").strip()

            # XPath-based extraction
            elif config["category"] == "xpath":
                xpath_data = config.get("xpathObject", {})
                for xpath, key in xpath_data.items():
                    element = target_page.locator(f"xpath={xpath}").first
                    if await element.count():
                        return_json[key] = (await element.text_content() or "").strip()

            # Input-based extraction
            elif config["category"] == "input":
                input_data = config.get("inputObject", {})
                for xpath, key in input_data.items():
                    element = target_page.locator(f"xpath={xpath}").first
                    if await element.count():
                        return_json[key] = (await element.input_value() or "").strip()

            else:
                raise ValueError("Invalid category specified")

            return return_json

        finally:
            if browser:
                await browser.close()


async def get_gotoRecreationPage_and_get_html_content(env, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asynchronously navigates recreation.gov and extracts specific content using Playwright.
    
    Args:
        env: Environment configuration object
        config: Dictionary containing:
            - selector: "class" 
            - class: class name to search
            - order: (optional) index for multi-element results
    
    Returns:
        Dictionary with extracted content under 'expected' key
        
    Raises:
        ConnectionError: If browser connection fails
        TimeoutError: If navigation steps timeout
    """
    # Connection parameters
    host = env.vm_ip
    port = env.chromium_port
    server_port = env.server_port
    remote_debugging_url = f"http://{host}:{port}"

    async with async_playwright() as p:
        browser = None
        try:
            # Connect to existing browser
            browser = await p.chromium.connect_over_cdp(remote_debugging_url)
        except Exception:
            # Launch new browser if connection fails
            machine_arch = platform.machine().lower()
            command = ["chromium", "--remote-debugging-port=1337"] if "arm" in machine_arch else ["google-chrome", "--remote-debugging-port=1337"]
            
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"http://{host}:{server_port}/setup/launch",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"command": command, "shell": False})
                )
            
            await asyncio.sleep(5)
            browser = await p.chromium.connect_over_cdp(remote_debugging_url)

        try:
            # Create new browsing context
            context = await browser.new_context()
            page = await context.new_page()
            
            # Step 1: Navigate to main page
            await page.goto("https://www.recreation.gov/", wait_until="networkidle", timeout=30000)
            
            # Step 2: Perform search
            await page.fill("input#hero-search-input", "Albion Basin")
            await page.click("button.nav-search-button")
            print("Search initiated")
            
            # Step 3: Handle search result popup
            async with page.expect_popup() as popup_info:
                await page.click(".search-result-highlight--success")
                print("Popup triggered")
            
            new_page = await popup_info.value
            await new_page.wait_for_load_state("networkidle", timeout=15000)
            print(f"New page title: {await new_page.title()}")
            
            # Step 4: Handle availability selection
            await new_page.click("button.next-available", timeout=10000)
            print("Availability selected")
            
            # Extract content
            result = {"expected": {}}
            if config["selector"] == "class":
                class_name = config["class"]
                
                # Wait for target elements
                await new_page.wait_for_selector(f".{class_name}", state="attached", timeout=10000)
                
                if "order" in config:
                    elements = await new_page.query_selector_all(f".{class_name}")
                    if len(elements) > config["order"]:
                        content = await elements[config["order"]].text_content()
                        result["expected"][class_name] = content.strip() if content else ""
                else:
                    element = await new_page.query_selector(f".{class_name}")
                    if element:
                        content = await element.text_content()
                        result["expected"][class_name] = content.strip() if content else ""
            
            return result
            
        finally:
            # Cleanup resources
            if 'new_page' in locals():
                await new_page.close()
            if 'page' in locals():
                await page.close()
            if 'context' in locals():
                await context.close()
            await browser.close()


def get_active_tab_url_parse(env, config: Dict[str, Any]):
    """
    This function is used to parse the url according to config["parse_keys"].
    config: 
        'parse_keys': must exist,
            a list of keys to extract from the query parameters of the url.
        'replace': optional, 
            a dict, used to replace the original key with the new key.
            ( { "original key": "new key" } )
    """
    active_tab_url = get_active_url_from_accessTree(env, config)
    if active_tab_url is None:
        return None

    # connect to remote Chrome instance
    # parse in a hard-coded way to find the specific info about task
    parsed_url = urlparse(active_tab_url)
    # Extract the query parameters
    query_params = parse_qs(parsed_url.query)
    # Define the keys of interest
    keys_of_interest = [key for key in config["parse_keys"]]
    # Extract the parameters of interest
    extracted_params = {key: query_params.get(key, [''])[0] for key in keys_of_interest}
    if "replace" in config:
        for key in config["replace"].keys():
            # change original key to new key, keep value unchange
            value = extracted_params.pop(key)
            extracted_params[config["replace"][key]] = value
    return extracted_params


def get_url_dashPart(env, config: Dict[str, str]):
    """
    This function is used to extract one of the dash-separated part of the URL.
    config
        'partIndex': must exist,
            the index of the dash-separated part to extract, starting from 0.
        'needDeleteId': optional,
            a boolean, used to indicate whether to delete the "id" part ( an example: "/part-you-want?id=xxx" )
        'returnType': must exist,
            a string, used to indicate the return type, "string" or "json".
    """
    active_tab_url = get_active_url_from_accessTree(env, config)
    if active_tab_url is None:
        return None

    # extract the last dash-separated part of the URL, and delete all the characters after "id"
    dash_part = active_tab_url.split("/")[config["partIndex"]]
    if config["needDeleteId"]:
        dash_part = dash_part.split("?")[0]
    # print("active_tab_title: {}".format(active_tab_info.get('title', 'None')))
    # print("active_tab_url: {}".format(active_tab_info.get('url', 'None')))
    # print("active_tab_content: {}".format(active_tab_info.get('content', 'None')))
    if config["returnType"] == "string":
        return dash_part
    elif config["returnType"] == "json":
        return {config["key"]: dash_part}
