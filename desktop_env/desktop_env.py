from __future__ import annotations

import logging
import os
import re
import time
import asyncio
from typing import Callable, Any, Optional, Tuple
from typing import List, Dict, Union

import gymnasium as gym

from desktop_env.controllers.python import PythonController
from desktop_env.controllers.setup import SetupController
from desktop_env.evaluators import metrics, getters
from desktop_env.providers import create_vm_manager_and_provider

logger = logging.getLogger("desktopenv.env")

Metric = Callable[[Any, Any], float]
Getter = Callable[[gym.Env, Dict[str, Any]], Any]


class DesktopEnv(gym.Env):
    """
    DesktopEnv with OpenAI Gym interface. It provides a desktop environment for setting and evaluating desktop automation tasks.
    """

    def __init__(
            self,
            provider_name: str = "docker",
            region: str = None,
            path_to_vm: str = None,
            snapshot_name: str = "init_state",
            action_space: str = "os_gym",
            cache_dir: str = "cache",
            screen_size: Tuple[int] = (1920, 1080),
            headless: bool = False,
            require_a11y_tree: bool = False,
            require_terminal: bool = False,
            os_type: str = "Ubuntu",
    ):
        """
        Args:
            provider_name (str): virtualization provider name, default to "vmware"
            region (str): the region for allocate machines, work for cloud services, default to  "us-east-1"
            path_to_vm (str): path to .vmx file
            snapshot_name (str): snapshot name to revert to, default to "init_state"
            action_space (str): "computer_13" | "pyautogui"
            cache_dir (str): cache directory to cache task-related stuffs like
              reference file for evaluation
            screen_size (Tuple[int]): screen size of the VM
            headless (bool): whether to run the VM in headless mode
            require_a11y_tree (bool): whether to require accessibility tree
            require_terminal (bool): whether to require terminal output
        """
        # Initialize VM manager and vitualization provider
        self.region = region

        # Default
        self.server_port = 5000
        self.chromium_port = 9222
        self.vnc_port = 8006
        self.vlc_port = 8080
        self.manager, self.provider = create_vm_manager_and_provider(provider_name, region)
        self.controller = None
        self.os_type = os_type

        # Initialize environment variables
        if path_to_vm:
            self.path_to_vm = os.path.abspath(os.path.expandvars(os.path.expanduser(path_to_vm))) \
                if provider_name in {"vmware", "virtualbox"} else path_to_vm
        else:
            self.path_to_vm = self.manager.get_vm_path(self.os_type, region)

        self.snapshot_name = snapshot_name
        self.cache_dir_base: str = cache_dir
        # todo: add the logic to get the screen size from the VM
        self.headless = headless
        self.require_a11y_tree = require_a11y_tree
        self.require_terminal = require_terminal

        # Initialize emulator and controller
        if provider_name != "docker": # Check if this is applicable to other VM providers
            logger.info("Initializing...")
            self._start_emulator()

        # mode: human or machine
        self.instruction = None
        assert action_space in ["computer_13", "pyautogui", "os_gym"]
        self.action_space = action_space  # todo: refactor it to the ActType

        # episodic stuffs, like counters, will be updated or reset
        # when calling self.reset()
        self._traj_no: int = -1
        self._step_no: int = 0
        self.action_history: List[Dict[str, any]] = []

    def _start_emulator(self):
        # Power on the virtual machine
        self.provider.start_emulator(self.path_to_vm, self.headless, self.os_type)

        # Get the ip from the virtual machine, and setup the controller
        vm_ip_ports = self.provider.get_ip_address(self.path_to_vm).split(':')
        self.vm_ip = vm_ip_ports[0]
        if len(vm_ip_ports) > 1:
            self.server_port = int(vm_ip_ports[1])
            self.chromium_port = int(vm_ip_ports[2])
            self.vnc_port = int(vm_ip_ports[3])
            self.vlc_port = int(vm_ip_ports[4])
        self.controller = PythonController(vm_ip=self.vm_ip, server_port=self.server_port)
        self.setup_controller = SetupController(vm_ip=self.vm_ip, server_port=self.server_port, chromium_port=self.chromium_port, vlc_port=self.vlc_port, cache_dir=self.cache_dir_base)

    def _revert_to_snapshot(self):
        # Revert to certain snapshot of the virtual machine, and refresh the path to vm and ip of vm
        # due to the fact it could be changed when implemented by cloud services
        path_to_vm = self.provider.revert_to_snapshot(self.path_to_vm, self.snapshot_name)
        if path_to_vm and not path_to_vm == self.path_to_vm:
            # path_to_vm has to be a new path
            self.manager.delete_vm(self.path_to_vm, self.region)
            self.manager.add_vm(path_to_vm, self.region)
            self.manager.occupy_vm(path_to_vm, os.getpid(), self.region)
            self.path_to_vm = path_to_vm

    def _save_state(self, snapshot_name=None):
        # Save the current virtual machine state to a certain snapshot name
        self.provider.save_state(self.path_to_vm, snapshot_name)

    def close(self):
        # Close (release) the virtual machine
        self.provider.stop_emulator(self.path_to_vm)

    async def reset(self, task_config: Optional[Dict[str, Any]] = None, seed=None, options=None) -> Dict[str, Any]:
        # Reset to certain task in OSWorld
        logger.info("Resetting environment...")
        logger.info("Switching task...")
        logger.info("Setting counters...")
        self._traj_no += 1
        self._step_no = 0
        self.action_history.clear()

        logger.info("Reverting to snapshot to {}...".format(self.snapshot_name))
        self._revert_to_snapshot()
        logger.info("Starting emulator...")
        self._start_emulator()
        logger.info("Emulator started.")

        if task_config is not None:
            self._set_task_info(task_config)
            self.setup_controller.reset_cache_dir(self.cache_dir)
            logger.info("Setting up environment...")
            await self.setup_controller.setup(self.config)
            logger.info("Environment setup complete.")

        time.sleep(5)
        observation = self._get_obs()
        return observation

    def _get_obs(self):
        # We provide screenshot, accessibility_tree (optional), terminal (optional), and instruction.
        # can be customized and scaled
        return {
            "screenshot": self.controller.get_screenshot(),
            "accessibility_tree": self.controller.get_accessibility_tree() if self.require_a11y_tree else None,
            "terminal": self.controller.get_terminal_output() if self.require_terminal else None,
            "instruction": self.instruction
        }

    @property
    def vm_platform(self):
        return self.controller.get_vm_platform()

    @property
    def vm_screen_size(self):
        return self.controller.get_vm_screen_size()

    def _get_unified_getter(self, type_name: str):
        func = getattr(getters, f"get_{type_name}", None)
        if type_name in [
            "info_from_website",
            "page_info",
            "open_tabs_info",
            "active_tab_info",
            "pdf_from_url",
            "chrome_saved_address",
            "number_of_search_results",
            "active_tab_html_parse",
            "gotoRecreationPage_and_get_html_content",
        ]:
            return func
        async def async_wrapper(*args, **kwargs):
            return await asyncio.to_thread(func, *args, **kwargs)
        return async_wrapper

    def _set_task_info(self, task_config: Dict[str, Any]):
        self.task_id: str = task_config["id"]
        self.cache_dir: str = os.path.join(self.cache_dir_base, self.task_id)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.instruction = task_config["instruction"]
        self.config = task_config["config"] if "config" in task_config else []

        # evaluator dict
        # func -> metric function string, or list of metric function strings
        # conj -> conjunction of multiple metrics if func is a list with length > 1, "and"/"or"
        # result -> result getter config, or list of result getter configs
        # expected (optional) -> expected getter config, or list of expected getter configs
        # options (optional) -> metric options, or list of metric options
        # if func is a str list, then result, expected (if exists), options (if exists) should also be lists of the same length
        # even if one of the metrics does not need expected or options field, it should be included in the list with None
        self.evaluator = task_config["evaluator"]
        self.metric: Metric = [getattr(metrics, func) for func in self.evaluator["func"]] \
            if isinstance(self.evaluator["func"], list) \
            else getattr(metrics, self.evaluator["func"])
        self.metric_conj: str = self.evaluator.get("conj", "and")  # take conjunction of multiple metrics
        if "result" in self.evaluator and len(self.evaluator["result"]) > 0:
            # logger.info("Result getter: %s", self.evaluator["result"])
            self.result_getter: Getter = [self._get_unified_getter(res["type"]) for res in
                                          self.evaluator["result"]] \
                if isinstance(self.evaluator["result"], list) \
                else self._get_unified_getter(self.evaluator["result"]["type"])
        else:
            self.result_getter = [None] * len(self.metric) \
                if isinstance(self.metric, list) \
                else None

        if "expected" in self.evaluator and len(self.evaluator["expected"]) > 0:
            # logger.info("Expected getter: %s", self.evaluator["expected"])
            self.expected_getter: Getter = [self._get_unified_getter(exp["type"]) if exp else None for exp in
                                           self.evaluator["expected"]] \
                if isinstance(self.evaluator["expected"], list) \
                else self._get_unified_getter(self.evaluator["expected"]["type"])
        else:
            self.expected_getter = [None] * len(self.metric) \
                if isinstance(self.metric, list) \
                else None
        self.metric_options: Union[List[Dict[str, Any]], Dict[str, Any]] = [opt if opt else {} for opt in
                                                                            self.evaluator["options"]] \
            if isinstance(self.evaluator.get("options", {}), list) \
            else self.evaluator["options"] \
            if "options" in self.evaluator \
            else [{}] * len(self.metric) \
            if isinstance(self.metric, list) \
            else {}

        assert (not isinstance(self.evaluator["func"], list)
                or (len(self.metric) == len(self.result_getter) == len(self.expected_getter) == len(
                    self.metric_options)))

    def step(self, action, pause=2, max_step=100):
        # Apply action form
        if self.action_space == "os_gym":
            action = self._apply_action_form(action)
        
        self._step_no += 1
        self.action_history.append(action)

        reward = 0  # todo: Define reward calculation for each example
        done = False  # todo: Define episode termination condition for each example
        info = {}

        # handle the special actions
        if action in ['WAIT', 'FAIL', 'DONE'] or (type(action) == dict and action['action_type'] in ['WAIT', 'FAIL', 'DONE']):
            if action == 'WAIT':
                time.sleep(pause)
            elif action == 'FAIL':
                done = True
                info = {"fail": True}
            elif action == 'DONE':
                done = True
                info = {"done": True}

        # Restrict max step
        if self._step_no >= max_step:
            done = True
            info = {"max_step": True}

        if self.action_space == "computer_13" or self.action_space == "os_gym":
            if isinstance(action, List):
                # action is a list of actions
                for act in action:
                    self.controller.execute_action(act)
            else:
                # the set of all possible actions defined in the action representation
                self.controller.execute_action(action)
        elif self.action_space == "pyautogui":
            if action in ['WAIT', 'FAIL', 'DONE']:
                self.controller.execute_action(action)
            else:
                # the set of all possible python commands insides `pyautogui`
                self.controller.execute_python_command(action)

        time.sleep(pause)
        observation = self._get_obs()

        return observation, reward, done, info

    async def evaluate(self):
        """
        Evaluate whether the task is successfully completed.
        """

        await self.setup_controller.setup(self.evaluator.get("postconfig", []))

        if self.evaluator['func'] == "infeasible":
            if len(self.action_history) > 0 and self.action_history[-1] == "FAIL":
                return 1
            else:
                return 0
        else:
            if len(self.action_history) > 0 and self.action_history[-1] == "FAIL":
                return 0

        if type(self.metric) == list:
            # Multiple metrics to evaluate whether the task is successfully completed
            results = []
            assert len(self.metric) == len(self.result_getter), "The number of metrics and result getters must be the same"
            if "expected" in self.evaluator:
                assert len(self.metric) == len(self.expected_getter), "The number of metrics and expected getters must be the same"
            for idx, metric in enumerate(self.metric):
                try:
                    config = self.evaluator["result"][idx]
                    result_state = await self.result_getter[idx](self, config)
                except FileNotFoundError:
                    logger.error("File not found!")
                    if self.metric_conj == 'and':
                        return 0

                if "expected" in self.evaluator and self.expected_getter and self.evaluator["expected"]:
                    expected_state = await self.expected_getter[idx](self, self.evaluator["expected"][idx])
                    metric: int = metric(result_state, expected_state, **self.metric_options[idx])
                else:
                    metric: int = metric(result_state, **self.metric_options[idx])

                if self.metric_conj == 'and' and float(metric) == 0.0:
                    return 0
                elif self.metric_conj == 'or' and float(metric) == 1.0:
                    return 1
                else:
                    results.append(metric)

            return sum(results) / len(results) if self.metric_conj == 'and' else max(results)
        else:
            # Single metric to evaluate whether the task is successfully completed
            try:
                result_state = await self.result_getter(self, self.evaluator["result"])
            except FileNotFoundError:
                logger.error("File not found!")
                return 0

            if "expected" in self.evaluator and self.expected_getter and self.evaluator["expected"]:
                expected_state = await self.expected_getter(self, self.evaluator["expected"])
                metric: float = self.metric(result_state, expected_state, **self.metric_options)
            else:
                metric: float = self.metric(result_state, **self.metric_options)

        return metric

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.controller.get_screenshot()
        else:
            raise ValueError('Unsupported render mode: {}'.format(mode))

    def _apply_action_form(self, action):
        """
        Apply action form to the action.
        Args:
            action: action to be applied
        Returns:
            action: action after applying the action form
        """
        if isinstance(action, str):
            pattern = r"^(\w+)\(([^)]*)\)$"
            match = re.match(pattern, action)
            if match:
                action = match.group(1)
                params_str = match.group(2)
                parameters = [arg.strip() for arg in params_str.split(",")] if params_str else []
                assert action in ["click", "left_double", "right_single", "drag", "hotkey", "type", "scroll", "wait", "call_user", "finish", "move", "press"], f"Invalid action: {action}"
                if action == "click":
                    return {
                        "action_type": "CLICK", 
                        "parameters": {
                            "x": int(parameters[0]),
                            "y": int(parameters[1]),
                        }
                    }
                elif action == "left_double":
                    return {
                        "action_type": "CLICK", 
                        "parameters": {
                            "x": int(parameters[0]),
                            "y": int(parameters[1]),
                            "num_clicks": 2,
                        }
                    }
                elif action == "right_single":
                    return {
                        "action_type": "RIGHT_CLICK", 
                        "parameters": {
                            "x": int(parameters[0]),
                            "y": int(parameters[1]),
                        }
                    }
                elif action == "drag":
                    return [
                    {
                        "action_type": "MOVE_TO",
                        "parameters": {
                            "x": int(parameters[0]),
                            "y": int(parameters[1]),
                        }
                    },
                    {
                        "action_type": "DRAG_TO", 
                        "parameters": {
                            "x": int(parameters[2]),
                            "y": int(parameters[3]),
                        }
                    }]
                elif action == "hotkey":
                    return {
                        "action_type": "HOTKEY", 
                        "parameters": {
                            "keys": parameters,
                        }
                    }
                elif action == "type":
                    return {
                        "action_type": "TYPING", 
                        "parameters": {
                            "text": params_str,
                        }
                    }
                elif action == "scroll":
                    return [
                    {
                        "action_type": "MOVE_TO",
                        "parameters": {
                            "x": int(parameters[0]),
                            "y": int(parameters[1]),
                        }
                    },
                    {
                        "action_type": "SCROLL", 
                        "parameters": {
                            "dy": int(parameters[2]),
                        }
                    }]
                elif action == "move":
                    return {
                        "action_type": "MOVE_TO", 
                        "parameters": {
                            "x": int(parameters[0]),
                            "y": int(parameters[1]),
                        }
                    }
                elif action == "press":
                    return {
                        "action_type": "PRESS", 
                        "parameters": {
                            "key": parameters[0],
                        }
                    }
                elif action == "wait":
                    return "WAIT"
                elif action == "call_user":
                    return "FAIL"
                elif action == "finish":
                    return "DONE"
            else:
                raise ValueError(f"Invalid action format: {action}")
        elif isinstance(action, list):
            # action is a list of actions
            return [self._apply_action_form(act) for act in action]
        else:
            raise ValueError(f"Invalid action type: {type(action)}")
