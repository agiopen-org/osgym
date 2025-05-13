from contextlib import asynccontextmanager
import datetime
import logging
import os
import sys
import threading
import time
import uvicorn
import base64
from desktop_env.desktop_env import DesktopEnv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Any, Dict, List, Union
from fastapi import responses, Query
from fastapi import HTTPException

#  Logger Configs {{{ # 
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler(os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8")
debug_handler = logging.FileHandler(os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8")
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8")

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s")
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
sdebug_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)
#  }}} Logger Configs # 

logger = logging.getLogger("desktopenv.main")
available_vms: List[int] = list(range(2, -1, -1)) # at most 2 VMs for each worker
active_vms: List[int] = []
vm_map: Dict[int, Dict[str, Any]] = {}
vm_lock = threading.Lock()

class ResetRequest(BaseModel):
    task_config: Dict[str, Any]
    timeout: int

class StepRequest(BaseModel):
    action: str
    vm_id: int

class ShutdownRequest(BaseModel):
    vm_id: Union[int, str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout_thread = threading.Thread(target=_check_timeout, daemon=True)
    timeout_thread.start()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/screenshot")
async def screenshot(vm_id: int = Query(..., alias="vmId")):
    try:
        vm_env = _get_vm_env(vm_id)
        obs = vm_env.render()
        logger.info(f"Taking screenshot for VM ID: {vm_id}")
        return {
            "screenshot": base64.b64encode(obs),
            "vm_id": vm_id,
        }
    except Exception as e:
        logger.error(f"Error taking screenshot for VM ID {vm_id}: {e}")
        return responses.JSONResponse(status_code=400, content={"message": str(e)})

@app.post("/reset")
async def reset(request: ResetRequest):
    try:
        logger.info(f"Before reset, closing all VMs")
        _release_vm("all")
        vm_id = _get_available_vm(request.timeout)
        task_config = request.task_config
        logger.info(f"Resetting VM ID: {vm_id} with task config: {task_config}")
        obs = await _get_vm_env(vm_id).reset(task_config)
        logger.info(f"Successfully reset, VM ID: {vm_id} with task config: {task_config}")
        return {
            "screenshot": base64.b64encode(obs["screenshot"]),
            "problem": obs["instruction"],
            "vm_id": vm_id,
        }
    except Exception as e:
        logger.error(f"Error resetting VM: {e}")
        return responses.JSONResponse(status_code=400, content={"message": str(e)})

@app.post("/step")
async def step(request: StepRequest):
    try:
        vm_id = request.vm_id
        vm_env = _get_vm_env(vm_id)
        action = request.action
        if "&" in action:
            action = [act.strip() for act in action.split("&")]
        logger.info(f"Stepping VM ID: {vm_id} with action: {action}")
        obs, reward, done, _ = vm_env.step(action)
        logger.info(f"Stepping VM ID: {vm_id} with action: {action} successfully")
        if done:
            reward = await vm_env.evaluate()
            _release_vm(vm_id)
        return {
            "screenshot": base64.b64encode(obs["screenshot"]),
            "is_finish": done,
            "reward": reward,
        }
    except Exception as e:
        logger.error(f"Error stepping VM ID {vm_id}: {e}")
        return responses.JSONResponse(status_code=400, content={"message": str(e)})

@app.post("/shutdown")
async def shutdown(request: ShutdownRequest):
    try:
        vm_id = request.vm_id
        _release_vm(vm_id)
        return {"vm_id": vm_id}
    except Exception as e:
        logger.error(f"Error shutting down VM ID {vm_id}: {e}")
        return responses.JSONResponse(status_code=400, content={"message": str(e)})

import uuid
from fastapi import Request

allowed_ips_str = os.getenv("OSGYM_ALLOWED_IPS", "")
if allowed_ips_str == "":
    RED = "\033[91m"
    RESET = "\033[0m"
    print(f"{RED}Please set OSGYM_ALLOWED_IPS environment variable. It should contain all your client IPs, split by comma. Example: export OSGYM_ALLOWED_IPS=11.22.33.44,11.44.77.99. Exiting the API app now...{RESET}")
    exit()

ALLOWED_IPS = [ip.strip() for ip in allowed_ips_str.split(",")]
YELLOW = "\033[93m"
RESET = "\033[0m"
print(f"{YELLOW}Allowed IPs: {ALLOWED_IPS}{RESET}")

@app.middleware("http")
async def ip_filter_middleware(request: Request, call_next):
    client_ip = request.client.host
    if client_ip not in ALLOWED_IPS:
        # If IP is not allowed, we return an error immediately
        # and never call the next middleware or route handler
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    response = await call_next(request)
    return response

def _get_available_vm(timeout: int) -> int:
    """
    Get an available VM ID.
    """
    with vm_lock:
        if available_vms.__len__() == 0:
            raise Exception("No available VMs")
        vm_id = available_vms.pop()
        active_vms.append(vm_id)
        if vm_map.get(vm_id) is None:
            vm_map[vm_id] = {
                "env": DesktopEnv (
                    provider_name="docker",
                    action_space="os_gym",
                    require_a11y_tree=False,
                    os_type="Ubuntu",
                ),
                "visited": True,
                "timeout": timeout,
                "lifetime": timeout,
            }
        logger.info(f"Allocated VM ID: {vm_id}")
        return vm_id

def _release_vm(vm_id: Union[int, str]):
    """
    Release a VM ID.
    """
    with vm_lock:
        if isinstance(vm_id, int):
            if vm_id <= 100:
                if vm_map[vm_id]["env"] is not None:
                    vm_map[vm_id]["env"].close()
                active_vms.remove(vm_id)
                available_vms.append(vm_id)
                logger.info(f"Released VM ID: {vm_id}")
            else:
                raise Exception(f"VM ID {vm_id} is not allocated or invalid")
        elif vm_id == "all":
            logger.info(f"active_vms: {active_vms}")
            for vm_id_i in active_vms:
                if vm_map[vm_id_i]["env"] is not None:
                    vm_map[vm_id_i]["env"].close()
                available_vms.append(vm_id_i)
                logger.info(f"Released VM ID: {vm_id_i}")
            active_vms.clear()
            logger.info("Released all VMs")

def _get_vm_env(vm_id: int) -> DesktopEnv:
    """
    Get a VM environment by ID.
    """
    with vm_lock:
        if vm_id not in vm_map or vm_id not in active_vms:
            raise Exception(f"VM ID {vm_id} not available")
        vm_map[vm_id]["visited"] = True
        return vm_map[vm_id]["env"]

def _check_timeout():
    """
    Check if any VM has timed out.
    """
    while True:
        with vm_lock:
            # logger.info("Checking for VM timeouts...")
            active_vms_copy = active_vms.copy()
            for vm_id in active_vms_copy:
                vm_info = vm_map[vm_id]
                if not vm_info["visited"]:
                    vm_info["lifetime"] = max(vm_info["lifetime"] - 60, 0)    # Decrease timeout by 60 seconds
                    logger.info(f"VM ID {vm_id} has not been visited, decreasing lifetime to {vm_info['lifetime']}")
                else:
                    vm_info["visited"] = False
                    vm_info["lifetime"] = vm_info["timeout"]    # Reset lifetime if visited
                if vm_info["lifetime"] == 0:
                    logger.info(f"VM ID {vm_id} has timed out, releasing it")
                    active_vms.remove(vm_id)
                    available_vms.append(vm_id)
        time.sleep(60)



if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")
    # Shut all vms down
    os.system("docker stop $(docker ps -aq) > /dev/null 2>&1")
    os.system("docker rm $(docker ps -aq) > /dev/null 2>&1")
    # Start the FastAPI server
    uvicorn.run(
        app="main:app",
        host="0.0.0.0",
        port=20000
    )
