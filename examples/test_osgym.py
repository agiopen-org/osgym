import requests
import multiprocessing
import time
import random
from osgym import OSGymEnvWorker
from filelock import Timeout, FileLock

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RED = '\033[91m'
RESET = '\033[0m'

# Create shared variables using Manager
def run_env_worker(worker_id, port, results_dict):
    print(f"{BLUE}Worker port: {port} starting...")
    env_config = {
        'server_url': f'http://localhost:{port}',
        'json_dir': './env_configs/libreoffice_calc',
        'img_h': 1080,
        'img_w': 1920,
        'max_step': 100,
        'max_hist': 10,
        'timeout': 1000
    }
    
    print(f"{BLUE}Worker {worker_id} starting...{RESET}")
    env = OSGymEnvWorker(env_config)
    
    # Initialize local counters
    worker_reset_time = 0
    worker_reset_count = 0
    worker_step_time = 0
    worker_step_count = 0
    worker_finish_time = 0
    worker_finish_count = 0
    n_steps = 100000
    wait_time_range = (10, 40)
    
    try:
        # Test basic functionality
        start_time = time.time()
        obs, meta_info = env.reset()
        reset_time = time.time() - start_time
        worker_reset_time += reset_time
        worker_reset_count += 1
        assert obs is not None
        assert meta_info is not None
        print(f"{YELLOW}Worker {worker_id} reset successful in {reset_time:.2f}s{RESET}")
        
        for i in range(n_steps):
            # Perform a simple action
            start_time = time.time()
            action = '<|think_start|><|think_end|><|action_start|>click(100,100)<|action_end|>'
            ret, meta_info = env.step(action)
            step1_time = time.time() - start_time
            worker_step_time += step1_time
            worker_step_count += 1
            for k, v in ret.items():
                if k not in ['obs', 'prev_obs']:
                    print(f"Worker {worker_id} {k}: {v}")
            assert ret is not None
            assert meta_info is not None
            print(f"{YELLOW}Worker {worker_id} click step successful in {step1_time:.2f}s{RESET}")
            time.sleep(random.uniform(wait_time_range[0], wait_time_range[1]))

        start_time = time.time()
        action = '<|think_start|><|think_end|><|action_start|>finish()<|action_end|>'
        ret, meta_info = env.step(action)
        step2_time = time.time() - start_time
        worker_finish_time += step2_time
        worker_finish_count += 1

        for k, v in ret.items():
            if k not in ['obs', 'prev_obs']:
                print(f"Worker {worker_id} {k}: {v}")
        assert ret is not None
        assert meta_info is not None
        assert ret['done']
        print(f"{GREEN}Worker {worker_id} finish successful in {step2_time:.2f}s{RESET}")
        
        # Keep the worker alive for a while to test stability
        time.sleep(5)
        
        # Store results in the shared dictionary
        results_dict[worker_id] = {
            'reset_time': worker_reset_time,
            'reset_count': worker_reset_count,
            'step_time': worker_step_time,
            'step_count': worker_step_count,
            'finish_time': worker_finish_time,
            'finish_count': worker_finish_count
        }
        
    except Exception as e:
        print(f"{RED}Worker {worker_id} encountered error {e}{RESET}")
        results_dict[worker_id] = {
            'error': str(e),
            'reset_time': worker_reset_time,
            'reset_count': worker_reset_count,
            'step_time': worker_step_time,
            'step_count': worker_step_count,
            'finish_time': worker_finish_time,
            'finish_count': worker_finish_count
        }

if __name__ == '__main__':
    import yaml
    from multiprocessing import Manager, Barrier
    
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    num_workers = len(config['ports'])
    processes = []
    
    # Create a Manager to share data between processes
    manager = Manager()
    results_dict = manager.dict()
    
    print(f"{BLUE}Starting {num_workers} environment workers...{RESET}")
    
    # Create and start processes
    for i in range(num_workers):
        p = multiprocessing.Process(target=run_env_worker, args=(i, config['ports'][i], results_dict))
        processes.append(p)
        p.start()
        # Small delay between starting workers to avoid overwhelming the server
        time.sleep(0.5)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Aggregate results
    reset_time_total = 0
    reset_count = 0
    step_time_total = 0
    step_count = 0
    finish_time_total = 0
    finish_count = 0
    
    for worker_id, result in results_dict.items():
        if 'error' not in result:
            reset_time_total += result['reset_time']
            reset_count += result['reset_count']
            step_time_total += result['step_time']
            step_count += result['step_count']
            finish_time_total += result['finish_time']
            finish_count += result['finish_count']
    
    print(f"{GREEN}All workers completed{RESET}")
    if reset_count > 0:
        print(f"{BLUE}Average reset time: {reset_time_total / reset_count:.2f}s{RESET}")
    if step_count > 0:
        print(f"{BLUE}Average step time: {step_time_total / step_count:.2f}s{RESET}")
    if finish_count > 0:
        print(f"{BLUE}Average finish time: {finish_time_total / finish_count:.2f}s{RESET}")
