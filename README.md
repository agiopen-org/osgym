# OSGym: Super-Scalable Distributed Data Engine for Generalizable Agents

## Setup
Setup environment:
```bash
conda create -n osgym python=3.10
```

Instal libGL:
```bash
sudo apt-get update
sudo apt-get install libgl1 libglx-mesa0
```

Install required Linux headers:
```bash
sudo apt-get install linux-headers-$(uname -r)
```
Install essential building tools:
```bash
sudo apt-get install python3-dev build-essential
```
Then install the dependencies:
```bash
pip install -r requirements.txt
```

<details>
<summary>Install Docker</summary>

Setup Docker `apt` repository:
```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

Install Docker:
```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Verify Installation:
```bash
sudo docker run hello-world
```

</details>

## Deployment

Launch server:
```bash
./start_workers.sh
```

Clean up server:
```bash
./clean.sh
```

## Benchmarking
Launch server locally:
```bash
./start_workers.sh --local
```

Benchmark speed:
```bash
cd examples
python test_osgym.py
```

## OS Replica State Manager

### Main Endpoints

#### `POST /reset`
- **Description:** Initializes a new environment with a given task configuration.
- **Request Body:**
  ```
  {
    "task_config": { ... },  // Task configuration JSON
    "timeout": 1000          // Timeout in seconds
  }
  ```
- **Response:**
  ```
  {
    "screenshot": "<base64-encoded image>",
    "problem": "<task instruction>",
    "vm_id": <int>
  }
  ```

#### `POST /step`
- **Description:** Executes an action in the environment.
- **Request Body:**
  ```
  {
    "action": "<action string>",
    "vm_id": <int>
  }
  ```
- **Response:**
  ```
  {
    "screenshot": "<base64-encoded image>",
    "is_finish": <bool>,
    "reward": <float>
  }
  ```

#### `POST /shutdown`
- **Description:** Shuts down and releases a VM.
- **Request Body:**
  ```
  {
    "vm_id": <int or 'all'>
  }
  ```
- **Response:**
  ```
  {
    "vm_id": <int or 'all'>
  }
  ```

#### `GET /screenshot`
- **Description:** Gets a screenshot of the current VM state.
- **Query Parameters:**
  - `vmId`: VM ID (integer)
- **Response:**
  ```
  {
    "screenshot": "<base64-encoded image>",
    "vm_id": <int>
  }
  ```

### Example Usage

**Reset an environment:**
```bash
curl -X POST http://localhost:20000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_config": {...}, "timeout": 1000}'
```

**Step in the environment:**
```bash
curl -X POST http://localhost:20000/step \
  -H "Content-Type: application/json" \
  -d '{"action": "<|think_start|><|think_end|><|action_start|>click(100,100)<|action_end|>", "vm_id": 0}'
```

**Shutdown a VM:**
```bash
curl -X POST http://localhost:20000/shutdown \
  -H "Content-Type: application/json" \
  -d '{"vm_id": 0}'
```

### Notes

- **Authentication:** The API restricts access by IP. Set the `OSGYM_ALLOWED_IPS` environment variable to allow your client IPs.
- **VM IDs:** VM IDs are managed by the server. Use the `vm_id` returned by `/reset` for subsequent calls.
- **Screenshots:** All screenshots are returned as base64-encoded PNG images.

## OSGym Workers

OSGym is designed for distributed, scalable RL/data collection. Each worker runs an environment server (FastAPI) on a different port, as specified in `config.yaml`.

### Launching Workers

To start all workers (one per port in `config.yaml`):

```bash
./start_workers.sh
```

- By default, this launches on all interfaces (`0.0.0.0`).  
- For local-only testing, use:
  ```bash
  ./start_workers.sh --local
  ```

Each worker runs a FastAPI server (see API Reference above) and manages its own set of VMs.

### Cleaning Up

To stop all workers and clean up Docker containers and processes:

```bash
./clean.sh
```

## Data Server and MultiTurnDataloader

OSGym provides a data server and dataloader for efficient RL training and experience replay.

### Data Server

The data server (see `examples/data_server.py`) manages:
- **ReplayBuffer**: Stores and samples experience tuples for RL.
- **MultiTurnDataloader**: Handles parallel environment rollout, batching, and preprocessing for RL agents.

#### Key Classes

- **ReplayBuffer**
  - Stores (state, action, reward, next_state, done) tuples.
  - Supports discounted reward calculation and outcome-only storage.
  - Sampling: `ReplayBuffer.sample(batch_size)`

- **MultiTurnDataloader**
  - Launches multiple environment workers in parallel (using multiprocessing).
  - Collects rollouts, preprocesses data (tokenization, multimodal support), and batches for training.
  - Example usage:
    ```python
    from examples.data_server import MultiTurnDataloader
    dataloader = MultiTurnDataloader(
        env_class=YourEnvClass,
        env_configs=[...],  # List of env configs (one per worker)
        tokenizer=your_tokenizer,
        processor=your_processor,  # Optional, for multimodal
        batch_size=8,
        ...
    )
    for batch in dataloader:
        # batch is a dict of tensors ready for model input
        ...
    ```

#### Example: Sampling from the Replay Buffer

```python
batch = dataloader.sample_from_buffer(batch_size=32)
# batch is a dict with keys: input_ids, attention_mask, position_ids, responses, reward, etc.
```

#### Example: Printing Replay Buffer Stats

```python
dataloader.print_stats_in_replay_buffer()
```

## Desktop Environment Server Endpoints (Advanced)

Each worker uses a desktop environment server (Flask, see `desktop_env/server/main.py`) to interact with the OS.  
Some useful endpoints (for advanced users):

| Endpoint                | Method | Description                        |
|-------------------------|--------|------------------------------------|
| `/screenshot`           | GET    | Capture current screen (with cursor) |
| `/terminal`             | GET    | Get terminal output (Linux only)   |
| `/accessibility`        | GET    | Get accessibility tree (A11y)      |
| `/execute`              | POST   | Execute a shell command            |
| `/list_directory`       | POST   | List directory contents            |
| `/file`                 | POST   | Download a file                    |

These are used internally by OSGym, but can be accessed directly for debugging or custom automation.

## Putting It All Together

- **Start workers**: `./start_workers.sh`
- **Interact via API**: Use `/reset`, `/step`, `/shutdown` as described above.
- **Collect data**: Use the `MultiTurnDataloader` and `ReplayBuffer` for RL training.
- **Benchmark**: Run `python examples/test_osgym.py` to test and benchmark the system.

## License
MIT License. Free for research and commercial use.

## Acknowledgement
Some part of the code were borrowed from [OSWorld](https://os-world.github.io/). Thanks for their open-source contribution! 