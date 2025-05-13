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