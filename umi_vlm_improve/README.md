# UMI Policy Evaluation and Iterative Rollout Collection

This repository provides a robot-side client for evaluating remote manipulation policies and collecting rollouts with value-model monitoring and vision-language-model (VLM) instruction switching. It includes UMI observation processing, robot and gripper control, camera recording, and replay-buffer storage.

The codebase is adapted from the official [Universal Manipulation Interface (UMI)](https://github.com/real-stanford/universal_manipulation_interface) repository. The hardware-side implementation is primarily adapted from [OneTwoVLA-UMI-Client](https://github.com/Fanqi-Lin/OneTwoVLA-UMI-Client).


## Environment and dependencies

Use a Linux host with access to the robot, gripper, cameras, and a graphical display for OpenCV visualization. The retained UMI environment specification targets Python 3.9 and PyTorch 2.1;

Create the base environment:

```bash
conda env create -f conda_environment.yaml
conda activate umi
```

Additional imports used by the client and VLM code include `openpi_client`, `zerorpc`, `requests`, `Pillow`, `python-dotenv`, and `google-genai`. Install `openpi_client` from the same compatible OpenPI codebase used by your policy server. For example, from its client package directory:

```bash
python -m pip install -e /path/to/openpi/packages/openpi-client
python -m pip install zerorpc requests Pillow python-dotenv google-genai
```

## Hardware and server configuration

1. Copy `example/eval_robots_config.yaml` to your own configuration file. Set the robot and gripper addresses, control latencies, TCP offset, camera configuration, and calibrated transforms for your installation. The included addresses and geometry are examples.
2. For Franka, start the Polymetis stack and configure the `RobotInterface` address in `scripts_real/launch_franka_interface_server.py`. Launch the interface server on the robot-control host:

   ```bash
   python scripts_real/launch_franka_interface_server.py
   ```


3. Configure the gripper connection. The `BimanualUmiEnv` implementation currently instantiates a WSG controller;
4. Start a compatible pi0.5 policy WebSocket server with the checkpoint to evaluate. The client defaults to `localhost:8000`; use `--remote_host` when the policy server runs elsewhere.
5. The value model service must be running for iterative collection. Its default address is `http://localhost:9000/predict_json`. 

## Run policy evaluation

One generic example:

```bash
python umi_client.py \
  --robot_config=/path/to/robot_config.yaml \
  -o /path/to/evaluation_output \
  --frequency 5 -j \
  --temporal_agg -si 1 \
  -ins "Complete the configured manipulation task." \
  -state_horizon 3,15 \
  -action_down_sample_steps 1 \
  -getitem_type necessary \
  --remote_port 8000
```

Replace the instruction with the natural-language task expected by the checkpoint. Here, `-ins` is passed directly to the policy as its prompt.


## Configure iterative rollout collection

The iterative entry point additionally requires:

- A value-model HTTP service. Configure `SERVER` in `vm_inter_func.py`; its current default is `http://localhost:9000/predict_json`.
- A configured VLM client in `vlm_request/gemini_request_genai.py`. Replace the project setting with your own project and provide credentials before import:

  ```bash
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
  ```

  Configure the model and location for your deployment. The current helper uses Vertex AI and initializes its client at import time.
- A task entry in `instr_dict.py`. The outer key selects the task, `'1'` is its complete instruction, other numbered entries specify available skills, and `'s'`, `'f'`, and `'d'` identify terminal outcomes. For example:

  ```python
  predefined_instr_dict = {
      "your_task": {
          "1": "Complete the configured manipulation task.",
          "2": "Execute the first configured skill.",
          "3": "Execute the second configured skill.",
          "s": "The task is successfully completed.",
          "f": "The task is failed, please try again.",
          "d": "The task directly succeed.",
      }
  }
  ```


### Run iterative collection

One generic example, after completing configuration and addressing the implementation limitations below:

```bash
python vm_inter_func.py \
  --robot_config=/path/to/robot_config.yaml \
  -o /path/to/iteration_output \
  --frequency 5 -j \
  --temporal_agg -si 1 \
  -ins "your_task" \
  -state_horizon 3,15 \
  -action_down_sample_steps 1 \
  -getitem_type necessary \
  --remote_port 8000
```

The intended workflow monitors policy rollouts, chooses a repair instruction when progress is insufficient, and returns to the complete task after repair. If you want human to intervene, you can use the code in the comment section and type `F2` to take over the control. 

## Attribution and licensing

- [Universal Manipulation Interface](https://github.com/real-stanford/universal_manipulation_interface): the original codebase and UMI infrastructure.
- [OneTwoVLA-UMI-Client](https://github.com/Fanqi-Lin/OneTwoVLA-UMI-Client): the primary source for the adapted hardware-side implementation.

