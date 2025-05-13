import uuid
import requests
import re
import json
from pathlib import Path
import random
from PIL import Image
import io
import base64
import numpy as np

DEBUG = True

def print_debug(msg):
    if DEBUG:
        print(msg)


def pillow_to_jpg_string(img):
    # Save to bytes buffer as JPG
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=80)  # quality=85 balances size and quality
    jpg_bytes = buffer.getvalue()
    # Encode bytes to base64 string
    jpg_string = base64.b64encode(jpg_bytes).decode('utf-8')
    return jpg_string


def jpg_string_to_pillow(jpg_string):
    # Decode base64 string to bytes
    jpg_bytes = base64.b64decode(jpg_string)
    # Convert bytes to PIL Image
    buffer = io.BytesIO(jpg_bytes)
    img = Image.open(buffer)
    return img


def rgb_to_jpg_string(rgb_array):
    """
    Convert RGB array to JPG-encoded base64 string.
    Input: numpy array of shape (height, width, 3) with values 0-255
    Output: base64-encoded string
    """
    # Convert RGB array to PIL Image
    img = Image.fromarray(rgb_array.astype('uint8'), 'RGB')

    # Save to bytes buffer as JPG
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=80)  # quality=85 balances size and quality
    jpg_bytes = buffer.getvalue()
    buffer.close()

    # Encode bytes to base64 string
    jpg_string = base64.b64encode(jpg_bytes).decode('utf-8')
    return jpg_string


def jpg_string_to_rgb(jpg_string):
    """
    Convert JPG-encoded base64 string back to RGB array.
    Input: base64-encoded string
    Output: numpy array of shape (height, width, 3) with values 0-255
    """
    # Decode base64 string to bytes
    jpg_bytes = base64.b64decode(jpg_string)

    # Convert bytes to PIL Image
    buffer = io.BytesIO(jpg_bytes)
    img = Image.open(buffer)

    # Convert to RGB array
    rgb_array = np.array(img)
    buffer.close()

    return rgb_array


def parse_action(action_string):
    pattern = r'(\w+)\((.*?)\)'
    match = re.match(pattern, action_string)
    if not match:
        return None, None
    function_name = match.group(1)
    args_string = match.group(2)
    if not args_string:
        return function_name, []
    args = []
    for arg in re.split(r',\s*', args_string):
        try:
            args.append(int(arg))
        except ValueError:
            args.append(arg)
    return function_name, args


def check_type(inputs, types):
    if len(inputs) != len(types):
        return False
    for i, t in zip(inputs, types):
        if not isinstance(i, t):
            return False
    return True


class OSGymEnvWorker:
    vision_start_token = '<|vision_start|>'
    vision_end_token = '<|vision_end|>'
    think_start_token = '<|think_start|>'
    think_end_token = '<|think_end|>'
    action_start_token = '<|action_start|>'
    action_end_token = '<|action_end|>'
    valid_fn_names = [
        'click', 'left_double', 'right_single', 'drag', 'hotkey', 'type', 'scroll', 'wait', 'call_user', 'finish'
    ]
    vlm_img_w = 1260
    vlm_img_h = 700

    def __init__(self, env_config):
        print_debug("[OSGymEnv] Initializing worker...")
        self.env_config = env_config
        self.server_url = env_config['server_url']
        self.max_step = env_config['max_step']
        self.max_hist = env_config['max_hist']
        self.json_dir = env_config['json_dir']
        self.img_h = env_config['img_h']
        self.img_w = env_config['img_w']
        self.timeout = env_config['timeout']
        self.vm_id = None
        self.uid = None
        self.step_count = 0
        self.done = False
        self.prompt = None

    def reset(self):
        ret, meta_info = self.reset_attempt()
        while ret is None:
            print_debug("[OSGymEnv] Reset failed, retrying...")
            ret, meta_info = self.reset_attempt()
        return ret, meta_info
        
    def reset_attempt(self):
        print_debug("[OSGymEnv] Resetting environment...")
        json_dir = Path(self.json_dir)
        json_files = list(json_dir.rglob('*.json'))
        json_path = random.choice(json_files)
        print_debug(f"[OSGymEnv] Selected JSON file: {json_path}")
        try:
            with json_path.open('r') as f:
                json_data = json.load(f)
            headers = {"Content-Type": "application/json"}
            print_debug(f"[OSGymEnv] Sending reset request to {self.server_url}/reset")
            reset = requests.post(f"{self.server_url}/reset",
                                  headers=headers,
                                  json={
                                      "task_config": json_data,
                                      "timeout": self.timeout
                                  })
            reset.raise_for_status()  # Raise an error for bad responses
            reset_data = reset.json()
            print_debug("[OSGymEnv] Reset successful, received response data")
            jpg_string = reset_data["screenshot"]
            instruction = reset_data["problem"]
            self.vm_id = reset_data["vm_id"]
            print_debug(f"[OSGymEnv] VM ID assigned: {self.vm_id}")

        except requests.RequestException as e:
            print_debug(f"[OSGymEnv] ERROR in reset processing: {e}")
            return None, None

        jpg_string = self.check_and_resize_image(jpg_string)
        self.prompt = {
            'instruction': instruction,
            'steps': [f"{self.vision_start_token}{jpg_string}{self.vision_end_token}"]
        }
        self.uid = str(uuid.uuid4())
        self.step_count = 0
        self.done = False

        obs = self.prompt_to_input_obs(self.prompt)
        ret = {'obs': obs, 'done': False, 'is_init': True}
        meta_info = {'uid': self.uid}
        return ret, meta_info

    def prompt_to_input_obs(self, prompt):
        obs = prompt['instruction']
        for s in prompt['steps']:
            obs = obs + s
        return obs

    def check_and_fix_action(self, action_str):
        print_debug(f"[OSGymEnv ID: {self.vm_id}] Processing action: {action_str}")
        fn_name, args = parse_action(action_str)
        if fn_name is None or fn_name not in self.valid_fn_names:
            return 'wait()', 'wait()'
        if fn_name == 'click':
            if not check_type(args, [int, int]):
                return 'wait()', 'wait()'
            # TODO: check xy or yx
            x = int(args[0] / 1000.0 * self.img_w)
            y = int(args[1] / 1000.0 * self.img_h)
            return f'click({args[0]},{args[1]})', f'click({x},{y})'
        elif fn_name == 'left_double':
            if not check_type(args, [int, int]):
                return 'wait()', 'wait()'
            x = int(args[0] / 1000.0 * self.img_w)
            y = int(args[1] / 1000.0 * self.img_h)
            return f'left_double({args[0]},{args[1]})', f'left_double({x},{y})'
        elif fn_name == 'right_single':
            if not check_type(args, [int, int]):
                return 'wait()', 'wait()'
            x = int(args[0] / 1000.0 * self.img_w)
            y = int(args[1] / 1000.0 * self.img_h)
            return f'right_single({args[0]},{args[1]})', f'right_single({x},{y})'
        elif fn_name == 'drag':
            if not check_type(args, [int, int, int, int]):
                return 'wait()', 'wait()'
            x1 = int(args[0] / 1000.0 * self.img_w)
            y1 = int(args[1] / 1000.0 * self.img_h)
            x2 = int(args[2] / 1000.0 * self.img_w)
            y2 = int(args[3] / 1000.0 * self.img_h)
            return f'drag({args[0]},{args[1]},{args[2]},{args[3]})', f'drag({x1},{y1},{x2},{y2})'
        elif fn_name == 'hotkey':
            args_str = ','.join(args)
            return f'hotkey({args_str})', f'hotkey({args_str})'
        elif fn_name == 'type':
            if not check_type(args, [str]):
                return 'wait()', 'wait()'
            return f'type({args[0]})', f'type({args[0]})'
        elif fn_name == 'scroll':
            if not check_type(args, [int, int, str]):
                return 'wait()', 'wait()'
            x = int(args[0] / 1000.0 * self.img_w)
            y = int(args[1] / 1000.0 * self.img_h)
            if args[2] == 'up':
                d = 50
            elif args[2] == 'down':
                d = -50
            else:
                return 'wait()', 'wait()'
            return f'scroll({args[0]},{args[1]},{args[2]})', f'scroll({x},{y},{d})'
        elif fn_name in ['wait', 'call_user', 'finish']:
            if len(args) != 0:
                return 'wait()', 'wait()'
            return fn_name + '()', fn_name + '()'
        else:
            # this is unlikely because we already considered all cases
            raise ValueError

    def reward_shaping(self, reward):
        return reward

    def check_and_resize_image(self, jpg_string):
        print_debug(f"[OSGymEnv ID: {self.vm_id}] Processing and resizing image...")
        img = jpg_string_to_pillow(jpg_string)
        w, h = img.size
        assert w == self.img_w and h == self.img_h
        img_rs = img.resize((self.vlm_img_w, self.vlm_img_h))
        jpg_string_rs = pillow_to_jpg_string(img_rs)
        return jpg_string_rs

    def step(self, action):
        ret, meta_info = self.step_attempt(action)
        while ret is None:
            print_debug(f"[OSGymEnv ID: {self.vm_id}] Step failed, retrying...")
            ret, meta_info = self.step_attempt(action)
        return ret, meta_info

    def step_attempt(self, action):
        print_debug(f"[OSGymEnv ID: {self.vm_id}] Step {self.step_count + 1} with action: {action}")
        assert not self.done
        assert isinstance(action, str)
        prev_obs = self.prompt_to_input_obs(self.prompt)
        try:
            print_debug(f"[OSGymEnv ID: {self.vm_id}] Parsing action tokens...")
            think_pattern = re.escape(self.think_start_token) + r"(.*?)" + re.escape(self.think_end_token)
            action_pattern = re.escape(self.action_start_token) + r"(.*?)" + re.escape(self.action_end_token)

            think_match = re.search(think_pattern, action)
            action_match = re.search(action_pattern, action)

            think_str = think_match.group(1).strip() if think_match else ''
            action_str = action_match.group(1).strip() if action_match else ''
            print_debug(
                f"[OSGymEnv ID: {self.vm_id}] Parsed action tokens: think_str={think_str}, action_str={action_str}")
        except Exception as e:
            # actually this is highly impossible. in the worse case the try will give empty str not error
            print_debug(f"[OSGymEnv ID: {self.vm_id}] ERROR in action parsing: {e}")
            return None, None

        action_str, action_denormalized = self.check_and_fix_action(action_str)
        print_debug(f"[OSGymEnv ID: {self.vm_id}] Normalized action: {action_str}")
        print_debug(f"[OSGymEnv ID: {self.vm_id}] Denormalized action: {action_denormalized}")

        if self.step_count >= self.max_step:
            print_debug(f"[OSGymEnv ID: {self.vm_id}] Max history reached, forcing finish()")
            action_to_send = 'finish()'
        else:
            action_to_send = action_denormalized
        try:
            print_debug(f"[OSGymEnv ID: {self.vm_id}] Sending step request to {self.server_url}/step")
            step_ret = requests.post(f"{self.server_url}/step", json={"action": action_to_send, "vm_id": self.vm_id})
            step_ret.raise_for_status()  # Raise an error for bad responses
            step_data = step_ret.json()
            print_debug(f"[OSGymEnv ID: {self.vm_id}] Step request successful")

            jpg_string = step_data['screenshot']
            reward = step_data['reward']
            is_finish = step_data['is_finish']
            print_debug(f"[OSGymEnv ID: {self.vm_id}] Step result - reward: {reward}, is_finish: {is_finish}")

            assert isinstance(reward, (int, float))
            assert isinstance(is_finish, bool)
            if action_to_send == 'finish()':
                assert is_finish
            self.done = is_finish
        except Exception as e:
            print_debug(f"[OSGymEnv ID: {self.vm_id}] ERROR in step execution: {e}")
            return None, None

        action_clean = f'{self.think_start_token}{think_str}{self.think_end_token}' + \
                       f'{self.action_start_token}{action_str}{self.action_end_token}'

        jpg_string = self.check_and_resize_image(jpg_string)
        reward = self.reward_shaping(reward)

        self.step_count += 1
        self.prompt['steps'].append(action_clean)
        self.prompt['steps'].append(f"{self.vision_start_token}{jpg_string}{self.vision_end_token}")
        curr_obs = self.prompt_to_input_obs(self.prompt)

        ret = {
            'prev_obs': prev_obs,
            'action': action_clean,
            'obs': curr_obs,
            'reward': reward,
            'done': self.done,
            'is_init': False
        }
        meta_info = {'uid': self.uid}
        print_debug(f"[OSGymEnv ID: {self.vm_id}] Step {self.step_count} completed successfully")
        return ret, meta_info

    def render(self):
        """
        Renders the current state in self.prompt as a sequence of text-image pairs into a single image
        Returns:
            PIL.Image: Combined image showing the instruction and interaction history
        """
        from PIL import Image, ImageDraw, ImageFont
        import re

        # Constants for rendering
        PADDING = 20
        TEXT_HEIGHT = 100
        FONT_SIZE = 20
        CLICK_RADIUS = 5
        CLICK_COLOR = 'red'
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
        except:
            font = ImageFont.load_default()

        # First calculate total height needed
        total_height = PADDING
        total_height += TEXT_HEIGHT  # Initial instruction
        
        for step in self.prompt['steps']:
            if self.vision_start_token in step:
                # Image height + padding
                total_height += self.vlm_img_h + PADDING
            else:
                # Text height + padding
                total_height += TEXT_HEIGHT + PADDING

        # Create blank image
        combined_img = Image.new('RGB', (self.vlm_img_w + 2*PADDING, total_height), 'white')
        draw = ImageDraw.Draw(combined_img)
        
        # Start drawing from top
        y_offset = PADDING
        
        # Draw instruction
        draw.text((PADDING, y_offset), self.prompt['instruction'], font=font, fill='black')
        y_offset += TEXT_HEIGHT + PADDING
        
        # Draw each step
        for step in self.prompt['steps']:
            if self.vision_start_token in step:
                # Extract and paste image
                img_str = step[len(self.vision_start_token):-len(self.vision_end_token)]
                img = jpg_string_to_pillow(img_str)
                combined_img.paste(img, (PADDING, y_offset))
                
                # Check previous step for click coordinates
                if len(self.prompt['steps']) > 1:
                    prev_step = self.prompt['steps'][-2]  # Get the step before the image
                    if '<|action_start|>' in prev_step and '<|action_end|>' in prev_step:
                        action = prev_step.split('<|action_start|>')[1].split('<|action_end|>')[0]
                        # Look for click(x,y) pattern
                        click_match = re.search(r'click\((\d+),(\d+)\)', action)
                        if click_match:
                            # Get normalized coordinates
                            norm_x, norm_y = map(int, click_match.groups())
                            # Denormalize coordinates (assuming coordinates are normalized to 100)
                            x = int((norm_x / 1000.0) * self.vlm_img_w)
                            y = int((norm_y / 1000.0) * self.vlm_img_h)
                            # Draw click point
                            draw.ellipse([
                                PADDING + x - CLICK_RADIUS, y_offset + y - CLICK_RADIUS,
                                PADDING + x + CLICK_RADIUS, y_offset + y + CLICK_RADIUS
                            ], fill=CLICK_COLOR)
                
                y_offset += self.vlm_img_h + PADDING
            else:
                # Draw text (action/thought)
                draw.text((PADDING, y_offset), step, font=font, fill='black')
                y_offset += TEXT_HEIGHT + PADDING
                
        return combined_img