import os
import sys
import time
import argparse

import numpy as np
from pynput import keyboard

cwd = os.getcwd()
sys.path.append(cwd)

import utils.geom as geom
from simulator import *
from simulator.wrapper import SimRobotEnvWrapper, SimEnvWrapper
from simulator.wrapper import unwrap_delta_action, unwrap_delta_obs
from simulator.render import CV2Renderer
from simulator.recorder import HDF5Recorder


## Define the thread receiving keyboard for debugging
class Keyboard():

    def __init__(self):

        self.single_click_and_hold = False

        self._reset_state = 0
        self._enabled = False

        self._flag_init = False
        self._t_last_click = -1
        self._t_click = -1

        self._succeed = False

        self.translation = np.zeros(3)
        self.rotation = np.zeros(3)
        self._keys_pressed = set()

        # launch a new listener thread to listen to keyboard
        self.thread = keyboard.Listener(on_press=self._on_press,
                                        on_release=self._on_release)
        self.thread.start()

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        # Reset grasp
        self.single_click_and_hold = False

        self._flag_init = False
        self._t_last_click = -1
        self._t_click = -1

    def _on_press(self, key):

        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)

        if key_char == 'e':
            self._t_last_click = -1
            self._t_click = time.time()
            elapsed_time = self._t_click - self._t_last_click
            self._t_last_click = self._t_click
            self.single_click_and_hold = True
        elif key_char == 's':
            self._succeed = True
            print('Recording successful')
        elif key == keyboard.Key.up:
            self._keys_pressed.add('up')
        elif key == keyboard.Key.down:
            self._keys_pressed.add('down')
        elif key == keyboard.Key.left:
            self._keys_pressed.add('left')
        elif key == keyboard.Key.right:
            self._keys_pressed.add('right')
        elif key == keyboard.Key.page_up:
            self._keys_pressed.add('page_up')
        elif key == keyboard.Key.page_down:
            self._keys_pressed.add('page_down')
        elif key_char in ['i', 'k', 'j', 'l', 'u', 'o']:
            self._keys_pressed.add(key_char)

    def _on_release(self, key):

        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)

        if key_char == 'e':
            self.single_click_and_hold = False

        elif key == keyboard.Key.esc or key_char == 'q':
            self._reset_state = 1
            self._enabled = False
            self._reset_internal_state()

        elif key_char == 'r':
            self._reset_state = 1
            self._enabled = True

        elif key == keyboard.Key.up:
            self._keys_pressed.discard('up')
        elif key == keyboard.Key.down:
            self._keys_pressed.discard('down')
        elif key == keyboard.Key.left:
            self._keys_pressed.discard('left')
        elif key == keyboard.Key.right:
            self._keys_pressed.discard('right')
        elif key == keyboard.Key.page_up:
            self._keys_pressed.discard('page_up')
        elif key == keyboard.Key.page_down:
            self._keys_pressed.discard('page_down')
        elif key_char in ['i', 'k', 'j', 'l', 'u', 'o']:
            self._keys_pressed.discard(key_char)

    def _update_motion_commands(self):
        # Reset commands
        self.translation = np.zeros(3)
        self.rotation = np.zeros(3)
        # Translation commands
        if 'up' in self._keys_pressed:
            self.translation[0] += 1  # Positive x
        if 'down' in self._keys_pressed:
            self.translation[0] -= 1  # Negative x
        if 'left' in self._keys_pressed:
            self.translation[1] -= 1  # Negative y
        if 'right' in self._keys_pressed:
            self.translation[1] += 1  # Positive y
        if 'page_up' in self._keys_pressed:
            self.translation[2] += 1  # Positive z
        if 'page_down' in self._keys_pressed:
            self.translation[2] -= 1  # Negative z
        # Rotation commands
        if 'i' in self._keys_pressed:
            self.rotation[0] += 1  # Positive roll
        if 'k' in self._keys_pressed:
            self.rotation[0] -= 1  # Negative roll
        if 'j' in self._keys_pressed:
            self.rotation[1] += 1  # Positive pitch
        if 'l' in self._keys_pressed:
            self.rotation[1] -= 1  # Negative pitch
        if 'u' in self._keys_pressed:
            self.rotation[2] += 1  # Positive yaw
        if 'o' in self._keys_pressed:
            self.rotation[2] -= 1  # Negative yaw

    @property
    def click(self):
        """
        Maps internal states into gripper commands.
        Returns:
            float: Whether we're using single click and hold or not
        """
        if self.single_click_and_hold:
            return 1.0
        return 0

    @property
    def enable(self):
        return self._enabled

    @property
    def succeed(self):
        return self._succeed

    def update(self):
        self._update_motion_commands()


def sim_demo(gui, save_path, task, robot_name="spot", cam_name="upview", gripper_tool=0):

    dir_name = "{}".format(int(time.time()))
    dir_path = os.path.join(save_path, dir_name)

    keyboard_control = Keyboard()

    env_class = ENVS[task]
    setup = SETUPS[robot_name]
    env_config = setup['env_config']
    if gripper_tool:
        env_config = tool_env_config_wrapper(env_config, gripper_name="right_gripper")
        setup['hand'] = setup['hand'].replace("eef", "tool")

    env = env_class(env_config=env_config)
    recorder = HDF5Recorder(config=env.config, sim=env.sim, file_path=dir_path)
    renderer = CV2Renderer(device_id=-1, sim=env.sim, height=900, width=1200, cam_name=cam_name, gui=gui)

    if robot_name == 'abstract':
        env_wrapper = SimEnvWrapper
    else:
        env_wrapper = SimRobotEnvWrapper

    env = env_wrapper(env, setup, angle_mode='rpy',
                        img_mode='gray',
                        unwrap_action_fn=unwrap_delta_action,
                        unwrap_obs_fn=unwrap_delta_obs)

    env.set_renderer(renderer)
    env.set_recorder(recorder)

    raw_ob = env.reset()
    print("Rendering", keyboard_control.enable)

    done = False
    while not keyboard_control.enable:
        pass

    while not done:
        keyboard_control.update()

        translation_step = 0.01  # meters per step
        rotation_step = 0.01     # radians per step

        act_pos = keyboard_control.translation * translation_step
        act_rpy = keyboard_control.rotation * rotation_step
        act_gripper = 1.0 * keyboard_control.click

        action = np.concatenate((act_pos, act_rpy, [act_gripper]))

        _ = env.step(action)

        done = not keyboard_control.enable

    renderer.close()
    if keyboard_control._succeed:
        print("Recording successful")
        recorder.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=int, default=1, help="")
    parser.add_argument("--robot", type=str, default='demo', help="")
    parser.add_argument('--gripper_tool', type=int, default=1)
    parser.add_argument('--task', type=str, default='ladle')
    parser.add_argument('--path', type=str, default='./data/lid')
    parser.add_argument('--hand_link', type=str, default='eef_point')
    parser.add_argument('--gripper_joint', type=str, default='joint_0')
    parser.add_argument(
        "--cam",
        type=str,
        default='demoview',
        help="",
    )
    args = parser.parse_args()

    gui = args.gui
    cam_name = args.cam
    gripper_tool = args.gripper_tool
    path = args.path
    task = args.task
    robot_name = args.robot

    while True:
        sim_demo(save_path=path, gui=gui, cam_name=cam_name, robot_name=robot_name, task=task, gripper_tool=gripper_tool)
