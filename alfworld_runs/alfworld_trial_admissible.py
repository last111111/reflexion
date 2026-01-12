"""Adapted from alfworld_trial.py - Admissible模式: 从候选动作中选择"""

import os
import sys
import json
import yaml
import openai
import importlib
import alfworld
import alfworld.agents.environment
from utils_qwen import Model, get_chat, get_completion
from env_history import EnvironmentHistory
import re

from typing import List, Dict, Any, Tuple

FOLDER = './prompts'
PROMPT_FILE = 'alfworld_3prompts_admissible.json'  # admissible模式使用专用prompts
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

def llm_choose_from_admissible(prompt: str, admissible: List[str], model: Model, stop: List[str] = ["\n"]):
    """从候选动作中选择最佳动作"""
    # 构建选择prompt
    instruction = d.get('instruction', '')
    choice_prompt = prompt + "\n\n" + instruction + "\n\nAvailable actions:\n"
    for i, cmd in enumerate(admissible):  # 显示所有候选动作
        choice_prompt += f"{i+1}. {cmd}\n"
    choice_prompt += "\nChoose the best action (reply with action text only):\n>"

    try:
        cur_try = 0
        while cur_try < 6:
            try:
                if model == "text-davinci-003":
                    text = get_completion(prompt=choice_prompt, temperature=cur_try * 0.2, stop_strs=stop)
                else:
                    text = get_chat(prompt=choice_prompt, model=model, temperature=cur_try * 0.2, stop_strs=stop)

                text = text.strip()

                # 匹配策略1: 直接匹配
                if text in admissible:
                    return text

                # 匹配策略2: 提取数字
                num_match = re.search(r'^(\d+)', text)
                if num_match:
                    idx = int(num_match.group(1)) - 1
                    if 0 <= idx < len(admissible):
                        return admissible[idx]

                # 匹配策略3: 模糊匹配
                text_clean = re.sub(r'[^\w\s]', '', text.lower())
                for cmd in admissible:
                    cmd_clean = re.sub(r'[^\w\s]', '', cmd.lower())
                    if text_clean in cmd_clean or cmd_clean in text_clean:
                        return cmd

            except Exception as gen_error:
                print(f"生成尝试 {cur_try+1} 失败: {gen_error}")
                # vLLM 引擎崩溃时，直接返回默认动作
                if "EngineDeadError" in str(gen_error) or "EngineCore" in str(gen_error):
                    print(f"⚠ vLLM引擎崩溃，使用默认第一个动作: {admissible[0]}")
                    return admissible[0]

            cur_try += 1

        return admissible[0]  # 默认返回第一个
    except Exception as e:
        print(f"选择错误: {e}")
        return admissible[0]

def llm(prompt: str, model: Model, stop: List[str] = ["\n"]):
    """标准LLM调用 - 用于generation模式"""
    try:
        cur_try = 0
        while cur_try < 6:
            if model == "text-davinci-003":
                text = get_completion(prompt=prompt, temperature=cur_try * 0.2, stop_strs=stop)
            else:
                text = get_chat(prompt=prompt, model=model, temperature=cur_try * 0.2, stop_strs=stop)
            if len(text.strip()) >= 5:
                return text
            cur_try += 1
        return ""
    except Exception as e:
        print(prompt)
        print(e)
        import sys
        sys.exit(1)

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob

def alfworld_run(env, base_prompt, memory: List[str], to_print=True, ob='', model: Model = "qwen-7b-chat", info=None) -> Tuple[EnvironmentHistory, bool]:
    """运行AlfWorld - Admissible模式 (只从候选动作中选择)"""
    if len(memory) > 3:
        env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])
    else:
        env_history = EnvironmentHistory(base_prompt, ob, memory, [])
    env_history.reset()
    if to_print:
        print(ob)
        sys.stdout.flush()

    cur_step = 0
    current_info = info  # 使用初始info

    while cur_step < 49:
        # 必须有admissible commands，否则报错
        if not current_info or 'admissible_commands' not in current_info or len(current_info['admissible_commands']) == 0:
            raise RuntimeError(f"Admissible模式要求环境必须提供admissible_commands，但在步骤{cur_step}未找到")

        admissible = current_info['admissible_commands'][0]
        # 添加 think 选项到候选列表
        admissible_with_think = ['think: [your thoughts here]'] + admissible
        action = llm_choose_from_admissible(str(env_history) + ">", admissible_with_think, model, stop=['\n'])

        # 如果选择了 think 模板，用 generation 模式生成真实思考内容
        if action == 'think: [your thoughts here]':
            action = llm(str(env_history) + ">", model=model, stop=['\n']).strip()
            # 确保生成的是 think 开头的内容
            if not action.startswith('think:'):
                action = 'think: ' + action

        env_history.add("action", action)

        # 如果选择了 think，不需要执行环境步骤
        if action.startswith('think:'):
            observation = 'OK.'
            done = False
            # current_info 保持不变，不调用 env.step()
        else:
            observation, _, done, current_info = env.step([action])
            observation, done = process_ob(observation[0]), current_info['won'][0]
        env_history.add("observation", observation)
        if to_print:
            print(f'> {action}\n{observation}')
            sys.stdout.flush()
        if done:
            return env_history, True
        elif env_history.check_is_exhausted():
            return env_history, False
        cur_step += 1
    return env_history, False

PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

def run_trial(
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        use_memory: bool,
        model: Model,
    ) -> List[Dict[str, Any]]:
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)

    with open('base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    split = "eval_out_of_distribution"

    env = alfworld.agents.environment.get_environment(config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)

    for z, env_config in enumerate(env_configs):
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])

        print(f"using {name}")

        if env_config["is_success"]:
            num_successes += 1
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            continue

        for i, (k, v) in enumerate(PREFIXES.items()):
            if name.startswith(k):
                base_prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0']
                final_env_history, is_success = alfworld_run(env, base_prompt, env_config["memory"] if use_memory else [], to_print=True, ob=ob, model=model, info=info)

                if is_success:
                    status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
                    env_configs[z]['is_success'] = True
                    num_successes += 1
                    num_additional_successes += 1
                else:
                    status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'

                with open(world_log_path, 'a') as f:
                    f.write(status_str + '\n')

                with open(trial_log_path, 'a') as wf:
                    wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

    env.close()

    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return env_configs
