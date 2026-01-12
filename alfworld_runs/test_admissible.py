"""快速测试Admissible模式"""

import os
import sys
import yaml
import alfworld
import alfworld.agents.environment

# 设置环境变量
os.environ['ALFWORLD_DATA'] = os.path.expandvars('$ALFWORLD_DATA')

def test_admissible_mode():
    print("="*60)
    print("测试 Admissible 模式")
    print("="*60)

    # 1. 加载配置
    print("\n[1/4] 加载配置...")
    with open('base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✓ 配置加载成功")

    # 2. 创建环境
    print("\n[2/4] 创建环境...")
    env_type = config['env']['type']
    env = alfworld.agents.environment.get_environment(env_type)(config, train_eval='eval_out_of_distribution')
    env = env.init_env(batch_size=1)
    print(f"✓ 环境创建成功: {env_type}")

    # 3. 重置环境
    print("\n[3/4] 重置环境...")
    ob, info = env.reset()
    ob_text = '\n'.join(ob[0].split('\n\n')[1:])
    name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
    print(f"✓ 任务: {name}")
    print(f"  初始观察: {ob_text[:100]}...")

    # 4. 检查admissible commands
    print("\n[4/4] 检查 admissible commands...")
    if 'admissible_commands' in info:
        admissible = info['admissible_commands'][0]
        print(f"✓ 找到 {len(admissible)} 个候选动作")
        print(f"\n  前5个动作:")
        for i, cmd in enumerate(admissible[:5]):
            print(f"    {i+1}. {cmd}")

        # 测试导入
        print("\n[额外] 测试导入 alfworld_trial_admissible...")
        try:
            from alfworld_trial_admissible import llm_choose_from_admissible, alfworld_run
            print("✓ alfworld_trial_admissible.py 导入成功")
            print(f"  - llm_choose_from_admissible: {llm_choose_from_admissible.__doc__.strip()}")
            print(f"  - alfworld_run: {alfworld_run.__doc__.strip()}")
        except Exception as e:
            print(f"✗ 导入失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ 没有找到 admissible_commands")
        print("  这个环境可能不支持admissible模式")

    env.close()

    print("\n" + "="*60)
    print("✓ 测试完成")
    print("="*60)
    print("\n下一步:")
    print("  1. 确认上面所有测试都通过 (✓)")
    print("  2. 在notebook的cell-29中修改导入:")
    print("     from alfworld_trial_admissible import run_trial")
    print("  3. 运行实验")

if __name__ == "__main__":
    test_admissible_mode()
