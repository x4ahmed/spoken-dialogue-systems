"""
Spoken Dialogue System Exercise - Task 4
Dialogue Systems and Machine Learning Group
Heinrich Heine University Düsseldorf

Description:
    This script plugs the RL-trained PPO policy (from Task 3) into the
    dialogue system pipeline from Exercise 2. It replaces the MLE policy
    with the PPO policy loaded from a finished experiment folder.

    Usage:
        python interact_ppo.py --model_path <path_to_finished_experiment>

    If no model_path is given, it defaults to looking in the
    finished_experiments/ directory under convlab/policy/ppo/.
"""

import sys
import os
# Add project root to sys.path so that data.unified_datasets can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Pre-import Database into vector_base namespace so exec/eval pattern works
from data.unified_datasets.simplemultiwoz21.database import Database as _Database
import convlab.policy.vector.vector_base as _vb
_vb.Database = _Database

from convlab.nlu.jointBERT.unified_datasets import BERTNLU
from convlab.dst.rule.simplemultiwoz21 import RuleDST
from convlab.policy.ppo import PPO
from convlab.nlg.template.simplemultiwoz21 import TemplateNLG
from convlab.dialog_agent import PipelineAgent

import random
import numpy as np
import torch
import argparse
import os
import glob


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


def find_latest_experiment():
    """Find the most recent finished experiment folder."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'convlab', 'policy', 'ppo', 'finished_experiments')
    base = os.path.abspath(base)
    if not os.path.exists(base):
        return None
    experiments = sorted(glob.glob(os.path.join(base, '*')))
    if not experiments:
        return None
    latest = experiments[-1]
    # Look for best_ppo.pol.mdl in the save/ subfolder
    save_dir = os.path.join(latest, 'save')
    if os.path.exists(os.path.join(save_dir, 'best_ppo.pol.mdl')):
        return os.path.join(save_dir, 'best_ppo.pol.mdl')
    if os.path.exists(os.path.join(save_dir, 'last_ppo.pol.mdl')):
        return os.path.join(save_dir, 'last_ppo.pol.mdl')
    return None


def interact(seed=20200202, model_path=None):
    set_seed(seed)

    # --- NLU: same pre-trained BERTNLU as Exercise 2 ---
    # Per the BERTNLU README, the system should use a model trained for speaker=user.
    sys_nlu = BERTNLU(
        mode='sys',
        config_file='simplemultiwoz21_user_context3.json',
        model_file='pretrained_models/nlu/bert-mini_bertnlu_unified_simplemultiwoz21_user_context3.zip'
    )

    # --- DST: same rule-based DST ---
    sys_dst = RuleDST()

    # --- Policy: PPO trained with RL (Task 3) ---
    # Determine the model path
    if model_path is None:
        model_path = find_latest_experiment()

    if model_path is None:
        print("No trained PPO model found. Please run Task 3 training first,")
        print("or provide --model_path <path_to_model>")
        print("Example: python interact_ppo.py --model_path finished_experiments/experiment_xxx/save/best")
        return

    print(f"Loading PPO policy from: {model_path}")

    # Try to read the ppo_config from the experiment's saved config
    import json as _json
    ppo_config_file = 'ppo_config.json'
    config_dir = os.path.join(os.path.dirname(model_path), '..', 'configs')
    config_path = os.path.join(config_dir, 'config_saved.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as _f:
            _cfg = _json.load(_f)
        ppo_config_file = _cfg.get('config', {}).get('model', {}).get('ppo_config', 'ppo_config.json')

    # Create PPO policy in inference mode with the correct config
    sys_policy = PPO(is_train=False,
                    seed=seed,
                    dataset_name='simplemultiwoz21',
                    use_masking=False,
                    manually_add_entity_names=False,
                    config_file=ppo_config_file)

    # Load the trained PPO weights
    sys_policy.load_from_pretrained(model_path)
    print("PPO policy loaded successfully.")

    # --- NLG: same template NLG ---
    sys_nlg = TemplateNLG(is_user=False)

    # --- Assemble the pipeline ---
    sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, 'sys')

    # --- Interaction loop ---
    sys_agent.init_session()
    print("\n" + "=" * 60)
    print("  PPO Dialogue System - Task 4")
    print("  Type 'bye' to end the conversation.")
    print("=" * 60)
    print("\nSystem: Hello, how can I help you today?")

    for turn in range(40):
        user_input = input("You: ").strip()
        if user_input.lower() == "bye":
            print("System: Goodbye! Have a nice day.")
            break
        if not user_input:
            continue
        sys_response = sys_agent.response(user_input)
        print(f"System: {sys_response}")

    print(f"\nDialogue ended after {turn + 1} turns.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PPO Dialogue Pipeline (Task 4)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the trained PPO model (without .pol.mdl extension). "
                             "If omitted, uses the latest finished experiment.")
    parser.add_argument("--seed", type=int, default=20200202,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    interact(seed=args.seed, model_path=args.model_path)