"""
Spoken Dialogue System Exercise  
Dialogue Systems and Machine Learning Group  
Heinrich Heine University Düsseldorf  

Author: Nurul Lubis  
Last update: 15 April 2026 

Description:  
    This script is part of exercise 2 on the spoken dialogue systems course,
    designed to help students understand and implement key concepts in dialogue system development.  

IMPORTANT:
    Follow instructions in the exercise sheet 
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
from convlab.policy.mle import MLEPolicy
from convlab.policy.mle.mle import MLEAbstract
from convlab.nlg.template.simplemultiwoz21 import TemplateNLG
from convlab.dialog_agent import PipelineAgent, BiSession

import random
import numpy as np
import torch
import logging



def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


def interact(seed=20200202, n_dialogues=1000):
    set_seed(seed)

    # go to README.md of each model for more information

    # BERT NLU
    #######################################################################################################
    ## TODO 1 which BERTNLU mode do we need for the system?
    ## TODO 2 plug the provided pre-trained NLU model into the pipeline
    ## Start of your code #################################################################################

    # The system agent needs to interpret user utterances.
    # Per the BERTNLU README, the system should use a model trained for speaker=user.
    # config_file matches the simplemultiwoz21 user config (bert-mini, context window 3).
    # model_file points to the local pre-trained zip.
    sys_nlu = BERTNLU(
        mode='sys',
        config_file='simplemultiwoz21_user_context3.json',
        model_file='pretrained_models/nlu/bert-mini_bertnlu_unified_simplemultiwoz21_user_context3.zip'
    )

    ## End of your code ###################################################################################

    # simple rule DST
    sys_dst = RuleDST()


    # rule policy
    #######################################################################################################
    ## TODO 3 plug the provided pre-trained MLE policy into the pipeline
    ## Start of your code #################################################################################

    # Load the pre-trained MLE policy from the local experiment folder.
    # We build it manually because MLEPolicy doesn't pass manually_add_entity_names
    # from the saved config, causing a state dimension mismatch.
    import json as _json
    import os as _os
    from convlab.policy.vector.vector_binary import VectorBinary as _VB
    from convlab.policy.rlmodule import MultiDiscretePolicy as _MDP
    _cfg_path = _os.path.join('pretrained_models/policy/experiment_2025-02-19-13-02-06', 'configs/config_saved.json')
    with open(_cfg_path, 'r') as _f:
        _cfg = _json.load(_f)
    _vector = _VB(dataset_name=_cfg['args']['dataset_name'],
                  use_masking=_cfg['args']['use_masking'],
                  manually_add_entity_names=False)
    _policy_net = _MDP(_vector.state_dim, _cfg['config']['h_dim'],
                      _vector.da_dim).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    _model_path = _os.path.join('pretrained_models/policy/experiment_2025-02-19-13-02-06',
                                _cfg['config']['load'] + '_mle.pol.mdl')
    _policy_net.load_state_dict(torch.load(_model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    logging.info(f'<<dialog policy>> loaded checkpoint from file: {_model_path}')
    sys_policy = MLEAbstract(_vector, _policy_net)

    ## End of your code ###################################################################################

    # template NLG
    sys_nlg = TemplateNLG(is_user=False)
    
    # assemble
    #######################################################################################################
    ## TODO 4 assemble the pipeline agent using the modules defined above. use the PipelineAgent class
    ## Start of your code #################################################################################

    sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, 'sys')

    ## End of your code ###################################################################################
    
    #######################################################################################################
    ## TODO 5 implement an interaction loop between you (user) and sys_agent (system) via the command line. 
    ## you can use the input() function for convenience.
    ## The maximum length of dialogue is 40 turns. The dialogue ends as soon as user inputs "bye". 
    ## Don't forget to initialize the agent before the dialogue starts.
    ## Start of your code #################################################################################

    sys_agent.init_session()
    print("System: Hello, how can I help you?")
    for turn in range(40):
        user_input = input("You: ")
        if user_input.strip().lower() == "bye":
            print("System: Goodbye!")
            break
        sys_response = sys_agent.response(user_input)
        print(f"System: {sys_response}")

    ## End of your code ###################################################################################

if __name__ == '__main__':

    interact(seed=20200202)
