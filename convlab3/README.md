# ConvLab-3 for Spoken Dialogue System Course SoSe 2026

![PyPI](https://img.shields.io/pypi/v/convlab) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/convlab) ![GitHub](https://img.shields.io/github/license/ConvLab/ConvLab-3)

**ConvLab-3** is a flexible dialog system platform based on a **unified data format** for task-oriented dialog (TOD) datasets. This repository hosts a significantly reduced version of the toolkit for the purpose of exercises accompanying the SDS lectures. The unified format serves as the adapter between TOD datasets and models: datasets are first transformed to the unified format and then loaded by models. In this way, the cost of adapting $M$ models to $N$ datasets is reduced from $M\times N$ to $M+N$. While retaining all features of [ConvLab-2](https://github.com/thu-coai/ConvLab-2),  ConvLab-3 greatly enlarges supported datasets and models thanks to the unified format, and enhances the utility of reinforcement learning (RL) toolkit for dialog policy module. For typical usage, see our [paper](http://arxiv.org/abs/2211.17148). Datasets and Trained models are also available on [Hugging Face Hub](https://huggingface.co/ConvLab).

- [Installation](#installation)
- [Unified Datasets](#unified-datasets)
- [Code Structure](#code-structure)
- [Team](#team)
- [Citing](#citing)
- [License](#license)

## Installation

Clone the repository:

```bash
git clone https://git.hhu.de/sds/convlab3
```

Install ConvLab-3 via pip:

```bash
cd convLab3
pip install -e .
```

## Unified Datasets

Current datasets in unified data format: (DA-U/DA-S stands for user/system dialog acts)

| Dataset       | Dialogs | Goal               | DA-U               | DA-S               | State              | API result         | DataBase           |
| ------------- | ------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| Camrest       | 676     | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    | :white_check_mark: |
| WOZ 2.0       | 1200    |                    | :white_check_mark: |                    | :white_check_mark: |                    |                    |
| KVRET         | 3030    |                    | :white_check_mark: |                    | :white_check_mark: | :white_check_mark: |                    |
| DailyDialog   | 13118   |                    | :white_check_mark: |                    |                    |                    |                    |
| Taskmaster-1  | 13175   |                    | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    |                    |
| Taskmaster-2  | 17303   |                    | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    |                    |
| MultiWOZ 2.1  | 10438   | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    | :white_check_mark: |
| Schema-Guided | 22825   |                    | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    |
| MetaLWOZ      | 40203   | :white_check_mark: |                    |                    |                    |                    |                    |
| CrossWOZ (zh) | 6012    | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Taskmaster-3  | 23757   |                    | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    |

Unified datasets are available under `data/unified_datasets` directory as well as [Hugging Face Hub](https://huggingface.co/ConvLab). We will continue adding more datasets listed in [this issue](https://github.com/ConvLab/ConvLab-3/issues/11). If you want to add a listed/custom dataset to ConvLab-3, you can create an issue for discussion and then create pull-request. We will list you as the [contributors](#Team) and highly appreciate your contributions!

For the exercises we will be using Simple MultiWOZ 2.1 which contains only the hotel and restaurant domains. 

## Code Structure

```bash
.
├── convlab                             # Source code, installed in pypi package
│   ├── dialog_agent                    # Interface for dialog agent and session
│   ├── nlu                             # NLU models, interface, and evaluation script
│   │   └── jointBERT                   # BERTNLU
│   │
│   ├── dst                             # DST models, interface, and evaluation script
│   │   ├── rule                        # RuleDST
│   │   └── trippy                      # TripPy
│   │
│   ├── policy                          # Policy models, interface, and RL toolkit
│   │   ├── vector                      # vectorizer class
│   │   ├── plot_results                # RL plotting tool
│   │   ├── mle                         # MLE (imitation learning) policy
│   │   ├── ppo                         # Proximal Policy Optimization
│   │   └── rule                        # Rule policies and rule-based user simulators 
│   │
│   ├── nlg                             # NLG models, interface, and evaluation script
│   │   ├── scgpt                       # SC-GPT
│   │   └── template                    # TemplateNLG*
│   │
│   ├── evaluator                       # Evaluator for interactive evaluation
│   ├── human_eval                      # Human evaluation with AMT
│   ├── task                            # Goal generators for MultiWOZ, CrossWOZ, and Camrest
│   ├── util
│   │   └── unified_datasets_util.py    # Utility function for unified data format
│   └── deploy                          # Deploy system for human conversion
│
├── data                                # Data dir, not included in pypi package
│   ├── ...                             # ConvLab-2 data, not available for pypi installation
│   └── unified_datasets                # Unified datasets, available for pypi installation
├── examples
│   └── agent_examples                  # Examples of building user and system agents
└── tutorials                           # Tutorials
```

*: models do not support unified datasets, only support MultiWOZ.

## Team

**ConvLab-3** is maintained and developed by [Tsinghua University Conversational AI](http://coai.cs.tsinghua.edu.cn/) group (THU-COAI), the [Dialogue Systems and Machine Learning Group](https://www.cs.hhu.de/en/research-groups/dialog-systems-and-machine-learning.html) at Heinrich Heine University, Düsseldorf, Germany and Microsoft Research (MSR).

We would like to thank all contributors of ConvLab:

Yan Fang, Zhuoer Feng, Jianfeng Gao, Qihan Guo, Kaili Huang, Minlie Huang, Sungjin Lee, Bing Li, Jinchao Li, Xiang Li, Xiujun Li, Jiexi Liu, Lingxiao Luo, Wenchang Ma, Mehrad Moradshahi, Baolin Peng, Runze Liang, Ryuichi Takanobu, Dazhen Wan, Hongru Wang, Jiaxin Wen, Yaoqin Zhang, Zheng Zhang, Qi Zhu, Xiaoyan Zhu, Carel van Niekerk, Christian Geishauser, Hsien-chin Lin, Nurul Lubis, Xiaochen Zhu, Michael Heck, Shutong Feng, Milica Gašić.

## Citing

If you use ConvLab-3 in your research, please cite:

```
@article{zhu2022convlab3,
    title={ConvLab-3: A Flexible Dialogue System Toolkit Based on a Unified Data Format},
    author={Qi Zhu and Christian Geishauser and Hsien-chin Lin and Carel van Niekerk and Baolin Peng and Zheng Zhang and Michael Heck and Nurul Lubis and Dazhen Wan and Xiaochen Zhu and Jianfeng Gao and Milica Gašić and Minlie Huang},
    journal={arXiv preprint arXiv:2211.17148},
    year={2022},
    url={http://arxiv.org/abs/2211.17148}
}
```

## License

Apache License 2.0
