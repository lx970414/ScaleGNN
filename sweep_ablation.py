import os
import subprocess
import yaml
from copy import deepcopy

base = yaml.safe_load(open('config.yaml'))
ablations = [
    {'fix_alpha': True,  'fix_mk': False},
    {'fix_alpha': False, 'fix_mk': True},
    {'fix_alpha': True,  'fix_mk': True},
    {'fix_alpha': False, 'fix_mk': False},
]
for i, abla in enumerate(ablations):
    cfg = deepcopy(base)
    cfg['model']['ablation'] = abla    # 规范一致，嵌套在model下
    yaml_file = f'config_ablation_{i}.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(cfg, f)
    subprocess.run(['python', 'main.py', '--config', yaml_file])
