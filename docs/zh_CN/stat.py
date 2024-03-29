#!/usr/bin/env python
import functools as func
import glob
import os
import re
from pathlib import Path

import numpy as np

MMCLS_ROOT = Path(__file__).absolute().parents[1]
url_prefix = 'https://github.com/open-mmlab/mmclassification/blob/master/'

papers_root = Path('papers')
papers_root.mkdir(exist_ok=True)
files = [Path(f) for f in sorted(glob.glob('../../configs/*/README.md'))]

stats = []
titles = []
num_ckpts = 0
num_configs = 0

for f in files:
    with open(f, 'r') as content_file:
        content = content_file.read()

    # Extract checkpoints
    ckpts = set(x.lower().strip()
                for x in re.findall(r'\[model\]\((https?.*)\)', content))
    if len(ckpts) == 0:
        continue
    num_ckpts += len(ckpts)

    # Extract paper title
    title = content.split('\n')[0].replace('# ', '').strip()
    titles.append(title)

    # Extract paper abbreviation
    abbr = [x for x in re.findall(r'<!-- {(.+)} -->', content)]
    abbr = abbr[0] if len(abbr) > 0 else title

    # Extract paper type
    _papertype = [x for x in re.findall(r'\[([A-Z]+)\]', content)]
    assert len(_papertype) > 0
    papertype = _papertype[0]
    paper = set([(papertype, title)])

    # Write a copy of README
    copy = papers_root / (f.parent.name + '.md')
    if copy.exists():
        os.remove(copy)

    def replace_link(matchobj):
        # Replace relative link to GitHub link.
        name = matchobj.group(1)
        link = matchobj.group(2)
        if not link.startswith('http') and (f.parent / link).exists():
            rel_link = (f.parent / link).absolute().relative_to(MMCLS_ROOT)
            link = url_prefix + str(rel_link)
        return f'[{name}]({link})'

    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_link, content)

    with open(copy, 'w') as copy_file:
        copy_file.write(content)

    statsmsg = f"""
\t* [{papertype}] [{title}]({copy}) ({len(ckpts)} ckpts)
"""
    stats.append(
        dict(
            paper=paper, ckpts=ckpts, statsmsg=statsmsg, abbr=abbr, copy=copy))

allpapers = func.reduce(lambda a, b: a.union(b),
                        [stat['paper'] for stat in stats])
msglist = '\n'.join(stat['statsmsg'] for stat in stats)

papertypes, papercounts = np.unique([t for t, _ in allpapers],
                                    return_counts=True)
countstr = '\n'.join(
    [f'   - {t}: {c}' for t, c in zip(papertypes, papercounts)])

modelzoo = f"""
# 模型库统计

* 论文数量： {len(set(titles))}
{countstr}

* 模型权重文件数量： {num_ckpts}
{msglist}
"""

with open('modelzoo_statistics.md', 'w') as f:
    f.write(modelzoo)

toctree = """
.. toctree::
   :maxdepth: 1
   :caption: 模型库
   :glob:

   modelzoo_statistics.md
   model_zoo.md
"""
with open('_model_zoo.rst', 'w') as f:
    f.write(toctree)
    for stat in stats:
        f.write(f'   {stat["abbr"]} <{stat["copy"]}>\n')
