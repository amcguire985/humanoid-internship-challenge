from pathlib import Path
p=Path('docs/DATA_003_COLAB.md'); s=p.read_text(encoding='utf-8-sig')
s=s.replace('In Colab, after pulling this commit:', 'GitHub rejected the original main push because an earlier unpublished commit contains oversized videos. Local main is preserved. The supported handoff branch is `colab/data-003-annotations`, based on remote main plus the current task files and the existing multi-transfer extractor/tests. The oversized earlier video blobs are not part of this branch.\n\nIn Colab:')
s=s.replace('!git pull --ff-only','!git fetch origin\n!git switch colab/data-003-annotations\n!git pull --ff-only origin colab/data-003-annotations')
p.write_text(s,encoding='utf-8')
