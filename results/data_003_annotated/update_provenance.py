from pathlib import Path
p=Path('scripts/annotate_transfer_demos.py'); s=p.read_text(encoding='utf-8-sig')
s=s.replace('import argparse\n','import argparse\nimport hashlib\n')
s=s.replace("geometry_limitations=m.get('limitations',[]),annotation_path=str(annotation_path),prepared_demo=str(output),", "geometry_limitations=m.get('limitations',[]),source_demo=str(demo),annotation_path=str(annotation_path),prepared_demo=str(output),\n        annotation_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),")
p.write_text(s,encoding='utf-8')
p=Path('scripts/record_transfer_demos.py'); s=p.read_text(encoding='utf-8-sig')
s=s.replace("    eligible=[v for v in validations if v['suitable_for_retargeting']]", "    from annotate_transfer_demos import prepare\n    validations=[prepare(Path(v['source_demo']),Path(v['annotation_path']),Path(v['prepared_demo']))\n        if sha256(v['annotation_path'])!=v['annotation_sha256'] else v for v in validations]\n    eligible=[v for v in validations if v['suitable_for_retargeting']]")
s=s.replace("            human=load_transport(demo/'processed_demo.csv'", "            if sha256(v['annotation_path'])!=v['annotation_sha256']: raise ValueError('Annotation changed after validation; rerun preparation.')\n            human=load_transport(demo/'processed_demo.csv'")
p.write_text(s,encoding='utf-8')
