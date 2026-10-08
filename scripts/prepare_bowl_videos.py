"""Create validated H.264 playback copies; never overwrite original recordings."""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def prepare(root, ffmpeg=None):
    root=Path(root)
    ffmpeg=ffmpeg or shutil.which('ffmpeg')
    if not ffmpeg: raise RuntimeError('ffmpeg is required: apt-get install -y ffmpeg')
    report=[]
    for source in sorted(root.rglob('*.mp4')):
        if source.stem.endswith('_playback'): continue
        destination=source.with_name(source.stem+'_playback.mp4')
        entry=dict(source=str(source),playback=str(destination))
        try:
            # Encode on local disk, validate all frames, then copy to Drive.
            with tempfile.TemporaryDirectory(prefix='bowl_video_') as directory:
                output=Path(directory)/'playback.mp4'
                command=[ffmpeg,'-nostdin','-hide_banner','-loglevel','error','-y','-i',str(source),
                    '-map','0:v:0','-an','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',
                    '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                    '-movflags','+faststart',str(output)]
                subprocess.run(command,check=True,capture_output=True,text=True,timeout=120)
                decoded=subprocess.run([ffmpeg,'-nostdin','-hide_banner','-loglevel','error','-xerror',
                    '-i',str(output),'-map','0:v:0','-progress','pipe:1','-f','null','-'],check=True,
                    capture_output=True,text=True,timeout=120)
                frames=max((int(line.split('=',1)[1]) for line in decoded.stdout.splitlines() if line.startswith('frame=')),default=0)
                if frames<1 or output.stat().st_size<100: raise RuntimeError('Empty encoded video')
                staged=destination.with_suffix('.mp4.partial')
                try:
                    shutil.copyfile(output,staged)
                    staged.replace(destination)
                finally:
                    if staged.exists(): staged.unlink()
            entry.update(status='ready',codec='H.264',pixel_format='yuv420p',faststart=True,
                bytes=destination.stat().st_size,decoded_frames=frames,full_decode_check=True)
        except Exception as exc:
            entry.update(status='failed',error=str(exc),stderr=getattr(exc,'stderr',None))
        report.append(entry)
    (root/'video_playback_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();results=prepare(args.root)
    if any(r['status']=='failed' for r in results): raise SystemExit(1)
