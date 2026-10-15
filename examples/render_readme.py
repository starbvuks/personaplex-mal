"""Regenerate README visuals from synthetic CPU state. Requires torch and Pillow.

Run from any directory: python examples/render_readme.py
No model, audio, conversation data or trained interface is loaded.
"""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from mal import MALBranch, tensor_hash

torch.manual_seed(7)
branches = torch.nn.ModuleList([MALBranch(16, 8, 2) for _ in range(4)])
history, query = torch.randn(1, 5, 16), torch.randn(1, 2, 16)
b = branches[0]
w1, w2 = b.initial_w1.detach().clone(), b.initial_w2.detach().clone()
trace = []
def observe(before, after, stats):
    trace.append(dict(write=after.writes, position=b.write_position,
                      w1_delta=(after.w1-w1).detach().tolist(),
                      w2_delta=(after.w2-w2).detach().tolist(),
                      change=float(stats['applied_state_change_norm'].detach())))
b.write_observer = observe
assert torch.equal(b(history, write=True), history)
b.freeze_state()
before = tensor_hash(b.fast.payload())
assert torch.equal(b(query), query)
after = tensor_hash(b.fast.payload())
assert before == after and len(trace) == 7
out = root / 'assets' / 'trace.json'
out.write_text(json.dumps(dict(seed=7, host_dim=16, memory_dim=8,
    prefix_tokens=2, history=history[0].tolist(), query=query[0].tolist(),
    writes=trace, state_hash=before, frozen_read_hash=after,
    gate=0, host_unchanged=True, provenance='Synthetic CPU fixture; no speech or recall evaluation.'), indent=2))
print('Captured seven writes; frozen reads preserve state; zero gate preserves host output.')

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import json, math

t=json.loads((root/'assets/trace.json').read_text())
W,H=936,432
BG='#141316'; INK='#e4dcd2'; MUTED='#9b9096'; LINE='#393038'; PINK='#d36a94'
def font(n):
    for candidate in ['Menlo.ttc', 'DejaVuSansMono.ttf', '/System/Library/Fonts/Menlo.ttc',
                      '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 'C:/Windows/Fonts/consola.ttf']:
        try: return ImageFont.truetype(candidate,n)
        except OSError: pass
    raise RuntimeError('Install a monospace TrueType font: Menlo, DejaVu Sans Mono or Consolas.')
def txt(d,xy,s,n=12,fill=INK):d.text(xy,s,font=font(n),fill=fill)
def heat(d,m,x,y,sz,limit):
    for r,row in enumerate(m):
        for c,v in enumerate(row):
            q=min(1,abs(v)/limit)
            base=(34,30,35); high=(224,123,163) if v>=0 else (139,158,178)
            col=tuple(round(a+(b-a)*q) for a,b in zip(base,high))
            d.rectangle((x+c*sz,y+r*sz,x+(c+1)*sz-2,y+(r+1)*sz-2),fill=col)
zero=[[0]*8 for _ in range(8)]
max1=max(abs(v) for w in t['writes'] for row in w['w1_delta'] for v in row)
max2=max(abs(v) for w in t['writes'] for row in w['w2_delta'] for v in row)
frames=[];dur=[]
# Each display step corresponds to a captured state, held long enough to read.
steps=[('RESET',0,900)]+[('WRITE',i,650) for i in range(1,8)]+[('FREEZE',7,1700),('READ',7,2100)]
for phase,count,ms in steps:
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    txt(d,(26,20),'MAL',42)
    txt(d,(149,34),'personaplex / memory as a layer',13,MUTED)
    txt(d,(796,34),'*  seed 07',12,PINK)
    d.line((26,83,910,83),fill=LINE)
    txt(d,(26,103),'01  /  INPUT',11,MUTED)
    txt(d,(307,103),'02  /  FAST WEIGHTS',11,MUTED)
    txt(d,(737,103),'03  /  HOST',11,MUTED)
    # Five actual 16-dimensional host frames, not an invented audio waveform.
    for k,row in enumerate(t['history']):
        x=27+k*47; active=(phase=='WRITE' and count==k+3)
        for j,v in enumerate(row):
            brightness=min(1,abs(v)/2.8)
            hi=(211,106,148) if v>0 else (139,158,178)
            col=tuple(round(32+(z-32)*brightness) for z in hi)
            if count<k+3:col=tuple(round(z*.45) for z in col)
            d.rectangle((x,141+j*7,x+34,145+j*7),fill=col)
        txt(d,(x+4,265),f'{k+1:02}',10,PINK if active else MUTED)
        if active:d.line((x,291,x+34,291),fill=PINK,width=2)
    state=t['writes'][count-1] if count else None
    heat(d,state['w1_delta'] if state else zero,307,141,18,max1)
    heat(d,state['w2_delta'] if state else zero,490,141,18,max2)
    txt(d,(307,298),'delta W1',11,MUTED);txt(d,(490,298),'delta W2',11,MUTED)
    txt(d,(737,145),'gate',12,MUTED);txt(d,(737,164),'0.000',25)
    txt(d,(737,212),'output',12,MUTED);txt(d,(737,235),'unchanged',16)
    txt(d,(737,275),'untrained interface',10,MUTED)
    d.line((26,334,910,334),fill=LINE)
    for j,p in enumerate(['RESET','WRITE','FREEZE','READ']):
        txt(d,(26+j*113,351),p,12,PINK if phase==p else MUTED)
    txt(d,(590,351),f'{count:02} writes / 02 prefix + 05 history',11)
    note={'RESET':'NEW EPISODE / state returns to its initial weights',
          'WRITE':f"WRITE {count:02} / "+('prefix token' if count<=2 else f'history frame {count-2:02}'),
          'FREEZE':f"FROZEN / sha256 {t['state_hash'][:16]}",
          'READ':'READ ONLY / 02 query frames; state hash unchanged'}[phase]
    txt(d,(26,382),note,11,MUTED)
    txt(d,(26,409),'SYNTHETIC CPU TRACE / matrix colours share a fixed scale per matrix / not a speech or recall result',9,MUTED)
    frames.append(im);dur.append(ms)
frames[8].save(root/'assets/memory-still.png')
frames[0].save(root/'assets/memory-loop.gif',save_all=True,append_images=frames[1:],duration=dur,loop=0,optimize=True,disposal=2)
# A second, more atmospheric treatment of the SAME captured matrix values.
# The enlargement is explicitly aesthetic; the calibrated mechanism display
# remains separately accessible and the trace JSON is the numeric source.
hero=[]
for phase,count,ms in steps:
    matrix=t['writes'][count-1]['w2_delta'] if count else zero
    small=Image.new('RGB',(8,8))
    for y,row in enumerate(matrix):
        for x,v in enumerate(row):
            q=max(-1,min(1,v/max2))
            a=abs(q)**.65
            base=(19,13,24); high=(216,49,111) if q>=0 else (92,68,144)
            small.putpixel((x,y),tuple(round(b+(h-b)*a) for b,h in zip(base,high)))
    im=small.resize((936,580),Image.Resampling.BICUBIC).crop((0,55,936,430)).filter(ImageFilter.GaussianBlur(7))
    d=ImageDraw.Draw(im)
    txt(d,(27,20),'starbvuks  /  personaplex-mal',12,'#e0c9d4')
    # Slightly off-centre typography; no fake chrome, random glitch or stock image.
    txt(d,(22,67),'MAL',126,'#eadbd7')
    txt(d,(30,221),'memory as a layer',16,'#eadbd7')
    d.line((30,299,472,299),fill='#90647d')
    txt(d,(30,314),'a writable network inside a speech model',12,'#eadbd7')
    txt(d,(30,341),f'{phase.lower():<7} / {count:02} writes / gate 0',11,'#d5b2c5')
    txt(d,(629,314),'W2 / enlarged state change',11,'#e0c9d4')
    txt(d,(629,341),'synthetic CPU fixture / seed 07',10,'#c6a2b8')
    hero.append(im)
hero[8].save(root/'assets/signal-still.png')
hero[0].save(root/'assets/signal.gif',save_all=True,append_images=hero[1:],duration=dur,loop=0,optimize=True,disposal=2)
# Plain, GitHub-compatible supporting architecture illustration.
(root/'assets/path.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="936" height="180" viewBox="0 0 936 180"><rect width="936" height="180" fill="#141316"/><g fill="none" stroke="#9b9096"><path d="M28 57H908M145 57V119H736V57"/><path d="M900 52l8 5-8 5M728 111l8 8-8 8"/></g><g font-family="monospace" font-size="13" fill="#e4dcd2"><text x="28" y="32">temporal-layer input</text><text x="765" y="32">host continues</text><text x="175" y="111">project down</text><text x="347" y="111">read fast weights</text><text x="556" y="111">project up</text><text x="663" y="111" fill="#d36a94">gate</text><text x="28" y="160" fill="#9b9096">Read before the current frame writes. Gate starts at zero.</text></g></svg>''')
print(f'Rendered {len(frames)} captured states, {(root/"assets/memory-loop.gif").stat().st_size:,} byte GIF.')
