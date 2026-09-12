from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('EVOKE_PLAYER_DATA', str(REPO / '.player-data')))
POOL = DATA / 'gpu-pool'
CONFIG = json.loads((DATA / 'deployment.json').read_text())
COUNT = int(CONFIG['workers'])
SP_SIZE = int(CONFIG.get('spSize', 1))
CLIENTS = DATA / 'clients'
os.environ.update(EVOKE_UI_CONTROL_ONLY='0',EVOKE_UI_JOBS=str(DATA/'jobs'),
                  EVOKE_UI_PROJECTS=str(DATA/'projects'),
                  EVOKE_UI_PYTHON_BIN=str(Path(sys.executable).parent),
                  EVOKE_REMOTE_CLIENT='1',
                  PYTHONDONTWRITEBYTECODE='1')
os.environ['PATH']=str(DATA/'bin')+os.pathsep+os.environ.get('PATH','')
try:
    from . import app as backend
except ImportError:
    import app as backend


def read(path):
    try: return json.loads(path.read_text())
    except (OSError,ValueError): return {}


def workers():
    heartbeat=read(POOL/'heartbeat.json')
    fresh=time.time()-float(heartbeat.get('updatedAt',0))<30
    live={v['slot']:v.get('alive',False) for v in heartbeat.get('workers',[])}
    result=[]
    for i in range(COUNT):
        directory=POOL/f'worker-{i}'
        state=read(directory/'state.json')
        marker=read(directory/'vigeo_ready.json')
        ranks=[read(directory/f'sp-rank-{r}.json') for r in range(SP_SIZE)] if SP_SIZE>1 else []
        sp_ready=(not ranks or (all(r.get('worldSize')==SP_SIZE for r in ranks)
                              and ranks[0].get('pid')==state.get('pid')))
        usable=(sp_ready and fresh and live.get(i) and not (directory/'restart').exists()
                and state.get('phase') in {'ready','running'}
                and marker.get('pid')==state.get('pid') and bool(marker.get('warmed')))
        result.append({**state,'slot':i,'usable':bool(usable),'geometry':marker})
    return result


def gpu_state():
    heartbeat=read(POOL/'heartbeat.json')
    available=time.time()-float(heartbeat.get('updatedAt',0))<30 and bool(heartbeat.get('devices'))
    return {'available':available,'capacityReady':available,'devices':heartbeat.get('devices',[]),
            'workers':workers(),'spSize':SP_SIZE,'updatedAt':heartbeat.get('updatedAt')}


def runtime_state():
    states=workers(); ready=sum(s['usable'] for s in states)
    active=sum(s.get('phase')=='running' and s['usable'] for s in states)
    if ready:
        return {'phase':'generating' if active else 'ready',
                'message':f'H200 {SP_SIZE} 卡 SP · {ready}/{COUNT} 实例就绪 · {active} 个任务生成中','jobId':None}
    errors=[s.get('message','') for s in states if s.get('phase')=='error']
    return {'phase':'error' if errors else 'loading',
            'message':errors[0] if errors else f'正在加载 H200 {SP_SIZE} 卡 SP 推理实例…','jobId':None}


def geometry_state():
    state=next((s for s in workers() if s['usable']),{})
    return {'enabled':True,'ready':bool(state),'warmed':bool(state),**state.get('geometry',{})}


leased=set()
lease_lock=asyncio.Lock()
original_run=backend._run_job
backend.GPU_LOCK=asyncio.Semaphore(COUNT)


async def run_job(job):
    slot=None
    try:
        while slot is None:
            if job.status=='cancelled': return
            async with lease_lock:
                slot=next((s['slot'] for s in workers() if s['usable'] and s['slot'] not in leased and s.get('phase')=='ready'),None)
                if slot is not None: leased.add(slot)
            if slot is None: await asyncio.sleep(.5)
        job.worker_root=CLIENTS/f'worker-{slot}'
        await original_run(job)
    finally:
        if slot is not None:
            async with lease_lock: leased.discard(slot)


async def restart_job(job):
    if job.worker_root is None: return
    directory=POOL/job.worker_root.name

    (directory/'requests'/f'{job.id}.json').unlink(missing_ok=True)
    (directory/'restart').write_text(str(time.time()))


async def startup():
    backend.JOBS_ROOT.mkdir(parents=True,exist_ok=True)
    backend.PROJECTS_ROOT.mkdir(parents=True,exist_ok=True)
    for slot in range(COUNT):
        client=CLIENTS/f'worker-{slot}'
        client.mkdir(parents=True,exist_ok=True)
        for name in ('requests','responses'):
            target=POOL/f'worker-{slot}'/name
            target.mkdir(parents=True,exist_ok=True)
            link=client/name
            if not os.path.lexists(link): link.symlink_to(target,target_is_directory=True)
            elif not link.is_symlink() or link.resolve()!=target.resolve():
                raise RuntimeError(f'Unexpected queue path: {link}')


        if (client/'state.json').exists(): raise RuntimeError(f'Unexpected GPU PID state in {client}')
    backend._restore_jobs()


backend._run_job=run_job
backend._restart_job_worker=restart_job
backend._gpu_state=gpu_state
backend._runtime_state=runtime_state
backend._vigeo_preload_state=geometry_state
backend.app.router.on_startup.clear()
if backend.shutdown in backend.app.router.on_shutdown:
    backend.app.router.on_shutdown.remove(backend.shutdown)
backend.app.router.add_event_handler('startup',startup)
app=backend.app

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host=os.environ.get('EVOKE_UI_HOST','0.0.0.0'),port=int(os.environ.get('EVOKE_UI_PORT','7861')))
