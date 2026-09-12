from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('EVOKE_PLAYER_DATA', str(REPO / '.player-data')))
RUNTIME = DATA / 'gpu-pool'
CONFIG = json.loads((DATA / 'deployment.json').read_text())
COUNT = int(CONFIG['workers'])
SP_SIZE = int(CONFIG.get('spSize', 1))
PYTHON = Path(CONFIG['inferencePython'])
STOPPING = False


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


def stop_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    except ProcessLookupError:
        pass


def start(slot):
    directory = RUNTIME / f'worker-{slot}'
    for name in ('requests', 'responses'):
        (directory / name).mkdir(parents=True, exist_ok=True)
    for claimed in directory.glob('running-*.json'):
        request_id=claimed.stem.removeprefix('running-')
        atomic(directory/'responses'/f'{request_id}.json',
               {'requestId':request_id,'returnCode':1,'error':'SP worker group restarted'})
        try:
            argv=json.loads(claimed.read_text())['argv']
            if '--live_control_path' in argv:
                live=Path(argv[argv.index('--live_control_path')+1])
                if live.resolve().is_relative_to((DATA/'live').resolve()):
                    state=json.loads((live/'state.json').read_text())
                    state.update(phase='error',message='The inference worker restarted. Please re-enter the world.',updatedAt=time.time())
                    atomic(live/'state.json',state)
                    atomic(live/'finished.json',{'error':'SP worker group restarted','finishedAt':time.time()})
        except (OSError,ValueError,KeyError,IndexError):pass
    for path in [directory/'state.json', directory/'vigeo_ready.json', *directory.glob('running-*.json'), *directory.glob('sp-rank-*.json')]:
        path.unlink(missing_ok=True)

    for path in (directory/'requests').glob('*.json'):
        atomic(directory/'responses'/path.name, {'requestId':path.stem,'returnCode':1,'error':'GPU worker restarted'})
        path.unlink()
    env = os.environ.copy()
    env.update({
        'PATH':str(PYTHON.parent)+os.pathsep+str(DATA/'bin')+os.pathsep+env.get('PATH',''),
        'PYTHONDONTWRITEBYTECODE':'1', 'EVOKE_PYTHON_BIN':str(PYTHON.parent), 'EVOKE_REMOTE_CLIENT':'0',
        'LD_LIBRARY_PATH':str(DATA/'native-libs')+os.pathsep+env.get('LD_LIBRARY_PATH',''),
        'MODE':'i2v', 'JSONL':str(REPO/'examples/segment_prompts/cases_meteor.jsonl'),
        'EVOKE_SP_SIZE':str(SP_SIZE), 'EVOKE_SP_VERIFY':str(int(CONFIG.get('spVerify', False))),
        'NUM_CHUNKS':'1', 'MAX_CASES':'1', 'LOCAL_GPUS':'1', 'GPU_OFFSET':str(slot * SP_SIZE),
        'OUT_ROOT':str(directory/'bootstrap'), 'JOYSTICK_HUD':'off', 'SAVE_SEGMENTS':'1',
        'IN_PROCESS_BATCH':'1', 'EVOKE_ARGV_SERVER_DIR':str(directory),'EVOKE_SERVER_PRELOAD':'1',
        'EVOKE_UI_PRELOAD_VIGEO':'1','EVOKE_UI_VIGEO_READY_STATE':str(directory/'vigeo_ready.json'),
        'PYTHONPATH':str(REPO/'ui/runtime')+os.pathsep+str(REPO/'ui')+os.pathsep+str(REPO),
        'BASE_CKPT':str((REPO/'models/evoke-base').resolve()),
        'TRANSFORMER_PATH':str((REPO/'models/evoke/stage3_post_distillation').resolve()),
        'VIGEO_WEIGHTS':str((REPO/'models/ViGeo1.1').resolve()),'EVOKE_VIGEO_WEIGHTS':str((REPO/'models/ViGeo1.1').resolve()),
        'DA3_WEIGHTS':str((REPO/'models/DA3').resolve()),'EVOKE_DA3_WEIGHTS':str((REPO/'models/DA3').resolve()),
        'OMP_NUM_THREADS':'16','MKL_NUM_THREADS':'16','OPENBLAS_NUM_THREADS':'16',
        'NUMEXPR_NUM_THREADS':'16','EVOKE_CPU_THREADS':'16',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'HF_HOME':str(DATA/'cache/huggingface'), 'TORCH_HOME':str(DATA/'cache/torch'),
    })
    if CONFIG.get('auxGpu') is not None:
        env['EVOKE_AUX_GPU']=str(CONFIG['auxGpu'])
        env['EVOKE_VISIBLE_GPUS']=','.join(str(i) for i in range(int(CONFIG['gpuCount'])))
    log=(directory/'worker.log').open('a')
    try:
        process=subprocess.Popen(['bash','scripts/inference/infer_post_distill.sh'],cwd=REPO,env=env,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    finally:
        log.close()
    print(f'[pool] GPU {slot}: launcher PID {process.pid}',flush=True)
    return process


def gpu_state():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free,utilization.gpu',
                           '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    devices=[]
    for line in result.stdout.splitlines():
        index,name,total,free,util=(v.strip() for v in line.split(','))
        devices.append({'index':int(index),'name':name,'totalMemoryMiB':int(total),
                        'freeMemoryMiB':int(free),'utilizationPercent':int(util)})
    return devices


def main():
    global STOPPING
    def on_stop(*_):
        global STOPPING
        STOPPING=True
    signal.signal(signal.SIGTERM,on_stop)
    signal.signal(signal.SIGINT,on_stop)
    RUNTIME.mkdir(parents=True,exist_ok=True)
    devices=gpu_state()
    expected=int(CONFIG.get('gpuCount',COUNT * SP_SIZE))
    if len(devices)!=expected:
        raise RuntimeError(f'Expected {expected} visible GPUs, found {len(devices)}')
    workers={}; retries={i:0 for i in range(COUNT)}
    try:
        for slot in range(COUNT):
            workers[slot]=start(slot)
        while not STOPPING and not (RUNTIME/'shutdown').exists():
            for slot,process in list(workers.items()):
                restart=RUNTIME/f'worker-{slot}'/'restart'
                if restart.exists():
                    stop_group(process)
                    workers[slot]=start(slot)
                    restart.unlink(missing_ok=True)
                elif process.poll() is not None and retries[slot]<2:
                    retries[slot]+=1
                    workers[slot]=start(slot)
            try:
                devices=gpu_state()
                atomic(RUNTIME/'heartbeat.json',{'updatedAt':time.time(),'devices':devices,'spSize':SP_SIZE,
                    'workers':[{'slot':i,'alive':p.poll() is None,'pid':p.pid,'restarts':retries[i]} for i,p in workers.items()]})
            except Exception as error:
                print(f'[pool] monitor error: {error}',flush=True)
            for _ in range(10):
                if STOPPING: break
                time.sleep(.5)
    finally:
        for process in workers.values(): stop_group(process)
        atomic(RUNTIME/'heartbeat.json',{'updatedAt':time.time(),'devices':[],'workers':[],'stopped':True})


if __name__=='__main__':
    main()
