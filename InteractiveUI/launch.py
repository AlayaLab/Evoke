import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
MODELS = {
    'base': 'models/evoke-base',
    'transformer': 'models/evoke/stage3_post_distillation',
    'vigeo': 'models/ViGeo1.1/vigeo.pt',
}


def main():
    parser = argparse.ArgumentParser(description='Launch the interactive EVOKE interface and five-GPU inference service.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    missing = [name for name in MODELS.values() if not (ROOT / name).exists()]
    if args.check:
        print(json.dumps({'modelPaths': MODELS, 'missing': missing, 'modelFilesIncluded': False}, indent=2))
        return 1 if missing else 0
    if missing:
        raise SystemExit('Missing model paths: ' + ', '.join(missing))
    if not shutil.which('ffmpeg'):
        raise SystemExit('ffmpeg must be installed and available on PATH')
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True).strip().splitlines()
    if len(devices) < 5:
        raise SystemExit('This configuration requires at least five CUDA GPUs')
    data = ROOT / '.runtime'
    data.mkdir(exist_ok=True)
    import fcntl
    lock = (data / 'service.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('An instance is already using this runtime directory')
    for name in ['bin', 'native-libs', 'cache', 'gpu-pool/worker-0']:
        (data / name).mkdir(parents=True, exist_ok=True)
    (data / 'gpu-pool/shutdown').unlink(missing_ok=True)
    (data / 'gpu-pool/worker-0/visible-gpus').write_text('0,1,2,3,4\n')
    config = dict(workers=1, spSize=4, auxGpu=4, gpuCount=len(devices),
                  inferencePython=sys.executable, port=args.port, pricePerHour=0)
    (data / 'deployment.json').write_text(json.dumps(config, indent=2))
    for name in ['geometry-policy.json', 'latency-policy.json']:
        shutil.copyfile(ROOT / 'configs' / name, data / name)
    socket_path = data / 'frames.sock'
    socket_path.unlink(missing_ok=True)
    env = os.environ.copy()
    env.update(EVOKE_PLAYER_DATA=str(data), EVOKE_FRAME_SOCKET=str(socket_path),
               EVOKE_UI_HOST=args.host, EVOKE_UI_PORT=str(args.port),
               PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT))
    children = []
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for module, name in [('ui.serving_app', 'web'), ('ui.gpu_worker_pool', 'gpu')]:
            with (data / f'{name}.log').open('a') as log:
                process = subprocess.Popen([sys.executable, '-m', module], cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(process)
            if name == 'web':
                deadline = time.monotonic() + 30
                while not socket_path.exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('Web server startup failed; inspect .runtime/web.log')
                    if stopping:
                        return 0
                    time.sleep(.1)
        print(f'Open http://{args.host}:{args.port}/play/ after model loading completes.', flush=True)
        while not stopping:
            if any(child.poll() is not None for child in children):
                raise RuntimeError('A service process exited; inspect .runtime/web.log and .runtime/gpu.log')
            time.sleep(.5)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        lock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
