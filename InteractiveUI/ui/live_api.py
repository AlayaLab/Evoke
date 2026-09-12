from __future__ import annotations
import asyncio
import base64
import contextlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from fastapi import File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, StreamingResponse
from .playback_settings import playback_settings
try:
    from .controls import parse_input
except ImportError:
    from controls import parse_input


def write_json(path, value):
    tmp=path.with_suffix('.web.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)


def read_json(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


@dataclass
class Session:
    id: str
    root: Path
    job_id: str = ''
    vae_precision: str = 'fp32'
    playback_mode: str = 'legacy'
    playback: dict = field(default_factory=dict)
    initial_prompt: str = ''
    prompt_request: dict | None = None
    prompt_requests: dict = field(default_factory=dict)
    connected: bool = False
    http_last_keys: set = field(default_factory=set)
    http_key_started: float = field(default_factory=time.monotonic)
    http_press_sequence: int = 0
    state: dict = field(default_factory=lambda:{'sequence':0,'keys':[],'speed':1.5,'lookSpeed':15,
                                                'paused':True,'connected':False,'stop':False,'playedFrame':0})
    def save(self):
        self.state['updatedAt']=time.time();write_json(self.root/'control.json',self.state)


def install(app, backend):
    sessions={}
    root=Path(os.environ.get('EVOKE_PLAYER_DATA', str(backend.REPO_ROOT/'.player-data')))/'live'

    references = {
        'meteor': '山间湖泊', 'aurora': '冰原旷野',
        'gateway': '星际之门', 'crystalstorm': '晶体风暴',
    }

    @app.get('/api/live/references')
    async def reference_list():
        items = []
        for name, title in references.items():
            case = read_json(backend.REPO_ROOT/'examples/segment_prompts'/f'cases_{name}.jsonl')
            if not (backend.REPO_ROOT/'examples/segment_prompts'/f'{name}.jpg').is_file():
                continue
            items.append({'id': name, 'title': title, 'prompt': case.get('prompt', ''),
                          'imageUrl': f'api/live/references/{name}/image'})
        return {'items': items}

    @app.get('/api/live/references/{name}/image')
    async def reference_image(name: str):
        if name not in references:
            raise HTTPException(404, '参考图不存在')
        return FileResponse(backend.REPO_ROOT/'examples/segment_prompts'/f'{name}.jpg', media_type='image/jpeg')

    def get(sid):
        if sid not in sessions:raise HTTPException(404,'探索会话不存在或已结束，请重新进入世界')
        return sessions[sid]

    def snapshot(session):
        result=read_json(session.root/'state.json')
        job=backend.JOBS.get(session.job_id)
        if job and job.status in {'failed','cancelled'}:
            result.update(phase='error',message=job.message)
        elif job and job.status=='queued':result.update(phase='queued',message='正在等待可用 GPU')
        return {**result,'sessionId':session.id,'jobId':session.job_id,
                'vaePrecision':session.vae_precision,'playbackMode':session.playback_mode,'playedFrame':session.state['playedFrame'],'inputSequence':session.state['sequence'],'inputKeys':session.state['keys'],
                'promptRequest':session.prompt_request,'playbackSettings':session.playback}

    @app.post('/api/live/sessions')
    async def start(reference: UploadFile=File(...), prompt: str=Form(...), quality: str=Form('quality'), latent_window: int | None=Form(None), vae_precision: str=Form('fp32'), playback_mode: str=Form('legacy'), warmup_chunks: int=Form(0), ahead_chunks: int | None=Form(None)):
        if not prompt.strip():raise HTTPException(400,'请填写场景描述')
        if quality == 'fast':raise HTTPException(400,'低分辨率模式已停用，请刷新网页；所有模式均使用原生 640×384')
        if quality == 'responsive':raise HTTPException(400,'短窗口模式已停用，请刷新网页；使用训练时的 9 latent 窗口')
        if quality != 'quality':raise HTTPException(400,'无效交互模式')
        if latent_window not in (None,9):raise HTTPException(400,'生成窗口固定为训练配置 9 latent，不支持缩短')
        if vae_precision not in {'fp32','fp16_mixed'}:raise HTTPException(400,'无效 VAE 精度，请选择 FP32 或 FP16 混合精度')
        if playback_mode not in {'legacy','continuous'}:raise HTTPException(400,'无效播放模式')
        try:playback=playback_settings(warmup_chunks,ahead_chunks)
        except ValueError as error:raise HTTPException(400,str(error))
        if playback_mode!='continuous' and (warmup_chunks or ahead_chunks is not None):
            raise HTTPException(400,'隐藏预热和提前生成设置需要连续播放模式')
        latent_window=9
        health=await backend.health()
        if not health['ready']:raise HTTPException(503,'GPU 模型尚未就绪')
        sid=uuid.uuid4().hex;directory=root/sid;directory.mkdir(parents=True)
        session=Session(sid,directory,vae_precision=vae_precision,playback_mode=playback_mode,playback=playback,initial_prompt=prompt.strip());session.save();sessions[sid]=session
        try:
            job=await backend.create_job(reference=reference,spec=json.dumps({'chunks':1,'prompts':[prompt.strip()],
                    'pathPoints':[{'x':.5,'y':.5}]*2,'followPath':False}))
            session.job_id=job['id']
            live_job=backend.JOBS[session.job_id]
            live_job.live_root=directory
            live_job.live_resolution=(384,640)
            write_json(directory/'session.json',{'id':sid,'jobId':job['id'],'quality':quality,'vaePrecision':vae_precision,'playbackMode':playback_mode,'warmupChunks':warmup_chunks,'aheadChunks':ahead_chunks,'initialPrompt':session.initial_prompt,'latentWindowSize':latent_window,'createdAt':time.time()})
        except BaseException:
            sessions.pop(sid,None);raise
        return {'id':sid,'jobId':session.job_id,'vaePrecision':vae_precision,'playbackMode':playback_mode,'streamUrl':f'api/live/{sid}/stream',
                'inputUrl':f'api/live/{sid}/input','promptUrl':f'api/live/{sid}/prompt','supportsLivePrompt':True,'playbackSettings':playback,'latentWindowSize':latent_window,'resolution':live_job.live_resolution[::-1],
                'streamTransport':'sse' if playback_mode=='continuous' or os.environ.get('EVOKE_FRAME_SOCKET') else 'mjpeg'}

    @app.get('/api/live/{sid}')
    async def status(sid: str):return snapshot(get(sid))

    @app.post('/api/live/{sid}/prompt')
    async def submit_prompt(sid: str, request: Request):
        session=get(sid)
        raw=bytearray()
        async for block in request.stream():
            if len(raw)+len(block)>16384:raise HTTPException(413,'事件描述请求过大')
            raw.extend(block)
        try:
            message=json.loads(raw)
        except (ValueError,UnicodeError):
            raise HTTPException(400,'事件描述请求必须为 JSON 对象')
        if not isinstance(message,dict):raise HTTPException(400,'事件描述请求必须为 JSON 对象')
        text=message.get('prompt')
        mode=message.get('mode','event')
        request_id=message.get('requestId')
        if not isinstance(text,str) or not text.strip() or len(text)>2000:
            raise HTTPException(400,'事件描述须为 1 至 2000 字符的非空文本')
        if mode not in ('event','replace'):raise HTTPException(400,'无效事件描述模式')
        if not isinstance(request_id,str) or not request_id.strip() or len(request_id)>64:
            raise HTTPException(400,'requestId 须为 1 至 64 字符的非空文本')
        state=read_json(session.root/'state.json')
        job=backend.JOBS.get(session.job_id)
        if (session.state['stop'] or state.get('phase') in {'stopped','error','failed','cancelled'}
                or (job and job.status in {'failed','cancelled','complete','cancelling'})):
            raise HTTPException(409,'探索会话已结束，请重新进入世界')
        previous=session.prompt_requests.get(request_id)
        if previous is not None:
            if previous['text']!=text or previous['mode']!=mode:
                raise HTTPException(409,'同一 requestId 不可用于不同事件描述')
            return previous
        revision=(session.prompt_request or {}).get('revision',0)+1
        prompt=text.strip() if mode=='replace' else session.initial_prompt+'\n\nCurrent event: '+text.strip()
        update={'revision':revision,'text':text,'prompt':prompt,'mode':mode,
                'requestId':request_id,'submittedAt':time.time()}


        write_json(session.root/'prompt-request.json',update)
        session.prompt_request=update
        session.prompt_requests[request_id]=update
        if len(session.prompt_requests)>32:session.prompt_requests.pop(next(iter(session.prompt_requests)))
        return update

    @app.post('/api/live/{sid}/stop')
    async def stop(sid: str):
        session=get(sid);session.state.update(stop=True,keys=[],paused=True);session.save()
        job=backend.JOBS.get(session.job_id)
        if job and job.status=='queued':
            job.status='cancelled';job.message='探索已结束';backend._persist_job(job)
        return {'stopping':True}

    def playback_ack(session, message):
        value=message.get('playedFrame')
        if value is None and session.playback_mode != 'continuous':
            return session.state['playedFrame']
        if type(value) is not int or value<0:
            raise ValueError('playedFrame must be a nonnegative integer')
        if os.environ.get('EVOKE_FRAME_SOCKET'):
            from .frame_hub import hub
            latest=hub.latest_index(session.id)
        else:
            latest=int(read_json(session.root/'state.json').get('latestFrame',0))
        if value>latest:raise ValueError('playedFrame exceeds published frames')
        return max(session.state['playedFrame'],value)

    @app.post('/api/live/{sid}/input-http')
    async def http_input(sid: str, request: Request):

        if not os.environ.get('EVOKE_FRAME_SOCKET') and get(sid).playback_mode!='continuous':raise HTTPException(404)
        session=get(sid)
        if session.connected or session.state['stop']:raise HTTPException(409,'会话已连接或结束')
        raw=await request.body()
        if len(raw)>4096:raise HTTPException(413)
        try:
            message=json.loads(raw)
            if message.get('type')=='disconnect':
                session.state.update(connected=False,paused=True,keys=[],events=[])
                session.http_last_keys=set();session.save()
                return snapshot(session)
            if message.get('type')!='input':raise ValueError('Expected input')
            keys,speed,look=parse_input(message)
            sequence=int(message.get('sequence',0))
            played=playback_ack(session,message)
            if sequence<session.state['sequence']:
                session.state['playedFrame']=played;session.save();return snapshot(session)
            paused=bool(message.get('paused',False))
            if paused:
                keys=set();session.state['events']=[];session.state['pauseSequence']=sequence
            if keys!=session.http_last_keys:
                if session.http_last_keys and not paused:
                    events=session.state.setdefault('events',[])
                    events.append({'keys':sorted(session.http_last_keys),
                        'duration':min(.33,time.monotonic()-session.http_key_started),
                        'sequence':session.http_press_sequence,'speed':session.state['speed'],'lookSpeed':session.state['lookSpeed']})
                    applied=read_json(session.root/'state.json').get('appliedSequence',-1)
                    session.state['events']=[e for e in events if e['sequence']>applied][-64:]
                if keys:session.http_press_sequence=sequence
                session.http_key_started=time.monotonic();session.http_last_keys=keys
            session.state.update(sequence=sequence,keys=sorted(keys),speed=speed,
                                 lookSpeed=math.degrees(look),paused=paused,connected=True,playedFrame=played)
            session.save()
        except (ValueError,TypeError,OverflowError,AttributeError) as error:
            raise HTTPException(400,str(error))
        return snapshot(session)

    @app.websocket('/api/live/{sid}/input')
    async def inputs(socket: WebSocket,sid: str):
        try:session=get(sid)
        except HTTPException:
            await socket.close(code=1008);return
        if session.connected or session.state['stop']:
            await socket.close(code=1008);return
        await socket.accept();session.connected=True;session.state.update(connected=True,paused=False);session.save()
        last_keys=set();key_started=time.monotonic();press_sequence=0;lock=asyncio.Lock()
        async def send(payload):
            async with lock:await socket.send_json(payload)
        async def receive():
            nonlocal last_keys,key_started,press_sequence
            while True:
                raw=await socket.receive_text()
                if len(raw)>4096:await socket.close(code=1009);return
                try:
                    message=json.loads(raw)
                    if not isinstance(message,dict):raise ValueError('Expected object')
                    if message.get('type')=='ping':
                        await send({'type':'pong','time':message.get('time')});continue
                    if message.get('type')!='input':raise ValueError('Expected input')
                    keys,speed,look=parse_input(message)
                    sequence=int(message.get('sequence',0))
                    played=playback_ack(session,message)
                    if sequence<session.state['sequence']:
                        session.state['playedFrame']=played;session.save();continue
                    paused=bool(message.get('paused',False))
                    if paused:
                        keys=set();session.state['events']=[];session.state['pauseSequence']=sequence
                    if keys!=last_keys:
                        if last_keys and not paused:


                            events=session.state.setdefault('events', [])
                            events.append({'keys':sorted(last_keys),
                                'duration':min(.33,time.monotonic()-key_started),
                                'sequence':press_sequence,'speed':session.state['speed'],
                                'lookSpeed':session.state['lookSpeed']})
                            applied=read_json(session.root/'state.json').get('appliedSequence',-1)
                            session.state['events']=[e for e in events if e['sequence']>applied][-64:]
                        if paused:session.state['events']=[]
                        if keys:press_sequence=sequence
                        key_started=time.monotonic();last_keys=keys
                    session.state.update(sequence=sequence,keys=sorted(keys),speed=speed,
                                         lookSpeed=math.degrees(look),paused=paused,playedFrame=played)
                    session.save()
                except (ValueError,TypeError,OverflowError) as error:
                    await send({'type':'error','message':str(error)})
        async def report():
            while True:
                await send({'type':'state',**snapshot(session)});await asyncio.sleep(.1)
        tasks=[asyncio.create_task(receive()),asyncio.create_task(report())]
        try:
            done,_=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
            for task in done:task.result()
        except (WebSocketDisconnect,OSError,RuntimeError):pass
        finally:
            session.connected=False;session.state.update(connected=False,paused=True,keys=[]);session.save()
            for task in tasks:task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)

    @app.get('/api/live/{sid}/stream')
    async def stream(sid: str, request: Request, after: int=0):
        session=get(sid)
        continuous=session.playback_mode=='continuous'
        socket_frames=bool(os.environ.get('EVOKE_FRAME_SOCKET'))
        sse=continuous or socket_frames
        try:


            cursor=int(request.headers.get('last-event-id',after)) if continuous else 0
            if cursor<0:raise ValueError()
        except (ValueError,TypeError):raise HTTPException(400,'Invalid frame cursor')

        def event(name, value):
            return ('event: '+name+'\ndata: '+json.dumps(value)+'\n\n').encode()

        async def frames():
            from .frame_hub import hub, FrameGap
            sent=cursor
            terminal_wait=None
            while True:
                if socket_frames:
                    try:entry=hub.next(sid,sent,continuous=continuous)
                    except FrameGap as gap:
                        yield event('gap',gap.details);return
                    if entry:
                        sent,jpeg=entry
                        yield b'event: frame\nid: '+str(sent).encode()+b'\ndata: '+base64.b64encode(jpeg)+b'\n\n'


                        await asyncio.sleep(0 if continuous else 1/24)
                        continue

                state=await asyncio.to_thread(snapshot,session)
                latest=int(state.get('latestFrame',0))
                if not socket_frames and latest>sent:
                    index=sent+1 if continuous else max(sent+1,latest-8)
                    try:jpeg=await asyncio.to_thread((session.root/'frames'/f'{index:09d}.jpg').read_bytes)
                    except OSError:
                        if continuous:
                            yield event('gap',{'expectedFrame':index,'oldestFrame':max(1,latest-session.playback['publicationLimitFrames']+1),'latestFrame':latest});return
                        await asyncio.sleep(.02);continue
                    sent=index
                    if sse:
                        yield b'event: frame\nid: '+str(sent).encode()+b'\ndata: '+base64.b64encode(jpeg)+b'\n\n'
                    else:
                        yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n'
                    await asyncio.sleep(0 if continuous else 1/24)
                    continue
                phase=state.get('phase')
                if phase=='error':
                    if continuous:yield event('error',{'message':state.get('message','生成已中断'),'lastFrame':sent})
                    return
                if phase=='stopped':


                    if continuous:
                        if sent<latest:


                            if socket_frames:
                                if terminal_wait is None:terminal_wait=time.monotonic()
                                if time.monotonic()-terminal_wait<1:
                                    await asyncio.sleep(.01);continue
                            yield event('gap',{'expectedFrame':sent+1,'latestFrame':latest})
                        else:yield event('done',{'lastFrame':sent})
                    return
                if not continuous and session.state['stop']:return
                await asyncio.sleep(.01 if socket_frames else .02)
        return StreamingResponse(frames(),media_type='text/event-stream' if sse else 'multipart/x-mixed-replace; boundary=frame',
                                 headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})

    async def shutdown():
        for session in sessions.values():
            session.state.update(stop=True,keys=[]);session.save()
    app.router.add_event_handler('shutdown',shutdown)
