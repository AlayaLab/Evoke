class ContinuousFramePlayer {
 constructor(canvas,{onFrame=()=>{},onState=()=>{},onError=()=>{},onEnd=()=>{},decode=null,initialBufferFrames=72,recoveryBufferFrames=60,publicationLimitFrames=96}={}){
  if(!Number.isInteger(publicationLimitFrames)||publicationLimitFrames<1||publicationLimitFrames>96||
    ![initialBufferFrames,recoveryBufferFrames].every(n=>Number.isInteger(n)&&n>=1&&n<=publicationLimitFrames))throw new Error('Invalid playback settings');
  this.initialBufferFrames=initialBufferFrames;this.recoveryBufferFrames=recoveryBufferFrames;this.publicationLimitFrames=publicationLimitFrames;
  this.canvas=canvas;this.context=canvas.getContext('2d',{alpha:false});
  this.onFrame=onFrame;this.onState=onState;this.onError=onError;this.onEnd=onEnd;
  this.decode=decode|| (async data=>{
   let bytes;if(typeof Uint8Array.fromBase64==='function')bytes=Uint8Array.fromBase64(data);
   else{const binary=atob(data);bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);}
   return createImageBitmap(new Blob([bytes],{type:'image/jpeg'}));
  });
  this.frames=new Map();this.pending=[];this.decoding=0;this.received=0;this.played=0;
  this.paused=false;this.stopped=false;this.done=false;this.started=false;this.buffering=true;this.nextAt=0;this.lastState='';
  this.metrics={arrivals:[],frames:[],states:[],maxQueue:0,rebufferCount:0,duplicateFrames:0};
  this._tick=this.tick.bind(this);this.raf=requestAnimationFrame(this._tick);this.emitState();
 }
 readyCount(){let count=0;for(let i=this.played+1;this.frames.get(i)?.image;i++)count++;return count;}
 emitState(){
  const state=this.stopped?'stopped':this.paused?'paused':this.buffering?'buffering':'playing';
  const ready=this.readyCount();const value={state,ready,queued:this.frames.size,played:this.played,received:this.received,initial:!this.started,targetFrames:this.started?this.recoveryBufferFrames:this.initialBufferFrames};
  const key=`${state}:${Math.floor(ready/4)}`;
  if(key!==this.lastState){this.lastState=key;this.onState(value);}
  const previous=this.metrics.states.at(-1);
  if(previous?.state!==state)this.metrics.states.push({...value,at:performance.now()});
 }
 enqueue(index,data,receivedAt=performance.now()){
  if(this.stopped)return;
  if(!Number.isSafeInteger(index)||index<1)return this.fail('Received an invalid frame number.');
  if(index<=this.received){this.metrics.duplicateFrames++;return;}
  if(index!==this.received+1)return this.fail('Frames were lost during a disconnect. Re-enter the world to resume continuous playback.');
  if(this.frames.size>=this.publicationLimitFrames)return this.fail('Playback buffer limit exceeded. Please re-enter the world.');
  const entry={index,data,receivedAt,image:null,decodedAt:null};this.received=index;
  this.frames.set(index,entry);this.pending.push(entry);this.metrics.arrivals.push({index,at:receivedAt});
  if(this.metrics.arrivals.length>12000)this.metrics.arrivals.shift();
  this.metrics.maxQueue=Math.max(this.metrics.maxQueue,this.frames.size);this.pump();this.emitState();
 }
 pump(){
  while(!this.stopped&&this.decoding<3&&this.pending.length){
   const entry=this.pending.shift();this.decoding++;
   Promise.resolve().then(()=>this.decode(entry.data)).then(image=>{
    if(this.stopped){image.close?.();return;}
    if(image.width!==640||image.height!==384){image.close?.();throw new Error('Generated frame dimensions are incorrect.');}
    entry.image=image;entry.data=null;entry.decodedAt=performance.now();
   }).catch(error=>this.fail(error.message||'Frame decoding failed. Please re-enter the world.')).finally(()=>{this.decoding--;this.pump();if(!this.stopped)this.emitState();});
  }
 }
 setPaused(value){if(this.stopped)return;this.paused=value;this.nextAt=0;this.emitState();}
 end(){if(this.stopped)return;this.done=true;this.emitState();}
 tick(now){
  if(this.stopped)return;
  if(!this.paused){
   const ready=this.readyCount();
   if(this.buffering){
    const reserve=this.started?this.recoveryBufferFrames:this.initialBufferFrames;
    const threshold=this.done?Math.min(reserve,this.received-this.played):reserve;
    if(threshold>0&&ready>=threshold){this.buffering=false;this.started=true;this.nextAt=now;this.emitState();}
   }
   if(!this.buffering&&(!this.nextAt||now+.25>=this.nextAt)){
    const remaining=this.received-this.played;
    if(ready>=1){
     const entry=this.frames.get(this.played+1);
     this.context.drawImage(entry.image,0,0,640,384);entry.image.close?.();
     this.frames.delete(entry.index);this.played=entry.index;this.canvas.dataset.frameIndex=String(entry.index);
     const previous=this.metrics.frames.at(-1);
     const record={index:entry.index,receivedAt:entry.receivedAt,decodedAt:entry.decodedAt,shownAt:now,
      gapMs:previous?now-previous.shownAt:null,queued:this.frames.size};
     this.metrics.frames.push(record);if(this.metrics.frames.length>12000)this.metrics.frames.shift();
     const deadline=(this.nextAt||now)+1000/24;
     this.nextAt=deadline<now?now+1000/24:deadline;
     this.onFrame(record);this.emitState();
    }else if(!(this.done&&remaining===0)){
     this.buffering=true;this.nextAt=0;this.metrics.rebufferCount++;this.emitState();
    }
   }
   if(this.done&&this.played===this.received&&this.decoding===0){this.stop();this.onEnd();return;}
  }
  this.raf=requestAnimationFrame(this._tick);
 }
 fail(message){if(this.stopped)return;this.stop();this.onError(message);}
 stop(){
  if(this.stopped)return;this.stopped=true;cancelAnimationFrame(this.raf);
  for(const entry of this.frames.values())entry.image?.close?.();this.frames.clear();this.pending=[];this.emitState();
 }
}

class InputMailbox {
 constructor(send,{onError=()=>{},clock=()=>performance.now()}={}){
  this.send=send;this.onError=onError;this.clock=clock;this.pending=[];this.flight=null;this.closed=false;this.played=0;
  this.metrics={sent:0,acknowledged:0,coalesced:0,maxPending:0,maxRoundTripMs:0};
 }
 push(value){
  if(this.closed)return;
  this.played=Math.max(this.played,value.playedFrame||0);
  const packet={...value,keys:[...value.keys]};
  if(this.pending.at(-1)?.sequence===packet.sequence){this.pending[this.pending.length-1]=packet;this.metrics.coalesced++;}
  else{
   if(this.pending.length>=64){this.close();this.onError('Control connection overloaded. Please re-enter the world.');return;}
   this.pending.push(packet);this.metrics.maxPending=Math.max(this.metrics.maxPending,this.pending.length);
  }
  this.flush();
 }
 flush(){
  if(this.closed||this.flight||!this.pending.length)return;
  const packet={...this.pending.shift(),playedFrame:this.played};
  this.flight={packet,sentAt:this.clock()};this.metrics.sent++;this.send(packet);
 }
 acknowledge(state){
  if(!this.flight||state.inputSequence<this.flight.packet.sequence||state.playedFrame<this.flight.packet.playedFrame)return;
  if(!Number.isInteger(state.inputSequence)||!Number.isInteger(state.playedFrame))return;
  this.metrics.acknowledged++;
  this.metrics.maxRoundTripMs=Math.max(this.metrics.maxRoundTripMs,this.clock()-this.flight.sentAt);
  this.flight=null;this.flush();
 }
 close(){this.closed=true;this.pending=[];this.flight=null;}
}

const $=id=>document.getElementById(id),base=new URL('../',location.href),url=path=>new URL(path,base).href;
const keys=new Set(),pointers=new Map(),codes=Object.fromEntries([...'wasdijkl'].map(k=>[`Key${k.toUpperCase()}`,k]));
let referencePath='api/default-case/reference',frameSource=null,framePlayer=null,playedSequence=-1;
let session=null,socket=null,inputMailbox=null,paused=false,starting=false,connected=false,seq=0,lastInput='',reference=null,referenceUrl=null,noticeTimer,reconnect;
const latency=window.evokeLatency={inputs:[],frames:[],windows:[],samples:[]};
const promptMetrics=window.evokePromptMetrics={submissions:[],applied:[],displayed:[]};
let latestPrompt=null,promptSending=false,promptWindows=new Map();
function promptMetric(kind,value){promptMetrics[kind].push(value);promptMetrics[kind].splice(0,Math.max(0,promptMetrics[kind].length-400));}
function promptEnabled(){return !!(session?.supportsLivePrompt&&session.promptUrl);}
function promptControls(){
 const enabled=promptEnabled();$('eventPrompt').disabled=!enabled;$('eventMode').disabled=!enabled;
 $('submitPrompt').disabled=!enabled||promptSending||!$('eventPrompt').value.trim();
 $('submitPrompt').textContent=promptSending?'Submitting…':latestPrompt?.error&&(latestPrompt.stage===0||latestPrompt.errorKind==='model')?'Retry →':'Submit event →';
 if(!session){$('eventStatus').textContent='Enter the world to submit live events.';$('eventStatus').dataset.error='false';}
 else if(!enabled){$('eventStatus').textContent='Live events are unavailable. Please refresh and try again later.';$('eventStatus').dataset.error='false';}
}
function renderPrompt(){
 promptControls();if(!promptEnabled()||!latestPrompt)return;
 const value=latestPrompt;
 $('eventStatus').dataset.error=String(!!value.error);
 $('eventStatus').textContent=value.error?`Submission failed: ${value.error}`:
  value.stage>=3?'Now showing the new prompt':value.stage>=2?'Applied to generation; awaiting playback':
  value.stage>=1?(paused?'Submitted; will apply after you resume':'Submitted; waiting for the model'):'Submitting event…';
}
function resetPrompt(){latestPrompt=null;promptSending=false;promptWindows=new Map();for(const key of Object.keys(promptMetrics))promptMetrics[key]=[];$('eventPrompt').value='';renderPrompt();}
function markPromptPlayed(frame){
 const value=latestPrompt;if(!value||!value.revision||value.stage>=3||value.error)return;
 const window=promptWindows.get(value.revision);if(!window||frame.index<window.firstOutputFrame)return;
 value.stage=3;promptMetric('displayed',{revision:value.revision,requestId:value.requestId,frame:frame.index,firstOutputFrame:window.firstOutputFrame,shownAt:frame.shownAt,submitToDisplayMs:frame.shownAt-value.at});renderPrompt();
}
function updatePromptState(data){
 for(const item of data.promptWindows||[]){
  if(!Number.isSafeInteger(item.revision)||item.revision<=0||!Number.isSafeInteger(item.firstOutputFrame))continue;
  if(!promptWindows.has(item.revision)){promptWindows.set(item.revision,item);promptMetric('applied',{...item,observedAt:performance.now()});}
 }
 while(promptWindows.size>400)promptWindows.delete(promptWindows.keys().next().value);
 const value=latestPrompt;if(!value)return;
 const accepted=data.promptRequest;
 if(accepted?.requestId===value.requestId&&Number.isSafeInteger(accepted.revision)&&accepted.revision>0){value.revision=accepted.revision;value.stage=Math.max(value.stage,1);if(value.errorKind==='network'){value.error=null;value.errorKind=null;}}
 if(value.revision){
  const window=promptWindows.get(value.revision);
  if(window||data.promptRevision===value.revision){value.stage=Math.max(value.stage,2);if(value.errorKind==='network'){value.error=null;value.errorKind=null;}}
  if(data.promptError?.revision===value.revision&&value.stage<2){value.error=data.promptError.message||'The model could not apply this event. Please resubmit.';value.errorKind='model';}
  if(window){const frame=latency.frames.find(f=>f.index>=window.firstOutputFrame&&f.index<=window.lastOutputFrame);if(frame)markPromptPlayed(frame);}
 }
 renderPrompt();
}
function promptRequestId(){if(globalThis.crypto?.randomUUID)return crypto.randomUUID();const bytes=new Uint8Array(16);crypto.getRandomValues(bytes);return Array.from(bytes,b=>b.toString(16).padStart(2,'0')).join('');}
async function submitPrompt(){
 if(!promptEnabled()||promptSending)return;
 const text=$('eventPrompt').value.trim(),mode=$('eventMode').value;if(!text)return;
 if(Array.from(text).length>2000){$('eventStatus').textContent='Event prompts can contain up to 2,000 characters.';$('eventStatus').dataset.error='true';return;}
 const sid=session.id;
 if(!latestPrompt||latestPrompt.text!==text||latestPrompt.mode!==mode||latestPrompt.errorKind==='model'){latestPrompt={text,mode,requestId:promptRequestId(),revision:null,stage:0,at:performance.now(),error:null};}
 const value=latestPrompt;value.error=null;value.errorKind=null;promptSending=true;
 promptMetric('submissions',{requestId:value.requestId,text,mode,at:performance.now(),paused});renderPrompt();
 const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),12000);
 try{
  const result=await(await request(session.promptUrl,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:text,mode,requestId:value.requestId}),signal:controller.signal})).json();
  if(session?.id!==sid||latestPrompt!==value)return;
  const accepted=result.promptRequest||result;
  if(!Number.isSafeInteger(accepted.revision)||accepted.revision<=0)throw new Error('No event confirmation received. Please retry.');
  value.revision=accepted.revision;value.stage=Math.max(value.stage,1);if(value.errorKind==='network'){value.error=null;value.errorKind=null;}
  updatePromptState({...result,promptRequest:{...accepted,requestId:value.requestId}});
 }catch(error){
  if(session?.id!==sid||latestPrompt!==value)return;
  if(value.stage===0){value.error=error.name==='AbortError'?'Connection timed out. Please retry; your event will not be duplicated.':error.message;value.errorKind='network';}
 }finally{clearTimeout(timer);if(session?.id===sid&&latestPrompt===value){promptSending=false;renderPrompt();}}
}
$('submitPrompt').onclick=submitPrompt;
$('eventPrompt').addEventListener('input',promptControls);
for(const id of ['eventPrompt','eventMode'])$(id).addEventListener('focus',()=>release());

function matchLatency(){
 for(const event of latency.inputs){
  if(event.matched)continue;
  const window=latency.windows.find(w=>w.inputs.some(i=>i.sequence===event.sequence));
  if(!window)continue;
  const adopted=window.inputs.find(i=>i.sequence===event.sequence);
  const frame=latency.frames.find(f=>f.index>=adopted.firstOutputFrame&&f.index<=window.lastOutputFrame&&f.shownAt>=event.at);
  if(!frame)continue;
  event.matched=true;
  latency.samples.push({sequence:event.sequence,keys:event.keys,chunk:window.chunk,frame:frame.index,
   eventToDisplayMs:frame.shownAt-event.at,receiveToLoadMs:frame.loadedAt-frame.receivedAt,
   loadToRafMs:frame.shownAt-frame.loadedAt});
 }
 latency.samples.splice(0,Math.max(0,latency.samples.length-400));
}
$('world').addEventListener('load',()=>{
 const element=$('world'),index=Number(element.dataset.frameIndex),src=element.src;
 if(!session||!index||!src.startsWith('data:image/jpeg'))return;
 const loadedAt=performance.now(),receivedAt=Number(element.dataset.receivedAt);
 requestAnimationFrame(()=>{
  if(element.src!==src)return;
  latency.frames.push({index,receivedAt,loadedAt,shownAt:performance.now()});
  latency.frames.splice(0,Math.max(0,latency.frames.length-3000));matchLatency();
 });
});
const phases={playback_wait:'Continuous playback',starting:'Entering world',queued:'Waiting for GPU',generating:'Generating world',streaming:'Generated view',waiting:'Preparing next update',paused:'Paused',stopped:'Exploration ended',error:'Generation error'};
let warmupCompleted=0;
function renderPlaybackStatus(value){
 if(!session)return;
 const status=$('modelStatus'),total=session.playbackSettings?.warmupChunks||0;
 const warming=warmupCompleted<total,buffering=value.state==='buffering';
 status.textContent=paused?'Paused':warming?`Warmup ${warmupCompleted}/${total}`:buffering?'Buffering':'Exploring';
 status.dataset.state=paused?'ready':warming||buffering?'loading':'active';
 status.title=warming?'Building world history; warmup frames are not shown':buffering?`${(value.ready/24).toFixed(1)} s of video ready`:'';
}
let workspaceActive=false;
function selectPanel(name,focus=false){
 for(const [key,tab,panel] of [['setup','tabSetup','setup'],['events','tabEvents','eventPanel']]){
  const active=key===name;$(panel).hidden=!active;$(tab).setAttribute('aria-selected',String(active));$(tab).tabIndex=active?0:-1;
  if(active&&focus)$(tab).focus();
 }
}
for(const [name,id] of [['setup','tabSetup'],['events','tabEvents']]){
 $(id).addEventListener('click',()=>selectPanel(name));
 $(id).addEventListener('keydown',event=>{
  if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();selectPanel(event.key==='Home'?'setup':event.key==='End'?'events':name==='setup'?'events':'setup',true);}
 });
}
function updateWorkspace(){const active=!!session;if(active!==workspaceActive){workspaceActive=active;selectPanel(active?'events':'setup');}}
function sceneControls(){updateWorkspace();promptControls();const locked=!!session||starting;$('prompt').disabled=locked;$('reference').disabled=locked;$('quality').disabled=locked;$('vaePrecision').disabled=locked;$('warmupChunks').disabled=locked;document.querySelectorAll('.reference-option').forEach(b=>b.disabled=locked);$('sceneHint').textContent=locked?'Hold WASD to move and IJKL to look. Use Live Events to change the prompt. End exploration to change the scene or precision.':'Choose a scene and enter. Hold WASD to move, IJKL to look; Esc to pause.';}
function selectReference(item){if(session||starting)return;reference=null;referencePath=item.imageUrl;if(referenceUrl?.startsWith('blob:'))URL.revokeObjectURL(referenceUrl);referenceUrl=url(item.imageUrl);$('world').src=referenceUrl;$('prompt').value=item.prompt;$('selectedReference').textContent=item.title;$('reference').value='';document.querySelectorAll('.reference-option').forEach(b=>{const selected=b.dataset.reference===item.id;b.classList.toggle('selected',selected);b.setAttribute('aria-pressed',String(selected));});}
function notice(text){$('notice').textContent=text;clearTimeout(noticeTimer);noticeTimer=setTimeout(()=>$('notice').textContent='',5000);}
function send(value){if(socket?.readyState===WebSocket.OPEN)inputMailbox?.push(value);}
function pressed(){return [...new Set([...keys,...pointers.values()])].sort();}
function input(){
 const state={keys:paused||(session?.playbackSettings?.warmupChunks>0&&!framePlayer?.started)?[]:pressed(),speed:Number($('speed').value),lookSpeed:Number($('lookSpeed').value),paused};
 const signature=JSON.stringify(state);if(signature!==lastInput){seq++;lastInput=signature;if(session){latency.inputs.push({sequence:seq,at:performance.now(),keys:state.keys,paused});latency.inputs.splice(0,Math.max(0,latency.inputs.length-400));}}
 send({type:'input',sequence:seq,...state,...(framePlayer?{playedFrame:framePlayer.played}:{})});
 document.querySelectorAll('[data-key]').forEach(el=>el.classList.toggle('active',state.keys.includes(el.dataset.key)));
}
function release(){keys.clear();pointers.clear();input();}
function pause(value){if(!session)return;paused=value;renderPrompt();framePlayer?.setPaused(value);release();$('pause').textContent=paused?'Resume':'Pause';$('overlay').hidden=!paused;
 if(paused){$('overlay').innerHTML='<strong>Exploration paused</strong><span>Click the view to resume. The current chunk may still be generating.</span>';}else{$('viewport').focus();if(framePlayer){framePlayer.lastState='';framePlayer.emitState();}input();}}
function connect(){
 if(!session)return;
 const address=new URL(session.inputUrl,base);address.protocol=location.protocol==='https:'?'wss:':'ws:';
 socket=new WebSocket(address);
 const current=socket;const mailbox=new InputMailbox(value=>current.send(JSON.stringify(value)),{onError:message=>{notice(message);finish();}});
 inputMailbox?.close();inputMailbox=mailbox;window.evokeInputDelivery=mailbox.metrics;
 socket.onopen=()=>{if(socket!==current)return;connected=true;$('connection').textContent='● Controls connected';input();};
 socket.onmessage=event=>{
  if(socket!==current||!session)return;
  const data=JSON.parse(event.data);
  if(data.type==='pong')return;
  if(data.type==='error'){notice(data.message);current.close();return;}
  if(data.type!=='state')return;
  mailbox.acknowledge(data);
  updatePromptState(data);
  warmupCompleted=Number(data.warmupCompleted||0);
  const warming=warmupCompleted<Number(session.playbackSettings?.warmupChunks||0);
  if(framePlayer)renderPlaybackStatus({state:framePlayer.buffering?'buffering':'playing',ready:framePlayer.readyCount()});
  if(data.inputWindows){latency.windows=data.inputWindows;matchLatency();}
  $('phase').textContent=framePlayer?(paused?'Paused':framePlayer.buffering?'World preview':'Continuous playback · 24 fps'):(phases[data.phase]||'Waiting for model');
  if(!framePlayer)$('worldTime').textContent=`${((data.latestFrame||0)/24).toFixed(1)} s`;
  if(data.camera)$('position').textContent=`X ${data.camera.x.toFixed(1)} / Z ${data.camera.z.toFixed(1)}`;
  $('inputStatus').textContent=paused?'Paused · Click the view to resume':seq>(framePlayer?playedSequence:(data.displayedSequence??data.appliedSequence??-1))?'Input received · Awaiting next update':'Current input applied to generation';
  if(data.chunkSeconds)$('latency').textContent=`Last update ${data.chunkSeconds.toFixed(1)} s`;
  if(warming){$('phase').textContent='Hidden warmup';$('inputStatus').textContent='Controls become available after warmup, when playback begins';}
  if(data.latestFrame>0&&!paused&&!framePlayer)$('overlay').hidden=true;
  if(data.phase==='error'){notice(data.message||'Generation failed');finish(false);}
  if(data.phase==='stopped'&&session&&!framePlayer){notice('Exploration ended. Choose a scene to start again.');finish(false);}
 };
 socket.onclose=event=>{mailbox.close();if(socket!==current)return;if(event.code===1008){finish(false);notice('This session has ended. Please re-enter the world.');return;}connected=false;$('connection').textContent='Disconnected · Reconnecting';release();if(session)reconnect=setTimeout(connect,1000);};
 socket.onerror=()=>current.close();
}
async function request(path,options){const r=await fetch(url(path),options);if(!r.ok){const v=await r.json().catch(()=>({}));throw new Error(typeof v.detail==='string'?v.detail:`Request failed ${r.status}`);}return r;}
async function finish(sendStop=true){
 frameSource?.close();frameSource=null;framePlayer?.stop();framePlayer=null;$('playback').hidden=true;$('world').hidden=false;
 const old=session;session=null;inputMailbox?.close();inputMailbox=null;connected=false;clearTimeout(reconnect);release();socket?.close();socket=null;
 if(sendStop&&old)try{await request(`api/live/${old.id}/stop`,{method:'POST'});}catch(error){notice(error.message);}
 document.body.classList.remove('playing');$('pause').disabled=true;$('stop').disabled=true;$('overlay').hidden=false;
 $('overlay').innerHTML='<strong>Start exploring from this image</strong><span>WASD to move · IJKL to look</span>';
 $('world').src=referenceUrl||url('api/default-case/reference');$('connection').textContent='Not started';$('phase').textContent='Starting image';
 $('inputStatus').textContent='Click Enter world to begin';sceneControls();health();
}
$('start').onclick=async()=>{
 if(starting||session)return;starting=true;$('start').disabled=true;sceneControls();
 try{
  const prompt=$('prompt').value.trim();if(!prompt)throw new Error('Please describe the world you want to explore.');
  const image=reference||await (await request(referencePath)).blob();
  const form=new FormData();form.append('reference',image,reference?.name||'reference.jpg');form.append('prompt',prompt);form.append('quality',$('quality').value);form.append('vae_precision',$('vaePrecision').value);form.append('playback_mode','continuous');form.append('warmup_chunks',$('warmupChunks').value);form.append('ahead_chunks','1');
  session=await (await request('api/live/sessions',{method:'POST',body:form})).json();
  if(session.vaePrecision!==$('vaePrecision').value||session.playbackMode!=='continuous'||session.playbackSettings?.aheadChunks!==1||session.playbackSettings?.warmupChunks!==Number($('warmupChunks').value)){await finish();throw new Error('Continuous playback is unavailable. Please refresh and try again later.');}
  paused=false;warmupCompleted=0;seq=0;lastInput='';resetPrompt();for(const key of Object.keys(latency))latency[key]=[];document.body.classList.add('playing');$('pause').disabled=false;$('stop').disabled=false;$('pause').textContent='Pause';
  $('overlay').hidden=true;
  if(session.streamTransport==='sse'){
   const sid=session.id;playedSequence=-1;
   framePlayer=new ContinuousFramePlayer($('playback'),{
    ...session.playbackSettings,
    onFrame:frame=>{
     $('world').hidden=true;$('playback').hidden=false;$('worldTime').textContent=`${(frame.index/24).toFixed(1)} s`;
     latency.frames.push({...frame,loadedAt:frame.decodedAt});latency.frames.splice(0,Math.max(0,latency.frames.length-3000));
     const window=latency.windows.find(w=>w.firstOutputFrame<=frame.index&&w.lastOutputFrame>=frame.index);
     if(window)for(const entry of window.inputs){if(entry.firstOutputFrame<=frame.index&&entry.sequence!=null)playedSequence=Math.max(playedSequence,entry.sequence);}
     matchLatency();markPromptPlayed(frame);
    },
    onState:value=>{
     if(!session||session.id!==sid)return;
     renderPlaybackStatus(value);
     if(value.state==='buffering'&&!paused){$('overlay').hidden=true;$('phase').textContent='World preview';}
     if(value.state==='playing'&&!paused){$('overlay').hidden=true;$('phase').textContent='Continuous playback · 24 fps';}
    },
    onError:message=>{notice(message);finish();},
    onEnd:()=>{if(session?.id===sid){notice('Exploration ended.');finish(false);}}
   });
   window.evokePlayback=framePlayer.metrics;
   frameSource=new EventSource(url(session.streamUrl));
   frameSource.addEventListener('frame',event=>{
    if(session?.id!==sid)return;
    framePlayer?.enqueue(Number(event.lastEventId),event.data,performance.now());
   });
   frameSource.addEventListener('gap',()=>framePlayer?.fail('Frames were lost during a disconnect. Re-enter the world to resume continuous playback.'));
   frameSource.addEventListener('done',()=>{frameSource?.close();framePlayer?.end();});
   frameSource.addEventListener('error',event=>{if(event.data){framePlayer?.fail('Generation was interrupted. Please re-enter the world.');}});
  }else{$('world').src=url(session.streamUrl);}
  connect();$('viewport').focus();
 }catch(error){notice(error.message);}finally{starting=false;sceneControls();health();}
};
$('pause').onclick=()=>pause(!paused);$('stop').onclick=()=>finish();
$('viewport').onclick=()=>{if(session){if(paused)pause(false);else $('viewport').focus();}};
window.addEventListener('keydown',event=>{
 if(event.code==='Escape'&&session){event.preventDefault();pause(true);return;}
 const focused=document.activeElement;
 if(!session||paused||!connected||(session.playbackSettings?.warmupChunks>0&&!framePlayer?.started)||(focused?.tagName==='TEXTAREA'||focused?.tagName==='SELECT'||(focused?.tagName==='INPUT'&&focused.type!=='range'))||event.ctrlKey||event.metaKey||event.altKey)return;
 const key=codes[event.code];if(key){event.preventDefault();keys.add(key);input();}
});
window.addEventListener('keyup',event=>{const key=codes[event.code];if(key){keys.delete(key);input();}});
window.addEventListener('blur',()=>{if(session)pause(true);});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&session)pause(true);});
for(const button of document.querySelectorAll('[data-key]')){
 button.addEventListener('pointerdown',event=>{if(!session||paused||(session.playbackSettings?.warmupChunks>0&&!framePlayer?.started))return;event.preventDefault();button.setPointerCapture(event.pointerId);pointers.set(event.pointerId,button.dataset.key);input();});
 for(const type of ['pointerup','pointercancel','lostpointercapture'])button.addEventListener(type,event=>{pointers.delete(event.pointerId);input();});
 button.addEventListener('contextmenu',event=>event.preventDefault());
}
$('speed').oninput=()=>{$('speedValue').value=Number($('speed').value).toFixed(1);input();};
$('lookSpeed').oninput=()=>{$('lookValue').value=`${$('lookSpeed').value}°/s`;input();};
$('fullscreen').onclick=async()=>{try{await $('viewport').requestFullscreen();$('viewport').focus();}catch{notice('Your browser could not enter fullscreen.');}};
$('reference').onchange=()=>{
 const file=$('reference').files[0];if(!file)return;
 if(!['image/png','image/jpeg','image/webp'].includes(file.type)||file.size>25*1024*1024){$('reference').value='';return notice('Choose a JPG, PNG, or WebP image under 25 MB.');}
 reference=file;if(referenceUrl?.startsWith('blob:'))URL.revokeObjectURL(referenceUrl);$('selectedReference').textContent=file.name;document.querySelectorAll('.reference-option').forEach(b=>{b.classList.remove('selected');b.setAttribute('aria-pressed','false');});referenceUrl=URL.createObjectURL(file);$('world').src=referenceUrl;
};
async function health(){try{const h=await (await request('api/health')).json();const status=$('modelStatus');status.textContent=h.ready?(session?'Exploring':h.runtime?.phase==='generating'?'Service running':'Ready'):h.runtime?.phase==='error'?'Unavailable':'Preparing';status.dataset.state=h.ready?(session?'active':'ready'):h.runtime?.phase==='error'?'error':'loading';$('start').disabled=!h.ready||starting||!!session;if(session&&framePlayer)renderPlaybackStatus({state:framePlayer.buffering?'buffering':'playing',ready:framePlayer.readyCount()});else status.title='';}catch{$('modelStatus').textContent='Connecting';$('modelStatus').dataset.state='loading';$('start').disabled=true;}}
setInterval(()=>{if(session)input();},100);setInterval(health,5000);health();
request('api/live/references').then(r=>r.json()).then(data=>{for(const item of data.items){const button=document.createElement('button');button.type='button';button.className='reference-option';button.dataset.reference=item.id;button.setAttribute('aria-label',item.title);const image=document.createElement('img');image.src=url(item.imageUrl);image.alt=item.title;const title=document.createElement('span');title.textContent=item.title;button.append(image,title);button.onclick=()=>selectReference(item);$('referenceList').append(button);}const initial=data.items.find(item=>item.id==='meteor')||data.items[0];if(initial)selectReference(initial);sceneControls();}).catch(error=>notice(error.message));
window.addEventListener('pagehide',()=>{framePlayer?.stop();frameSource?.close();if(session){navigator.sendBeacon(url(`api/live/${session.id}/stop`));socket?.close();}});
