import time
import statistics
import numpy as np


def benchmark_buffer(pixel):
    import cv2
    buffer=np.empty_like(pixel)
    params=[cv2.IMWRITE_JPEG_QUALITY,85]
    ok,expected=cv2.imencode('.jpg',cv2.cvtColor(pixel,cv2.COLOR_RGB2BGR),params)
    if not ok:raise RuntimeError('JPEG reference encoding failed')
    rows=[]

    for sweep in range(12):
        for reuse in ((False,True) if sweep%2==0 else (True,False)):
            started=time.perf_counter()
            bgr=(cv2.cvtColor(pixel,cv2.COLOR_RGB2BGR,dst=buffer) if reuse else
                 cv2.cvtColor(pixel,cv2.COLOR_RGB2BGR))
            ok,jpeg=cv2.imencode('.jpg',bgr,params)
            elapsed=time.perf_counter()-started
            if not ok or not np.array_equal(expected,jpeg):
                raise AssertionError('JPEG buffer candidate changed encoded bytes')
            rows.append({'reuse':reuse,'warmup':sweep<2,'seconds':elapsed,'jpegBytesEqual':True})
    return {'shape':list(pixel.shape),'samples':rows,'diagnosticOnly':True,
            'meanMs':{name:1000*statistics.mean(r['seconds'] for r in rows if not r['warmup'] and r['reuse']==reuse)
                      for name,reuse in [('baseline',False),('reuse',True)]}}
