"""Local web UI for labeling exceptional ``press_by_number`` intervals.

Run this script locally::

    python press_by_number_labeler.py --port 8765

The browser connects to this local server. The server uses SSH to read the
remote videos and to write the annotation JSON; remote videos are not copied
to the local disk.
"""

from __future__ import annotations

import argparse
import html
import json
import shlex
import tempfile
import subprocess
import os
import threading
import glob
try:
    import yaml
except ImportError:
    yaml = None
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


SSH = ["ssh", "-p", "41070", "root@183.233.148.6"]
REMOTE_VIDEO = (
    "/mnt/public2/liushengbang/data/RoboDojo_Dataset_to_VMB/press_by_number"
)
REMOTE_ANNOT = (
    "/mnt/public2/xiachenxiang/data/VOC-MEM/press_by_number/"
    "exceptional_intervals.json"
)
LOCAL_ANNOT = os.path.join(os.path.dirname(__file__), "organize_table_sia_annotations.json")
LOCAL_SCREENSHOTS = os.path.join(os.path.dirname(__file__), "press_by_number_screenshots")
FPS = 25
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "tasks.yaml")
ANNOTATION_LOCK = threading.Lock()
ALLOWED_VIDEO_GROUPS = ('ST-1', 'ST-HQ-EMB', 'ST-ENV', 'ST-HQ-ENV')

TASKS = {
    'organize_table': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/organize_table',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/organize_table/exceptional_intervals.json',
                       'metrics': {'SIA': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
    'store_laptop_and_headphones': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/store_laptop_and_headphone',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/store_laptop_and_headphones/exceptional_intervals.json',
                       'metrics': {'SIA': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
    'arrange_largest_number': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/arrange_largest_number',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/arrange_largest_number/exceptional_intervals.json',
                       'metrics': {'SIA': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
    'fold_clothes': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/fold_clothes',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/fold_clothes/exceptional_intervals.json',
                       'metrics': {'SIA': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
    'hang_mugs': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/hang_mugs',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/hang_mugs/exceptional_intervals.json',
                       'metrics': {'SIA': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
    'make_toast': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/make_toast',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/make_toast/exceptional_intervals.json',
                       'metrics': {'SIA': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
    'put_bottles_into_dustbin': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/put_bottles_into_dustbin',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/put_bottles_into_dustbin/exceptional_intervals.json',
                       'metrics': {'SIA': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
    'stack_blocks': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/stack_blocks',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/stack_blocks/exceptional_intervals.json',
                       'metrics': {'SIA': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
    'sweep_block': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/sweep_block',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/sweep_block/exceptional_intervals.json',
                       'metrics': {'SIA': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
}

def load_tasks():
    global TASKS
    if yaml and os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, encoding='utf-8') as f:
            TASKS = yaml.safe_load(f) or TASKS
load_tasks()
CURRENT_TASK = 'press_by_number'


def task_config(task: str, metric: str = 'SIA') -> dict:
    task_data = TASKS[task]
    metric_data = task_data['metrics'][metric]
    return {**task_data, **metric_data}


PAGE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>仿真数据标注</title>
  <style>
    body { font: 16px system-ui; margin: 24px; background: #f5f6f8; }
    main { max-width: 1100px; margin: auto; background: white; padding: 20px;
           border-radius: 12px; }
    video { width: 100%; max-height: 650px; background: #111; }
    select, button { font-size: 16px; padding: 8px; margin: 5px; }
    #timeline { position: relative; height: 28px; margin: 8px 0 16px; background: #dfe4ea; border-radius: 5px; cursor: pointer; }
    #progress { height: 100%; width: 0; background: #9bb7d4; border-radius: 5px; pointer-events: none; }
    .marker { position: absolute; top: -4px; width: 4px; height: 36px; transform: translateX(-2px); cursor: pointer; }
    .marker b { position: absolute; top: -24px; left: -5px; font-size: 13px; }
    .marker-b { background: #16803c; }.marker-s { background: #d28b00; }.marker-e { background: #c53030; }
    .key { display: inline-block; padding: 5px 10px; background: #eee;
           border-radius: 5px; }
    #status { margin: 12px 0; color: #174d2b; }
    table { border-collapse: collapse; }
    td, th { padding: 5px 12px; border-bottom: 1px solid #ddd; }
  </style>
</head>
<body>
  <main>
    <h1>仿真数据标注</h1>
    <p>选择 episode 的头视角，点击播放后使用
      <span id="key-help"></span>。按键记录当前视频帧；SIA 下按 <span class="key">c</span> 撤销最近一次标注。</p>
    <label>task <select id="task" onchange="changeConfig()"></select></label><label>metric <select id="metric" onchange="changeConfig()"></select></label>
    <select id="ep" onchange="loadVideo()"></select>
    <button onclick="loadVideo()">加载头视角</button>
    <button onclick="save()">保存并发送到远程</button>
    <button onclick="clearMarks()">清空本集</button>
    <button onclick="transfer()">Transfer all</button>
    <video id="v" controls tabindex="0"></video>
    <label>播放速度：<select id="speed" onchange="setPlaybackRate()"><option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="1.5">1.5x</option><option value="2">2x</option><option value="4">4x</option></select></label>
    <div id="timeline" title="点击跳转"><div id="progress"></div></div>
    <div id="status"></div>
    <table><thead><tr><th>类型</th><th>帧号</th><th>时间</th></tr></thead>
      <tbody id="rows"></tbody></table>
  </main>
  <script>
    const TASKS = __TASKS__;
    const DEFAULT_TASK = __DEFAULT_TASK__;
    const ALLOWED_KEYS = ['b', 's', 'e'];
    let task = DEFAULT_TASK, metric = 'SIA', episodes = [], marks = [], saved = {}, requestVersion = 0, v = document.getElementById('v');
    const speedEl = document.getElementById('speed');
    function setPlaybackRate() {
      const rate = Number(speedEl.value) || 1;
      v.defaultPlaybackRate = rate;
      v.playbackRate = rate;
    }
    v.addEventListener('loadedmetadata', setPlaybackRate);
    let ep = document.getElementById('ep');
    const manualLoadButton = document.querySelector('button[onclick="loadVideo()"]');
    if (manualLoadButton) manualLoadButton.remove();
    const legacySaveButton = document.querySelector('button[onclick^="save"]');
    if (legacySaveButton) legacySaveButton.remove();
    const taskEl=document.getElementById('task'), metricEl=document.getElementById('metric');
    taskEl.innerHTML=Object.keys(TASKS).map(x=>`<option value="${x}">${x}</option>`).join('');
    taskEl.value=task;
    function changeConfig(){ task=taskEl.value; fetchData(); }
    function fetchData(){
      const previousMetric=metricEl.value;
      const metricNames=Object.keys(TASKS[task].metrics);
      metricEl.innerHTML=metricNames.map(x=>`<option value="${x}">${x}</option>`).join('');
      metricEl.value=metricNames.includes(previousMetric) ? previousMetric : metricNames[0];
      metric=metricEl.value;
      const requestedTask=task, requestedMetric=metric, currentRequest=++requestVersion;
      Promise.all([
        fetch('/annotations?task='+encodeURIComponent(requestedTask)+'&metric='+encodeURIComponent(requestedMetric)).then(r=>r.json()),
        fetch('/episodes?task='+encodeURIComponent(requestedTask)).then(r=>r.json()),
      ]).then(([a,xs])=>{
        if (currentRequest !== requestVersion || task !== requestedTask || metric !== requestedMetric) return;
        saved=a; episodes=xs;
        ep.innerHTML=xs.map(x=>`<option value="${x}">${saved[x]?'✓':'○'} ${x.split('/').slice(-5,-4)[0]}</option>`).join('');
        if (xs.length) loadVideo();
        else { marks=[]; v.removeAttribute('src'); v.load(); render(); document.getElementById('status').textContent='当前任务没有可标注视频'; }
      }).catch(error=>document.getElementById('status').textContent='加载失败：'+error.message);
    }
    fetchData();
    /* legacy initialization disabled */
    /* fetch('/annotations').then(r=>r.json()).then(a=>{saved=a; return fetch('/episodes');}).then(r => r.json()).then(xs => {
      ep.innerHTML = xs.map(x => `<option value="${x}">${saved[x] ? '✓ ' : '○ '}${x.replace(/^.*press_by_number\//, '')}</option>`).join('');
      loadVideo();
    }); */
    function savedMarksFor(episode) {
      const record=saved[episode];
      if (Array.isArray(record)) return record;
      return Array.isArray(record?.marks) ? record.marks : [];
    }
    function loadVideo() {
      marks = savedMarksFor(ep.value).map(x=>({...x}));
      render();
      setPlaybackRate();
      v.pause();
      v.removeAttribute('src');
      v.load();
      v.src = '/video?task='+encodeURIComponent(task)+'&metric='+encodeURIComponent(metric)+'&episode=' + encodeURIComponent(ep.value);
      v.load();
      setPlaybackRate();
      v.focus();
    }
    function frame() { return Math.max(0, Math.round(v.currentTime * 25)); }
    function addMark(key) {
      key = String(key).toLowerCase();
      marks.push({type: key, frame: frame(), time: Number(v.currentTime.toFixed(3))});
      render();
      persistLocal();
      document.getElementById('status').textContent = 'Marked ' + key.toUpperCase() + ' at frame ' + frame();
    }
    function undoLastSiaMark() {
      if (metric !== 'SIA') return false;
      if (!marks.length) {
        document.getElementById('status').textContent = '当前没有可撤销的 SIA 标注';
        return true;
      }
      const removed = marks.pop();
      render();
      persistLocal();
      document.getElementById('status').textContent = '已撤销第 ' + removed.frame + ' 帧的 SIA 标注';
      return true;
    }
    document.addEventListener('keydown', e => {
      const key = (e.key || '').toLowerCase();
      if ((key === 'j' || key === 'k' || key === 'd') && e.target.tagName !== 'SELECT') { e.preventDefault(); const n=ep.selectedIndex+(key==='j'?-1:1); ep.selectedIndex=(n+episodes.length)%episodes.length; loadVideo(); return; }
      if (key === 'c' && e.target.tagName !== 'SELECT' && !e.ctrlKey && !e.altKey && !e.metaKey && undoLastSiaMark()) { e.preventDefault(); return; }
      if (!ALLOWED_KEYS.includes(key) || e.target.tagName === 'SELECT' || e.ctrlKey || e.altKey || e.metaKey) return;
      e.preventDefault();
      addMark(key);
      document.getElementById('status').textContent = '已标注 ' + key.toUpperCase() + '：第 ' + frame() + ' 帧（' + v.currentTime.toFixed(3) + ' 秒）';
    });
    v.addEventListener('timeupdate', updateTimeline);
    v.addEventListener('loadedmetadata', updateTimeline);
    document.getElementById('timeline').addEventListener('click', e => {
      if (!v.duration) return;
      const r=e.currentTarget.getBoundingClientRect();
      v.currentTime=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*v.duration;
    });
    function updateTimeline() {
      const hasDuration = Number.isFinite(v.duration) && v.duration > 0;
      const pct=hasDuration ? 100*v.currentTime/v.duration : 0;
      document.getElementById('progress').style.width=pct+'%';
      document.querySelectorAll('.marker').forEach(x=>x.remove());
      marks.forEach(m=>{
        const el=document.createElement('div'); el.className='marker marker-'+m.type;
        const position = hasDuration ? Math.max(0, Math.min(100, 100*m.time/v.duration)) : 0;
        el.style.left=position+'%'; el.title=m.type+' @ '+m.time+'s';
        el.innerHTML='<b>'+m.type+'</b>';
        el.onclick=ev=>{ev.stopPropagation();v.currentTime=m.time};
        document.getElementById('timeline').appendChild(el);
      });
    }
    function render() { document.getElementById('rows').innerHTML = marks.map(m =>
      `<tr><td>${m.type}</td><td>${m.frame}</td><td>${m.time}s</td></tr>`).join('');
      updateTimeline();
      document.getElementById('status').textContent = `当前 ${ep.value || ''}：${marks.length} 个标注点`; }
    function clearMarks() { marks = []; render(); }
    function persistLocal() { fetch('/save', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({episode:ep.value, marks, duration:v.duration, task, metric})}).then(r => r.json()).then(x =>
      { if(x.ok){saved[ep.value]={marks:marks.map(mark=>({...mark}))}; ep.options[ep.selectedIndex].textContent='✓ '+ep.options[ep.selectedIndex].textContent.replace(/^[✓○] /,''); render();} document.getElementById('status').textContent = x.message || x.error; }); }
    async function save() {
      const saveImages = confirm('是否同时在本地保存各标注时间点的视频截图？');
      if (saveImages) {
        if (!v.videoWidth || !v.videoHeight) { alert('视频尚未加载完成，无法截图'); return; }
        const canvas = document.createElement('canvas');
        canvas.width = v.videoWidth; canvas.height = v.videoHeight;
        const ctx = canvas.getContext('2d');
        async function seekToFrame(time) {
          v.pause();
          if (Math.abs(v.currentTime - time) > 0.001) {
            await new Promise(resolve => {
              const onSeeked = () => { v.removeEventListener('seeked', onSeeked); resolve(); };
              v.addEventListener('seeked', onSeeked);
              v.currentTime = time;
            });
          }
          // seeked means the timestamp was reached, but not necessarily that
          // the decoded frame has already been painted to the video element.
          if (v.requestVideoFrameCallback) {
            await new Promise(resolve => v.requestVideoFrameCallback(() => resolve()));
          } else {
            await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
          }
        }
        for (const mark of marks) {
          await seekToFrame(mark.time);
          ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
          const response = await fetch('/screenshot', {method:'POST', headers:{'Content-Type':'application/json'},
            body:JSON.stringify({episode: ep.value, frame: mark.frame})});
          if (!response.ok) throw new Error('截图保存失败：第 ' + mark.frame + ' 帧');
        }
      }
      persistLocal();
    }
    function transfer() {
      const directory = prompt('请输入远端存储目录的绝对路径：');
      if (!directory || !directory.startsWith('/')) return;
      fetch('/transfer', {method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({directory, task, metric})}).then(r=>r.json()).then(x=>
        document.getElementById('status').textContent=x.message||x.error); }
  </script>
</body>
</html>'''


def remote(command: str, data: bytes | None = None) -> bytes:
    """Run a command on the remote host through SSH."""
    result = subprocess.run(
        SSH + [command], input=data, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True,
    )
    return result.stdout


def local_task_root(task: str) -> str | None:
    root = task_config(task, 'SIA')['video_root']
    return root if os.path.isdir(root) else None


def allowed_video_path(path: str, root: str) -> bool:
    relative = os.path.relpath(path, root)
    group = relative.split(os.sep, 1)[0]
    return not relative.startswith('..' + os.sep) and group in ALLOWED_VIDEO_GROUPS


def episodes(task=None) -> list[str]:
    task = task or CURRENT_TASK
    root = local_task_root(task)
    if root:
        paths = []
        for group in ALLOWED_VIDEO_GROUPS:
            paths.extend(glob.glob(
                os.path.join(root, group, '**', 'observation.images.cam_high', '*.mp4'),
                recursive=True,
            ))
        return sorted(paths)
    remote_searches = ' '.join(
        f"if [ -d {root} ]; then find {root} -type f -path '*/observation.images.cam_high/*.mp4'; fi;"
        for root in (
            shlex.quote(os.path.join(task_config(task, 'SIA')['video_root'], group))
            for group in ALLOWED_VIDEO_GROUPS
        )
    )
    output = remote(f"{{ {remote_searches} }} | sort")
    return output.decode().splitlines()

def annotation_path(task=CURRENT_TASK, metric='SIA'):
    return os.path.join(os.path.dirname(__file__), f'{task}_{metric.lower()}_annotations.json')

def read_local(task=CURRENT_TASK, metric='SIA'):
    try:
        with open(annotation_path(task, metric), encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return {}

def write_local(data, task=CURRENT_TASK, metric='SIA'):
    path = annotation_path(task, metric)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def update_local_annotation(record):
    """Incrementally replace one episode while preserving every other record."""
    task = record['task']
    metric = record['metric']
    with ANNOTATION_LOCK:
        data = read_local(task, metric)
        data[record['episode']] = record
        write_local(data, task, metric)
    return data

def make_frames(record):
    marks = record.get('marks', [])
    config = task_config(record.get('task', CURRENT_TASK), record.get('metric', 'SIA'))
    if any(m.get('type') not in config['markers'] for m in marks):
        raise ValueError(f"invalid marker type in {record.get('episode')}")
    if config['kind'] == 'nodes':
        return sorted({int(m['frame']) for m in marks})
    types = [m['type'] for m in marks]
    if not types or types[0] != 'b':
        raise ValueError(f"markers must start with b: {record.get('episode')}")
    special, intervals, start = set(), [], None
    for m in marks:
        t, f = m['type'], int(m['frame'])
        special.add(f)
        if t == 'b':
            if start is not None: raise ValueError('b before previous e')
            start = f
        elif t == 'e':
            if start is None or f < start: raise ValueError('e without valid b')
            intervals.append((start, f)); start = None
        elif start is None: raise ValueError('s outside b-e interval')
    if start is not None: raise ValueError('missing e for b')
    total = max(max(e for _, e in intervals) + 1, round(float(record.get('duration', 0)) * FPS))
    return sorted(special | {i for i in range(0, total, 20)
                             if not any(b <= i <= e for b, e in intervals)})


def parse_byte_range(value: str | None, file_size: int) -> tuple[int, int] | None:
    if not value:
        return None
    if not value.startswith('bytes=') or ',' in value:
        raise ValueError('unsupported range')
    bounds = value[6:].split('-', 1)
    if len(bounds) != 2 or not any(bounds):
        raise ValueError('invalid range')
    if not bounds[0]:
        length = int(bounds[1])
        if length <= 0:
            raise ValueError('invalid suffix range')
        start = max(0, file_size - length)
        return start, file_size - 1
    start = int(bounds[0])
    end = int(bounds[1]) if bounds[1] else file_size - 1
    if start < 0 or start >= file_size or end < start:
        raise ValueError('range outside file')
    return start, min(end, file_size - 1)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def serve_video(self, path: str) -> None:
        file_size = os.path.getsize(path)
        try:
            requested_range = parse_byte_range(self.headers.get('Range'), file_size)
        except (TypeError, ValueError):
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{file_size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return

        start, end = requested_range or (0, file_size - 1)
        content_length = end - start + 1
        self.send_response(206 if requested_range else 200)
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(content_length))
        if requested_range:
            self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
        self.end_headers()

        with open(path, 'rb') as source:
            source.seek(start)
            remaining = content_length
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        request_task = query.get('task', [CURRENT_TASK])[0]
        request_metric = query.get('metric', ['SIA'])[0]
        try:
            if parsed.path == '/':
                body, content_type = PAGE.encode(), 'text/html; charset=utf-8'
            elif parsed.path == '/episodes':
                body, content_type = json.dumps(episodes(request_task)).encode(), 'application/json'
            elif parsed.path == '/annotations':
                all_data = read_local(request_task, request_metric); body, content_type = json.dumps(all_data).encode(), 'application/json'
            elif parsed.path == '/video':
                path = query.get('episode', [''])[0]
                root = task_config(request_task, request_metric)['video_root']
                if not (path.startswith(root + '/') and allowed_video_path(path, root) and
                        '/observation.images.cam_high/' in path and path.endswith('.mp4')):
                    raise ValueError('invalid video path')
                if local_task_root(request_task) and os.path.isfile(path):
                    self.serve_video(path)
                    return
                else:
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                        tmp_path = tmp.name
                    try:
                        with open(tmp_path, 'wb') as out:
                            subprocess.run(SSH + [f"cat -- {shlex.quote(path)}"], stdout=out, check=True)
                        self.serve_video(tmp_path)
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                    return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self.send_error(500, html.escape(str(exc)))

    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path == '/screenshot':
            try:
                length = int(self.headers.get('Content-Length', 0))
                obj = json.loads(self.rfile.read(length))
                episode = obj.get('episode', '')
                frame = int(obj.get('frame'))
                if frame < 0:
                    raise ValueError('invalid frame')
                selected_task = obj.get('task', CURRENT_TASK)
                root = task_config(selected_task)['video_root']
                if not (episode.startswith(root + '/') and allowed_video_path(episode, root) and episode.endswith('.mp4') and
                        '/observation.images.cam_high/' in episode):
                    raise ValueError('invalid episode path')
                episode_name = os.path.basename(episode.split('/videos/')[0].rstrip('/'))
                if not episode_name:
                    raise ValueError('invalid episode')
                timestamp = frame / FPS
                command = (f"ffmpeg -hide_banner -loglevel error -ss {timestamp:.6f} "
                           f"-i {shlex.quote(episode)} -frames:v 1 -f image2pipe "
                           "-vcodec mjpeg -")
                if local_task_root(selected_task) and os.path.isfile(episode):
                    result = subprocess.run(
                        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', f'{timestamp:.6f}',
                         '-i', episode, '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-'],
                        stdout=subprocess.PIPE, check=True,
                    )
                    raw = result.stdout
                else:
                    raw = remote(command)
                if not raw:
                    raise ValueError('ffmpeg did not return an image')
                directory = os.path.join(LOCAL_SCREENSHOTS, episode_name)
                os.makedirs(directory, exist_ok=True)
                with open(os.path.join(directory, f'{frame:08d}.jpg'), 'wb') as out:
                    out.write(raw)
                self.send_json({'ok': True})
            except Exception as exc:
                self.send_json({'ok': False, 'error': str(exc)}, 400)
            return
        if self.path == '/transfer':
            try:
                length = int(self.headers.get('Content-Length', 0))
                request = json.loads(self.rfile.read(length) or b'{}')
                directory = request.get('directory', '')
                if not isinstance(directory, str) or not directory.startswith('/'):
                    raise ValueError('远端存储目录必须是绝对路径')
                if any(part in ('', '.', '..') for part in directory.split('/')[1:]):
                    raise ValueError('远端存储目录路径不合法')
                data = read_local(request.get('task', CURRENT_TASK), request.get('metric', 'SIA'))
                if not data: raise ValueError('No locally processed episodes to transfer')
                output = [{'episode': k, 'frames': make_frames(v)} for k, v in data.items()]
                payload = json.dumps(output, ensure_ascii=False, indent=2).encode()
                destination = directory.rstrip('/') + '/exceptional_intervals.json'
                if os.path.isdir(os.path.dirname(directory)) or os.path.isdir(directory):
                    os.makedirs(directory, exist_ok=True)
                    with open(destination, 'wb') as output_file:
                        output_file.write(payload)
                else:
                    dest = shlex.quote(destination)
                    remote(f"mkdir -p {shlex.quote(directory)} && cat > {dest}", payload)
                self.send_json({'ok': True, 'message': f'Transferred {len(output)} episodes to {directory}'})
            except Exception as exc: self.send_json({'ok': False, 'error': str(exc)}, 400)
            return
        if self.path != '/save':
            self.send_error(404)
            return
        try:
            length = int(self.headers['Content-Length'])
            obj = json.loads(self.rfile.read(length))
            parts = obj.get('episode', '').split('/')
            try:
                selected_task = obj.get('task', CURRENT_TASK)
                video_root = task_config(selected_task, obj.get('metric', 'SIA'))['video_root']
                if not allowed_video_path(obj.get('episode', ''), video_root):
                    raise ValueError('selected video is outside the allowed ST groups')
                task_root = video_root.rstrip('/').split('/')[-1]
                idx = parts.index(task_root)
                obj.update(task=selected_task, metric=obj.get('metric', 'SIA'), group=parts[idx + 1],
                           episode_name=parts[idx + 2], gap=20)
            except (ValueError, IndexError):
                raise ValueError('selected video path does not contain task/group/episode')
            update_local_annotation(obj)
            self.send_json({'ok': True, 'message': 'Saved locally'})
            return
            body = json.dumps({'message': '已保存到远程 VOC-MEM/press_by_number/exceptional_intervals.json'}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self.send_error(500, html.escape(str(exc)))


def main() -> None:
    global CURRENT_TASK, REMOTE_VIDEO, REMOTE_ANNOT, LOCAL_ANNOT, LOCAL_SCREENSHOTS, PAGE
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--task', choices=sorted(TASKS), default='organize_table',
                        help='任务名；Verified_Data 下的新任务默认只使用 s 键')
    args = parser.parse_args()
    CURRENT_TASK = args.task
    config = task_config(CURRENT_TASK)
    REMOTE_VIDEO = config['video_root']
    REMOTE_ANNOT = config['annotation']
    LOCAL_ANNOT = annotation_path(CURRENT_TASK, 'SIA')
    LOCAL_SCREENSHOTS = os.path.join(os.path.dirname(__file__), f'{CURRENT_TASK}_screenshots')
    PAGE = PAGE.replace('__TASKS__', json.dumps(TASKS))
    PAGE = PAGE.replace('__DEFAULT_TASK__', json.dumps(CURRENT_TASK))
    PAGE = PAGE.replace(
        "const TASKS = __TASKS__;",
        f"const TASKS = {json.dumps(TASKS)}; const ALLOWED_KEYS = ['b','s','e'];",
    )
    PAGE = PAGE.replace(
        '<span id="key-help"></span>',
        ' '.join(f'<span class="key">{key}</span>' for key in config['markers']),
    )
    PAGE = PAGE.replace('press_by_number /', CURRENT_TASK + ' /')
    print(f'Open http://127.0.0.1:{args.port}')
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
