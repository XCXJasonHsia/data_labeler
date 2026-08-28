"""Local web UI for simulation-data labeling and read-only review.

Run this script locally::

    python labeler.py --port 8765
"""

from __future__ import annotations

import argparse
import fnmatch
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
LABEL_DATA_ROOT = '/mnt/public2/liushengbang/data/Veified_Data'
VERIFY_DATA_ROOT = '/mnt/public2/liushengbang/data/RoboDojo_Dataset_to_VMB'
REMOTE_VIDEO = (
    "/mnt/public2/liushengbang/data/Veified_Data/press_by_number"
)
REMOTE_ANNOT = (
    "/mnt/public2/xiachenxiang/data/VOC-MEM/press_by_number/"
    "exceptional_intervals.json"
)
LOCAL_ANNOT = os.path.join(os.path.dirname(__file__), "organize_table_sia_cspc_annotations.json")
LOCAL_SCREENSHOTS = os.path.join(os.path.dirname(__file__), "press_by_number_screenshots")
FPS = 25
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "tasks.yaml")
ANNOTATION_LOCK = threading.Lock()
DEFAULT_METRIC = 'SIA+CSPC'
METRIC_DEFINITIONS = {
    'SIA+CSPC': {
        'markers': ['s'],
        'kind': 'nodes',
        'groups': ['ST-1', 'ST-HQ-EMB', 'ST-HQ-ENV', 'ST-2'],
    },
    'VOC-MEM': {
        'markers': ['b', 's', 'e'],
        'kind': 'interval',
        'groups': ['ST-1', 'ST-HQ-EMB', 'ST-HQ-ENV'],
    },
    'FPL+TRR': {
        'markers': ['s'],
        'kind': 'nodes',
        'groups': ['FRT-*'],
    },
}
LEGACY_METRICS = {'SIA': 'SIA+CSPC'}
METRIC_FILE_SLUGS = {
    'SIA+CSPC': 'sia_cspc',
    'VOC-MEM': 'voc-mem',
    'FPL+TRR': 'fpl_trr',
}
VIDEO_CAMERA_DIRECTORIES = (
    'observation.images.cam_high',
    'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist',
)
DATASET_SIM_ROOT = '/mnt/public2/liushengbang/vmbmk/dataset_sim'
DATASET_TASK_ALIASES = {'sweep_block': 'sweep_blocks'}
DATASET_GROUP_DOMAINS = {
    'ST-1': 'id',
    'ST-HQ-EMB': 'emb',
    'ST-ENV': 'env',
    'ST-HQ-ENV': 'env',
}

TASKS = {
    'organize_table': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/organize_table',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/organize_table/exceptional_intervals.json',
                       'metrics': {'SIA+CSPC': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
    'store_laptop_and_headphones': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/store_laptop_and_headphone',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/store_laptop_and_headphones/exceptional_intervals.json',
                       'metrics': {'SIA+CSPC': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
    'arrange_largest_number': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/arrange_largest_number',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/arrange_largest_number/exceptional_intervals.json',
                       'metrics': {'SIA+CSPC': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
    'fold_clothes': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/fold_clothes',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/fold_clothes/exceptional_intervals.json',
                       'metrics': {'SIA+CSPC': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
    'hang_mugs': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/hang_mugs',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/hang_mugs/exceptional_intervals.json',
                       'metrics': {'SIA+CSPC': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
    'make_toast': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/make_toast',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/make_toast/exceptional_intervals.json',
                       'metrics': {'SIA+CSPC': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
    'put_bottles_into_dustbin': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/put_bottles_into_dustbin',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/put_bottles_into_dustbin/exceptional_intervals.json',
                       'metrics': {'SIA+CSPC': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
    'stack_blocks': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/stack_blocks',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/stack_blocks/exceptional_intervals.json',
                       'metrics': {'SIA+CSPC': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
    'sweep_block': {'video_root': '/mnt/public2/liushengbang/data/Veified_Data/sweep_blocks',
                       'annotation': '/mnt/public2/xiachenxiang/data/VOC-MEM/sweep_block/exceptional_intervals.json',
                       'metrics': {'SIA+CSPC': {'markers': ['s'], 'kind': 'nodes'},
                                   'VOC-MEM': {'markers': ['b', 's', 'e'], 'kind': 'interval'}}},
}

def canonical_metric(metric: str) -> str:
    return LEGACY_METRICS.get(metric, metric)


def path_within(path: str, root: str) -> bool:
    try:
        normalized_root = os.path.realpath(root)
        return os.path.commonpath((os.path.realpath(path), normalized_root)) == normalized_root
    except (TypeError, ValueError):
        return False


def load_tasks():
    global TASKS
    if yaml and os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, encoding='utf-8') as f:
            TASKS = yaml.safe_load(f) or TASKS
    for task, task_data in TASKS.items():
        if not path_within(task_data.get('video_root', ''), LABEL_DATA_ROOT):
            raise ValueError(f'{task} 的标注视频目录必须位于 {LABEL_DATA_ROOT}')
        configured_metrics = task_data.get('metrics', {})
        normalized_metrics = {}
        for metric, defaults in METRIC_DEFINITIONS.items():
            legacy_name = 'SIA' if metric == 'SIA+CSPC' else metric
            configured = configured_metrics.get(metric, configured_metrics.get(legacy_name, {}))
            normalized_metrics[metric] = {**defaults, **configured, 'groups': defaults['groups']}
        task_data['metrics'] = normalized_metrics
load_tasks()
CURRENT_TASK = 'press_by_number'
MODE = 'label'


def task_config(task: str, metric: str = DEFAULT_METRIC) -> dict:
    metric = canonical_metric(metric)
    task_data = TASKS[task]
    metric_data = task_data['metrics'][metric]
    return {**task_data, **metric_data}


PAGE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>仿真数据标注</title>
  <style>
    :root {
      color-scheme: light;
      --primary: #2563eb;
      --primary-dark: #1d4ed8;
      --ink: #0f172a;
      --muted: #64748b;
      --line: #e2e8f0;
      --surface: rgba(255, 255, 255, 0.96);
      --soft: #f8fafc;
      --video: #0b1220;
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      min-height: 100vh;
      margin: 0;
      padding: 28px;
      color: var(--ink);
      font: 15px/1.6 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 12% 4%, rgba(59, 130, 246, 0.16), transparent 30%),
        radial-gradient(circle at 92% 12%, rgba(14, 165, 233, 0.12), transparent 28%),
        #eef3f9;
    }
    main {
      max-width: 1440px;
      margin: auto;
      padding: 28px;
      border: 1px solid rgba(255, 255, 255, 0.8);
      border-radius: 22px;
      background: var(--surface);
      box-shadow: 0 24px 70px rgba(30, 64, 175, 0.12);
      backdrop-filter: blur(16px);
    }
    .page-header { display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-bottom: 18px; }
    .eyebrow { margin: 0 0 2px; color: var(--primary); font-size: 12px; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; }
    h1 { margin: 0; font-size: clamp(26px, 3vw, 38px); line-height: 1.2; letter-spacing: -0.04em; }
    .sync-badge { flex: none; padding: 8px 13px; border: 1px solid #bfdbfe; border-radius: 999px; color: #1e40af; background: #eff6ff; font-size: 13px; font-weight: 700; }
    .guide {
      margin: 0 0 18px;
      padding: 14px 16px;
      border: 1px solid #dbeafe;
      border-left: 4px solid var(--primary);
      border-radius: 12px;
      color: #334155;
      background: linear-gradient(135deg, #eff6ff, #f8fafc);
    }
    .key {
      display: inline-block;
      min-width: 28px;
      margin: 0 2px;
      padding: 1px 7px;
      border: 1px solid #cbd5e1;
      border-bottom-width: 2px;
      border-radius: 6px;
      color: #1e293b;
      background: white;
      font: 700 13px/1.7 ui-monospace, SFMono-Regular, Menlo, monospace;
      text-align: center;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }
    #verify-controls, .toolbar {
      margin-bottom: 18px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--soft);
    }
    #verify-controls p { margin: 10px 2px 0; color: var(--muted); }
    .verify-row { display: flex; align-items: end; gap: 10px; }
    .verify-field { flex: 1; }
    .toolbar { display: flex; align-items: end; flex-wrap: wrap; gap: 10px; }
    #label-controls { display: contents; }
    .field { display: flex; min-width: 150px; flex-direction: column; gap: 5px; color: #475569; font-size: 12px; font-weight: 700; }
    .episode-field { min-width: 260px; flex: 1 1 360px; }
    .episode-context { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin: 0 0 16px; }
    .context-item { padding: 10px 12px; border: 1px solid var(--line); border-radius: 10px; background: white; }
    .context-item span { display: block; margin-bottom: 2px; color: var(--muted); font-size: 11px; font-weight: 700; letter-spacing: 0.04em; }
    .context-item strong { color: #1e3a8a; font-size: 15px; }
    select, input {
      width: 100%;
      min-height: 40px;
      padding: 8px 11px;
      border: 1px solid #cbd5e1;
      border-radius: 9px;
      outline: none;
      color: var(--ink);
      background: white;
      font: inherit;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    select:focus, input:focus { border-color: #60a5fa; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.14); }
    button {
      min-height: 40px;
      padding: 8px 14px;
      border: 1px solid #cbd5e1;
      border-radius: 9px;
      color: #334155;
      background: white;
      font: inherit;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
    }
    button:hover { border-color: #94a3b8; background: #f8fafc; box-shadow: 0 5px 14px rgba(15, 23, 42, 0.08); transform: translateY(-1px); }
    button:active { transform: translateY(0); }
    .primary-button { border-color: var(--primary); color: white; background: var(--primary); }
    .primary-button:hover { border-color: var(--primary-dark); background: var(--primary-dark); }
    .danger-button { border-color: #fecaca; color: #b91c1c; background: #fff7f7; }
    .video-layout { display: grid; grid-template-columns: minmax(0, 2fr) minmax(270px, 1fr); gap: 14px; align-items: stretch; }
    .video-panel { display: flex; min-width: 0; flex-direction: column; gap: 8px; padding: 10px; border: 1px solid #1e293b; border-radius: 15px; background: var(--video); box-shadow: 0 12px 28px rgba(15, 23, 42, 0.16); }
    .video-label { display: flex; align-items: center; gap: 7px; color: #dbeafe; font-size: 13px; font-weight: 700; }
    .video-label::before { width: 7px; height: 7px; border-radius: 50%; background: #38bdf8; box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.14); content: ""; }
    .video-layout video { display: block; width: 100%; min-height: 0; border-radius: 10px; background: #020617; object-fit: contain; }
    #v { flex: 1; max-height: 650px; }
    .wrist-stack { display: grid; min-height: 0; grid-template-rows: 1fr 1fr; gap: 14px; }
    .wrist-stack video { flex: 1; max-height: 310px; }
    .playback-toolbar { display: flex; align-items: center; justify-content: flex-end; gap: 9px; margin: 14px 0 10px; color: #475569; font-size: 13px; font-weight: 700; }
    .playback-toolbar select { width: 110px; }
    #timeline { position: relative; height: 18px; margin: 12px 0 18px; border: 1px solid #cbd5e1; border-radius: 999px; background: #e8eef5; cursor: pointer; box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.06); }
    #progress { height: 100%; width: 0; border-radius: inherit; background: linear-gradient(90deg, #3b82f6, #06b6d4); pointer-events: none; }
    .marker { position: absolute; top: -7px; width: 4px; height: 30px; border-radius: 999px; transform: translateX(-2px); cursor: pointer; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.25); }
    .marker b { position: absolute; top: -23px; left: -6px; padding: 0 4px; border-radius: 4px; color: white; font-size: 11px; text-transform: uppercase; }
    .marker-b, .marker-b b { background: #16a34a; }.marker-s, .marker-s b { background: #d97706; }.marker-e, .marker-e b { background: #dc2626; }
    #status { margin: 12px 0; padding: 11px 14px; border: 1px solid #bbf7d0; border-radius: 10px; color: #166534; background: #f0fdf4; font-size: 14px; }
    .table-card { overflow: hidden; border: 1px solid var(--line); border-radius: 12px; background: white; }
    table { width: 100%; border-collapse: collapse; }
    td, th { padding: 10px 14px; border-bottom: 1px solid var(--line); text-align: left; }
    th { color: #475569; background: #f8fafc; font-size: 12px; letter-spacing: 0.04em; }
    tbody tr:last-child td { border-bottom: 0; }
    tbody tr:hover { background: #f8fbff; }
    @media (max-width: 850px) {
      body { padding: 14px; }
      main { padding: 18px; border-radius: 16px; }
      .video-layout { grid-template-columns: 1fr; }
      .wrist-stack { grid-template-columns: 1fr 1fr; grid-template-rows: none; }
      .sync-badge { display: none; }
    }
    @media (max-width: 580px) {
      .page-header { align-items: flex-start; }
      .wrist-stack { grid-template-columns: 1fr; }
      .verify-row { align-items: stretch; flex-direction: column; }
      #verify-load { width: 100%; }
      .toolbar, .field, .episode-field { width: 100%; min-width: 0; }
      .toolbar button { flex: 1 1 calc(50% - 8px); }
      .episode-context { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header class="page-header">
      <div>
        <p class="eyebrow">Simulation Data Workspace</p>
        <h1>仿真数据标注</h1>
      </div>
      <div class="sync-badge">三视角同步 · 25 FPS</div>
    </header>
    <p id="label-help" class="guide annotation-only">选择 episode 后，在主视角点击播放并使用
      <span id="key-help"></span>。按键记录当前视频帧；点标注指标下按 <span class="key">c</span> 撤销最近一次标注。按 <span class="key">a / ←</span> 后退一帧，按 <span class="key">d / →</span> 前进一帧。按 <span class="key">j</span> 切换上一个视频，按 <span class="key">k</span> 切换下一个视频。</p>
    <div id="verify-controls" hidden>
      <div class="verify-row">
        <label class="field verify-field">审核任务<select id="verify-task" onchange="loadVerifyVideos()"></select></label>
        <button id="verify-load" class="primary-button" onclick="loadVerifyVideos()">加载视频</button>
      </div>
      <p>审核数据固定读取 <code id="verify-data-root"></code>。选择任务后，将根据 <code>observation.images.cam_high/*.mp4</code> 查找 episode，并同时加载主视角、左腕和右腕视频。按 <span class="key">k</span> 查看下一个，按 <span class="key">j</span> 查看上一个；按 <span class="key">a / ←</span> 后退一帧，按 <span class="key">d / →</span> 前进一帧。</p>
    </div>
    <section class="toolbar">
      <div id="label-controls" class="annotation-only">
        <label class="field">任务<select id="task" onchange="changeConfig()"></select></label>
        <label class="field">标注类型<select id="metric" onchange="changeConfig()"></select></label>
      </div>
      <label class="field episode-field">Episode<select id="ep" onchange="loadVideo()"></select></label>
      <button class="annotation-only primary-button" onclick="save()">保存并发送到远程</button>
      <button class="annotation-only danger-button" onclick="clearMarks()">清空本集</button>
      <button class="annotation-only" onclick="transfer()">Transfer all</button>
      <button class="annotation-only" onclick="syncDataset()">同步全部到 dataset_sim</button>
    </section>
    <section id="episode-context" class="episode-context annotation-only">
      <div class="context-item"><span>当前指标</span><strong id="current-metric">—</strong></div>
      <div class="context-item"><span>数据分组</span><strong id="current-group">—</strong></div>
      <div class="context-item"><span>错误类型编号</span><strong id="current-error-type">—</strong></div>
      <div class="context-item"><span>Episode 类别</span><strong id="current-episode-kind">—</strong></div>
    </section>
    <div class="video-layout">
      <div class="video-panel">
        <div class="video-label">主视角</div>
        <video id="v" controls playsinline preload="auto" tabindex="0"></video>
      </div>
      <div class="wrist-stack">
        <div class="video-panel">
          <div class="video-label">左手腕部相机视角</div>
          <video id="left-wrist-video" muted playsinline preload="auto"></video>
        </div>
        <div class="video-panel">
          <div class="video-label">右手腕部相机视角</div>
          <video id="right-wrist-video" muted playsinline preload="auto"></video>
        </div>
      </div>
    </div>
    <label class="playback-toolbar">播放速度<select id="speed" onchange="setPlaybackRate()"><option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="1.5">1.5x</option><option value="2">2x</option><option value="4">4x</option></select></label>
    <div id="timeline" class="annotation-only" title="点击跳转"><div id="progress"></div></div>
    <div id="status"></div>
    <div class="table-card annotation-only">
      <table><thead><tr><th>类型</th><th>帧号</th><th>时间</th></tr></thead>
        <tbody id="rows"></tbody></table>
    </div>
  </main>
  <script>
    const TASKS = __TASKS__;
    const DEFAULT_TASK = __DEFAULT_TASK__;
    const APP_MODE = __APP_MODE__;
    const VERIFY_DATA_ROOT = __VERIFY_DATA_ROOT__;
    const ALLOWED_KEYS = ['b', 's', 'e'];
    const VIDEO_FPS = 25;
    let task = DEFAULT_TASK, metric = 'SIA+CSPC', episodes = [], marks = [], saved = {}, requestVersion = 0, v = document.getElementById('v');
    const leftWristVideo = document.getElementById('left-wrist-video');
    const rightWristVideo = document.getElementById('right-wrist-video');
    const wristVideos = [leftWristVideo, rightWristVideo];
    const allVideos = [v, ...wristVideos];
    const speedEl = document.getElementById('speed');
    function setPlaybackRate() {
      const rate = Number(speedEl.value) || 1;
      allVideos.forEach(video => {
        video.defaultPlaybackRate = rate;
        video.playbackRate = rate;
      });
    }
    allVideos.forEach(video => video.addEventListener('loadedmetadata', setPlaybackRate));
    let ep = document.getElementById('ep');
    if (APP_MODE === 'verify') {
      document.title = '仿真数据审核';
      document.querySelector('h1').textContent = '仿真数据审核';
      document.getElementById('verify-controls').hidden = false;
      document.getElementById('verify-data-root').textContent = VERIFY_DATA_ROOT;
      document.querySelectorAll('.annotation-only').forEach(x => x.hidden = true);
      document.getElementById('ep').disabled = false;
    }
    const manualLoadButton = document.querySelector('button[onclick="loadVideo()"]');
    if (manualLoadButton) manualLoadButton.remove();
    const legacySaveButton = document.querySelector('button[onclick^="save"]');
    if (legacySaveButton) legacySaveButton.remove();
    const taskEl=document.getElementById('task'), metricEl=document.getElementById('metric');
    taskEl.innerHTML=Object.keys(TASKS).map(x=>`<option value="${x}">${x}</option>`).join('');
    taskEl.value=task;
    function changeConfig(){ task=taskEl.value; fetchData(); }
    function fetchData(){
      if (APP_MODE === 'verify') return loadVerifyVideos();
      const previousMetric=metricEl.value;
      const metricNames=Object.keys(TASKS[task].metrics);
      metricEl.innerHTML=metricNames.map(x=>`<option value="${x}">${x}</option>`).join('');
      metricEl.value=metricNames.includes(previousMetric) ? previousMetric : metricNames[0];
      metric=metricEl.value;
      const metricConfig=TASKS[task].metrics[metric];
      document.getElementById('key-help').innerHTML=metricConfig.markers.map(key=>`<span class="key">${key}</span>`).join(' ');
      const requestedTask=task, requestedMetric=metric, currentRequest=++requestVersion;
      Promise.all([
        fetch('/annotations?task='+encodeURIComponent(requestedTask)+'&metric='+encodeURIComponent(requestedMetric)).then(r=>r.json()),
        fetch('/episodes?task='+encodeURIComponent(requestedTask)+'&metric='+encodeURIComponent(requestedMetric)).then(r=>r.json()),
      ]).then(([a,xs])=>{
        if (currentRequest !== requestVersion || task !== requestedTask || metric !== requestedMetric) return;
        saved=a; episodes=xs;
        ep.innerHTML=xs.map(x=>`<option value="${escapeHtml(x)}">${saved[x]?'✓':'○'} ${escapeHtml(episodeLabel(x))}</option>`).join('');
        if (xs.length) loadVideo();
        else { marks=[]; clearVideos(); render(); document.getElementById('status').textContent='当前任务没有可标注视频'; }
      }).catch(error=>document.getElementById('status').textContent='加载失败：'+error.message);
    }
    if (APP_MODE === 'verify') loadVerifyTasks();
    else fetchData();
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
    function episodeInfo(episode) {
      const normalized=String(episode || '').replaceAll('\\', '/');
      const root=(TASKS[task]?.video_root || '').replace(/\/$/, '');
      const relative=root && normalized.startsWith(root + '/') ? normalized.slice(root.length + 1) : normalized;
      const parts=relative.split('/');
      const group=parts[0] || '—';
      const episodeName=parts.includes('videos') ? parts[parts.indexOf('videos') - 1] : (parts.at(-1) || '—');
      const regular=relative.match(/(?:^|\/)FRT-(\d+)\/FRT-\d+-(\d+)\/FRT-\d+-\d+-([abc])(?:\/|$)/i);
      const domain=relative.match(/(?:^|\/)FRT-(\d+)\/FRT-\d+-(EMB|ENV)(?:\/|$)/i);
      if (regular) return {group, episodeName, errorType:`${regular[1]}-${regular[2]}`, episodeKind:regular[3].toLowerCase()};
      if (domain) return {group, episodeName, errorType:domain[1], episodeKind:domain[2].toUpperCase()};
      return {group, episodeName, errorType:'—', episodeKind:'—'};
    }
    function episodeLabel(episode) {
      const info=episodeInfo(episode);
      if (metric === 'FPL+TRR') return `${info.group} · 错误 ${info.errorType} · ${info.episodeKind} · ${info.episodeName}`;
      return `${info.group} · ${info.episodeName}`;
    }
    function updateEpisodeContext() {
      const info=episodeInfo(ep.value);
      document.getElementById('current-metric').textContent=metric;
      document.getElementById('current-group').textContent=info.group;
      document.getElementById('current-error-type').textContent=info.errorType;
      document.getElementById('current-episode-kind').textContent=info.episodeKind;
    }
    function wristEpisode(episode, side) {
      return episode.replace('/observation.images.cam_high/', `/observation.images.cam_${side}_wrist/`);
    }
    function videoUrl(episode) {
      return '/video?task='+encodeURIComponent(task)+'&metric='+encodeURIComponent(metric)+'&episode=' + encodeURIComponent(episode);
    }
    function clearVideos() {
      allVideos.forEach(video => {
        video.pause();
        video.removeAttribute('src');
        video.load();
      });
    }
    function syncWristTimes(time, tolerance = 0.015) {
      wristVideos.forEach(video => {
        if (video.readyState < 1) return;
        const target = Number.isFinite(video.duration) ? Math.min(time, video.duration) : time;
        if (Math.abs(video.currentTime - target) > tolerance) video.currentTime = target;
      });
    }
    function playWristVideos() {
      syncWristTimes(v.currentTime, 0.08);
      wristVideos.forEach(video => video.play().catch(() => {}));
    }
    function pauseWristVideos() {
      wristVideos.forEach(video => video.pause());
    }
    v.addEventListener('play', playWristVideos);
    v.addEventListener('playing', playWristVideos);
    v.addEventListener('pause', pauseWristVideos);
    v.addEventListener('seeking', () => syncWristTimes(v.currentTime));
    v.addEventListener('ratechange', () => {
      wristVideos.forEach(video => {
        video.defaultPlaybackRate = v.playbackRate;
        video.playbackRate = v.playbackRate;
      });
    });
    v.addEventListener('timeupdate', () => {
      if (!v.paused) syncWristTimes(v.currentTime, 0.12);
    });
    wristVideos.forEach(video => video.addEventListener('loadedmetadata', () => {
      syncWristTimes(v.currentTime);
      if (!v.paused) video.play().catch(() => {});
    }));
    function loadVideo() {
      marks = savedMarksFor(ep.value).map(x=>({...x}));
      updateEpisodeContext();
      render();
      clearVideos();
      const mainEpisode = ep.value;
      const sources = [mainEpisode, wristEpisode(mainEpisode, 'left'), wristEpisode(mainEpisode, 'right')];
      allVideos.forEach((video, index) => {
        video.src = videoUrl(sources[index]);
        video.load();
      });
      setPlaybackRate();
      v.focus();
    }
    function loadVerifyTasks() {
      fetch('/verify-tasks').then(r => r.json().then(data => {
        if (!r.ok) throw new Error(data.error || '加载审核任务失败');
        return data;
      })).then(tasks => {
        const selector = document.getElementById('verify-task');
        selector.innerHTML = tasks.map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join('');
        if (tasks.length) loadVerifyVideos();
        else document.getElementById('status').textContent = '审核数据目录下没有可用任务';
      }).catch(error => document.getElementById('status').textContent = '加载失败：'+error.message);
    }
    function loadVerifyVideos() {
      const verifyTask = document.getElementById('verify-task').value;
      if (!verifyTask) return;
      task = verifyTask;
      fetch('/episodes?task='+encodeURIComponent(verifyTask)).then(r => r.json().then(data => {
        if (!r.ok) throw new Error(data.error || '加载失败');
        return data;
      })).then(xs => {
        episodes = xs; saved = {}; marks = [];
        ep.innerHTML = xs.map(x => `<option value="${escapeHtml(x)}">${escapeHtml(x)}</option>`).join('');
        if (xs.length) loadVideo();
        else { clearVideos(); render(); document.getElementById('status').textContent = '该路径下没有可核验的 episode'; }
        document.getElementById('status').textContent = `找到 ${xs.length} 个 episode`;
      }).catch(error => document.getElementById('status').textContent = '加载失败：'+error.message);
    }
    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
    }
    function frame() { return Math.max(0, Math.round(v.currentTime * VIDEO_FPS)); }
    function stepFrame(direction) {
      if (!Number.isFinite(v.duration) || v.duration <= 0) return;
      v.pause();
      const maxFrame = Math.floor(v.duration * VIDEO_FPS);
      const targetFrame = Math.max(0, Math.min(maxFrame, frame() + direction));
      const targetTime = targetFrame / VIDEO_FPS;
      v.currentTime = targetTime;
      syncWristTimes(targetTime);
      document.getElementById('status').textContent = `${direction < 0 ? '后退' : '前进'}一帧：第 ${targetFrame} 帧（${targetTime.toFixed(3)} 秒）`;
    }
    function addMark(key) {
      if (APP_MODE === 'verify') return;
      key = String(key).toLowerCase();
      marks.push({type: key, frame: frame(), time: Number(v.currentTime.toFixed(3))});
      render();
      persistLocal();
      document.getElementById('status').textContent = 'Marked ' + key.toUpperCase() + ' at frame ' + frame();
    }
    function undoLastNodeMark() {
      if (TASKS[task].metrics[metric].kind !== 'nodes') return false;
      if (!marks.length) {
        document.getElementById('status').textContent = `当前没有可撤销的 ${metric} 标注`;
        return true;
      }
      const removed = marks.pop();
      render();
      persistLocal();
      document.getElementById('status').textContent = `已撤销 ${metric} 第 ${removed.frame} 帧的标注`;
      return true;
    }
    document.addEventListener('keydown', e => {
      const key = (e.key || '').toLowerCase();
      const isFormInput = ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName);
      if (e.ctrlKey || e.altKey || e.metaKey || isFormInput) return;
      if ((key === 'j' || key === 'k') && episodes.length) { e.preventDefault(); const n=ep.selectedIndex+(key==='j'?-1:1); ep.selectedIndex=(n+episodes.length)%episodes.length; loadVideo(); return; }
      if (key === 'a' || key === 'arrowleft') { e.preventDefault(); stepFrame(-1); return; }
      if (key === 'd' || key === 'arrowright') { e.preventDefault(); stepFrame(1); return; }
      if (APP_MODE === 'verify') return;
      if (key === 'c' && undoLastNodeMark()) { e.preventDefault(); return; }
      if (!ALLOWED_KEYS.includes(key)) return;
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
      const info=episodeInfo(ep.value);
      const errorText=metric === 'FPL+TRR' ? `，错误 ${info.errorType}，类别 ${info.episodeKind}` : '';
      document.getElementById('status').textContent = `当前指标 ${metric}，${info.group} / ${info.episodeName}${errorText}：${marks.length} 个标注点`; }
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
            body:JSON.stringify({episode: ep.value, frame: mark.frame, task, metric})});
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
    function syncDataset() {
      if (!confirm('将所有已完成的本地标注同步到 dataset_sim 中对应 episode 的 annotation.json，是否继续？')) return;
      fetch('/sync-dataset', {method:'POST'}).then(r=>r.json()).then(x=>
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
    root = TASKS[task]['video_root']
    return root if os.path.isdir(root) else None


def allowed_video_path(path: str, root: str, metric: str = DEFAULT_METRIC) -> bool:
    relative = os.path.relpath(path, root)
    group = relative.split(os.sep, 1)[0]
    patterns = METRIC_DEFINITIONS[canonical_metric(metric)]['groups']
    return (relative != '..' and not relative.startswith('..' + os.sep) and
            any(fnmatch.fnmatchcase(group, pattern) for pattern in patterns))


def episodes(task=None, metric: str = DEFAULT_METRIC) -> list[str]:
    task = task or CURRENT_TASK
    metric = canonical_metric(metric)
    config = task_config(task, metric)
    root = local_task_root(task)
    if root:
        paths = []
        for group in config['groups']:
            paths.extend(glob.glob(
                os.path.join(root, group, '**', 'observation.images.cam_high', '*.mp4'),
                recursive=True,
            ))
        return sorted(paths)
    quoted_root = shlex.quote(config['video_root'])
    output = remote(
        f"find {quoted_root} -type f -path '*/observation.images.cam_high/*.mp4' | sort"
    )
    return [path for path in output.decode().splitlines()
            if allowed_video_path(path, config['video_root'], metric)]

def verify_tasks() -> list[str]:
    """List task directories available in the dedicated verification dataset."""
    if not os.path.isdir(VERIFY_DATA_ROOT):
        raise ValueError('审核数据目录不存在')
    tasks = []
    for entry in os.scandir(VERIFY_DATA_ROOT):
        if not entry.is_dir() or entry.name.startswith('.'):
            continue
        if any(group.is_dir() and (group.name.startswith('ST-') or group.name.startswith('FRT-'))
               for group in os.scandir(entry.path)):
            tasks.append(entry.name)
    return sorted(tasks)


def verify_task_root(task: str) -> str:
    if task not in verify_tasks():
        raise ValueError('审核任务不存在')
    root = os.path.abspath(os.path.join(VERIFY_DATA_ROOT, task))
    if not path_within(root, VERIFY_DATA_ROOT):
        raise ValueError('invalid verify task path')
    return root


def verify_episodes(task: str) -> list[str]:
    """Find all head-view videos for one verification task."""
    root = verify_task_root(task)
    return sorted(glob.glob(
        os.path.join(root, '**', 'observation.images.cam_high', '*.mp4'),
        recursive=True,
    ))

def annotation_path(task=CURRENT_TASK, metric=DEFAULT_METRIC):
    metric = canonical_metric(metric)
    slug = METRIC_FILE_SLUGS.get(metric, metric.lower().replace('+', '_'))
    return os.path.join(os.path.dirname(__file__), f'{task}_{slug}_annotations.json')

def read_local(task=CURRENT_TASK, metric=DEFAULT_METRIC):
    metric = canonical_metric(metric)
    paths = [annotation_path(task, metric)]
    if metric == 'SIA+CSPC':
        paths.append(os.path.join(os.path.dirname(__file__), f'{task}_sia_annotations.json'))
    for path in paths:
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return {}

def write_local(data, task=CURRENT_TASK, metric=DEFAULT_METRIC):
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
    config = task_config(record.get('task', CURRENT_TASK), record.get('metric', DEFAULT_METRIC))
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


def dataset_task_directory(task: str) -> str:
    candidates = [
        DATASET_TASK_ALIASES.get(task, task),
        os.path.basename(task_config(task)['video_root'].rstrip('/')),
    ]
    for name in dict.fromkeys(candidates):
        path = os.path.join(DATASET_SIM_ROOT, name)
        if os.path.isdir(path):
            return path
    raise ValueError(f'dataset_sim 中不存在 task: {task}')


def dataset_episode_map(task: str) -> dict[str, str]:
    """Map source videos to converted dataset episodes by group order."""
    task_directory = dataset_task_directory(task)
    target_groups: dict[str, list[tuple[int, str]]] = {
        group: [] for group in DATASET_GROUP_DOMAINS
    }
    for metadata_file in glob.glob(os.path.join(task_directory, 'episodes', '*', 'metadata.json')):
        with open(metadata_file, encoding='utf-8') as source:
            metadata = json.load(source)
        if metadata.get('success') is not True:
            continue
        episode_id = metadata.get('episode_id', '')
        try:
            episode_number = int(episode_id.rsplit('_', 1)[1])
        except (IndexError, ValueError):
            continue
        for group, domain in DATASET_GROUP_DOMAINS.items():
            if metadata.get('domain') == domain:
                target_groups[group].append((episode_number, os.path.dirname(metadata_file)))

    video_root = task_config(task)['video_root']
    mapping = {}
    for group in DATASET_GROUP_DOMAINS:
        source_videos = sorted(glob.glob(
            os.path.join(video_root, group, '**', 'observation.images.cam_high', '*.mp4'),
            recursive=True,
        ))
        target_episodes = [path for _, path in sorted(target_groups[group])]
        for source_video, target_episode in zip(source_videos, target_episodes):
            target_video = os.path.join(target_episode, 'videos', 'front.mp4')
            if (os.path.isfile(source_video) and os.path.isfile(target_video) and
                    os.path.getsize(source_video) != os.path.getsize(target_video)):
                continue
            mapping[source_video] = os.path.join(target_episode, 'annotation.json')
    return mapping


def dataset_annotation_value(record: dict) -> tuple[str, list[dict]]:
    metric = canonical_metric(record.get('metric', DEFAULT_METRIC))
    if metric == 'SIA+CSPC':
        frames = sorted({int(mark['frame']) for mark in record.get('marks', [])})
        return 'subtask_segments', [
            {'id': f'{index:03d}', 'subtask_frame': frame}
            for index, frame in enumerate(frames, 1)
        ]
    if metric == 'VOC-MEM':
        return 'voc_mem', [{'frame_index': frame} for frame in make_frames(record)]
    raise ValueError(f'不支持的 metric: {metric}')


def sync_dataset_annotations(dry_run: bool = False) -> tuple[int, list[str]]:
    """Merge every local label record into its dataset_sim annotation file."""
    updates: dict[str, dict[str, list[dict]]] = {}
    skipped = []
    for task in TASKS:
        task_records = [
            record
            for metric in TASKS[task]['metrics']
            for record in read_local(task, metric).values()
        ]
        if not task_records:
            continue
        try:
            episode_map = dataset_episode_map(task)
        except (OSError, ValueError) as exc:
            skipped.extend(f'{task}/{record.get("episode_name", "unknown")}: {exc}'
                           for record in task_records)
            continue
        for record in task_records:
            try:
                annotation_file = episode_map.get(record.get('episode', ''))
                if not annotation_file or not os.path.isfile(annotation_file):
                    raise ValueError('dataset_sim 中没有匹配的视频')
                field, value = dataset_annotation_value(record)
                updates.setdefault(annotation_file, {})[field] = value
            except (KeyError, TypeError, ValueError) as exc:
                skipped.append(f'{task}/{record.get("episode_name", "unknown")}: {exc}')

    payloads = {}
    for annotation_file, fields in updates.items():
        with open(annotation_file, encoding='utf-8') as source:
            annotation = json.load(source)
        annotation.update(fields)
        payloads[annotation_file] = annotation

    if dry_run:
        return len(payloads), skipped

    for annotation_file, annotation in payloads.items():
        temporary = annotation_file + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as output:
            json.dump(annotation, output, ensure_ascii=False, indent=2)
            output.write('\n')
        os.replace(temporary, annotation_file)
    return len(updates), skipped


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
        request_metric = canonical_metric(query.get('metric', [DEFAULT_METRIC])[0])
        try:
            if parsed.path == '/':
                body, content_type = PAGE.encode(), 'text/html; charset=utf-8'
            elif parsed.path == '/verify-tasks':
                if MODE != 'verify':
                    self.send_error(404)
                    return
                body, content_type = json.dumps(verify_tasks()).encode(), 'application/json'
            elif parsed.path == '/episodes':
                if MODE == 'verify':
                    try:
                        body = json.dumps(verify_episodes(request_task)).encode()
                        content_type = 'application/json'
                    except ValueError as exc:
                        self.send_json({'error': str(exc)}, 400)
                        return
                else:
                    body, content_type = json.dumps(episodes(request_task, request_metric)).encode(), 'application/json'
            elif parsed.path == '/annotations':
                if MODE == 'verify':
                    self.send_error(403, 'verify mode is read-only')
                    return
                all_data = read_local(request_task, request_metric); body, content_type = json.dumps(all_data).encode(), 'application/json'
            elif parsed.path == '/video':
                path = query.get('episode', [''])[0]
                if MODE == 'verify':
                    normalized = os.path.abspath(path)
                    verify_root = verify_task_root(request_task)
                    if (not path.startswith('/') or normalized != path or
                            not path_within(normalized, verify_root) or
                            not os.path.isfile(normalized) or
                            os.path.basename(os.path.dirname(normalized)) not in VIDEO_CAMERA_DIRECTORIES or
                            not normalized.endswith('.mp4')):
                        raise ValueError('invalid verify video path')
                    self.serve_video(normalized)
                    return
                root = task_config(request_task, request_metric)['video_root']
                if not (path.startswith(root + '/') and allowed_video_path(path, root, request_metric) and
                        os.path.basename(os.path.dirname(path)) in VIDEO_CAMERA_DIRECTORIES and
                        path.endswith('.mp4')):
                    raise ValueError('invalid video path')
                if local_task_root(request_task):
                    if not os.path.isfile(path):
                        raise ValueError('video file does not exist')
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
        if MODE == 'verify':
            length = int(self.headers.get('Content-Length', 0))
            if length:
                self.rfile.read(length)
            self.send_json({'ok': False, 'error': 'verify 模式不支持标注或导出'}, 403)
            return
        if self.path == '/screenshot':
            try:
                length = int(self.headers.get('Content-Length', 0))
                obj = json.loads(self.rfile.read(length))
                episode = obj.get('episode', '')
                frame = int(obj.get('frame'))
                if frame < 0:
                    raise ValueError('invalid frame')
                selected_task = obj.get('task', CURRENT_TASK)
                selected_metric = canonical_metric(obj.get('metric', DEFAULT_METRIC))
                root = task_config(selected_task, selected_metric)['video_root']
                if not (episode.startswith(root + '/') and allowed_video_path(episode, root, selected_metric) and episode.endswith('.mp4') and
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
        if self.path == '/sync-dataset':
            try:
                with ANNOTATION_LOCK:
                    updated, skipped = sync_dataset_annotations()
                message = f'已同步 {updated} 个 episode 到 {DATASET_SIM_ROOT}'
                if skipped:
                    message += f'；跳过 {len(skipped)} 条未匹配或不完整记录'
                    message += '：' + '；'.join(skipped[:3])
                self.send_json({
                    'ok': True,
                    'message': message,
                    'updated': updated,
                    'skipped': skipped,
                })
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
                data = read_local(request.get('task', CURRENT_TASK), request.get('metric', DEFAULT_METRIC))
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
                selected_metric = canonical_metric(obj.get('metric', DEFAULT_METRIC))
                video_root = task_config(selected_task, selected_metric)['video_root']
                if not allowed_video_path(obj.get('episode', ''), video_root, selected_metric):
                    raise ValueError('selected video is outside the groups allowed for this metric')
                task_root = video_root.rstrip('/').split('/')[-1]
                idx = parts.index(task_root)
                episode_name = os.path.basename(obj['episode'].split('/videos/')[0].rstrip('/'))
                obj.update(task=selected_task, metric=selected_metric, group=parts[idx + 1],
                           episode_name=episode_name, gap=20)
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
    global CURRENT_TASK, MODE, REMOTE_VIDEO, REMOTE_ANNOT, LOCAL_ANNOT, LOCAL_SCREENSHOTS, PAGE
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--mode', choices=('label', 'verify'), default='label',
                        help='label 标注模式；verify 从固定审核数据源读取的只读核验模式')
    parser.add_argument('--task', choices=sorted(TASKS), default='organize_table',
                        help='任务名；Veified_Data 下的新任务默认只使用 s 键')
    args = parser.parse_args()
    CURRENT_TASK = args.task
    MODE = args.mode
    config = task_config(CURRENT_TASK)
    REMOTE_VIDEO = config['video_root']
    REMOTE_ANNOT = config['annotation']
    LOCAL_ANNOT = annotation_path(CURRENT_TASK, DEFAULT_METRIC)
    LOCAL_SCREENSHOTS = os.path.join(os.path.dirname(__file__), f'{CURRENT_TASK}_screenshots')
    PAGE = PAGE.replace('__TASKS__', json.dumps(TASKS))
    PAGE = PAGE.replace('__DEFAULT_TASK__', json.dumps(CURRENT_TASK))
    PAGE = PAGE.replace('__APP_MODE__', json.dumps(MODE))
    PAGE = PAGE.replace('__VERIFY_DATA_ROOT__', json.dumps(VERIFY_DATA_ROOT))
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
