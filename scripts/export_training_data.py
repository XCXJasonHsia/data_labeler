"""Export the requested official RoboDojo episodes without re-encoding video.

Requires pyarrow and ffmpeg/ffprobe; these are export-time dependencies only.
Original episode/frame/task indices and all state/action columns are retained.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


# Official task indices, checked against the instruction strings in tasks.parquet
# and task/RoboDojo/tasks (in particular swap_blocks is index 33, not swap_T).
TASK_INDICES = {
    'arrange_largest_number': 0,
    'fill_pen_holder': 8,
    'fold_clothes': 9,
    'hang_mugs': 11,
    'insert_tubes': 14,
    'make_toast': 16,
    'organize_table': 18,
    'pack_objects_into_box': 19,
    'play_stacking_toy': 21,
    'press_by_number': 26,
    'put_bottles_into_dustbin': 10,
    'stack_blocks': 29,
    'store_laptop_and_headphones': 31,
    'swap_blocks': 33,
    'sweep_blocks': 34,
}
CAMERAS = ('observation.images.cam_high', 'observation.images.cam_left_wrist',
           'observation.images.cam_right_wrist')


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def probe_video(path, frames, fps):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,nb_frames,start_time,duration,width,height',
        '-of', 'json', str(path),
    ], check=True, capture_output=True, text=True)
    video = json.loads(result.stdout)['streams'][0]
    if (int(video['nb_frames']) != frames or
            abs(float(video['start_time'])) > 0.00001 or
            abs(float(video['duration']) - frames / fps) > 0.0001):
        raise ValueError(f'Video/frame boundary mismatch: {path}: {video}, expected {frames}')
    return video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(
        '/mnt/public2/liushengbang/RoboDojo/data/RoboDojo_lerobot_v30_video'))
    parser.add_argument('--destination', type=Path, default=Path(
        '/mnt/public2/liushengbang/data/Training_Data'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    source, dest = args.source.resolve(), args.destination.resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        raise ValueError('Source and destination must be separate directories')
    info = json.loads((source / 'meta/info.json').read_text())
    fps = info['fps']
    tasks = pq.read_table(source / 'meta/tasks.parquet')
    instructions = {r['task_index']: r['__index_level_0__'] for r in tasks.to_pylist()}
    by_instruction = {instructions[index]: name for name, index in TASK_INDICES.items()}
    episode_table = pq.read_table(source / 'meta/episodes')
    selected = [(by_instruction[r['tasks'][0]], r) for r in episode_table.to_pylist()
                if len(r['tasks']) == 1 and r['tasks'][0] in by_instruction]
    counts = Counter(task for task, _ in selected)
    if counts != Counter({task: 100 for task in TASK_INDICES}):
        raise ValueError(f'Expected 100 episodes for each of 15 tasks, found {counts}')
    print(json.dumps({'tasks': dict(counts), 'total_episodes': len(selected)}, indent=2), flush=True)
    source_mismatches = []
    for task, row in selected:
        for camera in CAMERAS:
            prefix = 'videos/' + camera
            video_frames = round((row[prefix + '/to_timestamp'] - row[prefix + '/from_timestamp']) * fps)
            if video_frames != row['length']:
                source_mismatches.append({'task': task, 'episode_index': row['episode_index'],
                    'camera': camera, 'state_frames': row['length'], 'video_frames': video_frames})

    # Cache the three source parquet shards once; episode rows are contiguous.
    tables = {}
    for _, row in selected:
        key = (row['data/chunk_index'], row['data/file_index'])
        if key not in tables:
            path = source / info['data_path'].format(chunk_index=key[0], file_index=key[1])
            table = pq.read_table(path)
            tables[key] = (table, table.column('index')[0].as_py())

    dest.mkdir(parents=True, exist_ok=True)
    for task, task_index in TASK_INDICES.items():
        meta = dest / task / 'meta'
        meta.mkdir(parents=True, exist_ok=True)
        chosen_ids = [r['episode_index'] for t, r in selected if t == task]
        pq.write_table(episode_table.filter(pc.is_in(episode_table['episode_index'],
                       value_set=pa.array(chosen_ids, type=pa.int64()))), meta / 'source_episodes.parquet')
        pq.write_table(tasks.filter(pc.equal(tasks['task_index'], task_index)), meta / 'source_tasks.parquet')
        shutil.copy2(source / 'meta/info.json', meta / 'source_info.json')
        # This is a per-episode export, not a rewritten LeRobot v3 dataset.
        # source_info/source_episodes describe the original archive faithfully.
        write_json(meta / 'export_info.json', {
            'format': 'robodojo_per_episode_export_v1', 'task': task,
            'instruction': instructions[task_index], 'source_task_index': task_index,
            'total_episodes': counts[task], 'fps': fps, 'cameras': CAMERAS,
            'source_root': str(source), 'source_repo': 'RoboDojo-Benchmark/RoboDojo',
            'source_episode_indices': chosen_ids, 'video_encoding': 'lossless stream copy (AV1)',
            'data_path': 'episode_{episode_index:06d}/data/episode.parquet',
            'video_path': 'episode_{episode_index:06d}/videos/{camera}/episode.mp4',
        })

    def export_episode(item):
        task, row = item
        index, frames = row['episode_index'], row['length']
        directory = dest / task / f'episode_{index:06d}'
        complete = directory / 'meta/export.json'
        if complete.exists():
            existing = json.loads(complete.read_text())
            if existing['source_root'] != str(source) or existing['source_episode_index'] != index:
                raise ValueError(f'Existing export has a different source: {directory}')
        directory.mkdir(parents=True, exist_ok=True)
        data, first_index = tables[(row['data/chunk_index'], row['data/file_index'])]
        episode = data.slice(row['dataset_from_index'] - first_index, frames)
        if (episode.num_rows != frames or
                pc.unique(episode['episode_index']).to_pylist() != [index] or
                pc.unique(episode['task_index']).to_pylist() != [TASK_INDICES[task]] or
                episode['index'][-1].as_py() != row['dataset_to_index'] - 1):
            raise ValueError(f'State/action rows do not match episode {index}')
        parquet = directory / 'data/episode.parquet'
        parquet.parent.mkdir(exist_ok=True)
        if not parquet.exists():
            tmp = parquet.with_suffix('.tmp')
            pq.write_table(episode, tmp, compression='zstd')
            tmp.replace(parquet)
        elif not pq.read_table(parquet).equals(episode):
            raise ValueError(f'Existing state/action data differs: {parquet}')
        videos = {}
        for camera in CAMERAS:
            prefix = 'videos/' + camera
            rel = info['video_path'].format(video_key=camera,
                chunk_index=row[prefix + '/chunk_index'], file_index=row[prefix + '/file_index'])
            start, end = row[prefix + '/from_timestamp'], row[prefix + '/to_timestamp']
            video_frames = round((end - start) * fps)
            if video_frames <= 0 or abs(end - start - video_frames / fps) > 0.0001:
                raise ValueError(f'Invalid original video timestamps: {index} {camera}')
            video = directory / 'videos' / camera / 'episode.mp4'
            video.parent.mkdir(parents=True, exist_ok=True)
            if not video.exists():
                tmp = video.with_name('episode.partial.mp4')
                subprocess.run([
                    'ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', f'{start:.6f}',
                    '-i', str(source / rel), '-t', f'{video_frames / fps:.6f}',
                    '-map', '0:v:0', '-c:v', 'copy', '-an', '-movflags', '+faststart', str(tmp),
                ], check=True, capture_output=True)
                probe_video(tmp, video_frames, fps)
                tmp.replace(video)
            else:
                probe_video(video, video_frames, fps)
            videos[camera] = {'path': str(video.relative_to(directory)), 'source_path': rel,
                             'source_from_timestamp': start, 'source_to_timestamp': end,
                             'frames': video_frames, 'bytes': video.stat().st_size}
        write_json(directory / 'meta/source_episode.json', row)
        write_json(complete, {
            'task': task, 'source_root': str(source), 'source_episode_index': index,
            'source_task_index': TASK_INDICES[task], 'instruction': instructions[TASK_INDICES[task]],
            'frames': frames, 'fps': fps, 'videos': videos,
            'source_frame_count_mismatches': [v for v in source_mismatches if v['episode_index'] == index],
            'data_path': 'data/episode.parquet', 'indices': 'original source indices retained',
        })
        return sum(v['bytes'] for v in videos.values()) + parquet.stat().st_size

    started = time.monotonic()
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(export_episode, item) for item in selected]
        for done, job in enumerate(as_completed(jobs), 1):
            total_bytes += job.result()
            if done % 25 == 0 or done == len(jobs):
                print(f'{done}/{len(jobs)} episodes, {total_bytes / 1e9:.2f} GB, '
                      f'{time.monotonic() - started:.1f}s', flush=True)
    write_json(dest / 'manifest.json', {
        'source_repo': 'RoboDojo-Benchmark/RoboDojo', 'source_root': str(source),
        'format': 'robodojo_per_episode_export_v1', 'total_episodes': len(selected),
        'total_frames': sum(r['length'] for _, r in selected), 'total_videos': len(selected) * 3,
        'tasks': dict(counts), 'data_and_video_bytes': total_bytes,
        'video_validation': 'All videos start at 0; frame counts and durations match source camera timestamps',
        'source_frame_count_mismatches': source_mismatches,
    })
    print('Export complete:', dest / 'manifest.json', flush=True)


if __name__ == '__main__':
    main()
