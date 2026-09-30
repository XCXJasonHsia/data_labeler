"""Exercise training validity through the real HTTP handler with isolated files."""

import copy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import subprocess
import sys
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

import labeler


class QuietHandler(labeler.Handler):
    def log_message(self, *args):
        pass


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.training = self.root / 'Training_Data/organize_table'
        self.legacy = self.root / 'Veified_Data/organize_table'
        self.episode = self.make_video(self.training / 'episode_001800')
        self.old_episode = self.make_video(self.legacy / 'ST-1/episode_001800')
        tasks = copy.deepcopy(labeler.TASKS)
        tasks['organize_table']['video_root'] = str(self.legacy)
        tasks['organize_table']['metrics']['training data']['video_root'] = str(self.training)
        self.patch = patch.multiple(labeler, TASKS=tasks, MODE='label',
                                    __file__=str(self.root / 'labeler.py'))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.server = labeler.ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def make_video(self, directory):
        video = directory / 'videos/observation.images.cam_high/episode.mp4'
        video.parent.mkdir(parents=True)
        video.write_bytes(b'video bytes for range requests')
        return str(video)

    def request(self, method, endpoint, payload=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        body = None if payload is None else json.dumps(payload)
        connection.request(method, endpoint, body, headers or {'Content-Type': 'application/json'})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def validity(self, state, reason='', **extra):
        payload = dict(task='organize_table', metric='training data', episode=self.episode,
                       state=state, reason=reason)
        payload.update(extra)
        status, _, body = self.request('POST', '/validity', payload)
        return status, json.loads(body)

    def test_invalid_requires_manual_reason_and_round_trips(self):
        for reason in ('', '  \n ', None, ['bad']):
            status, result = self.validity('invalid', reason)
            self.assertEqual(status, 400)
            self.assertFalse(result['ok'])
        self.assertFalse(Path(labeler.validity_path('organize_table', 'training data')).exists())
        status, result = self.validity('invalid', '  抓取失败，物体滑落 <test>  ')
        self.assertEqual(status, 200)
        self.assertEqual(result['record']['reason'], '抓取失败，物体滑落 <test>')
        query = urlencode({'task': 'organize_table', 'metric': 'training data'})
        status, _, body = self.request('GET', '/validity?' + query)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)[self.episode]['reason'], '抓取失败，物体滑落 <test>')
        status, result = self.validity('invalid', '更新后的原因')
        self.assertEqual(result['record']['reason'], '更新后的原因')
        status, result = self.validity('valid', 'previous reason')
        self.assertEqual(result['record']['reason'], '')
        self.assertEqual(result['record']['state'], 'valid')
        self.validity(None)
        self.assertEqual(labeler.read_validity('organize_table', 'training data'), {})

    def test_training_and_existing_records_are_separate(self):
        labeler.update_validity('organize_table', self.old_episode, 'invalid')
        old_file = Path(labeler.validity_path('organize_table'))
        before = old_file.read_bytes()
        self.validity('invalid', '训练样本失效')
        self.assertEqual(old_file.read_bytes(), before)
        self.assertEqual(labeler.read_validity('organize_table'), {self.old_episode: 'invalid'})
        self.assertNotIn(self.old_episode, labeler.read_validity('organize_table', 'training data'))
        status, _ = self.validity('valid', episode=self.old_episode)
        self.assertEqual(status, 400)
        status, _ = self.validity('valid', metric='SIA+CSPC')
        self.assertEqual(status, 400)

    def test_episode_listing_and_video_ranges_use_selected_source(self):
        self.assertEqual(labeler.episodes('organize_table', 'training data'), [self.episode])
        self.assertEqual(labeler.episodes('organize_table', 'SIA+CSPC'), [self.old_episode])
        query = urlencode({'task': 'organize_table', 'metric': 'training data', 'episode': self.episode})
        status, headers, body = self.request('GET', '/video?' + query, headers={'Range': 'bytes=2-6'})
        self.assertEqual(status, 206)
        self.assertEqual(body, b'deo b')
        self.assertTrue(headers['Content-Range'].startswith('bytes 2-6/'))
        outside = self.make_video(self.root / 'elsewhere/episode_000001')
        link = self.training / 'episode_escape'
        link.symlink_to(Path(outside).parents[2], target_is_directory=True)
        escaped = str(link / 'videos/observation.images.cam_high/episode.mp4')
        status, _ = self.validity('valid', episode=escaped)
        self.assertEqual(status, 400)

    def test_training_marker_and_deletion_endpoints_are_disabled(self):
        payload = dict(task='organize_table', metric='training data', episode=self.episode,
                       marks=[{'type': 's', 'frame': 1}], frame=1, directory=str(self.root / 'out'))
        for endpoint in ('/save', '/screenshot', '/transfer', '/delete-episode'):
            status, _, body = self.request('POST', endpoint, payload)
            self.assertEqual(status, 400, (endpoint, body))
        self.assertTrue(Path(self.episode).is_file())
        self.assertEqual(list(self.root.glob('*_annotations.json')), [])

    def test_concurrent_saves_do_not_lose_episodes(self):
        episodes = [self.make_video(self.training / f'episode_{n:06d}') for n in range(12)]
        results = []
        threads = [threading.Thread(target=lambda ep=ep: results.append(
            self.validity('invalid', '人工原因', episode=ep)[0])) for ep in episodes]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(results, [200] * 12)
        self.assertEqual(len(labeler.read_validity('organize_table', 'training data')), 12)

    def test_verify_mode_stays_read_only(self):
        with patch.object(labeler, 'MODE', 'verify'):
            status, _ = self.validity('valid')
            self.assertEqual(status, 403)
            status, _, _ = self.request('GET', '/validity?task=organize_table&metric=training%20data')
            self.assertEqual(status, 403)
        self.assertFalse(Path(labeler.validity_path('organize_table', 'training data')).exists())

    def test_multiple_server_processes_do_not_overwrite_each_other(self):
        code = """
import labeler, sys
labeler.__file__ = sys.argv[1]
for n in range(12):
    labeler.update_validity('organize_table', sys.argv[2] + str(n),
                           'invalid', 'training data', '人工原因')
"""
        jobs = [subprocess.Popen([sys.executable, '-c', code, str(self.root / 'labeler.py'),
                                  self.episode + f'-process-{i}-'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(4)]
        for job in jobs:
            _, err = job.communicate(timeout=20)
            self.assertEqual(job.returncode, 0, err.decode())
        self.assertEqual(len(labeler.read_validity('organize_table', 'training data')), 48)


if __name__ == '__main__':
    unittest.main()
