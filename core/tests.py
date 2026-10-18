import os
import tempfile

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from . import logs


def _logging(tmpdir):
    return {
        'version': 1,
        'handlers': {
            'console': {'class': 'logging.StreamHandler'},
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': os.path.join(tmpdir, 'debug.log'),
                'backupCount': 2,
            },
        },
    }


SAMPLE = (
    "[2026-10-18 09:00:00] INFO django.server: GET / 200\n"
    "[2026-10-18 09:00:01] ERROR ai_lab_chatbot.views: Mycroft stream failed\n"
    "Traceback (most recent call last):\n"
    "httpcore.ConnectError: [Errno 111] Connection refused\n"
    "[2026-10-18 09:00:02] WARNING django.security.DisallowedHost: Invalid HTTP_HOST header: 'tossdown.com'\n"
    "[2026-10-18 09:00:03] DEBUG core.views: Parsed UPS data\n"
)


class LogHelperTests(SimpleTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, 'debug.log')
        override = override_settings(LOGGING=_logging(self.tmp.name))
        override.enable()
        self.addCleanup(override.disable)

    def test_only_file_handlers_listed(self):
        open(self.path, 'w').close()
        open(self.path + '.2', 'w').close()
        [log] = logs.available_logs()
        self.assertEqual(log['key'], 'file')
        self.assertEqual([b['index'] for b in log['backups']], [2])

    def test_resolve_log_whitelist(self):
        self.assertEqual(logs.resolve_log('file'), self.path)
        self.assertEqual(logs.resolve_log('file', 2), self.path + '.2')
        self.assertIsNone(logs.resolve_log('file', 3))
        self.assertIsNone(logs.resolve_log('file', -1))
        self.assertIsNone(logs.resolve_log('console'))
        self.assertIsNone(logs.resolve_log('../../etc/passwd'))

    def test_tail_lines_spans_blocks(self):
        with open(self.path, 'w', newline='\n') as f:
            for i in range(1000):
                f.write(f'line {i}\n')
        self.assertEqual(logs.tail_lines(self.path, 3, block_size=16), ['line 997', 'line 998', 'line 999'])
        self.assertEqual(len(logs.tail_lines(self.path, 5000)), 1000)

    def test_group_entries_keeps_traceback(self):
        entries = logs.group_entries(SAMPLE.splitlines())
        self.assertEqual(len(entries), 4)
        self.assertEqual(entries[1]['level'], 'ERROR')
        self.assertIn('Connection refused', entries[1]['message'])

    def test_group_entries_orphan_lines(self):
        entries = logs.group_entries(['  File "x.py", line 1', SAMPLE.splitlines()[0]])
        self.assertIsNone(entries[0]['level'])
        self.assertEqual(entries[1]['level'], 'INFO')

    def test_filter_entries(self):
        entries = logs.group_entries(SAMPLE.splitlines())
        self.assertEqual([e['level'] for e in logs.filter_entries(entries, 'WARNING')], ['ERROR', 'WARNING'])
        self.assertEqual(len(logs.filter_entries(entries, query='connection REFUSED')), 1)
        self.assertEqual(len(logs.filter_entries(entries, query='disallowedhost')), 1)


class LogViewTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        with open(os.path.join(self.tmp.name, 'debug.log'), 'w', newline='\n') as f:
            f.write(SAMPLE)
        override = override_settings(LOGGING=_logging(self.tmp.name))
        override.enable()
        self.addCleanup(override.disable)
        self.staff = User.objects.create_user('staff', password='pw', is_staff=True)
        self.user = User.objects.create_user('plain', password='pw')

    def test_non_staff_redirected(self):
        self.client.force_login(self.user)
        self.assertEqual(self.client.get(reverse('logs')).status_code, 302)
        self.assertEqual(self.client.get(reverse('log_download')).status_code, 302)

    def test_staff_sees_entries(self):
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('logs'), {'level': 'ERROR'})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.context['entries']), 1)
        self.assertContains(resp, 'Mycroft stream failed')

    def test_unknown_log_or_backup_404(self):
        self.client.force_login(self.staff)
        self.assertEqual(self.client.get(reverse('logs'), {'log': '../../etc/passwd'}).status_code, 404)
        self.assertEqual(self.client.get(reverse('logs'), {'backup': '99'}).status_code, 404)
        self.assertEqual(self.client.get(reverse('logs'), {'backup': 'x'}).status_code, 404)

    def test_missing_backup_shows_message(self):
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('logs'), {'backup': '1'})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "doesn&#x27;t exist yet")

    def test_download(self):
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('log_download'))
        self.assertEqual(resp.status_code, 200)
        self.assertIn('attachment', resp['Content-Disposition'])
        self.assertEqual(b''.join(resp.streaming_content).decode(), SAMPLE)
        resp.close()
