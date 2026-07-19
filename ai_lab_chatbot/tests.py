import json
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from ai_lab_chatbot.models import Personality, HouseholdFact, UserContext
from ai_lab_chatbot.mycroft.prompts import build_system_prompt, STANDARD_GUARDRAILS


# Models live in the `ai_lab` DB and auth.User in `default`; both are needed.
DBS = {'default', 'ai_lab'}


class PersonalityModelTests(TestCase):
    databases = DBS

    def test_setting_active_deactivates_others(self):
        first = Personality.objects.create(
            name='A', description='d', instructions='i', is_active=True)
        second = Personality.objects.create(
            name='B', description='d', instructions='i', is_active=True)

        first.refresh_from_db()
        self.assertFalse(first.is_active)
        self.assertTrue(second.is_active)

    def test_get_active_returns_active(self):
        Personality.objects.create(
            name='A', description='d', instructions='i', is_active=False)
        active = Personality.objects.create(
            name='B', description='d', instructions='i', is_active=True)
        self.assertEqual(Personality.get_active(), active)

    def test_get_active_none_when_no_active(self):
        Personality.objects.create(
            name='A', description='d', instructions='i', is_active=False)
        self.assertIsNone(Personality.get_active())


class UserContextTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='x')

    def test_for_user_creates_empty(self):
        ctx = UserContext.for_user(self.user)
        self.assertEqual(ctx.user_id, self.user.id)
        self.assertEqual(ctx.username, 'leo')
        self.assertEqual(ctx.content, '')

    def test_for_user_is_idempotent(self):
        a = UserContext.for_user(self.user)
        b = UserContext.for_user(self.user)
        self.assertEqual(a.pk, b.pk)
        self.assertEqual(UserContext.objects.filter(user_id=self.user.id).count(), 1)


class BuildSystemPromptTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='x')

    def test_includes_personality_facts_context_and_guardrails(self):
        Personality.objects.create(
            name='Mycroft', description='You are Mycroft.',
            instructions='Be concise.', is_active=True)
        HouseholdFact.objects.create(content='Based in Raleigh, NC.')
        ctx = UserContext.for_user(self.user)
        ctx.content = 'Leo runs a blog.'
        ctx.save()

        prompt = build_system_prompt(self.user)

        self.assertIn('You are Mycroft.', prompt)
        self.assertIn('Be concise.', prompt)
        self.assertIn('About the household:', prompt)
        self.assertIn('- Based in Raleigh, NC.', prompt)
        self.assertIn('About the current user:', prompt)
        self.assertIn('Leo runs a blog.', prompt)
        self.assertIn(STANDARD_GUARDRAILS, prompt)

    def test_omits_empty_sections(self):
        Personality.objects.create(
            name='Mycroft', description='You are Mycroft.',
            instructions='', is_active=True)
        prompt = build_system_prompt(self.user)
        self.assertNotIn('About the household:', prompt)
        self.assertNotIn('About the current user:', prompt)
        self.assertIn(STANDARD_GUARDRAILS, prompt)

    def test_tolerates_no_active_personality(self):
        prompt = build_system_prompt(self.user)
        self.assertIn('Mycroft', prompt)
        self.assertIn(STANDARD_GUARDRAILS, prompt)


class SendMessageViewTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='secret')
        self.url = reverse('ai_lab_chatbot:send')
        Personality.objects.create(
            name='Mycroft', description='You are Mycroft.',
            instructions='', is_active=True)

    def test_requires_login(self):
        resp = self.client.post(self.url, data='{}',
                                content_type='application/json')
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/login', resp['Location'])

    def _frames(self, resp):
        body = b''.join(resp.streaming_content).decode()
        return [json.loads(line) for line in body.splitlines() if line.strip()]

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_streams_token_frames_then_done(self, mock_stream):
        mock_stream.return_value = iter(['Hello', ', ', 'Leo.'])
        self.client.login(username='leo', password='secret')

        resp = self.client.post(
            self.url,
            data='{"messages": [{"role": "user", "content": "hi"}]}',
            content_type='application/json',
        )

        frames = self._frames(resp)
        tokens = [f['content'] for f in frames if f['type'] == 'token']
        self.assertEqual(''.join(tokens), 'Hello, Leo.')
        # A terminal 'done' is how the client knows it wasn't truncated.
        self.assertEqual(frames[-1]['type'], 'done')

        # The system prompt is prepended before the user's message.
        sent_messages = mock_stream.call_args.args[0]
        self.assertEqual(sent_messages[0]['role'], 'system')
        self.assertEqual(sent_messages[-1], {'role': 'user', 'content': 'hi'})

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_mid_stream_failure_yields_error_frame_and_no_done(self, mock_stream):
        def blow_up():
            yield 'Partial'
            raise RuntimeError('ollama went away')

        mock_stream.return_value = blow_up()
        self.client.login(username='leo', password='secret')

        resp = self.client.post(
            self.url,
            data='{"messages": [{"role": "user", "content": "hi"}]}',
            content_type='application/json',
        )

        frames = self._frames(resp)
        self.assertEqual(frames[0], {'type': 'token', 'content': 'Partial'})
        self.assertEqual(frames[-1]['type'], 'error')
        self.assertNotIn('done', [f['type'] for f in frames])

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_rejects_empty_history(self, mock_stream):
        self.client.login(username='leo', password='secret')
        resp = self.client.post(
            self.url, data='{"messages": []}',
            content_type='application/json')
        self.assertEqual(resp.status_code, 400)
        mock_stream.assert_not_called()
