import json
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse

from ai_lab_chatbot.models import (
    Personality, HouseholdFact, UserContext, Conversation, Message,
)
from ai_lab_chatbot.mycroft import memory
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


class ConversationModelTests(TestCase):
    databases = DBS

    def test_display_title_prefers_title(self):
        conv = Conversation.objects.create(user_id=1, username='leo', title='Weather chat')
        self.assertEqual(conv.display_title(), 'Weather chat')

    def test_display_title_falls_back_to_first_user_message(self):
        conv = Conversation.objects.create(user_id=1, username='leo')
        Message.objects.create(conversation=conv, role='user', content='How tall is Everest?')
        self.assertEqual(conv.display_title(), 'How tall is Everest?')

    def test_display_title_empty_conversation(self):
        conv = Conversation.objects.create(user_id=1, username='leo')
        self.assertEqual(conv.display_title(), 'New conversation')


class MemoryTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='x')
        self.other = User.objects.create_user(username='sam', password='x')

    def test_get_or_create_makes_new_when_id_falsy(self):
        conv = memory.get_or_create_conversation(self.user, None)
        self.assertEqual(conv.user_id, self.user.id)
        self.assertEqual(conv.username, 'leo')

    def test_get_or_create_fetches_owned(self):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        fetched = memory.get_or_create_conversation(self.user, conv.id)
        self.assertEqual(fetched.id, conv.id)

    def test_get_or_create_rejects_other_users_conversation(self):
        conv = Conversation.objects.create(user_id=self.other.id, username='sam')
        with self.assertRaises(Conversation.DoesNotExist):
            memory.get_or_create_conversation(self.user, conv.id)

    @override_settings(MYCROFT_HISTORY_WINDOW=3)
    def test_history_for_prompt_windows_and_orders(self):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        for i in range(5):
            memory.add_message(conv, 'user', f'm{i}')
        window = memory.history_for_prompt(conv)
        self.assertEqual([m['content'] for m in window], ['m2', 'm3', 'm4'])

    def test_touch_bumps_updated_at(self):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        before = conv.updated_at
        memory.add_message(conv, 'user', 'hi')
        memory.touch(conv)
        conv.refresh_from_db()
        self.assertGreater(conv.updated_at, before)

    def test_recent_conversations_scoped_to_user(self):
        mine = Conversation.objects.create(user_id=self.user.id, username='leo')
        Conversation.objects.create(user_id=self.other.id, username='sam')
        result = list(memory.recent_conversations(self.user))
        self.assertEqual(result, [mine])

    def test_favorite_and_other_partition_conversations(self):
        fav = Conversation.objects.create(
            user_id=self.user.id, username='leo', is_favorite=True)
        plain = Conversation.objects.create(user_id=self.user.id, username='leo')
        Conversation.objects.create(user_id=self.other.id, username='sam', is_favorite=True)

        self.assertEqual(list(memory.favorite_conversations(self.user)), [fav])
        self.assertEqual(list(memory.other_conversations(self.user)), [plain])

    def test_toggle_favorite_flips_and_preserves_updated_at(self):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        before = conv.updated_at

        toggled = memory.toggle_favorite(self.user, conv.id)
        self.assertTrue(toggled.is_favorite)
        toggled.refresh_from_db()
        # Favoriting is not activity — updated_at must not move.
        self.assertEqual(toggled.updated_at, before)

        again = memory.toggle_favorite(self.user, conv.id)
        self.assertFalse(again.is_favorite)

    def test_toggle_favorite_rejects_other_users_conversation(self):
        conv = Conversation.objects.create(user_id=self.other.id, username='sam')
        with self.assertRaises(Conversation.DoesNotExist):
            memory.toggle_favorite(self.user, conv.id)

    def test_delete_conversation_scoped(self):
        conv = Conversation.objects.create(user_id=self.other.id, username='sam')
        with self.assertRaises(Conversation.DoesNotExist):
            memory.delete_conversation(self.user, conv.id)
        self.assertTrue(Conversation.objects.filter(id=conv.id).exists())


class SendMessageViewTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='secret')
        self.url = reverse('ai_lab_chatbot:send')
        Personality.objects.create(
            name='Mycroft', description='You are Mycroft.',
            instructions='', is_active=True)

    def _post(self, **body):
        return self.client.post(
            self.url, data=json.dumps(body), content_type='application/json')

    def _frames(self, resp):
        body = b''.join(resp.streaming_content).decode()
        return [json.loads(line) for line in body.splitlines() if line.strip()]

    def test_requires_login(self):
        resp = self._post(conversation_id=None, content='hi')
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/login', resp['Location'])

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_creates_conversation_and_persists_both_turns(self, mock_stream):
        mock_stream.return_value = iter(['Hello', ', ', 'Leo.'])
        self.client.login(username='leo', password='secret')

        resp = self._post(conversation_id=None, content='hi')
        frames = self._frames(resp)

        tokens = [f['content'] for f in frames if f['type'] == 'token']
        self.assertEqual(''.join(tokens), 'Hello, Leo.')
        done = frames[-1]
        self.assertEqual(done['type'], 'done')
        self.assertTrue(done['is_new'])

        conv = Conversation.objects.get(id=done['conversation_id'])
        self.assertEqual(conv.user_id, self.user.id)
        roles = list(conv.messages.values_list('role', 'content'))
        self.assertEqual(roles, [('user', 'hi'), ('assistant', 'Hello, Leo.')])

        # System prompt prepended; the persisted user turn is the last message.
        sent_messages = mock_stream.call_args.args[0]
        self.assertEqual(sent_messages[0]['role'], 'system')
        self.assertEqual(sent_messages[-1], {'role': 'user', 'content': 'hi'})

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_continues_existing_conversation(self, mock_stream):
        mock_stream.return_value = iter(['ok'])
        self.client.login(username='leo', password='secret')
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        memory.add_message(conv, 'user', 'earlier')
        memory.add_message(conv, 'assistant', 'reply')

        resp = self._post(conversation_id=str(conv.id), content='again')
        frames = self._frames(resp)
        self.assertFalse(frames[-1]['is_new'])
        self.assertEqual(Conversation.objects.count(), 1)
        self.assertEqual(conv.messages.count(), 4)

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_other_users_conversation_is_404(self, mock_stream):
        other = User.objects.create_user(username='sam', password='x')
        conv = Conversation.objects.create(user_id=other.id, username='sam')
        self.client.login(username='leo', password='secret')
        resp = self._post(conversation_id=str(conv.id), content='hi')
        self.assertEqual(resp.status_code, 404)
        mock_stream.assert_not_called()

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_mid_stream_failure_saves_partial_and_no_done(self, mock_stream):
        def blow_up():
            yield 'Partial'
            raise RuntimeError('ollama went away')

        mock_stream.return_value = blow_up()
        self.client.login(username='leo', password='secret')

        resp = self._post(conversation_id=None, content='hi')
        frames = self._frames(resp)

        self.assertEqual(frames[0], {'type': 'token', 'content': 'Partial'})
        self.assertEqual(frames[-1]['type'], 'error')
        self.assertNotIn('done', [f['type'] for f in frames])

        # The partial assistant text is persisted (transcript-visible).
        conv = Conversation.objects.get(id=frames[-1]['conversation_id'])
        self.assertEqual(
            list(conv.messages.values_list('role', 'content')),
            [('user', 'hi'), ('assistant', 'Partial')],
        )

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_rejects_empty_content(self, mock_stream):
        self.client.login(username='leo', password='secret')
        resp = self._post(conversation_id=None, content='   ')
        self.assertEqual(resp.status_code, 400)
        mock_stream.assert_not_called()
        self.assertEqual(Conversation.objects.count(), 0)


class ResumeAndHistoryViewTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='secret')
        self.other = User.objects.create_user(username='sam', password='x')

    def test_resume_renders_transcript_for_owner(self):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        memory.add_message(conv, 'user', 'hello there')
        memory.add_message(conv, 'assistant', 'general kenobi')
        self.client.login(username='leo', password='secret')

        resp = self.client.get(
            reverse('ai_lab_chatbot:conversation', args=[conv.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'hello there')
        self.assertContains(resp, 'general kenobi')

    def test_resume_other_users_conversation_is_404(self):
        conv = Conversation.objects.create(user_id=self.other.id, username='sam')
        self.client.login(username='leo', password='secret')
        resp = self.client.get(
            reverse('ai_lab_chatbot:conversation', args=[conv.id]))
        self.assertEqual(resp.status_code, 404)

    def test_history_lists_only_own_conversations(self):
        mine = Conversation.objects.create(
            user_id=self.user.id, username='leo', title='Mine')
        Conversation.objects.create(
            user_id=self.other.id, username='sam', title='Theirs')
        self.client.login(username='leo', password='secret')

        resp = self.client.get(reverse('ai_lab_chatbot:history'))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Mine')
        self.assertNotContains(resp, 'Theirs')

    def test_history_splits_favorites_from_others(self):
        Conversation.objects.create(
            user_id=self.user.id, username='leo', title='Pinned', is_favorite=True)
        Conversation.objects.create(
            user_id=self.user.id, username='leo', title='Plain')
        self.client.login(username='leo', password='secret')

        resp = self.client.get(reverse('ai_lab_chatbot:history'))
        self.assertEqual(list(resp.context['favorites'].values_list('title', flat=True)),
                         ['Pinned'])
        self.assertEqual(list(resp.context['conversations'].values_list('title', flat=True)),
                         ['Plain'])


class FavoriteViewTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='secret')
        self.other = User.objects.create_user(username='sam', password='x')

    def test_toggle_returns_new_state(self):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        self.client.login(username='leo', password='secret')
        url = reverse('ai_lab_chatbot:favorite', args=[conv.id])

        resp = self.client.post(url)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()['is_favorite'])
        conv.refresh_from_db()
        self.assertTrue(conv.is_favorite)

        resp = self.client.post(url)
        self.assertFalse(resp.json()['is_favorite'])

    def test_toggle_requires_post(self):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        self.client.login(username='leo', password='secret')
        resp = self.client.get(reverse('ai_lab_chatbot:favorite', args=[conv.id]))
        self.assertEqual(resp.status_code, 405)

    def test_cannot_favorite_other_users_conversation(self):
        conv = Conversation.objects.create(user_id=self.other.id, username='sam')
        self.client.login(username='leo', password='secret')
        resp = self.client.post(reverse('ai_lab_chatbot:favorite', args=[conv.id]))
        self.assertEqual(resp.status_code, 404)
        conv.refresh_from_db()
        self.assertFalse(conv.is_favorite)


class TitleViewTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='secret')
        self.conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        memory.add_message(self.conv, 'user', 'How do I center a div?')
        memory.add_message(self.conv, 'assistant', 'Use flexbox.')
        self.url = reverse('ai_lab_chatbot:title', args=[self.conv.id])
        self.client.login(username='leo', password='secret')

    @patch('ai_lab_chatbot.views.complete_chat')
    def test_generates_and_saves_title(self, mock_complete):
        mock_complete.return_value = '"Centering a div"'
        resp = self.client.post(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['title'], 'Centering a div')
        self.conv.refresh_from_db()
        self.assertEqual(self.conv.title, 'Centering a div')

    @patch('ai_lab_chatbot.views.complete_chat')
    def test_noop_if_title_already_set(self, mock_complete):
        self.conv.title = 'Existing'
        self.conv.save()
        resp = self.client.post(self.url)
        self.assertEqual(resp.json()['title'], 'Existing')
        mock_complete.assert_not_called()

    @patch('ai_lab_chatbot.views.complete_chat',
           side_effect=RuntimeError('ollama down'))
    def test_failure_leaves_title_blank(self, mock_complete):
        resp = self.client.post(self.url)
        self.assertEqual(resp.json()['title'], '')
        self.conv.refresh_from_db()
        self.assertEqual(self.conv.title, '')

    @patch('ai_lab_chatbot.views.complete_chat')
    def test_other_users_conversation_is_404(self, mock_complete):
        other = User.objects.create_user(username='sam', password='x')
        conv = Conversation.objects.create(user_id=other.id, username='sam')
        resp = self.client.post(
            reverse('ai_lab_chatbot:title', args=[conv.id]))
        self.assertEqual(resp.status_code, 404)
        mock_complete.assert_not_called()


class DeleteViewTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='secret')
        self.other = User.objects.create_user(username='sam', password='x')

    def test_delete_removes_conversation_and_messages(self):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        memory.add_message(conv, 'user', 'hi')
        self.client.login(username='leo', password='secret')

        resp = self.client.post(reverse('ai_lab_chatbot:delete', args=[conv.id]))
        self.assertRedirects(resp, reverse('ai_lab_chatbot:history'))
        self.assertFalse(Conversation.objects.filter(id=conv.id).exists())
        self.assertFalse(Message.objects.filter(conversation_id=conv.id).exists())

    def test_cannot_delete_other_users_conversation(self):
        conv = Conversation.objects.create(user_id=self.other.id, username='sam')
        self.client.login(username='leo', password='secret')
        resp = self.client.post(reverse('ai_lab_chatbot:delete', args=[conv.id]))
        self.assertEqual(resp.status_code, 404)
        self.assertTrue(Conversation.objects.filter(id=conv.id).exists())

    def test_delete_requires_post(self):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        self.client.login(username='leo', password='secret')
        resp = self.client.get(reverse('ai_lab_chatbot:delete', args=[conv.id]))
        self.assertEqual(resp.status_code, 405)
