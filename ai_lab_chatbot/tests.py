import json
from types import SimpleNamespace
from unittest.mock import patch

from django.conf import settings
from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from ai_lab_chatbot.models import (
    Personality, HouseholdFact, UserContext, Conversation, Message, Knowledge,
    MycroftConfig,
)
from ai_lab_chatbot.mycroft import memory, tools, weather
from ai_lab_chatbot.mycroft.client import (
    stream_chat, complete_chat, list_models, supports_tools,
)
from ai_lab_chatbot.mycroft.prompts import (
    build_system_prompt, LEARNING_INSTRUCTIONS, STANDARD_GUARDRAILS,
)


# Models live in the `ai_lab` DB and auth.User in `default`; both are needed.
DBS = {'default', 'ai_lab'}

EMBED_DIM = 768


def _vec(pairs=()):
    """A 768-dim test embedding with the given (index, value) components set,
    the rest zero — lets tests place vectors at known cosine distances without
    calling Ollama."""
    v = [0.0] * EMBED_DIM
    for index, value in pairs:
        v[index] = value
    return v


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

    def test_superseded_facts_excluded(self):
        HouseholdFact.objects.create(content='Lives at 12 Oak St.')
        HouseholdFact.objects.create(content='Lives at 9 Elm Rd.', is_current=False)
        prompt = build_system_prompt(self.user)
        self.assertIn('12 Oak St', prompt)
        self.assertNotIn('9 Elm Rd', prompt)

    def test_learning_instructions_only_in_learning_mode(self):
        self.assertNotIn(LEARNING_INSTRUCTIONS, build_system_prompt(self.user))
        self.assertIn(
            LEARNING_INSTRUCTIONS,
            build_system_prompt(self.user, mode='learning'),
        )


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

    def test_rename_sets_title_and_preserves_updated_at(self):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        before = conv.updated_at

        renamed = memory.rename_conversation(self.user, conv.id, '  Groceries  ')
        # Trimmed, and renaming is not activity — updated_at must not move.
        self.assertEqual(renamed.title, 'Groceries')
        renamed.refresh_from_db()
        self.assertEqual(renamed.title, 'Groceries')
        self.assertEqual(renamed.updated_at, before)

    def test_rename_blank_keeps_existing_title(self):
        conv = Conversation.objects.create(
            user_id=self.user.id, username='leo', title='Old name')

        # A blank save is a no-op — the existing name must survive.
        renamed = memory.rename_conversation(self.user, conv.id, '   ')
        self.assertEqual(renamed.title, 'Old name')
        renamed.refresh_from_db()
        self.assertEqual(renamed.title, 'Old name')

    def test_rename_truncates_to_200(self):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        renamed = memory.rename_conversation(self.user, conv.id, 'x' * 250)
        self.assertEqual(len(renamed.title), 200)

    def test_rename_rejects_other_users_conversation(self):
        conv = Conversation.objects.create(user_id=self.other.id, username='sam')
        with self.assertRaises(Conversation.DoesNotExist):
            memory.rename_conversation(self.user, conv.id, 'mine now')

    def test_delete_conversation_scoped(self):
        conv = Conversation.objects.create(user_id=self.other.id, username='sam')
        with self.assertRaises(Conversation.DoesNotExist):
            memory.delete_conversation(self.user, conv.id)
        self.assertTrue(Conversation.objects.filter(id=conv.id).exists())


class EmbeddingHelperTests(TestCase):
    """embed_message / embed_knowledge store the vector, and are best-effort —
    an embed failure is swallowed so the chat path never breaks."""
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='x')
        self.conv = Conversation.objects.create(user_id=self.user.id, username='leo')

    @patch('ai_lab_chatbot.mycroft.memory.embed_text')
    def test_embed_message_stores_vector(self, mock_embed):
        mock_embed.return_value = _vec([(0, 1.0)])
        msg = memory.add_message(self.conv, 'user', 'remember trains')

        returned = memory.embed_message(msg)

        self.assertIsNotNone(returned)
        msg.refresh_from_db()
        self.assertIsNotNone(msg.embedding)
        self.assertEqual(len(list(msg.embedding)), EMBED_DIM)
        mock_embed.assert_called_once_with('remember trains')

    @patch('ai_lab_chatbot.mycroft.memory.embed_text',
           side_effect=RuntimeError('ollama down'))
    def test_embed_message_swallows_failure(self, mock_embed):
        msg = memory.add_message(self.conv, 'user', 'remember trains')

        # assertLogs both asserts the error is logged and keeps the expected
        # traceback out of the test console.
        with self.assertLogs('ai_lab_chatbot.mycroft.memory', level='ERROR'):
            returned = memory.embed_message(msg)

        self.assertIsNone(returned)
        msg.refresh_from_db()
        self.assertIsNone(msg.embedding)  # still saved, just unretrievable

    @patch('ai_lab_chatbot.mycroft.memory.embed_text')
    def test_embed_knowledge_stores_vector_from_topic_and_content(self, mock_embed):
        mock_embed.return_value = _vec([(3, 1.0)])
        k = Knowledge.objects.create(topic='Car', content='Leo drives an Outback')

        memory.embed_knowledge(k)

        k.refresh_from_db()
        self.assertIsNotNone(k.embedding)
        mock_embed.assert_called_once_with('Car\nLeo drives an Outback')


class RetrieveMemoriesTests(TestCase):
    """Semantic retrieval: cosine ordering, the distance ceiling, and the
    privacy/de-dup scoping rules. Requires the `vector` extension in the test DB."""
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='x')
        self.other = User.objects.create_user(username='sam', password='x')
        self.conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        self.old_conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        # Query points along axis 0. Distances (cosine): near ~0.02, mid ~0.29,
        # far = 1.0 (orthogonal, beyond the 0.55 default ceiling).
        self.query = _vec([(0, 1.0)])
        self.near = _vec([(0, 1.0), (1, 0.2)])
        self.mid = _vec([(0, 1.0), (1, 1.0)])
        self.far = _vec([(1, 1.0)])

    def _msg(self, conv, content, embedding, role='user'):
        return Message.objects.create(
            conversation=conv, role=role, content=content, embedding=embedding)

    def test_orders_by_distance_and_drops_beyond_ceiling(self):
        self._msg(self.old_conv, 'mid', self.mid)
        self._msg(self.old_conv, 'near', self.near)
        self._msg(self.old_conv, 'far', self.far)

        _, messages = memory.retrieve_memories(
            self.user, self.query, exclude_conversation_id=self.conv.id)

        # Closest first; the orthogonal 'far' match is filtered out.
        self.assertEqual([m.content for m in messages], ['near', 'mid'])

    def test_excludes_current_conversation(self):
        self._msg(self.conv, 'in current window', self.near)
        self._msg(self.old_conv, 'older recall', self.near)

        _, messages = memory.retrieve_memories(
            self.user, self.query, exclude_conversation_id=self.conv.id)

        self.assertEqual([m.content for m in messages], ['older recall'])

    def test_scoped_to_user(self):
        theirs = Conversation.objects.create(user_id=self.other.id, username='sam')
        self._msg(theirs, "sam's secret", self.near)
        self._msg(self.old_conv, 'my memory', self.near)

        _, messages = memory.retrieve_memories(
            self.user, self.query, exclude_conversation_id=self.conv.id)

        self.assertEqual([m.content for m in messages], ['my memory'])

    def test_ignores_rows_without_embedding(self):
        self._msg(self.old_conv, 'no vector', None)

        _, messages = memory.retrieve_memories(
            self.user, self.query, exclude_conversation_id=self.conv.id)

        self.assertEqual(messages, [])

    def test_knowledge_is_global_and_thresholded(self):
        Knowledge.objects.create(topic='near', content='relevant', embedding=self.near)
        Knowledge.objects.create(topic='far', content='irrelevant', embedding=self.far)

        knowledge, _ = memory.retrieve_memories(
            self.user, self.query, exclude_conversation_id=self.conv.id)

        self.assertEqual([k.content for k in knowledge], ['relevant'])

    @override_settings(MYCROFT_RETRIEVAL_MESSAGES=1)
    def test_message_limit_honored(self):
        self._msg(self.old_conv, 'near', self.near)
        self._msg(self.old_conv, 'mid', self.mid)

        _, messages = memory.retrieve_memories(
            self.user, self.query, exclude_conversation_id=self.conv.id)

        self.assertEqual([m.content for m in messages], ['near'])

    def test_none_query_returns_empty(self):
        self._msg(self.old_conv, 'near', self.near)
        self.assertEqual(
            memory.retrieve_memories(self.user, None, exclude_conversation_id=self.conv.id),
            ([], []),
        )


class RetrievedPromptTests(TestCase):
    """build_system_prompt renders the retrieved knowledge / past sections when
    given them, and omits them otherwise."""
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='x')
        Personality.objects.create(
            name='Mycroft', description='You are Mycroft.', instructions='',
            is_active=True)

    def test_injects_knowledge_and_past(self):
        knowledge = [SimpleNamespace(content='Leo drives an Outback')]
        past = [SimpleNamespace(role='user', content='I love trains')]

        prompt = build_system_prompt(self.user, knowledge=knowledge, past=past)

        self.assertIn('Relevant knowledge:', prompt)
        self.assertIn('- Leo drives an Outback', prompt)
        self.assertIn('Relevant past conversation:', prompt)
        self.assertIn('- user: I love trains', prompt)

    def test_omits_sections_when_empty(self):
        prompt = build_system_prompt(self.user, knowledge=[], past=[])
        self.assertNotIn('Relevant knowledge:', prompt)
        self.assertNotIn('Relevant past conversation:', prompt)

    def test_default_args_unchanged(self):
        # No retrieval args → behaves exactly as before.
        self.assertNotIn('Relevant knowledge:', build_system_prompt(self.user))


class StreamChatStatsTests(TestCase):
    """The client wrapper: num_ctx is sent, and the final chunk's metrics land
    in stats_out with durations converted from nanoseconds to milliseconds."""

    @staticmethod
    def _chunk(content, done=False, **metrics):
        return SimpleNamespace(
            message=SimpleNamespace(content=content), done=done, **metrics)

    @patch('ai_lab_chatbot.mycroft.client._client')
    def test_sends_num_ctx_and_captures_final_metrics(self, mock_client):
        chunks = [
            self._chunk('Hello'),
            self._chunk(' world'),
            self._chunk('', done=True, prompt_eval_count=1500, eval_count=300,
                        eval_duration=5_000_000_000, total_duration=6_000_000_000),
        ]
        mock_client.return_value.chat.return_value = iter(chunks)

        stats = {}
        out = list(stream_chat([{'role': 'user', 'content': 'hi'}], stats_out=stats))

        self.assertEqual(out, ['Hello', ' world'])
        self.assertEqual(
            mock_client.return_value.chat.call_args.kwargs['options'],
            {'num_ctx': settings.MYCROFT_NUM_CTX},
        )
        self.assertEqual(stats['prompt_tokens'], 1500)
        self.assertEqual(stats['completion_tokens'], 300)
        self.assertEqual(stats['eval_duration_ms'], 5000.0)   # ns → ms
        self.assertEqual(stats['total_duration_ms'], 6000.0)

    @patch('ai_lab_chatbot.mycroft.client._client')
    def test_stats_out_optional(self, mock_client):
        mock_client.return_value.chat.return_value = iter([
            self._chunk('hi', done=True, prompt_eval_count=1, eval_count=1,
                        eval_duration=0, total_duration=0),
        ])
        # No stats_out passed — must not raise.
        self.assertEqual(list(stream_chat([])), ['hi'])


class SendMessageViewTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='secret')
        self.url = reverse('ai_lab_chatbot:send')
        patcher = patch('ai_lab_chatbot.views.supports_tools', return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        Personality.objects.create(
            name='Mycroft', description='You are Mycroft.',
            instructions='', is_active=True)

    def _post(self, **body):
        return self.client.post(
            self.url, data=json.dumps(body), content_type='application/json')

    def _frames(self, resp):
        body = b''.join(resp.streaming_content).decode()
        return [json.loads(line) for line in body.splitlines() if line.strip()]

    @staticmethod
    def _stream_with_stats(tokens, stats):
        """A stream_chat stand-in that yields `tokens` and fills the caller's
        stats_out dict, mirroring a clean Ollama completion."""
        def _side_effect(messages, stats_out=None, model=None, **kwargs):
            if stats_out is not None:
                stats_out.update(stats)
            return iter(tokens)
        return _side_effect

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
    def test_stores_token_counts_on_assistant_message(self, mock_stream):
        mock_stream.side_effect = self._stream_with_stats(
            ['Hi'],
            {'prompt_tokens': 1200, 'completion_tokens': 5,
             'eval_duration_ms': 500.0, 'total_duration_ms': 800.0},
        )
        self.client.login(username='leo', password='secret')

        resp = self._post(conversation_id=None, content='hi')
        frames = self._frames(resp)

        conv = Conversation.objects.get(id=frames[-1]['conversation_id'])
        assistant = conv.messages.get(role='assistant')
        self.assertEqual(assistant.prompt_tokens, 1200)
        self.assertEqual(assistant.completion_tokens, 5)
        # The user turn never carries counts.
        self.assertIsNone(conv.messages.get(role='user').prompt_tokens)

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_emits_stats_frame_before_done(self, mock_stream):
        mock_stream.side_effect = self._stream_with_stats(
            ['Hi'],
            {'prompt_tokens': 1200, 'completion_tokens': 5,
             'eval_duration_ms': 500.0, 'total_duration_ms': 800.0},
        )
        self.client.login(username='leo', password='secret')

        frames = self._frames(self._post(conversation_id=None, content='hi'))
        types = [f['type'] for f in frames]

        self.assertIn('stats', types)
        self.assertLess(types.index('stats'), types.index('done'))
        stats = frames[types.index('stats')]
        self.assertEqual(stats['prompt_tokens'], 1200)
        self.assertEqual(stats['completion_tokens'], 5)
        self.assertEqual(stats['eval_duration_ms'], 500.0)
        self.assertEqual(stats['total_duration_ms'], 800.0)

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_no_stats_frame_when_metrics_absent(self, mock_stream):
        # Stream that never populates stats_out (metrics unavailable): no stats
        # frame, and the assistant message keeps null counts.
        mock_stream.return_value = iter(['Hi'])
        self.client.login(username='leo', password='secret')

        frames = self._frames(self._post(conversation_id=None, content='hi'))
        self.assertNotIn('stats', [f['type'] for f in frames])

        conv = Conversation.objects.get(id=frames[-1]['conversation_id'])
        assistant = conv.messages.get(role='assistant')
        self.assertIsNone(assistant.prompt_tokens)
        self.assertIsNone(assistant.completion_tokens)

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_rejects_empty_content(self, mock_stream):
        self.client.login(username='leo', password='secret')
        resp = self._post(conversation_id=None, content='   ')
        self.assertEqual(resp.status_code, 400)
        mock_stream.assert_not_called()
        self.assertEqual(Conversation.objects.count(), 0)

    @patch('ai_lab_chatbot.mycroft.memory.embed_text')
    @patch('ai_lab_chatbot.views.stream_chat')
    def test_retrieved_knowledge_feeds_system_prompt(self, mock_stream, mock_embed):
        # The incoming turn embeds to a vector that matches a seeded Knowledge row.
        mock_embed.return_value = _vec([(0, 1.0)])
        mock_stream.return_value = iter(['ok'])
        Knowledge.objects.create(
            topic='Car', content='Leo drives a 2020 Subaru Outback',
            embedding=_vec([(0, 1.0)]))
        self.client.login(username='leo', password='secret')

        self._frames(self._post(conversation_id=None, content='what car?'))

        system_prompt = mock_stream.call_args.args[0][0]['content']
        self.assertIn('Relevant knowledge:', system_prompt)
        self.assertIn('Leo drives a 2020 Subaru Outback', system_prompt)

    @patch('ai_lab_chatbot.mycroft.memory.embed_text')
    @patch('ai_lab_chatbot.views.stream_chat')
    def test_embeds_both_turns(self, mock_stream, mock_embed):
        mock_embed.return_value = _vec([(0, 1.0)])
        mock_stream.return_value = iter(['reply'])
        self.client.login(username='leo', password='secret')

        frames = self._frames(self._post(conversation_id=None, content='hi'))

        conv = Conversation.objects.get(id=frames[-1]['conversation_id'])
        self.assertIsNotNone(conv.messages.get(role='user').embedding)
        self.assertIsNotNone(conv.messages.get(role='assistant').embedding)

    @patch('ai_lab_chatbot.mycroft.memory.embed_text',
           side_effect=RuntimeError('embed model down'))
    @patch('ai_lab_chatbot.views.stream_chat')
    def test_chat_survives_embed_failure(self, mock_stream, mock_embed):
        mock_stream.return_value = iter(['still works'])
        self.client.login(username='leo', password='secret')

        # Both turns fail to embed; assertLogs captures those expected errors so
        # the traceback stays out of the console.
        with self.assertLogs('ai_lab_chatbot.mycroft.memory', level='ERROR'):
            frames = self._frames(self._post(conversation_id=None, content='hi'))

        # Reply streamed and completed despite embedding failing on both turns.
        self.assertEqual(frames[-1]['type'], 'done')
        conv = Conversation.objects.get(id=frames[-1]['conversation_id'])
        self.assertEqual(
            list(conv.messages.values_list('role', 'content')),
            [('user', 'hi'), ('assistant', 'still works')])
        self.assertIsNone(conv.messages.get(role='assistant').embedding)


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

    def test_resume_bootstrap_carries_context_from_last_assistant(self):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        memory.add_message(conv, 'user', 'hi')
        memory.add_message(conv, 'assistant', 'hello',
                           prompt_tokens=1500, completion_tokens=42)
        self.client.login(username='leo', password='secret')

        resp = self.client.get(
            reverse('ai_lab_chatbot:conversation', args=[conv.id]))
        self.assertEqual(
            resp.context['bootstrap']['context'],
            {'prompt_tokens': 1500, 'completion_tokens': 42},
        )
        self.assertEqual(resp.context['mycroft_num_ctx'], settings.MYCROFT_NUM_CTX)

    def test_resume_context_none_for_pre_feature_messages(self):
        # An assistant message with null counts (predates the feature) resumes
        # without error and reports no context.
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        memory.add_message(conv, 'user', 'hi')
        memory.add_message(conv, 'assistant', 'hello')
        self.client.login(username='leo', password='secret')

        resp = self.client.get(
            reverse('ai_lab_chatbot:conversation', args=[conv.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.context['bootstrap']['context'])
        self.assertContains(resp, 'hello')  # transcript still renders

    def test_fresh_chat_has_no_context(self):
        self.client.login(username='leo', password='secret')
        resp = self.client.get(reverse('ai_lab_chatbot:chat'))
        self.assertIsNone(resp.context['bootstrap']['context'])
        self.assertEqual(resp.context['mycroft_num_ctx'], settings.MYCROFT_NUM_CTX)

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


class RenameViewTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='secret')
        self.other = User.objects.create_user(username='sam', password='x')

    def _post(self, conv, title):
        return self.client.post(
            reverse('ai_lab_chatbot:rename', args=[conv.id]),
            data=json.dumps({'title': title}),
            content_type='application/json',
        )

    def test_rename_returns_title_and_display_title(self):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        self.client.login(username='leo', password='secret')

        resp = self._post(conv, 'Weekend plans')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(),
                         {'title': 'Weekend plans', 'display_title': 'Weekend plans'})
        conv.refresh_from_db()
        self.assertEqual(conv.title, 'Weekend plans')

    def test_blank_title_keeps_previous(self):
        conv = Conversation.objects.create(
            user_id=self.user.id, username='leo', title='Old name')
        self.client.login(username='leo', password='secret')

        # Blank save is a no-op; the response echoes the unchanged title.
        resp = self._post(conv, '   ')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['title'], 'Old name')
        self.assertEqual(resp.json()['display_title'], 'Old name')
        conv.refresh_from_db()
        self.assertEqual(conv.title, 'Old name')

    def test_rename_requires_post(self):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        self.client.login(username='leo', password='secret')
        resp = self.client.get(reverse('ai_lab_chatbot:rename', args=[conv.id]))
        self.assertEqual(resp.status_code, 405)

    def test_cannot_rename_other_users_conversation(self):
        conv = Conversation.objects.create(user_id=self.other.id, username='sam')
        self.client.login(username='leo', password='secret')
        resp = self._post(conv, 'mine now')
        self.assertEqual(resp.status_code, 404)
        conv.refresh_from_db()
        self.assertEqual(conv.title, '')

    def test_rename_requires_login(self):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        resp = self._post(conv, 'x')
        self.assertEqual(resp.status_code, 302)


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


class ParseFactOpsTests(TestCase):
    """Unit tests for the defensive JSON-object parse (no DB, no Ollama)."""

    def test_full_object(self):
        raw = json.dumps({
            'add': ['A fact.'],
            'update': [{'id': 3, 'content': 'New text.'}],
            'retire': [7],
        })
        self.assertEqual(memory._parse_fact_ops(raw), {
            'add': ['A fact.'],
            'update': [{'id': 3, 'content': 'New text.'}],
            'retire': [7],
        })

    def test_code_fence_and_prose_tolerated(self):
        raw = 'Sure:\n```json\n{"add": ["Leo has a garden."]}\n```'
        ops = memory._parse_fact_ops(raw)
        self.assertEqual(ops['add'], ['Leo has a garden.'])
        self.assertEqual(ops['update'], [])
        self.assertEqual(ops['retire'], [])

    def test_non_json_yields_empty(self):
        self.assertEqual(
            memory._parse_fact_ops('Nothing to change.'),
            {'add': [], 'update': [], 'retire': []},
        )

    def test_bad_entries_dropped(self):
        raw = json.dumps({
            'add': ['Real fact.', '', '   ', 42, None],
            'update': [{'id': 'x', 'content': 'Bad id.'},
                       {'id': '4', 'content': 'Quoted id ok.'},
                       {'id': 5, 'content': ''}, 'nope'],
            'retire': ['9', True, None, 'abc'],
        })
        self.assertEqual(memory._parse_fact_ops(raw), {
            'add': ['Real fact.'],
            'update': [{'id': 4, 'content': 'Quoted id ok.'}],
            'retire': [9],
        })

    def test_non_list_values_ignored(self):
        ops = memory._parse_fact_ops('{"add": "Not a list.", "retire": 3}')
        self.assertEqual(ops, {'add': [], 'update': [], 'retire': []})

    def test_count_is_capped(self):
        raw = json.dumps({'add': [f"Fact {i}." for i in range(20)]})
        self.assertEqual(len(memory._parse_fact_ops(raw)['add']),
                         memory._MAX_LEARNED_PER_EXCHANGE)


class ApplyLearningTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='secret')
        self.conv = Conversation.objects.create(
            user_id=self.user.id, username='leo', mode='learning')
        memory.add_message(self.conv, 'assistant', 'Do you have any pets?')
        memory.add_message(self.conv, 'user', 'A dog named Biscuit.')
        self.url = reverse('ai_lab_chatbot:learn', args=[self.conv.id])
        self.client.login(username='leo', password='secret')

    def _reply(self, mock_complete, add=(), update=(), retire=()):
        mock_complete.return_value = json.dumps({
            'add': list(add), 'update': list(update), 'retire': list(retire),
        })

    @patch('ai_lab_chatbot.mycroft.memory.complete_chat')
    def test_adds_learned_facts_attributed_to_user(self, mock_complete):
        self._reply(mock_complete, add=[
            "Leo's dog is named Biscuit.", 'Leo has a garden.'])
        resp = self.client.post(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()['added']), 2)

        facts = HouseholdFact.objects.filter(source='learned')
        self.assertEqual(facts.count(), 2)
        for fact in facts:
            self.assertTrue(fact.is_current)
            self.assertEqual(fact.source_user_id, self.user.id)
            self.assertEqual(fact.source_username, 'leo')

    @patch('ai_lab_chatbot.mycroft.memory.complete_chat')
    def test_prompt_uses_question_as_context_and_lists_current_facts(
            self, mock_complete):
        HouseholdFact.objects.create(content='The household is in Raleigh.')
        HouseholdFact.objects.create(content='Old fact.', is_current=False)
        self._reply(mock_complete)
        self.client.post(self.url)
        body = mock_complete.call_args.args[0][-1]['content']
        self.assertIn('Mycroft asked: Do you have any pets?', body)
        self.assertIn('answered: A dog named Biscuit.', body)
        self.assertIn('The household is in Raleigh.', body)
        self.assertNotIn('Old fact.', body)

    @patch('ai_lab_chatbot.mycroft.memory.complete_chat')
    def test_duplicates_skipped(self, mock_complete):
        HouseholdFact.objects.create(content="Leo's dog is named Biscuit.")
        # Same fact (casing/whitespace/period) plus a within-batch repeat.
        self._reply(mock_complete, add=[
            "  leo's DOG is named   Biscuit  ",
            'Leo has a garden.', 'leo has a garden'])
        resp = self.client.post(self.url)
        self.assertEqual(resp.json()['added'], ['Leo has a garden.'])
        self.assertEqual(HouseholdFact.objects.count(), 2)

    @patch('ai_lab_chatbot.mycroft.memory.complete_chat')
    def test_update_supersedes_old_fact(self, mock_complete):
        old = HouseholdFact.objects.create(
            content='The household lives at 9 Elm Rd.')
        self._reply(mock_complete, update=[
            {'id': old.id, 'content': 'The household lives at 12 Oak St.'}])
        resp = self.client.post(self.url)

        self.assertEqual(resp.json()['updated'], [{
            'from': 'The household lives at 9 Elm Rd.',
            'to': 'The household lives at 12 Oak St.',
        }])
        old.refresh_from_db()
        new = HouseholdFact.objects.get(content__contains='12 Oak St')
        self.assertFalse(old.is_current)
        self.assertEqual(old.superseded_by, new)
        self.assertIsNotNone(old.superseded_at)
        self.assertTrue(new.is_current)
        self.assertEqual(new.source, 'learned')

    @patch('ai_lab_chatbot.mycroft.memory.complete_chat')
    def test_update_with_same_text_is_noop(self, mock_complete):
        old = HouseholdFact.objects.create(content='Leo has a garden.')
        self._reply(mock_complete, update=[
            {'id': old.id, 'content': 'leo has a garden'}])
        resp = self.client.post(self.url)
        self.assertEqual(resp.json()['updated'], [])
        old.refresh_from_db()
        self.assertTrue(old.is_current)
        self.assertEqual(HouseholdFact.objects.count(), 1)

    @patch('ai_lab_chatbot.mycroft.memory.complete_chat')
    def test_retire_marks_not_current(self, mock_complete):
        fact = HouseholdFact.objects.create(content='Leo has a cat named Tom.')
        self._reply(mock_complete, retire=[fact.id])
        resp = self.client.post(self.url)
        self.assertEqual(resp.json()['retired'], ['Leo has a cat named Tom.'])
        fact.refresh_from_db()
        self.assertFalse(fact.is_current)
        self.assertIsNone(fact.superseded_by)
        self.assertIsNotNone(fact.superseded_at)

    @patch('ai_lab_chatbot.mycroft.memory.complete_chat')
    def test_unknown_and_superseded_ids_ignored(self, mock_complete):
        gone = HouseholdFact.objects.create(content='Old.', is_current=False)
        self._reply(
            mock_complete,
            update=[{'id': 9999, 'content': 'Invented.'},
                    {'id': gone.id, 'content': 'Revived.'}],
            retire=[9999, gone.id],
        )
        resp = self.client.post(self.url)
        self.assertEqual(
            resp.json(), {'added': [], 'updated': [], 'retired': []})
        self.assertEqual(HouseholdFact.objects.count(), 1)

    @patch('ai_lab_chatbot.mycroft.memory.complete_chat')
    def test_garbage_reply_changes_nothing(self, mock_complete):
        mock_complete.return_value = 'I did not find any facts.'
        resp = self.client.post(self.url)
        self.assertEqual(
            resp.json(), {'added': [], 'updated': [], 'retired': []})
        self.assertEqual(HouseholdFact.objects.count(), 0)

    @patch('ai_lab_chatbot.mycroft.memory.complete_chat',
           side_effect=RuntimeError('ollama down'))
    def test_failure_is_swallowed(self, mock_complete):
        with self.assertLogs('ai_lab_chatbot.mycroft.memory', level='ERROR'):
            resp = self.client.post(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json(), {'added': [], 'updated': [], 'retired': []})

    @patch('ai_lab_chatbot.mycroft.memory.complete_chat')
    def test_chat_mode_conversation_never_learns(self, mock_complete):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        memory.add_message(conv, 'user', 'We got a dog named Biscuit.')
        memory.add_message(conv, 'assistant', 'Congrats!')
        resp = self.client.post(
            reverse('ai_lab_chatbot:learn', args=[conv.id]))
        self.assertEqual(
            resp.json(), {'added': [], 'updated': [], 'retired': []})
        mock_complete.assert_not_called()
        self.assertEqual(HouseholdFact.objects.count(), 0)

    @patch('ai_lab_chatbot.mycroft.memory.complete_chat')
    def test_no_user_turn_is_noop(self, mock_complete):
        conv = Conversation.objects.create(
            user_id=self.user.id, username='leo', mode='learning')
        memory.add_message(conv, 'assistant', 'Who lives here?')
        resp = self.client.post(
            reverse('ai_lab_chatbot:learn', args=[conv.id]))
        self.assertEqual(resp.json()['added'], [])
        mock_complete.assert_not_called()

    @patch('ai_lab_chatbot.mycroft.memory.complete_chat')
    def test_other_users_conversation_is_404(self, mock_complete):
        other = User.objects.create_user(username='sam', password='x')
        conv = Conversation.objects.create(
            user_id=other.id, username='sam', mode='learning')
        resp = self.client.post(
            reverse('ai_lab_chatbot:learn', args=[conv.id]))
        self.assertEqual(resp.status_code, 404)
        mock_complete.assert_not_called()


class StartLearningViewTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='secret')
        self.client.login(username='leo', password='secret')
        self.url = reverse('ai_lab_chatbot:learn_new')

    @patch('ai_lab_chatbot.mycroft.memory.complete_chat')
    def test_creates_learning_conversation_with_opener(self, mock_complete):
        mock_complete.return_value = 'Hi! Who lives in your household?'
        resp = self.client.post(self.url)

        conv = Conversation.objects.get(user_id=self.user.id)
        self.assertRedirects(
            resp, reverse('ai_lab_chatbot:conversation', args=[conv.id]),
            fetch_redirect_response=False)
        self.assertEqual(conv.mode, 'learning')
        self.assertEqual(conv.title, memory.LEARNING_TITLE)
        msgs = list(conv.messages.all())
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].role, 'assistant')
        self.assertEqual(msgs[0].content, 'Hi! Who lives in your household?')
        # The opener is generated with the learning-mode system prompt.
        system = mock_complete.call_args.args[0][0]['content']
        self.assertIn('This is a learning conversation', system)

    @patch('ai_lab_chatbot.mycroft.memory.complete_chat',
           side_effect=RuntimeError('ollama down'))
    def test_falls_back_to_canned_opener(self, mock_complete):
        with self.assertLogs('ai_lab_chatbot.mycroft.memory', level='ERROR'):
            self.client.post(self.url)
        conv = Conversation.objects.get(user_id=self.user.id)
        self.assertEqual(
            conv.messages.get().content, memory.FALLBACK_OPENER)

    def test_requires_post(self):
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 405)

    def test_requires_login(self):
        self.client.logout()
        resp = self.client.post(self.url)
        self.assertEqual(resp.status_code, 302)
        self.assertFalse(Conversation.objects.exists())


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


class ListModelsTests(TestCase):
    """The installed-models helper: names extracted, non-chat models dropped by
    Ollama capabilities (with a name-based fallback), sorted, best-effort on
    failure."""

    @staticmethod
    def _resp(*names):
        return SimpleNamespace(
            models=[SimpleNamespace(model=n) for n in names])

    @staticmethod
    def _show(caps):
        """A `.show()` return with the given capabilities list (or None for a
        server that doesn't report them)."""
        return SimpleNamespace(capabilities=caps)

    def _wire(self, mock_client, list_resp, caps_by_name):
        """Point the mocked client at a `.list()` result and a per-name
        `.show()` capabilities map (value None -> no capabilities reported;
        an Exception instance -> `.show()` raises for that model)."""
        mock_client.return_value.list.return_value = list_resp

        def show(name):
            caps = caps_by_name.get(name)
            if isinstance(caps, Exception):
                raise caps
            return self._show(caps)

        mock_client.return_value.show.side_effect = show

    @patch('ai_lab_chatbot.mycroft.client._client')
    def test_keeps_chat_models_drops_embedder_by_capability(self, mock_client):
        # The embedder carries a :latest tag the old exact-match would miss;
        # capabilities catch it regardless.
        self._wire(
            mock_client,
            self._resp('mistral', 'llama3.1:8b', 'nomic-embed-text:latest'),
            {
                'mistral': ['completion', 'tools'],
                'llama3.1:8b': ['completion'],
                'nomic-embed-text:latest': ['embedding'],
            },
        )
        self.assertEqual(list_models(), ['llama3.1:8b', 'mistral'])

    @patch('ai_lab_chatbot.mycroft.client._client')
    def test_falls_back_to_name_when_capabilities_absent(self, mock_client):
        # No capabilities reported -> drop the configured embed model by base
        # name (tag-tolerant), keep the rest.
        embed = settings.OLLAMA_EMBED_MODEL
        self._wire(
            mock_client,
            self._resp('llama3.1:8b', f'{embed}:latest'),
            {'llama3.1:8b': None, f'{embed}:latest': None},
        )
        self.assertEqual(list_models(), ['llama3.1:8b'])

    @patch('ai_lab_chatbot.mycroft.client._client')
    def test_show_raising_falls_back_without_blowing_up(self, mock_client):
        # A `.show()` failure for one model is tolerated (treated as unknown ->
        # name fallback); the rest of the list still resolves.
        embed = settings.OLLAMA_EMBED_MODEL
        self._wire(
            mock_client,
            self._resp('llama3.1:8b', embed),
            {'llama3.1:8b': RuntimeError('boom'), embed: RuntimeError('boom')},
        )
        self.assertEqual(list_models(), ['llama3.1:8b'])

    @patch('ai_lab_chatbot.mycroft.client._client')
    def test_tolerates_dict_entries(self, mock_client):
        self._wire(
            mock_client,
            SimpleNamespace(models=[{'model': 'llama3.1:8b'}, {'name': 'phi3'}]),
            {'llama3.1:8b': ['completion'], 'phi3': ['completion']},
        )
        self.assertEqual(list_models(), ['llama3.1:8b', 'phi3'])

    @patch('ai_lab_chatbot.mycroft.client._client')
    def test_failure_returns_empty(self, mock_client):
        mock_client.return_value.list.side_effect = RuntimeError('down')
        self.assertEqual(list_models(), [])


class ModelPassthroughTests(TestCase):
    """stream_chat / complete_chat send the requested model, or the configured
    default when none is given."""

    @patch('ai_lab_chatbot.mycroft.client._client')
    def test_stream_chat_uses_given_model(self, mock_client):
        mock_client.return_value.chat.return_value = iter([])
        list(stream_chat([], model='mistral'))
        self.assertEqual(
            mock_client.return_value.chat.call_args.kwargs['model'], 'mistral')

    @patch('ai_lab_chatbot.mycroft.client._client')
    def test_stream_chat_defaults_to_setting(self, mock_client):
        mock_client.return_value.chat.return_value = iter([])
        list(stream_chat([]))
        self.assertEqual(
            mock_client.return_value.chat.call_args.kwargs['model'],
            settings.OLLAMA_CHAT_MODEL)

    @patch('ai_lab_chatbot.mycroft.client._client')
    def test_complete_chat_uses_given_model(self, mock_client):
        mock_client.return_value.chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content='hi'))
        complete_chat([], model='mistral')
        self.assertEqual(
            mock_client.return_value.chat.call_args.kwargs['model'], 'mistral')


class MycroftConfigTests(TestCase):
    databases = DBS

    def test_default_model_falls_back_to_setting_when_blank(self):
        self.assertEqual(
            MycroftConfig.default_model(), settings.OLLAMA_CHAT_MODEL)

    def test_default_model_returns_stored_value(self):
        cfg = MycroftConfig.get_solo()
        cfg.default_chat_model = 'mistral'
        cfg.save()
        self.assertEqual(MycroftConfig.default_model(), 'mistral')

    def test_is_singleton(self):
        MycroftConfig.get_solo()
        second = MycroftConfig(default_chat_model='phi3')
        second.save()
        self.assertEqual(MycroftConfig.objects.count(), 1)
        self.assertEqual(MycroftConfig.objects.get().pk, 1)


class ModelSelectorViewTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='secret')
        self.client.login(username='leo', password='secret')
        patcher = patch('ai_lab_chatbot.views.supports_tools', return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch('ai_lab_chatbot.views.list_models',
           return_value=['llama3.1:8b', 'mistral'])
    def test_chat_view_bootstraps_models_and_default(self, _mock):
        resp = self.client.get(reverse('ai_lab_chatbot:chat'))
        boot = resp.context['bootstrap']
        self.assertEqual(boot['selected_model'], settings.OLLAMA_CHAT_MODEL)
        self.assertIn('mistral', boot['models'])
        # The selected model is always present as an option, even if not listed.
        self.assertIn(settings.OLLAMA_CHAT_MODEL, boot['models'])

    @patch('ai_lab_chatbot.views.list_models', return_value=[])
    def test_resume_view_uses_conversation_model(self, _mock):
        conv = Conversation.objects.create(
            user_id=self.user.id, username='leo', model='mistral')
        resp = self.client.get(
            reverse('ai_lab_chatbot:conversation', args=[conv.id]))
        boot = resp.context['bootstrap']
        self.assertEqual(boot['selected_model'], 'mistral')
        self.assertIn('mistral', boot['models'])

    @patch('ai_lab_chatbot.views.list_models', return_value=['mistral'])
    @patch('ai_lab_chatbot.views.stream_chat')
    def test_send_uses_and_persists_chosen_model(self, mock_stream, _mock):
        mock_stream.return_value = iter(['Hi'])
        resp = self.client.post(
            reverse('ai_lab_chatbot:send'),
            data=json.dumps({'conversation_id': None, 'content': 'hello',
                             'model': 'mistral'}),
            content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        b''.join(resp.streaming_content)  # drain the generator (persists turns)

        self.assertEqual(mock_stream.call_args.kwargs['model'], 'mistral')
        conv = Conversation.objects.get(user_id=self.user.id)
        self.assertEqual(conv.model, 'mistral')
        assistant = conv.messages.get(role='assistant')
        self.assertEqual(assistant.model, 'mistral')

    @patch('ai_lab_chatbot.views.list_models', return_value=['llama3.1:8b'])
    @patch('ai_lab_chatbot.views.stream_chat')
    def test_send_without_model_uses_default(self, mock_stream, _mock):
        MycroftConfig.get_solo()  # blank default → setting fallback
        mock_stream.return_value = iter(['Hi'])
        resp = self.client.post(
            reverse('ai_lab_chatbot:send'),
            data=json.dumps({'conversation_id': None, 'content': 'hello'}),
            content_type='application/json')
        b''.join(resp.streaming_content)
        self.assertEqual(
            mock_stream.call_args.kwargs['model'], settings.OLLAMA_CHAT_MODEL)
        conv = Conversation.objects.get(user_id=self.user.id)
        self.assertEqual(conv.model, settings.OLLAMA_CHAT_MODEL)


# --- Phase 4: tool calling ------------------------------------------------

def _forecast_payload():
    """A trimmed Open-Meteo forecast response (US units, timezone=auto)."""
    return {
        'timezone': 'America/New_York',
        'timezone_abbreviation': 'EDT',
        'current': {
            'time': '2026-10-18T14:00', 'temperature_2m': 71.6,
            'apparent_temperature': 70.2, 'relative_humidity_2m': 55,
            'precipitation': 0.0, 'weather_code': 2, 'wind_speed_10m': 8.4,
            'wind_gusts_10m': 15.1, 'is_day': 1,
        },
        'hourly': {
            'time': [f'2026-10-18T{h:02d}:00' for h in range(14, 24)]
                    + ['2026-10-19T00:00', '2026-10-19T01:00'],
            'temperature_2m': [72, 71, 70, 68, 65, 62, 60, 58, 57, 56, 55, 54],
            'precipitation_probability': [0, 0, 5, 10, 20, 40, 60, 60, 30, 10, 5, 0],
            'weather_code': [2, 2, 3, 3, 61, 61, 63, 61, 3, 3, 2, 1],
        },
        'daily': {
            'time': ['2026-10-18', '2026-10-19', '2026-10-20'],
            'weather_code': [61, 0, 3],
            'temperature_2m_max': [74.2, 68.0, 66.5],
            'temperature_2m_min': [54.4, 48.9, 50.0],
            'precipitation_probability_max': [60, 0, 15],
            'sunset': ['2026-10-18T18:42', '2026-10-19T18:41', '2026-10-20T18:40'],
        },
    }


def _response(payload):
    return SimpleNamespace(json=lambda: payload, raise_for_status=lambda: None)


class WeatherTests(SimpleTestCase):
    """The Open-Meteo lookup, with HTTP mocked (no network)."""

    @patch('ai_lab_chatbot.mycroft.weather.requests.get')
    def test_blank_location_uses_raleigh_without_geocoding(self, mock_get):
        mock_get.return_value = _response(_forecast_payload())
        result = weather.get_current_weather('')

        self.assertEqual(mock_get.call_count, 1)  # forecast only
        url, kwargs = mock_get.call_args.args[0], mock_get.call_args.kwargs
        self.assertEqual(url, weather.FORECAST_URL)
        self.assertEqual(kwargs['params']['latitude'], weather.DEFAULT_PLACE.latitude)
        self.assertEqual(kwargs['params']['temperature_unit'], 'fahrenheit')
        self.assertEqual(result['location'], weather.DEFAULT_PLACE.name)

    @patch('ai_lab_chatbot.mycroft.weather.requests.get')
    def test_shapes_forecast(self, mock_get):
        mock_get.return_value = _response(_forecast_payload())
        result = weather.get_current_weather()

        self.assertEqual(result['observed_at'], 'Sunday 2026-10-18 2:00 PM EDT')
        self.assertEqual(result['current'], {
            'conditions': 'Partly cloudy', 'temperature_f': 72,
            'feels_like_f': 70, 'humidity_pct': 55, 'wind_mph': 8,
            'wind_gusts_mph': 15, 'precipitation_in': 0.0, 'daytime': True,
        })
        # Every other hour of the 12-hour outlook.
        self.assertEqual(len(result['next_hours']), 6)
        self.assertEqual(result['next_hours'][0]['time'], 'Sun 2 PM')
        self.assertEqual(result['next_hours'][3]['conditions'], 'Rain')
        self.assertEqual(
            [d['day'] for d in result['forecast']],
            ['today', 'tomorrow', 'Tuesday'])
        today = result['forecast'][0]
        self.assertEqual(
            (today['conditions'], today['high_f'], today['low_f'],
             today['precipitation_chance_pct'], today['sunset']),
            ('Light rain', 74, 54, 60, '6:42 PM'))

    @patch('ai_lab_chatbot.mycroft.weather.requests.get')
    def test_geocode_qualifier_picks_matching_state(self, mock_get):
        mock_get.side_effect = [
            _response({'results': [
                {'name': 'Cary', 'admin1': 'Illinois', 'country': 'United States',
                 'country_code': 'US', 'latitude': 42.2, 'longitude': -88.2},
                {'name': 'Cary', 'admin1': 'North Carolina',
                 'country': 'United States', 'country_code': 'US',
                 'latitude': 35.79, 'longitude': -78.78},
            ]}),
            _response(_forecast_payload()),
        ]
        result = weather.get_current_weather('Cary, NC')

        geo_params = mock_get.call_args_list[0].kwargs['params']
        self.assertEqual(geo_params['name'], 'Cary')
        self.assertEqual(
            mock_get.call_args_list[1].kwargs['params']['latitude'], 35.79)
        self.assertEqual(
            result['location'], 'Cary, North Carolina, United States')

    @patch('ai_lab_chatbot.mycroft.weather.requests.get')
    def test_unknown_place_raises(self, mock_get):
        mock_get.return_value = _response({})  # Open-Meteo omits "results"
        with self.assertRaisesMessage(weather.WeatherError, "Atlantis"):
            weather.get_current_weather('Atlantis')

    @patch('ai_lab_chatbot.mycroft.weather.requests.get',
           side_effect=weather.requests.ConnectionError('offline'))
    def test_network_failure_raises_weather_error(self, mock_get):
        with self.assertRaises(weather.WeatherError):
            weather.get_current_weather()

    @patch('ai_lab_chatbot.mycroft.weather.requests.get')
    def test_malformed_payload_raises_weather_error(self, mock_get):
        mock_get.return_value = _response({'current': {}})
        with self.assertRaises(weather.WeatherError):
            weather.get_current_weather()


class ToolDispatchTests(SimpleTestCase):
    """The registry and `run_tool`, using a stand-in tool."""

    class Boom(Exception):
        pass

    def setUp(self):
        self.calls = []

        def fake(location=None):
            self.calls.append(location)
            if location == 'bad':
                raise self.Boom('No such place.')
            if location == 'crash':
                raise RuntimeError('bug')
            return {'location': location}

        self.tool = tools.Tool(
            name='fake_tool', description='A fake.',
            parameters={'type': 'object',
                        'properties': {'location': {'type': 'string'}}},
            func=fake, prompt_hint='fake_tool: for tests.',
            router_hint='fake_tool: when testing.',
            user_errors=(self.Boom,),
        )
        patcher = patch.dict(tools._REGISTRY, {'fake_tool': self.tool}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_definitions_in_ollama_format(self):
        self.assertEqual(tools.tool_definitions(), [{
            'type': 'function',
            'function': {'name': 'fake_tool', 'description': 'A fake.',
                         'parameters': self.tool.parameters},
        }])

    def test_description_lists_hints(self):
        self.assertIn('- fake_tool: for tests.', tools.describe_registered_tools())

    def test_runs_tool_and_returns_json(self):
        out = tools.run_tool('fake_tool', {'location': 'Cary'})
        self.assertEqual(json.loads(out), {'location': 'Cary'})

    def test_string_arguments_parsed_and_extras_dropped(self):
        out = tools.run_tool('fake_tool', '{"location": "Cary", "units": "metric"}')
        self.assertEqual(json.loads(out), {'location': 'Cary'})

    def test_non_string_value_coerced_for_string_param(self):
        tools.run_tool('fake_tool', {'location': 27601})
        self.assertEqual(self.calls, ['27601'])

    def test_unknown_tool_is_error_result(self):
        out = json.loads(tools.run_tool('nope', {}))
        self.assertIn('nope', out['error'])

    def test_user_error_relayed(self):
        out = json.loads(tools.run_tool('fake_tool', {'location': 'bad'}))
        self.assertEqual(out, {'error': 'No such place.'})

    def test_unexpected_error_is_generic_and_logged(self):
        with self.assertLogs('ai_lab_chatbot.mycroft.tools', level='ERROR'):
            out = json.loads(tools.run_tool('fake_tool', {'location': 'crash'}))
        self.assertIn('failed unexpectedly', out['error'])


class ToolRegistryTests(SimpleTestCase):
    def test_weather_tool_registered(self):
        names = [d['function']['name'] for d in tools.tool_definitions()]
        self.assertIn('get_current_weather', names)
        self.assertIn('get_current_weather', tools.describe_registered_tools())


class ToolPromptTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='x')

    def test_tools_section_only_when_enabled(self):
        self.assertNotIn('get_current_weather', build_system_prompt(self.user))
        self.assertIn(
            'get_current_weather',
            build_system_prompt(self.user, tool_names=['get_current_weather']))


class StreamChatToolCallTests(SimpleTestCase):

    @patch('ai_lab_chatbot.mycroft.client._client')
    def test_tools_passed_and_calls_captured(self, mock_client):
        call = SimpleNamespace(function=SimpleNamespace(
            name='get_current_weather', arguments={'location': 'Cary'}))
        mock_client.return_value.chat.return_value = iter([
            SimpleNamespace(message=SimpleNamespace(content='', tool_calls=[call]),
                            done=True, prompt_eval_count=1, eval_count=1,
                            eval_duration=0, total_duration=0),
        ])
        defs = tools.tool_definitions()
        calls = []
        self.assertEqual(
            list(stream_chat([], tools=defs, tool_calls_out=calls)), [])
        self.assertEqual(mock_client.return_value.chat.call_args.kwargs['tools'], defs)
        self.assertEqual(
            calls, [{'name': 'get_current_weather', 'arguments': {'location': 'Cary'}}])

    @patch('ai_lab_chatbot.mycroft.client._client')
    def test_no_tools_kwarg_when_none(self, mock_client):
        mock_client.return_value.chat.return_value = iter([])
        list(stream_chat([]))
        self.assertNotIn('tools', mock_client.return_value.chat.call_args.kwargs)


class SupportsToolsTests(SimpleTestCase):

    @patch('ai_lab_chatbot.mycroft.client._client')
    def test_reads_capabilities(self, mock_client):
        mock_client.return_value.show.return_value = SimpleNamespace(
            capabilities=['completion', 'tools'])
        self.assertTrue(supports_tools('llama3.1:8b'))
        mock_client.return_value.show.return_value = SimpleNamespace(
            capabilities=['completion'])
        self.assertFalse(supports_tools('gemma'))

    @patch('ai_lab_chatbot.mycroft.client._client')
    def test_unknown_counts_as_supported(self, mock_client):
        mock_client.return_value.show.side_effect = RuntimeError('down')
        self.assertTrue(supports_tools('llama3.1:8b'))


WEATHER_CALL = {'name': 'get_current_weather', 'arguments': {'location': 'Cary, NC'}}
WEATHER_RESULT = json.dumps({'location': 'Cary', 'current': {'temperature_f': 72}})


@patch('ai_lab_chatbot.views.supports_tools', return_value=True)
class ToolLoopViewTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='secret')
        self.client.login(username='leo', password='secret')
        self.seen = []  # (messages snapshot, tools offered) per stream call
        patcher = patch('ai_lab_chatbot.mycroft.tools.select_tools',
                        return_value=['get_current_weather'])
        self.select = patcher.start()
        self.addCleanup(patcher.stop)

    def _script(self, rounds):
        """A stream_chat stand-in replaying `rounds` of (tokens, tool calls)."""
        it = iter(rounds)

        def _side_effect(messages, stats_out=None, model=None, tools=None,
                         tool_calls_out=None):
            self.seen.append(([dict(m) for m in messages], tools))
            tokens, calls = next(it)
            if calls and tool_calls_out is not None:
                tool_calls_out.extend(calls)
            if stats_out is not None:
                stats_out.update({'prompt_tokens': 100, 'completion_tokens': 5,
                                  'eval_duration_ms': 1.0, 'total_duration_ms': 2.0})
            return iter(tokens)
        return _side_effect

    def _send(self, content='Weather in Cary?', conversation_id=None):
        resp = self.client.post(
            reverse('ai_lab_chatbot:send'),
            data=json.dumps({'conversation_id': conversation_id, 'content': content}),
            content_type='application/json')
        body = b''.join(resp.streaming_content).decode()
        return [json.loads(line) for line in body.splitlines() if line.strip()]

    @patch('ai_lab_chatbot.mycroft.tools.run_tool', return_value=WEATHER_RESULT)
    @patch('ai_lab_chatbot.views.stream_chat')
    def test_tool_round_then_answer(self, mock_stream, mock_run, _supports):
        mock_stream.side_effect = self._script([
            ([], [WEATHER_CALL]),
            (['It is ', '72°F in Cary.'], []),
        ])
        frames = self._send()
        types = [f['type'] for f in frames]

        self.assertEqual(
            types, ['tool_call', 'tool_result', 'token', 'token', 'stats', 'done'])
        self.assertEqual(frames[0]['name'], 'get_current_weather')
        self.assertEqual(frames[0]['arguments'], {'location': 'Cary, NC'})
        self.assertEqual(frames[1]['content'], WEATHER_RESULT)
        mock_run.assert_called_once_with('get_current_weather', {'location': 'Cary, NC'})

        # Transcript: user, tool-call turn, tool result, answer.
        conv = Conversation.objects.get(id=frames[-1]['conversation_id'])
        msgs = list(conv.messages.all())
        self.assertEqual([m.role for m in msgs], ['user', 'assistant', 'tool', 'assistant'])
        self.assertEqual(msgs[1].tool_calls, [WEATHER_CALL])
        self.assertEqual(msgs[1].content, '')
        self.assertEqual(msgs[2].tool_name, 'get_current_weather')
        self.assertEqual(msgs[2].content, WEATHER_RESULT)
        self.assertEqual(msgs[3].content, 'It is 72°F in Cary.')
        self.assertEqual(msgs[3].prompt_tokens, 100)

        # Round 1 offers the tools; round 2 sees the call + result.
        first_msgs, first_tools = self.seen[0]
        self.assertTrue(first_tools)
        self.assertIn('get_current_weather', first_msgs[0]['content'])
        second_msgs, _ = self.seen[1]
        self.assertEqual(second_msgs[-2]['tool_calls'], [
            {'function': {'name': 'get_current_weather',
                          'arguments': {'location': 'Cary, NC'}}}])
        self.assertEqual(second_msgs[-1], {
            'role': 'tool', 'content': WEATHER_RESULT,
            'tool_name': 'get_current_weather'})

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_plain_answer_makes_no_tool_turns(self, mock_stream, _supports):
        mock_stream.side_effect = self._script([(['4'], [])])
        frames = self._send('What is 2+2?')
        self.assertNotIn('tool_call', [f['type'] for f in frames])
        conv = Conversation.objects.get(id=frames[-1]['conversation_id'])
        self.assertEqual(
            list(conv.messages.values_list('role', flat=True)), ['user', 'assistant'])
        self.assertEqual(len(self.seen), 1)

    @patch('ai_lab_chatbot.mycroft.tools.run_tool', return_value=WEATHER_RESULT)
    @patch('ai_lab_chatbot.views.stream_chat')
    @override_settings(MYCROFT_TOOL_MAX_ROUNDS=2)
    def test_rounds_are_capped(self, mock_stream, mock_run, _supports):
        mock_stream.side_effect = self._script([
            ([], [WEATHER_CALL]), ([], [WEATHER_CALL]), (['Done.'], []),
        ])
        frames = self._send()
        self.assertEqual(frames[-1]['type'], 'done')
        self.assertEqual(mock_run.call_count, 2)
        # The final round withholds tools so the model has to answer.
        self.assertEqual([bool(t) for _, t in self.seen], [True, True, False])

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_no_tools_for_model_without_support(self, mock_stream, supports):
        supports.return_value = False
        mock_stream.side_effect = self._script([(['Hi'], [])])
        self._send('hello')
        messages, offered = self.seen[0]
        self.assertIsNone(offered)
        self.assertNotIn('get_current_weather', messages[0]['content'])
        self.select.assert_not_called()

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_no_tools_in_learning_chat(self, mock_stream, _supports):
        conv = Conversation.objects.create(
            user_id=self.user.id, username='leo', mode='learning')
        mock_stream.side_effect = self._script([(['Nice!'], [])])
        self._send('We have a dog.', conversation_id=str(conv.id))
        self.assertIsNone(self.seen[0][1])

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_text_with_tool_call_stays_on_call_turn(self, mock_stream, _supports):
        mock_stream.side_effect = self._script([
            (['Let me check.'], [WEATHER_CALL]),
            (['Sunny.'], []),
        ])
        with patch('ai_lab_chatbot.mycroft.tools.run_tool', return_value='{}'):
            frames = self._send()
        conv = Conversation.objects.get(id=frames[-1]['conversation_id'])
        assistants = list(conv.messages.filter(role='assistant')
                          .values_list('content', flat=True))
        self.assertEqual(assistants, ['Let me check.', 'Sunny.'])

    @patch('ai_lab_chatbot.mycroft.tools.run_tool', return_value=WEATHER_RESULT)
    @patch('ai_lab_chatbot.views.stream_chat')
    def test_tool_turns_not_embedded(self, mock_stream, mock_run, _supports):
        mock_stream.side_effect = self._script([
            ([], [WEATHER_CALL]), (['Warm.'], []),
        ])
        with patch('ai_lab_chatbot.mycroft.memory.embed_text',
                   return_value=_vec([(0, 1.0)])):
            frames = self._send()
        conv = Conversation.objects.get(id=frames[-1]['conversation_id'])
        embedded = {
            (m.role, bool(m.tool_calls)): m.embedding is not None
            for m in conv.messages.all()
        }
        self.assertEqual(embedded, {
            ('user', False): True, ('assistant', True): False,
            ('tool', False): False, ('assistant', False): True,
        })

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_router_says_none_offers_no_tools(self, mock_stream, _supports):
        self.select.return_value = []
        mock_stream.side_effect = self._script([(['4'], [])])
        self._send('What is 2+2?')
        messages, offered = self.seen[0]
        self.assertIsNone(offered)
        self.assertNotIn('get_current_weather', messages[0]['content'])

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_router_gets_latest_and_earlier_text_turns(self, mock_stream, _supports):
        conv = Conversation.objects.create(user_id=self.user.id, username='leo')
        memory.add_message(conv, 'user', 'Weather in Cary?')
        memory.add_message(conv, 'assistant', '', tool_calls=[WEATHER_CALL])
        memory.add_message(conv, 'tool', WEATHER_RESULT,
                           tool_name='get_current_weather')
        memory.add_message(conv, 'assistant', 'Sunny.')
        mock_stream.side_effect = self._script([(['Rain.'], [])])
        self._send('And tomorrow?', conversation_id=str(conv.id))

        recent, latest = self.select.call_args.args
        self.assertEqual(latest, 'And tomorrow?')
        # Text turns only: no tool-call or tool-result turns, no current turn.
        self.assertEqual(
            recent, [('user', 'Weather in Cary?'), ('assistant', 'Sunny.')])

    @patch('ai_lab_chatbot.mycroft.tools.run_tool', return_value=WEATHER_RESULT)
    @patch('ai_lab_chatbot.views.stream_chat')
    def test_text_written_call_is_run(self, mock_stream, mock_run, _supports):
        mock_stream.side_effect = self._script([
            (['{"name": "get_current_weather", ',
              '"parameters": {"location": "Cary, NC"}}'], []),
            (['Sunny.'], []),
        ])
        frames = self._send()
        types = [f['type'] for f in frames]
        self.assertEqual(types[:2], ['tool_call', 'tool_result'])
        # The JSON never reached the chat as text.
        tokens = ''.join(f['content'] for f in frames if f['type'] == 'token')
        self.assertEqual(tokens, 'Sunny.')
        mock_run.assert_called_once_with(
            'get_current_weather', {'location': 'Cary, NC'})
        conv = Conversation.objects.get(id=frames[-1]['conversation_id'])
        self.assertEqual(
            list(conv.messages.values_list('role', flat=True)),
            ['user', 'assistant', 'tool', 'assistant'])

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_junk_text_call_discarded_and_retried_without_tools(
            self, mock_stream, _supports):
        mock_stream.side_effect = self._script([
            (['{"name": "None", "parameters": {}}'], []),
            (['4'], []),
        ])
        with self.assertLogs('ai_lab_chatbot.views', level='INFO'):
            frames = self._send('What is 2+2?')
        tokens = ''.join(f['content'] for f in frames if f['type'] == 'token')
        self.assertEqual(tokens, '4')
        self.assertEqual([bool(t) for _, t in self.seen], [True, False])
        conv = Conversation.objects.get(id=frames[-1]['conversation_id'])
        self.assertEqual(
            list(conv.messages.values_list('role', 'content')),
            [('user', 'What is 2+2?'), ('assistant', '4')])

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_held_brace_reply_released_when_not_a_call(self, mock_stream, _supports):
        mock_stream.side_effect = self._script([(['{', 'braces} are fun'], [])])
        frames = self._send()
        tokens = ''.join(f['content'] for f in frames if f['type'] == 'token')
        self.assertEqual(tokens, '{braces} are fun')

    @patch('ai_lab_chatbot.views.stream_chat')
    def test_plain_reply_streams_when_tools_offered(self, mock_stream, _supports):
        mock_stream.side_effect = self._script([(['It ', 'is ', 'sunny.'], [])])
        frames = self._send()
        tokens = [f['content'] for f in frames if f['type'] == 'token']
        # Released on the first non-JSON piece, then streamed piece by piece.
        self.assertEqual(tokens, ['It ', 'is ', 'sunny.'])


class SelectToolsTests(SimpleTestCase):
    """The per-message tool router, with the completion mocked."""

    @patch('ai_lab_chatbot.mycroft.tools.complete_chat',
           return_value='["get_current_weather"]')
    def test_picks_named_tool_deterministically(self, mock_complete):
        self.assertEqual(
            tools.select_tools([], 'Jacket tonight?', model='llama3.1:8b'),
            ['get_current_weather'])
        kwargs = mock_complete.call_args.kwargs
        self.assertEqual(kwargs['model'], 'llama3.1:8b')
        self.assertEqual(kwargs['options'], {'temperature': 0})
        system = mock_complete.call_args.args[0][0]['content']
        self.assertIn('get_current_weather:', system)

    @patch('ai_lab_chatbot.mycroft.tools.complete_chat', return_value='[]')
    def test_empty_array_is_no_tools(self, mock_complete):
        self.assertEqual(tools.select_tools([], 'What is 2+2?'), [])

    @patch('ai_lab_chatbot.mycroft.tools.complete_chat',
           return_value='["get_current_weather"]')
    def test_decision_logged(self, mock_complete):
        with self.assertLogs('ai_lab_chatbot.mycroft.tools', level='INFO') as cm:
            tools.select_tools([], 'Give me 10 more')
        [line] = cm.output
        self.assertIn("['get_current_weather']", line)
        self.assertIn("'Give me 10 more'", line)

    @patch('ai_lab_chatbot.mycroft.tools.complete_chat', return_value='[]')
    def test_prompt_covers_bare_follow_ups(self, mock_complete):
        tools.select_tools([], 'x')
        system = mock_complete.call_args.args[0][0]['content']
        self.assertIn('"Give me 10 more" -> []', system)

    @patch('ai_lab_chatbot.mycroft.tools.complete_chat')
    def test_reply_parsing(self, mock_complete):
        cases = {
            'Sure: ["get_current_weather", "get_current_weather"]':
                ['get_current_weather'],
            '["launch_rockets"]': [],
            'You need get_current_weather for that.': ['get_current_weather'],
            'No tools needed.': [],
            '[not json': [],
        }
        for raw, expected in cases.items():
            mock_complete.return_value = raw
            self.assertEqual(tools.select_tools([], 'x'), expected, raw)

    @patch('ai_lab_chatbot.mycroft.tools.complete_chat')
    def test_only_recent_turns_sent(self, mock_complete):
        mock_complete.return_value = '[]'
        recent = [('user', f'turn {i}') for i in range(10)]
        tools.select_tools(recent, 'And tomorrow?')
        body = mock_complete.call_args.args[0][-1]['content']
        self.assertNotIn('turn 5', body)
        self.assertIn('turn 6', body)
        self.assertIn('Latest message: And tomorrow?', body)

    @patch('ai_lab_chatbot.mycroft.tools.complete_chat',
           side_effect=RuntimeError('ollama down'))
    def test_failure_means_no_tools(self, mock_complete):
        with self.assertLogs('ai_lab_chatbot.mycroft.tools', level='ERROR'):
            self.assertEqual(tools.select_tools([], 'Weather?'), [])


class ParseTextToolCallTests(SimpleTestCase):

    def test_parses_call_shapes(self):
        self.assertEqual(
            tools.parse_text_tool_call(
                '{"name": "get_current_weather", "parameters": {"location": "Cary"}}'),
            {'name': 'get_current_weather', 'arguments': {'location': 'Cary'}})
        self.assertEqual(
            tools.parse_text_tool_call(
                '```json\n{"name": "None", "arguments": {}}\n```'),
            {'name': 'None', 'arguments': {}})

    def test_rejects_non_calls(self):
        for text in ['4', '{"answer": 4}', '{"name": "x", "extra": 1}',
                     '{not json}', 'The answer is {"name": "x"}']:
            self.assertIsNone(tools.parse_text_tool_call(text), text)


class CompleteChatOptionsTests(SimpleTestCase):

    @patch('ai_lab_chatbot.mycroft.client._client')
    def test_num_ctx_matches_stream_and_options_merge(self, mock_client):
        mock_client.return_value.chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content=' ok '))
        self.assertEqual(complete_chat([], options={'temperature': 0}), 'ok')
        self.assertEqual(
            mock_client.return_value.chat.call_args.kwargs['options'],
            {'num_ctx': settings.MYCROFT_NUM_CTX, 'temperature': 0})


class ToolHistoryTests(TestCase):
    databases = DBS

    def setUp(self):
        self.user = User.objects.create_user(username='leo', password='secret')
        self.conv = Conversation.objects.create(user_id=self.user.id, username='leo')

    def _tool_exchange(self):
        memory.add_message(self.conv, 'user', 'Weather?')
        memory.add_message(self.conv, 'assistant', '', tool_calls=[WEATHER_CALL])
        memory.add_message(self.conv, 'tool', WEATHER_RESULT,
                           tool_name='get_current_weather')
        memory.add_message(self.conv, 'assistant', 'Warm.')

    def test_history_replays_tool_turns(self):
        self._tool_exchange()
        self.assertEqual(memory.history_for_prompt(self.conv), [
            {'role': 'user', 'content': 'Weather?'},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'function': {'name': 'get_current_weather',
                              'arguments': {'location': 'Cary, NC'}}}]},
            {'role': 'tool', 'content': WEATHER_RESULT,
             'tool_name': 'get_current_weather'},
            {'role': 'assistant', 'content': 'Warm.'},
        ])

    def test_window_never_starts_with_orphan_tool_result(self):
        self._tool_exchange()
        window = memory.history_for_prompt(self.conv, limit=2)
        self.assertEqual(window, [{'role': 'assistant', 'content': 'Warm.'}])

    def test_resume_bootstrap_includes_tool_fields(self):
        self._tool_exchange()
        self.client.login(username='leo', password='secret')
        resp = self.client.get(
            reverse('ai_lab_chatbot:conversation', args=[self.conv.id]))
        msgs = resp.context['bootstrap']['messages']
        self.assertEqual(msgs[1]['tool_calls'], [WEATHER_CALL])
        self.assertEqual(msgs[2]['tool_name'], 'get_current_weather')

    @patch('ai_lab_chatbot.views.complete_chat', return_value='Weather check')
    def test_title_skips_tool_call_turn(self, mock_complete):
        self._tool_exchange()
        self.client.login(username='leo', password='secret')
        self.client.post(reverse('ai_lab_chatbot:title', args=[self.conv.id]))
        exchange = mock_complete.call_args.args[0][-1]['content']
        self.assertIn('Assistant: Warm.', exchange)
