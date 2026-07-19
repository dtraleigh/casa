# Mycroft — Chatbot Planning Doc

**Project 1 of the AI Lab initiative.** Foundation (Ollama, pgvector, Django integration) is complete — see `ai-lab-plan.md` for infrastructure context.

## What Mycroft is

Mycroft is a locally-hosted, named chatbot with progressively expanding capabilities. It runs as a Django app inside the existing `casa` project, uses Ollama for inference, and stores conversation history and knowledge in the `ai_lab` Postgres database.

## Design principles

- **Iterative delivery.** Each phase produces something usable. No "big bang" launches.
- **Everything local by default.** External API calls only when a capability genuinely requires live data (weather, sports scores). AI inference stays on emo-server.
- **Conversational, not agentic.** Mycroft answers questions from his own knowledge (general facts, explanations, conversation) and calls tools for specific real-time data he can't know (weather, etc.). He is not designed for long-running research tasks, autonomous multi-step workflows, or agentic behavior. If it takes 20 minutes of thinking, it's out of scope. If it's a natural chat exchange or a bounded tool call, it's in scope.
- **Own the plumbing.** No LangChain, LlamaIndex, or agent frameworks. Direct Django + Ollama + Postgres. Adopt higher-level tools only when a concrete pain point demands them.
- **Transcript-visible.** Every conversation is stored in Postgres and reviewable. No black-box "the AI decided" — you can always see what was sent, what came back, and why.

## Architecture at a glance

```
Browser
  │
  ▼  HTTP
Django (casa) — new app: ai_lab_chatbot
  │
  ├──▶ Ollama (localhost:11434) — inference + embeddings
  │
  └──▶ Postgres (ai_lab database) — conversations, knowledge, embeddings
```

Every request flow is: browser sends message → Django view assembles context (system prompt + history + optional retrieved knowledge + tool definitions) → Ollama generates response → Django stores both messages → returns to browser.

## Django app layout

New app: `ai_lab_chatbot` (prefix keeps the router happy)

```
ai_lab_chatbot/
├── migrations/
├── templates/ai_lab_chatbot/
│   ├── chat.html
│   └── _message.html
├── models.py          # Personality, Conversation, Message, (later) Knowledge
├── admin.py           # Personality admin registration
├── views.py           # chat page, send_message endpoint
├── urls.py            # /mycroft/ prefix
├── mycroft/
│   ├── __init__.py
│   ├── client.py      # Ollama wrapper for Mycroft
│   ├── prompts.py     # assembles system prompt from active Personality + tools
│   ├── tools.py       # tool definitions and dispatch (Phase 3+)
│   └── memory.py      # history assembly, retrieval (Phase 4+)
└── apps.py
```

The `mycroft/` submodule inside the app keeps the "chatbot brain" logic separate from Django's HTTP layer. Views should be thin — they handle web concerns; `mycroft/` handles chatbot concerns. This separation pays off when you eventually add a voice interface (Phase 5) that calls the same `mycroft/` code from outside a web request.

## URL layout

Mounted at `/mycroft/` under casa's existing domain:

- `GET  /mycroft/` — chat interface (main page)
- `POST /mycroft/send/` — send a message, get response
- `GET  /mycroft/history/` — list past conversations (Phase 2)
- `GET  /mycroft/conversation/<uuid>/` — view a specific past conversation (Phase 2)

## Phase 1: Basic chat with configurable personality

> **Status: ✅ Complete — 2026-07-19.** Shipped as the `ai_lab_chatbot` app. All 12
> tests pass and the Definition of Done was walked end-to-end (streaming, personality
> switching, household facts, per-user context isolation). See **As-built notes**
> below for where the implementation deviated from this plan.

**Goal:** A working web page where you can talk to Mycroft. He has a name, a personality driven by an admin-editable model, and responds coherently. No memory between page loads yet.

**What ships:**

- `/mycroft/` page with a simple chat interface: message list, input box, send button
- POST endpoint that sends the current message + assembled system prompt to Ollama and returns the response
- **Personality model** editable via Django admin, with exactly one active at a time
- **HouseholdFact model** — shared knowledge, admin-curated (auto-learning deferred to Phase 4)
- **UserContext model** — per-user context, auto-created empty on first use
- Seed migration creating an initial "Mycroft v1" personality, starter HouseholdFacts, and a UserContext for `leo`
- System prompt assembled at request time from active Personality + all HouseholdFacts + current user's UserContext
- Each page load starts a fresh conversation (no persistence yet)
- Streaming response tokens to the browser so replies appear as they generate (feels much more alive than waiting for the whole response)
- Auth handled by casa's existing login system — `/mycroft/` is behind login

### The three data models

Personality, HouseholdFact, and UserContext together form the "what does Mycroft know" layer.

**Personality** — Mycroft's voice and rules. One row active at a time. Same for everyone.

```python
class Personality(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(
        help_text="Personality and voice — how Mycroft speaks."
    )
    instructions = models.TextField(
        help_text="Rules and considerations for every response."
    )
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        if self.is_active:
            Personality.objects.exclude(pk=self.pk).update(is_active=False)
        super().save(*args, **kwargs)

    @classmethod
    def get_active(cls):
        return cls.objects.filter(is_active=True).first()
```

**HouseholdFact** — shared knowledge about the household. Every fact visible to every user's conversations. Phase 1 is admin-curated only; Phase 4 adds auto-learning from conversations.

```python
class HouseholdFact(models.Model):
    """Facts about the household, shared across all users' conversations."""
    content = models.TextField(help_text="A single fact about the household.")
    source = models.CharField(
        max_length=20,
        choices=[('admin', 'Admin-curated'), ('learned', 'Learned from conversation')],
        default='admin'
    )
    source_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        help_text="For learned facts: who was talking when this was extracted."
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.content[:80]
```

**UserContext** — per-user context, one row per Django user. What Mycroft knows about the person he's currently talking to.

```python
class UserContext(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='mycroft_context'
    )
    content = models.TextField(
        blank=True,
        help_text="What Mycroft should know about this user specifically."
    )
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Mycroft's context for {self.user.username}"

    @classmethod
    def for_user(cls, user):
        """Get or create-empty; never returns None."""
        obj, _ = cls.objects.get_or_create(user=user)
        return obj
```

**Design notes:**

- **One HouseholdFact per row, not a text blob.** Individual facts are easier to add, edit, and delete. Also makes future auto-learning (Phase 4) much cleaner — each learned fact becomes its own row with source attribution.
- **`source_user` on HouseholdFact** is unused in Phase 1 but present in the schema so no migration is needed when Phase 4 arrives.
- **`UserContext.for_user()` classmethod** guarantees the prompt assembly code never has to null-check — every user has *a* context, even if empty.

### System prompt assembly

The final prompt sent to Ollama is not stored anywhere — it's assembled fresh each request from:

1. **Personality content** (from the active `Personality` row): description + instructions
2. **Household knowledge** (all HouseholdFact rows — shared across users)
3. **User context** (the UserContext for the requesting user)
4. **Capability declarations** (auto-generated from registered tools — empty in Phase 1, populated in Phase 3+)
5. **Behavioral guardrails** (hardcoded rules that always apply regardless of personality — e.g. "never claim to have capabilities you don't have")

Editing personality or context via admin can never accidentally break tool behavior, because tool descriptions come from code. Adding a new tool automatically updates what Mycroft claims he can do.

`mycroft/prompts.py` has one function, roughly:

```python
def build_system_prompt(user) -> str:
    personality = Personality.get_active()
    household_facts = HouseholdFact.objects.all()  # small table, load all
    user_context = UserContext.for_user(user)
    tool_descriptions = describe_registered_tools()  # empty in Phase 1
    guardrails = STANDARD_GUARDRAILS
    return assemble(personality, household_facts, user_context, tool_descriptions, guardrails)
```

The assembled prompt has a rough structure like:

```
[personality description]

[personality instructions]

About the household:
- [fact 1]
- [fact 2]
...

About the current user:
[user context content]

[tool descriptions]

[guardrails]
```

Empty sections get omitted rather than shown as "None" — cleaner prompt.

### Initial seed data (draft for review)

Seeded via data migration:

**Personality "Mycroft v1":**

- **description:** "You are Mycroft, a locally-hosted assistant running on Leo's home server. You are intelligent, dry, and economical with words. You skip corporate-assistant phrases like 'Great question!' or 'I'd be happy to help!' and just help. You have a subtle sense of humor but don't force it. You are conversational, not formal."
- **instructions:** "Answer general questions from your own knowledge — facts, explanations, discussion, opinions when asked. Keep responses concise unless asked to elaborate. Prefer prose over bullet lists in conversation. If a question requires real-time information you don't have (current weather, live sports scores, recent news), acknowledge that plainly rather than guessing. You are a conversational assistant, not a research agent — don't attempt long multi-step research tasks."

**HouseholdFacts (starter set — admin-editable afterward):**

- "The household is based in Raleigh, NC."
- (Additional facts you want Mycroft to know from day one — add as many as makes sense before launch)

**UserContext for `leo`:**

- "Leo runs an urban planning blog focused on Raleigh, NC, serves on the local transit board (RTA), and is an urbanism advocate. He works with Django, runs a home server, and is technically fluent. Reference this context only when directly relevant; don't force it."

Other users get empty UserContexts on first login (via `UserContext.for_user()`), fillable via admin.

### Technical scope

- One Django view for the page, one for the send endpoint
- Ollama client wrapper in `mycroft/client.py`
- Prompt assembly in `mycroft/prompts.py`
- Client-side JavaScript for the chat UX (vanilla JS, no framework)
- Server-Sent Events (SSE) for streaming
- Basic HTML/CSS — functional, not pretty. Matches casa's existing style.
- Django admin registration for Personality model
- Data migration creating the initial "Mycroft v1" personality with `is_active=True`

### Model choice

`llama3.1:8b` — already installed. Strong tool calling support for Phase 3.

### What is deliberately NOT in Phase 1

- Conversation persistence
- Memory across page loads
- Tools (weather, etc.)
- Multi-conversation UI
- Auth (using casa's — no new work)
- Fancy UI

### Definition of done

- Navigate to `/mycroft/` in a browser (after logging into casa)
- Send a message, watch response stream in
- Follow-up messages within the same page load feel coherent (browser sends full history each turn)
- Refresh the page and everything resets — expected for Phase 1
- Edit the active Personality's `description` in admin, save, refresh chat, verify Mycroft's voice changed
- Create a second Personality, mark it active, verify the first was automatically deactivated
- Add a HouseholdFact via admin ("Leo's wife is [name]"). Chat as `leo` and verify Mycroft references it when relevant
- Log in as a different user, chat, and verify:
  - Mycroft still knows the household facts (shared)
  - Mycroft does *not* have leo's UserContext (isolated)
  - An empty UserContext row was created for the new user (check admin)

### Phase 1 decisions (resolved)

- **Personality voice** — "Mycroft v1" draft above is approved.
- **Personality edit propagation** — changes take effect on the next message. No reload action needed; `build_system_prompt` fetches the active Personality fresh each request.
- **Streaming** — ~~Server-Sent Events (SSE)~~ **NDJSON over `fetch`** (see As-built notes; SSE is GET-only and we need a POST body).
- **UI aesthetic** — match casa's existing style.
- **Auth** — casa's existing login system.
- **CSRF** — Django's standard protection; works fine with `fetch` + the `X-CSRFToken` header.

### As-built notes (deviations from this plan)

What actually shipped differs from the sketch above in a few deliberate places.
Recorded here so Phase 2+ builds on reality, not the original guess.

- **Streaming transport is NDJSON, not SSE.** `POST /mycroft/send/` returns a
  `StreamingHttpResponse` of newline-delimited JSON frames (`{"type": "token"|"error"
  |"done", ...}`), read in the browser via `fetch()` + a `ReadableStream` reader.
  SSE (`EventSource`) was dropped because it's GET-only and we need a POST body (CSRF
  + the full history as JSON). The framing also exists for a correctness reason: a
  gunicorn worker killed mid-stream writes its own raw HTTP 500 onto the connection,
  and unframed text let that error page leak into the chat as if Mycroft had said it.
  Framing lets the client discard anything that isn't a frame it recognises, and a
  missing terminal `done` frame is how it detects truncation. See
  `docs/server-config.md` → "Streaming responses: the timeout trap."

- **No cross-database foreign keys — user references are decoupled integers.**
  `auth.User` lives in the `default` DB; these models live in `ai_lab`, and Postgres
  can't enforce an FK across databases (the `AiLabRouter` also blocks the relation).
  So the model sketches above that show `ForeignKey(AUTH_USER_MODEL)` were **not**
  implemented as written:
  - `HouseholdFact.source_user` (FK) → `source_user_id` (`IntegerField`, null/blank)
    + `source_username` (`CharField`). Unused in Phase 1; present for Phase 4.
  - `UserContext` keys on `user_id` (`IntegerField`, `unique=True`) + a convenience
    `username`, rather than a user FK. `for_user()` does get-or-create by `user_id`.
  Every model still has its own normal auto-increment `id` PK; `user_id` is just a
  hand-maintained cross-DB reference (an FK in all but the DB-enforced constraint).

- **Deployment needed timeout + buffering changes for streaming.** A long streamed
  response outlives gunicorn's default 30s worker timeout and gets SIGKILLed. Fixed
  with `--timeout 300 --graceful-timeout 300` on `gunicorn-casa` and a dedicated nginx
  `location /mycroft/send/` (`proxy_buffering off`, `proxy_read_timeout 300s`), plus
  `X-Accel-Buffering: no` on the response. Documented in `docs/server-config.md`.

- **UI went a little past "simple."** Beyond the planned message-list/input/send: the
  input is an auto-growing `<textarea>` (Enter sends, Shift+Enter newlines), the chat
  card fills the viewport height (measured, not a magic number), and a header **Width**
  selector (Narrow/Medium/Wide/Full) is remembered per browser via `localStorage`.
  Still vanilla JS + Bootstrap, no framework — consistent with the plan's intent.

- **New docs.** `docs/server-config.md` captures the emo-server deployment (service
  names, request path, the streaming timeout trap, the admin-static permission fix).

---

## Phase 2: Persistent conversations

**Goal:** Conversations survive page reloads. You can start a new conversation, resume an old one, and browse history.

**What ships:**

- `Conversation` and `Message` models in `ai_lab` database
- Auto-save every user message and assistant response
- URL `/mycroft/conversation/<uuid>/` to resume a specific conversation
- URL `/mycroft/history/` listing recent conversations with a snippet
- "New conversation" button
- Conversation titles auto-generated by asking Mycroft to summarize the first exchange in a few words

**Data model (draft):**

```python
class Conversation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Message(models.Model):
    id = models.BigAutoField(primary_key=True)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=20)  # 'user', 'assistant', 'system', 'tool'
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
```

Deliberately no embedding column yet — that comes in Phase 4 when it's needed.

**Sliding window:** send only the last N messages (say 20) to Ollama, plus the system prompt. Simple, effective. Context management comes in Phase 4.

**Definition of done:**

- Start a conversation, close the browser, come back a day later, resume it
- History page shows recent conversations sorted by most-recent-activity
- Titles are auto-generated and reasonably descriptive

---

## Phase 3: Tool calling — weather integration

**Goal:** Mycroft can answer questions about current weather for Raleigh (and other locations if asked). This is the first "capability" — the pattern all future capabilities will follow.

**What ships:**

- Tool definition schema for `get_current_weather(location)`
- Tool dispatch layer in `mycroft/tools.py`
- Weather API integration (Open-Meteo — free, no key required, respects the "as offline as possible" principle since the tool call is minimal and only fires when needed)
- Multi-turn tool workflow: model requests tool → Django calls API → result goes back to model → model synthesizes natural response
- Tool call and result stored as messages in the conversation (with `role='tool'`) — full audit trail

**Architecture note:** Ollama's `/api/chat` endpoint has native tool-calling support with Llama 3.1. The model returns a `tool_calls` array when it wants to invoke a function; Django's job is to see that, call the actual function, and send the result back with `role='tool'` in the next turn. Loop until the model returns a plain response.

**System prompt updates:** Add "You can check current weather using the get_current_weather tool. Use it when Leo asks about weather; don't guess." Update the "capabilities are limited" language to reflect what's now possible.

**Tools designed for later expansion:** the tool dispatch pattern is generic. Adding future tools (calendar lookup, blog search, home network status) is "define the tool schema, write the function, register it" — not a rewrite.

**Definition of done:**

- Ask "what's the weather in Raleigh?" → Mycroft returns actual current conditions
- Ask "should I bring a jacket to dinner tonight?" → Mycroft calls weather tool and reasons about it
- Ask "what's 2+2?" → Mycroft answers directly without calling a tool
- Conversation transcript shows the tool call and result as visible messages

**Open questions for Phase 3:**

- Weather provider — Open-Meteo (free, no key, good enough) vs. something you already use? I'm defaulting to Open-Meteo.
- Location default: assume Raleigh when not specified? Recommendation: yes, since that's where you are 95% of the time.

---

## Phase 4: Semantic memory + knowledge base

**Goal:** Mycroft can recall things you told him weeks ago, and can be pre-loaded with facts about you and your world.

**What ships:**

- `embedding` column added to `Message` model (via `pgvector.django.VectorField`)
- Every message embedded with `nomic-embed-text` at write time
- New `Knowledge` model for curated facts (not conversation-derived)
- Retrieval layer in `mycroft/memory.py`: given a user message, find top-N semantically relevant historical messages and knowledge entries
- Retrieved context injected into the system prompt for that turn
- Admin interface for curating the Knowledge table (Django admin, since it's free)
- **Auto-learning of HouseholdFacts** — background task runs after each conversation exchange, evaluates whether a durable household fact was surfaced, and creates a `HouseholdFact` row with `source='learned'` and `source_user` set. Admin reviews learned facts and can approve, edit, or delete them.

### Auto-learning design notes

This is the hard part of Phase 4 and deserves its own scoping when we get there. Key concerns:

- **Extraction prompt quality.** How reliably can Mycroft identify "durable household facts" vs. transient information?
- **Deduplication.** If the same fact gets learned twice with slightly different wording, we need to consolidate.
- **Contradiction handling.** New fact contradicts an existing one — which wins? Manual review probably right for now.
- **Privacy.** A user might mention something in confidence; Mycroft shouldn't promote every offhand comment to shared knowledge. Conservative extraction — only clear, obviously-durable facts.
- **Auditability.** Every learned fact keeps its `source_user` attribution so you can see who was talking when it was extracted.

**Data model additions:**

```python
class Message(models.Model):
    # ... existing fields ...
    embedding = VectorField(dimensions=768, null=True)

class Knowledge(models.Model):
    id = models.BigAutoField(primary_key=True)
    topic = models.CharField(max_length=200)
    content = models.TextField()
    embedding = VectorField(dimensions=768)
    created_at = models.DateTimeField(auto_now_add=True)
```

**Retrieval strategy:**

- Last N messages by time (short-term working memory — same as Phase 2)
- Plus top K semantically similar older messages (long-term recall)
- Plus top K semantically similar Knowledge entries (curated facts)
- All combined into the context sent to Ollama

**Prompt structure becomes:**

```
[system prompt]

[relevant knowledge]
- Fact about Leo
- Fact about Raleigh transit
...

[relevant past conversation]
- Earlier exchange about X
...

[recent conversation]
- last 15 messages
```

**Indexing:** Skip for now — pgvector's exhaustive search handles thousands of vectors easily. Revisit at 50K+.

**Definition of done:**

- Have a long conversation about a topic. Come back weeks later, mention the topic obliquely, and Mycroft brings up relevant details from the earlier exchange.
- Add a knowledge entry via admin ("Leo drives a 2020 Subaru Outback"). Ask "what car should I look at?" and see if it references the Outback context appropriately.

**Open questions for Phase 4:**

- Do we want to embed *every* message, or only user messages? (Every message is more complete; user messages only saves ~half the embedding calls and is often enough.) Recommendation: embed everything for full coverage.
- How aggressive should recall be? Too much retrieved context and Mycroft over-references old stuff; too little and he feels forgetful. Tunable.

---

## Phase 5: Voice I/O

**Goal:** Talk to Mycroft with your voice; hear him respond. Hardware experiment — this is the phase where things get physically fun.

**What ships:**

- Speech-to-text: Whisper (via `whisper.cpp` or `faster-whisper`) running locally
- Text-to-speech: Piper — fast, local, decent voice quality
- Web audio capture from the browser microphone
- Streaming audio back for playback
- Hardware exploration: raspi-based dedicated Mycroft device? Or just laptop mic + speakers as MVP?

**Deliberately vague at this point** — voice hardware needs its own scoping session, and the software layer depends on what hardware direction you go. Getting through Phases 1-4 informs those choices.

**Open questions for Phase 5 (revisit later):**

- Dedicated hardware device (Pi Zero 2 W + mic hat + small speaker) or browser-based?
- Wake word ("Hey Mycroft") or push-to-talk?
- One voice profile or should Mycroft have different voices for different contexts?

---

## Future phases (not scoped yet)

Ideas to hold for later:

- **More tools:** blog search over dtraleigh.com, calendar integration, home network status queries, controlling Wemo/Home Assistant devices
- **Multi-user support:** if others (family) should be able to use Mycroft with separate memory
- **Proactive Mycroft:** background jobs that surface information ("your daily agenda includes...")
- **RTA-related tools:** committee schedule lookup, R-Line data queries — direct crossover with existing civic work

---

## Model choices — current

| Purpose | Model | Notes |
|---------|-------|-------|
| Conversation | `llama3.1:8b` | Already installed. Strong tool calling. Reevaluate if quality issues emerge. |
| Embeddings | `nomic-embed-text` | Already installed. 768 dims, cosine or L2. |

Both stay resident with `OLLAMA_KEEP_ALIVE=24h` once Phase 4 ships (so retrieval doesn't pay the load cost per query). Decision deferred until Phase 4.

## Open architectural questions (later phases)

- **Static vs. dynamic tool registration** — hard-coded in Phase 3, tool registry pattern later if needed.
- **Auto-learning extraction quality** — Phase 4 concern; will need its own scoping session.
- **Voice hardware direction** — Phase 5; depends on what we learn in Phases 1-4.

## Next actions

Phase 1 is scoped and ready to build. Rough implementation order:

1. Create the `ai_lab_chatbot` app (`python manage.py startapp ai_lab_chatbot`), add to `INSTALLED_APPS`
2. Build the three models (Personality, HouseholdFact, UserContext) and register in admin
3. Data migration to seed Mycroft v1, starter HouseholdFacts, and leo's UserContext
4. `mycroft/prompts.py` — `build_system_prompt(user)` function
5. `mycroft/client.py` — Ollama client wrapper with streaming support
6. Views: chat page + SSE streaming send endpoint
7. Template + minimal JS for the chat interface
8. URL routing under `/mycroft/`
9. Work through the Definition of Done checklist

Estimated effort: a focused evening or two.
