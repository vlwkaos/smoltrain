# smolroute — Specification

## Overview

smolroute is a lightweight routing decision engine that sits between a caller and multiple LLM/tool backends. It receives a text request, classifies it using one or more configurable methods, and returns a routing decision telling the caller which backend to use.

smolroute does NOT forward requests. It only classifies and decides. The caller is responsible for sending the request to the returned backend.

```
caller → smolroute → routing decision (backend name)
                           ↓
caller → backend[decision] → response
```

---

## What smolroute is NOT

- Not a proxy (does not forward requests or responses)
- Not an LLM (does not generate text)
- Not a load balancer (fallback chains are for availability, not distribution)
- Not a rate limiter or auth layer

---

## Routing Methods

smolroute supports six primary routing methods and three heuristic methods. All methods are composable and can be combined in a pipeline. Within the pipeline, the first method that produces a match wins.

### 1. smoltrain

Uses a trained ONNX classifier served by `smoltrain serve` over a Unix socket. This is the highest-accuracy method and the primary use case. Adds approximately 21ms overhead per request.

The classifier returns a label (e.g. "code", "edit", "chat") which maps to a route.

Config:

```yaml
- method: smoltrain
  socket: /tmp/smoltrain.sock
  task: routing
  timeout_ms: 100         # abort and fall through if no response
  label_map:              # optional: remap classifier labels to route labels
    code_generation: code
    text_editing: edit
    question_answer: chat
```

### 2. regex

A list of regex patterns. The first pattern to match the input text wins. Zero latency, fully deterministic. Best for known exact patterns (URLs, file paths, command syntax, etc.).

Patterns are matched against the full input text. Flags (case-insensitive, multiline, etc.) are configurable per pattern.

Config:

```yaml
- method: regex
  rules:
    - pattern: "(?i)^(fix|debug|error|traceback|exception)"
      label: code
    - pattern: "(?i)\\b(translate|in (spanish|french|german|japanese|chinese))\\b"
      label: translate
    - pattern: "(?i)^(summarize|tldr|summary of)"
      label: research
    - pattern: "https?://\\S+"
      label: research
    - pattern: "```[\\s\\S]*```"
      label: code
```

### 3. length

Routes by character or token count. Useful for directing short conversational requests to fast cheap models and long detailed requests to capable models. Thresholds are configurable.

Token counting uses a simple whitespace-split estimate by default. Optionally integrates with tiktoken if installed.

Config:

```yaml
- method: length
  unit: chars             # chars | tokens (whitespace-split estimate)
  rules:
    - max: 80
      label: chat         # very short → conversational model
    - max: 500
      label: edit         # medium → capable but cheap model
    - min: 501
      label: research     # long → most capable model
```

Rules are evaluated in order. The first rule where the input length satisfies the condition wins. A rule may use `min`, `max`, or both to define a range.

### 4. keyword

Checks for presence of words from a configured list. Cheaper than regex — uses simple word boundary matching or substring search. Good for domain vocabulary that clearly indicates intent.

Config:

```yaml
- method: keyword
  case_sensitive: false
  match: any              # any | all (all = AND logic for multi-keyword rules)
  rules:
    - words: [refactor, rewrite, rename, extract function, clean up]
      label: code
    - words: [summarize, summary, overview, brief, tldr]
      label: research
    - words: [translate, translation, in english, en français]
      label: translate
    - words: [schedule, reminder, calendar, meeting, appointment]
      label: tools
```

### 5. prefix

Routes based on a command prefix at the start of the message. Zero latency, no computation. Gives users explicit control over routing by typing a command like `/code` or `/search`.

The prefix is stripped from the text before forwarding (configurable).

Config:

```yaml
- method: prefix
  strip_prefix: true      # remove the prefix from text before routing decision output
  rules:
    - prefix: "/code"
      label: code
    - prefix: "/search"
      label: research
    - prefix: "/edit"
      label: edit
    - prefix: "/chat"
      label: chat
    - prefix: "!gpt"
      label: openai
    - prefix: "!local"
      label: local
```

### 6. always

Unconditional route. Always matches. Used as the final fallback at the end of a pipeline to ensure every request gets a routing decision. Takes a single label.

Config:

```yaml
- method: always
  label: default
```

---

## Heuristic Methods

These are cheap single-purpose classifiers. They can be added to the pipeline like any other method.

### question

Matches if the text looks like a conversational question. Checks:
- Ends with `?`
- Starts with what / why / how / when / where / who / is / are / can / could / should / would / do / does / did

Config:

```yaml
- method: question
  label: chat
```

### code_block

Matches if the text contains a fenced code block (triple backticks) or a 4-space-indented block of at least 2 consecutive lines.

Config:

```yaml
- method: code_block
  label: code
```

### language

Detects non-English text using a lightweight heuristic (character n-gram frequency or the `langdetect` library if installed). Routes non-English text to a dedicated backend or passthrough.

Config:

```yaml
- method: language
  non_english_label: translate   # label to use if non-English detected
  english_label: null            # null = no match (fall through), or set a label
  min_confidence: 0.85           # langdetect confidence threshold
```

---

## Fallback Chains

Routes map labels to ordered lists of backends. If the first backend in the list is unavailable or returns an error on health check, smolroute returns the next backend in the list.

```yaml
routes:
  code:
    backends: [claude-sonnet, claude-haiku]   # fallback chain
  edit:
    backends: [claude-haiku]
  chat:
    backends: [local-llama, claude-haiku]     # prefer local, fall back to cloud
  research:
    backends: [claude-sonnet]
  translate:
    backends: [claude-haiku]
  tools:
    backends: [claude-sonnet]
  default:
    backends: [claude-haiku]                  # catch-all
```

Backend availability is determined by a lightweight health check (TCP connect or HTTP GET to a /health endpoint). Health state is cached with a configurable TTL to avoid adding latency to every request.

```yaml
health:
  check_interval_s: 30     # how often to re-check backends in the background
  cache_ttl_s: 60          # how long to trust a healthy/unhealthy result
  timeout_ms: 500          # health check connection timeout
```

---

## Router Pipeline

The pipeline is an ordered list of routing methods. smolroute evaluates each step in order. The first step that produces a label wins, and evaluation stops.

```yaml
router:
  pipeline:
    - method: prefix        # first: explicit user commands (/code, /search)
    - method: code_block    # detect embedded code
    - method: question      # detect questions → chat
    - method: regex         # known structural patterns
    - method: keyword       # cheap vocabulary match
    - method: language      # non-English detection
    - method: smoltrain     # semantic classification (highest accuracy)
    - method: length        # fallback by size
    - method: always        # unconditional catch-all
      label: default
```

Methods higher in the pipeline are cheaper and more specific. smoltrain is placed near the end because it adds ~21ms overhead — only reached if cheaper methods don't match.

---

## Backend Definitions

Backends are named and typed. Supported types:

- `anthropic` — Anthropic API
- `openai` — OpenAI API
- `openai_compat` — Any OpenAI-compatible API (Ollama, vLLM, LM Studio, etc.)
- `smoltrain` — smoltrain serve Unix socket (used as a classifier backend, not for text generation)

```yaml
backends:
  claude-sonnet:
    type: anthropic
    model: claude-sonnet-4-6
    api_key_env: ANTHROPIC_API_KEY      # env var name (not the value)
    timeout_s: 60

  claude-haiku:
    type: anthropic
    model: claude-haiku-4-5
    api_key_env: ANTHROPIC_API_KEY

  openai-gpt4:
    type: openai
    model: gpt-4o
    api_key_env: OPENAI_API_KEY

  local-llama:
    type: openai_compat
    base_url: http://localhost:11434/v1
    model: llama3.2
    api_key: ollama                     # literal value (Ollama requires any non-empty string)
    timeout_s: 120

  smoltrain-classifier:
    type: smoltrain
    socket: /tmp/smoltrain-routing.sock
    task: routing
    timeout_ms: 100
```

API keys can be provided as:
- `api_key_env: VAR_NAME` — read from environment variable at startup
- `api_key: literal-value` — inline literal (not recommended for production)

---

## Full Example Config

```yaml
# smolroute.yaml

backends:
  claude-sonnet:
    type: anthropic
    model: claude-sonnet-4-6
    api_key_env: ANTHROPIC_API_KEY

  claude-haiku:
    type: anthropic
    model: claude-haiku-4-5
    api_key_env: ANTHROPIC_API_KEY

  local-llama:
    type: openai_compat
    base_url: http://localhost:11434/v1
    model: llama3.2
    api_key: ollama

  smoltrain-classifier:
    type: smoltrain
    socket: /tmp/smoltrain-routing.sock
    task: routing
    timeout_ms: 100

routes:
  code:
    backends: [claude-sonnet, claude-haiku]
  edit:
    backends: [claude-haiku]
  chat:
    backends: [local-llama, claude-haiku]
  research:
    backends: [claude-sonnet]
  translate:
    backends: [claude-haiku]
  default:
    backends: [claude-haiku]

health:
  check_interval_s: 30
  cache_ttl_s: 60
  timeout_ms: 500

router:
  pipeline:
    - method: prefix
      strip_prefix: true
      rules:
        - prefix: "/code"
          label: code
        - prefix: "/search"
          label: research
        - prefix: "/edit"
          label: edit
        - prefix: "/chat"
          label: chat
        - prefix: "!local"
          label: chat           # forces local-llama via chat route

    - method: code_block
      label: code

    - method: regex
      rules:
        - pattern: "(?i)^(fix|debug|why (is|does|won't|doesn't)|error:|exception:)"
          label: code
        - pattern: "(?i)^(refactor|rewrite|rename|extract)"
          label: edit
        - pattern: "(?i)^(summarize|tldr|what does .{5,50} (do|mean))"
          label: research
        - pattern: "(?i)\\b(translate|in (spanish|french|german|japanese))\\b"
          label: translate

    - method: keyword
      case_sensitive: false
      rules:
        - words: [def , class , function, import , async def, "return ", "if __name__"]
          label: code
        - words: [refactor, rewrite, rename, clean up, improve this]
          label: edit
        - words: [summarize, summary, overview, explain, tldr]
          label: research
        - words: [translate, in english, en français, auf deutsch]
          label: translate

    - method: language
      non_english_label: translate
      english_label: null
      min_confidence: 0.85

    - method: question
      label: chat

    - method: smoltrain
      socket: /tmp/smoltrain-routing.sock
      task: routing
      timeout_ms: 100
      label_map:
        code_generation: code
        text_editing: edit
        question_answer: chat
        research: research

    - method: length
      unit: chars
      rules:
        - max: 120
          label: chat
        - min: 121
          label: research

    - method: always
      label: default

server:
  unix_socket: /tmp/smolroute.sock
  http_host: 127.0.0.1     # set to 0.0.0.0 to expose on network
  http_port: 8080
  http_enabled: false       # disabled by default
```

---

## Interface

### Input

JSON object sent over Unix socket or HTTP POST:

```json
{
  "text": "can you help me refactor this function to use a list comprehension",
  "context": {
    "session_id": "abc123",
    "user_id": "user42",
    "previous_label": "code"
  }
}
```

The `context` field is optional and passed through to the routing decision output. smolroute does not use context for routing decisions in the base implementation (reserved for future use).

### Output

```json
{
  "backend": "claude-haiku",
  "label": "edit",
  "method": "smoltrain",
  "confidence": 0.94,
  "latency_ms": 21,
  "fallback_used": false,
  "text": "can you help me refactor this function to use a list comprehension"
}
```

Fields:
- `backend` — the backend name from config to send the request to
- `label` — the route label that was matched
- `method` — which pipeline step produced the match
- `confidence` — confidence score (1.0 for deterministic methods like prefix/regex/always, float for smoltrain/language)
- `latency_ms` — total time spent in smolroute classification
- `fallback_used` — true if backend[0] was unavailable and a fallback was selected
- `text` — the (possibly prefix-stripped) text to send to the backend

### Unix Socket Protocol

Line-delimited JSON (one JSON object per line, newline-terminated). Connections are persistent; multiple requests can be sent on the same connection.

```
{"text": "fix the bug in this code"}\n
{"backend": "claude-sonnet", "label": "code", "method": "regex", "confidence": 1.0, "latency_ms": 0, "fallback_used": false, "text": "fix the bug in this code"}\n
```

### HTTP Endpoint

`POST /route` with `Content-Type: application/json`. Same input/output format.

`GET /health` — returns `{"status": "ok", "backends": {"claude-haiku": "up", "local-llama": "down"}}`.

---

## Implementation

### Language and Dependencies

Python 3.10+. Target: single file (`smolroute.py`) or minimal package (`smolroute/`).

Required dependencies:
- `pyyaml` — config parsing
- `requests` — HTTP backend health checks

Optional dependencies (graceful degradation if not installed):
- `tiktoken` — accurate token counting for `length` method (falls back to whitespace split)
- `langdetect` — accurate language detection (falls back to ASCII ratio heuristic)

Standard library only for core socket communication (`socket`, `json`, `re`, `threading`, `time`).

### Entry Points

```
smolroute serve --config smolroute.yaml        # start Unix socket + optional HTTP server
smolroute route --config smolroute.yaml        # read one request from stdin, print decision
smolroute explain --config smolroute.yaml      # trace pipeline step-by-step for a request
smolroute validate --config smolroute.yaml     # validate config without starting server
```

### File Layout (package form)

```
smolroute/
  __init__.py
  cli.py           # entry points
  config.py        # YAML loading and validation
  pipeline.py      # pipeline runner
  methods/
    __init__.py
    smoltrain.py   # smoltrain socket client
    regex.py
    length.py
    keyword.py
    prefix.py
    always.py
    question.py
    code_block.py
    language.py
  backends.py      # backend registry and health checks
  server.py        # Unix socket and HTTP server
```

### Pipeline Runner (pseudocode)

```python
def route(text, context, config):
    for step in config.pipeline:
        method = load_method(step)
        result = method.match(text, context)
        if result is not None:
            label = result.label
            backend = resolve_backend(label, config.routes, config.health_cache)
            return RoutingDecision(
                backend=backend,
                label=label,
                method=step.method,
                confidence=result.confidence,
                text=result.text,   # may be prefix-stripped
            )
    raise RuntimeError("pipeline exhausted without match — add an 'always' step")
```

### Error Handling

- If smoltrain socket is unavailable, log a warning and fall through to the next pipeline step
- If a backend health check times out, mark it unhealthy and use the next backend in the fallback chain
- If all backends in a fallback chain are unhealthy, return an error response: `{"error": "no_backends_available", "label": "code"}`
- If config is invalid at startup, exit with a descriptive error message

---

## Performance Targets

| Method        | Expected latency |
|---------------|-----------------|
| prefix        | < 0.1ms         |
| always        | < 0.1ms         |
| code_block    | < 0.1ms         |
| question      | < 0.1ms         |
| keyword       | < 0.5ms         |
| regex         | < 1ms           |
| length        | < 0.5ms         |
| language      | < 2ms           |
| smoltrain     | ~21ms           |

Total pipeline latency (typical, smoltrain not reached): < 2ms
Total pipeline latency (smoltrain reached): ~23ms

---

## Design Decisions

**Why first-match-wins instead of scoring all methods?**
Simpler, faster, and more predictable. The pipeline order is explicit configuration — operators control priority directly. Scoring across heterogeneous methods (regex vs neural classifier) would require arbitrary weight tuning.

**Why not forward the request?**
Decoupling routing from forwarding keeps smolroute simple and lets callers handle auth, streaming, retries, and response parsing using their preferred HTTP client. smolroute's only job is the routing decision.

**Why Unix socket as primary interface?**
Matches the smoltrain serve pattern. Sub-millisecond IPC overhead, no TCP stack, no port management. HTTP is opt-in for remote/cross-host use cases.

**Why Python?**
Matches smoltrain's implementation language. Easy to extend with new routing methods. The ~21ms smoltrain overhead dominates any Python interpreter overhead for the other methods.
