RepoTrim turns a messy codebase into a focused context packet for AI coding agents.

# RepoTrim

RepoTrim is a local CLI context engine for Codex, Claude Code, Cursor, and other AI coding tools. It scans a repository, filters out junk and risky files, understands the task, ranks the files most likely to matter, optionally applies semantic search, and writes an agent-ready context packet.

## Why RepoTrim

Most AI coding sessions start with too much context or the wrong context.

| Before | After |
| --- | --- |
| Paste or attach a whole repo | Generate a focused packet |
| Generated files, caches, lock files, and duplicates add noise | Junk and generated paths are ignored |
| Secrets can slip into prompts | Secret-looking files are skipped |
| The agent guesses where to start | RepoTrim ranks task-relevant entry points |
| Context windows fill with unrelated files | Token count is usually reduced by 70-90% |
| Hard to audit what was sent | Packet, token savings, and agent reports are written to disk |

RepoTrim is a pre-step. Run it before opening Codex, Claude Code, or Cursor, then give the generated packet to the agent instead of the entire repository.

## Install

```bash
pip install repotrim
```

Optional semantic search support:

```bash
pip install "repotrim[semantic]"
```

## Quick Start

```bash
repotrim task "restrict admin screens" --semantic
```

Sample output:

```text
RepoTrim Task Report

Task:
restrict admin screens

Task type:
Access control / RBAC

Primary files:
1. lib/rbac.ts
   Score: 142
   Tokens: 420
   Why:
   - prioritized core RBAC policy file

2. middleware.ts
   Score: 118
   Tokens: 310
   Why:
   - prioritized middleware/protected route file

Semantic search: 12 chunks from lexical-fallback (repotrim-lexical-fallback)
context_packet.md written - paste into Claude or Codex to one-shot this.

RepoTrim Token Savings
----------------------
Files counted:              184
Full repo estimate:         154,963 tokens
Context packet estimate:    15,994 tokens
Tokens saved:               138,969 tokens
Reduction:                  89.7%
Compression ratio:          9.7x smaller
```

RepoTrim writes the main packet to:

```text
.repotrim/context_packet.md
```

## How It Works

Pipeline:

```text
task -> scan -> classify -> rank -> semantic boost -> token fit -> packet -> agent report
```

Briefly:

1. Parse the coding task and classify the intent.
2. Scan the repo with shared ignore rules for caches, generated output, secrets, binaries, and oversized files.
3. Extract lightweight file metadata: language, symbols, imports, comments, size, and token estimate.
4. Rank files using deterministic task-aware scoring.
5. Optionally chunk and semantically search source files, then blend those matches into the ranking.
6. Fit selected files into a context budget.
7. Write `context_packet.md`, token savings data, and agent planning artifacts.

## Key Features

- Local-first: no repo upload required
- No AI API required for the core workflow
- Deterministic ranking by default
- Secret and generated-file filtering
- Task-aware file ranking for common engineering work
- Optional semantic search with local fallback
- Token savings report
- Agent plan report
- JSON output for automation
- Designed as a pre-step before Codex, Claude Code, and Cursor

## Who It's For

- Developers using AI agents on medium or large repositories
- Teams that want repeatable context selection instead of ad hoc file picking
- Maintainers who need to avoid sending secrets, caches, and generated output to AI tools
- Engineers building automation around AI-assisted code changes
- Anyone trying to keep agent prompts small enough to be useful

## Roadmap

- Better scoring for framework-specific project layouts
- Evaluation benchmark for ranking quality and token reduction
- Dedicated Codex, Claude Code, and Cursor output formats

## Contributing

Issues and pull requests are welcome. Keep changes focused, add tests for ranking or packet behavior, and include a real CLI example when changing user-facing output.

For local development:

```bash
pip install -e .
pytest
```

## License

MIT
