# RepoTrim

RepoTrim is a local Python CLI that scans a repository, ranks files for a coding task, and writes a focused context packet for AI coding agents.

## Install

```powershell
git clone https://github.com/Sudhakhar-Katta/repotrim.git
cd repotrim
py -m pip install -e .
```

Optional semantic search support:

```powershell
py -m pip install -e .[semantic]
```

## CLI examples

```powershell
repotrim scan .
repotrim index .
repotrim task "implement RBAC access for tools"
repotrim task "fix login redirect" --semantic
repotrim savings .
repotrim savings . --packet .repotrim/context_packet.md
```

`repotrim task` writes `.repotrim/context_packet.md` and appends a token-savings report. `repotrim savings [REPO_PATH]` can be run later to compare the full repository token estimate with that packet and writes `.repotrim/token_savings.json`.

Semantic mode chunks Python, TypeScript, JavaScript, and C# code into file/class/function/method sections, embeds those chunks, caches them by file hash in `.repotrim/semantic_index.json`, and blends semantic matches into the normal rule-based ranking. For sentence-transformers support:

```powershell
py -m pip install -e .[semantic]
repotrim index .
repotrim task "restrict admin screens" --semantic
```

## License

MIT
