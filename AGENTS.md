# Agent Guidance

Before creating new files, read `docs/FILE_STRUCTURE.md` and place files by
semantic purpose, not by whichever directory is currently open.

`graphify/` and `graphify-out/` are local generated navigation artifacts. They
can help an agent find relevant files with fewer tokens, but they are not a
source of truth and should not be committed. Verify claims against live code,
configs, tests, and command output.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After material file-structure changes, run `bash scripts/graphify_refresh.sh` before the final response. Material structure changes include creating, deleting, moving, or renaming files/directories; adding new modules, configs, scripts, docs, tests, checkpoint roots, or output roots; or changing this file-placement policy.
- After code-only changes, run `graphify update .` when graphify-out/graph.json exists to keep the graph current (AST-only, no API cost).
