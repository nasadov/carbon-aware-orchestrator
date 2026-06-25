# Archived: Go/gRPC deployment half (FogAtlas scheduler)

Frozen, unused deployment half of the original FogAtlas-derived carbon-aware Kubernetes scheduler.
Archived (not deleted) on 2026-06-25 by the repo-restructure task.

## Why archived
- Last functional change was **March 2025**; everything since is the Python research codebase.
- The paper's experiments are pure-Python (`import carbon_aware` directly), and its real-Kubernetes
  (KWOK) validation uses the **unmodified kube-scheduler** — neither uses this Go/gRPC scheduler.

## What's here
- `idl/` — gRPC IDL (`idl.proto`).
- `generated-go/` — generated Go gRPC code.
- `server-go/` — Go scheduler service.
- `client/` — Go client.
- `main.py` — Python gRPC service entrypoint (was `pkg/carbon-aware/server-python/main.py`).
- `test_client.py` — manual gRPC test client (was `tests/test_client.py`).

NOTE: the generated Python stubs `idl_pb2.py` / `idl_pb2_grpc.py` and `carbon_aware/server.py` are
left in place under `pkg/carbon-aware/server-python/` (they may sit in the `carbon_aware` import
surface); archiving them is deferred until verified safe.

## Restore
Everything moved with `git mv`, nothing committed — restore with the inverse `git mv` (or
`git restore`), then revert the Makefile proto/build targets.
