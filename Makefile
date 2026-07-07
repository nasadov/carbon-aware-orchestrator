#
# Copyright 2023 FBK.
#  
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


# NOTE: the Go/gRPC deployment half (Docker/silly/protoc/PyInstaller targets) was archived to
# legacy/deployment/ (frozen since Mar-2025, unused by the Python research codebase) and removed
# from the working tree in the 2026-07-06 cleanup; recover the old targets from git history.

PYTHON ?= python3

.PHONY: test lint lint-full syntax-check clean-python-artifacts

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

lint-full:
	$(PYTHON) -m ruff check --select E,F,I,UP,B .

syntax-check:
	$(PYTHON) scripts/check_python_syntax.py

clean-python-artifacts:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
