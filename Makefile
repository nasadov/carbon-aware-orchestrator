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


REGISTRY := gitlab-registry.fbk.eu
ACCOUNT := fogatlas-k8s
REPO := algorithms
IMAGE := $(ACCOUNT)/$(REPO)

registry-login:    
	docker login $(REGISTRY)


build-silly:                 
	@docker build -f ./pkg/silly/server-go/Dockerfile -t $(REGISTRY)/$(IMAGE)/silly:latest .

push-silly:                  
	@docker push $(REGISTRY)/$(IMAGE)/silly:latest

build-silly-local:                 
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o silly pkg/silly/server-go/silly.go

generate-go:
	@rm -rf ./pkg/generated-go; mkdir ./pkg/generated-go
	@protoc -I./pkg/idl --go_out=./pkg/generated-go --go-grpc_out=./pkg/generated-go ./pkg/idl/idl.proto

generate-py:
	@echo "Recall to activate the conda grpc environment otherwise the following commands will fail"
	@python -m grpc_tools.protoc -I ./pkg/idl/ --python_out=./pkg/carbon-aware/server-python --pyi_out=./pkg/carbon-aware/server-python --grpc_python_out=./pkg/carbon-aware/server-python ./pkg/idl/idl.proto

build-carbon-aware:
	@echo "⚠️ Make sure your conda environment is activated"
	@echo "Installing PyInstaller if needed..."
	@pip install pyinstaller > /dev/null || (echo "Failed to install PyInstaller"; exit 1)
	@echo "Building carbon-aware executable..."
	@cd ./pkg/carbon-aware/server-python && pyinstaller --onefile --name carbon-aware \
        --distpath $(CURDIR)/bin \
        --hidden-import grpc \
        --hidden-import google.protobuf \
        --hidden-import google.protobuf.internal \
        --hidden-import grpcio \
        --hidden-import carbon_aware.models \
        --hidden-import carbon_aware.utils \
        --hidden-import carbon_aware.server \
        --hidden-import carbon_aware.state \
        --hidden-import carbon_aware.algorithms \
        --hidden-import carbon_aware.algorithms.base \
        --hidden-import carbon_aware.algorithms.heuristic \
        --add-data "$(CURDIR)/pkg/carbon-aware/server-python/all_forecasts.json:." \
        main.py
	@echo "✅ Executable created at $(CURDIR)/bin/carbon-aware"
	@echo "📄 Copying carbon intensity data file to bin directory for convenience..."
	@cp $(CURDIR)/pkg/carbon-aware/server-python/all_forecasts.json $(CURDIR)/bin/

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
