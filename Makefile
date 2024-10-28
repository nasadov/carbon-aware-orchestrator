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
	@python -m grpc_tools.protoc -I ./pkg/idl/ --python_out=./pkg/silly/server-python --pyi_out=./pkg/silly/server-python --grpc_python_out=./pkg/silly/server-python ./pkg/idl/idl.proto		