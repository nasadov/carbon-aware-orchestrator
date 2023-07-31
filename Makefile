REGISTRY := gitlab-registry.fbk.eu
ACCOUNT := fogatlas-k8s
REPO := algorithms
IMAGE := $(ACCOUNT)/$(REPO)

registry-login:    
	docker login $(REGISTRY)

build-tradeoffboard:                 
	@docker build -f ./pkg/tradeoffboard/server/Dockerfile -t $(REGISTRY)/$(IMAGE)/tradeoffboard:latest .

push-tradeoffboard:                  
	@docker push $(REGISTRY)/$(IMAGE)/tradeoffboard:latest

build-silly:                 
	@docker build -f ./pkg/silly/server-go/Dockerfile -t $(REGISTRY)/$(IMAGE)/silly:latest .

push-silly:                  
	@docker push $(REGISTRY)/$(IMAGE)/silly:latest