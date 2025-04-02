.PHONY: build-dev build-runtime shell sweep-dev train-test render-grid minikube-up minikube-down pvc-up pvc-down pvc-clean monitor-jobs jobs-clean tensorboard-up tensorboard-restart tensorboard-down image-push registry-up registry-down minikube-cache-add

build-dev:
	DOCKER_BUILDKIT=1 docker build -t flex-rl:latest -f docker/Dockerfile

build-runtime:
	DOCKER_BUILDKIT=1 docker build -t flex-rl-runtime:latest -f docker/runtime.Dockerfile .

registry-up:
	docker run -d -p 5000:5000 --restart=always --name registry registry:2 || true

registry-down:
	docker stop registry || true
	docker rm registry || true

image-push:
	docker push flex-rl-runtime:latest

image-load:
	minikube image load flex-rl-runtime:latest

minikube-cache-add:
	minikube cache add flex-rl-runtime:latest

shell:
	docker run --rm -it \
		--gpus all \
		-v $(PWD):/project \
		-v /var/run/docker.sock:/var/run/docker.sock \
		flex-rl:latest

train-test:
	python train.py --name test --config examples/humanoid_walk_forward.yaml

render-grid:
	docker run --rm -it \
		--gpus all \
		-v $(PWD):/project \
		flex-rl:latest \
		python orchestrator/scripts/render_grid.py --runs_dir workspace/ --output workspace/grid.gif

minikube-up:
	./orchestrator/k8s/minikube_setup.sh up

minikube-down:
	./orchestrator/k8s/minikube_setup.sh down

sweep-dev:
	./orchestrator/k8s/minikube_setup.sh sweep-dev

monitor-jobs:
	./orchestrator/k8s/minikube_setup.sh monitor

pvc-up:
	./orchestrator/k8s/minikube_setup.sh pvc-up

pvc-down:
	./orchestrator/k8s/minikube_setup.sh pvc-down

pvc-clean:
	kubectl delete jobs --all
	kubectl delete deployment tensorboard || true
	kubectl delete svc tensorboard || true
	./orchestrator/k8s/minikube_setup.sh pvc-down

jobs-clean:
	kubectl delete jobs --all

tensorboard-up:
	kubectl apply -f orchestrator/k8s/tensorboard.yaml
	@echo "⏳ Waiting for TensorBoard Pod to become Ready..."
	@kubectl wait --for=condition=ready pod -l app=tensorboard --timeout=300s
	@echo "🌐 TensorBoard is ready. Access at http://localhost:6006"
	kubectl port-forward svc/tensorboard 6006:6006

tensorboard-restart:
	make tensorboard-down
	sleep 3
	make tensorboard-up

tensorboard-down:
	kubectl delete svc tensorboard || true
	kubectl delete deployment tensorboard || true
