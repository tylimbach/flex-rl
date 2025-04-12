.PHONY: build-dev build-runtime shell sweep-dev train-test render-grid minikube-up minikube-down pvc-up pvc-down pvc-clean monitor-jobs jobs-clean tensorboard-up tensorboard-restart tensorboard-down image-push registry-up registry-down minikube-cache-add

build-dev:
	DOCKER_BUILDKIT=1 docker build -t flex-rl:latest -f docker/Dockerfile

build-runtime:
	# eval(minikube docker-env)
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

minikube-up:
	./k8s/minikube_setup.sh up

minikube-down:
	./k8s/minikube_setup.sh down

monitor-jobs:
	./k8s/minikube_setup.sh monitor

pvc-up:
	./k8s/minikube_setup.sh pvc-up

pvc-down:
	./k8s/minikube_setup.sh pvc-down

pvc-clean:
	kubectl delete jobs --all
	kubectl delete deployment tensorboard || true
	kubectl delete svc tensorboard || true
	./k8s/minikube_setup.sh pvc-down

jobs-clean:
	kubectl delete jobs --all

tensorboard-up:
	kubectl apply -f orchestrator/k8s/tensorboard.yaml
	@echo "⏳ Waiting for TensorBoard Pod to become Ready..."
	@kubectl wait --for=condition=ready pod -l app=tensorboard --timeout=300s
	@POD=$$(kubectl get pod -l app=tensorboard -o jsonpath="{.items[0].metadata.name}"); \
	echo "🌐 TensorBoard Pod $$POD is ready. Waiting for server to start..."; \
	until kubectl logs $$POD | grep -q "Press CTRL+C to quit"; do \
		echo "Waiting for TensorBoard log output..."; \
		sleep 1; \
	done; \
	echo "✅ TensorBoard is up. Starting port-forward..."; \
	kubectl port-forward svc/tensorboard 6006:6006

tensorboard-restart:
	make tensorboard-down
	sleep 3
	make tensorboard-up

tensorboard-down:
	kubectl delete svc tensorboard || true
	kubectl delete deployment tensorboard || true

gcp-create-cluster-t4:
	gcloud container clusters create llm-cluster \
		--zone=us-central1-a \
		--num-nodes=2 \
		--machine-type=n1-standard-8 \
		--accelerator type=nvidia-tesla-t4,count=1 \
		--image-type=UBUNTU_CONTAINERD \
		--scopes=https://www.googleapis.com/auth/cloud-platform \
		--enable-ip-alias \
		--no-enable-basic-auth \
		--metadata disable-legacy-endpoints=true \
		--enable-autoupgrade

gcp-create-dev-compute-vm:
	gcloud compute instances create llm-dev \
		--zone=us-west1-b \
		--machine-type=n1-standard-8 \
		--accelerator type=nvidia-tesla-t4,count=1 \
		--maintenance-policy TERMINATE \
		--preemptible \
		--image-family pytorch-latest-gpu \
		--image-project deeplearning-platform-release \
		--boot-disk-size=100GB \
		--metadata "install-nvidia-driver=True" \
		--scopes=cloud-platform

gcp-create-registry:
	gcloud artifacts repositories create llm-infer-repo \
	--repository-format=docker \
	--location=us-central1

gcp-push-image:
	docker tag llm-infer:latest us-central1-docker.pkg.dev/YOUR_PROJECT_ID/llm-infer-repo/llm-infer:latest
	docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/llm-infer-repo/llm-infer:latest

# === INFRA ===
.PHONY: infra-dev-up infra-dev-down infra-gke-up infra-gke-down infra-apply infra-destroy

infra-dev-up:
	cd infra/terraform/dev-t4-vm && terraform init && terraform apply -auto-approve

infra-dev-down:
	cd infra/terraform/dev-t4-vm && terraform destroy -auto-approve

infra-gke-up:
	cd infra/terraform/gke-cluster && terraform init && terraform apply -auto-approve

infra-gke-down:
	cd infra/terraform/gke-cluster && terraform destroy -auto-approve

# === HELM ===
.PHONY: deploy-llm deploy-rl uninstall-llm uninstall-rl

deploy-llm:
	helm upgrade --install llm-infer infra/helm/llm-infer \
		--values infra/helm/llm-infer/values.yaml

deploy-rl:
	helm upgrade --install rl-infer infra/helm/rl-infer \
		--values infra/helm/rl-infer/values.yaml

uninstall-llm:
	helm uninstall llm-infer || true

uninstall-rl:
	helm uninstall rl-infer || true

# === KUBECTL SHORTCUTS ===
.PHONY: logs-llm logs-rl

logs-llm:
	kubectl logs deployment/llm-infer -c llm-infer --tail=100 -f

logs-rl:
	kubectl logs deployment/rl-infer -c rl-infer --tail=100 -f

mlflow-up:
	mlflow server \
		--host 127.0.0.1 \
		--port 5001 \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns

