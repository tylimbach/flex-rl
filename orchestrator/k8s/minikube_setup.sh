#!/bin/bash

# === Minikube Setup & Management for flex-rl ===

minikube-up() {
	minikube start \
		--driver=docker \
		--container-runtime=docker \
		--cpus=8 \
		--memory=16g \
		--gpus all

	kubectl config use-context minikube
	echo "✅ Minikube cluster is up and ready."
}

minikube-down() {
	minikube delete
	echo "🗑️ Minikube cluster deleted."
}

sweep-dev() {
	python orchestrator/scripts/submit_sweep.py --sweep examples/humanoid_sweep.yaml
}

monitor-jobs() {
	kubectl get jobs
	kubectl get pods
}

pvc-up() {
	kubectl apply -f orchestrator/k8s/pvc.yaml
	echo "📂 PVC created."
}

pvc-down() {
	kubectl delete pvc flex-rl-pvc || true
	echo "🗑️ PVC deleted."
}

# === Usage ===
# ./minikube_setup.sh up            -> start cluster
# ./minikube_setup.sh down          -> delete cluster
# ./minikube_setup.sh sweep-dev     -> submit sweep
# ./minikube_setup.sh monitor       -> monitor jobs
# ./minikube_setup.sh pvc-up        -> create pvc
# ./minikube_setup.sh pvc-down      -> delete pvc

case "$1" in
	up)
		minikube-up
		;;
	down)
		minikube-down
		;;
	sweep-dev)
		sweep-dev
		;;
	monitor)
		monitor-jobs
		;;
	pvc-up)
		pvc-up
		;;
	pvc-down)
		pvc-down
		;;
	*)
		echo "Usage: $0 {up|down|sweep-dev|monitor|pvc-up|pvc-down}"
		exit 1
		;;
esac
