#!/bin/bash

minikube-up() {
	minikube start \
		--driver=docker \
		--container-runtime=docker \
		--cpus=6 \
		--memory=12g \
		--gpus all

	kubectl config use-context minikube
	echo "✅ Minikube cluster is up and ready."
}

minikube-down() {
	minikube delete
	echo "🗑️ Minikube cluster deleted."
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

case "$1" in
	up)
		minikube-up
		;;
	down)
		minikube-down
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
