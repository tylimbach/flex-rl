.PHONY: build-dev build-runtime shell sweep-test train-test cluster-up cluster-down pvc-up

build-dev:
	docker build -t flex-rl:latest -f docker/Dockerfile docker/

build-runtime:
	docker build -t flex-rl-runtime:latest -f docker/runtime.Dockerfile .

shell:
	docker run --rm -it \
	  --gpus all \
	  -v $(PWD):/project \
	  -v /var/run/docker.sock:/var/run/docker.sock \
	  flex-rl:latest

cluster-up:
	kind create cluster --name flex-rl

cluster-down:
	kind delete cluster --name flex-rl

pvc-up:
	kubectl apply -f orchestrator/k8s/pvc.yaml

sweep-test:
	  python orchestrator/scripts/submit_sweep.py --sweep examples/humanoid_sweep.yaml

train-test:
	  python train.py --name test --config examples/humanoid_walk_forward.yaml

render-grid:
	docker run --rm -it \
	  --gpus all \
	  -v $(PWD):/project \
	  flex-rl:latest \
	  python orchestrator/scripts/render_grid.py --runs_dir workspace/ --output workspace/grid.gif
