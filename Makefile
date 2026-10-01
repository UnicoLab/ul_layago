.PHONY: build test race purego vet generate check
build:
	mkdir -p bin
	CGO_ENABLED=0 go build -trimpath -o bin/layago ./cmd/layago
	CGO_ENABLED=0 go build -trimpath -o bin/laya-train ./cmd/laya-train
test:
	go test ./...
race:
	go test -race ./...
purego:
	go test -tags purego ./...
vet:
	go vet ./...
generate:
	go generate ./tensor
check: test purego vet
	python3 -m unittest discover -s scripts -p 'test_*.py'
