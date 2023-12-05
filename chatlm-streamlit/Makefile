# Makefile to build and run the docker container for the chat app

.PHONY: build run build_and_run

build_and_run: stop build run

stop:
	docker stop $$(docker ps -a -q --filter ancestor=chatlm-app --format="{{.ID}}")

build:
	docker build -t chatlm-app .

run:
	docker run -d -p 8501:8501 --env-file .env chatlm-app
