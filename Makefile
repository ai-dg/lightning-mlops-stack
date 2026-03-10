COMPOSE = srcs/docker-compose.yml
DIRECTORIES_DATABASES=./srcs/data ./srcs/data/postgres ./srcs/data/postgres_mlflow ./srcs/data/mlflow ./srcs/data/minio ./srcs/data/artifacts



build:
	@sudo mkdir -p $(DIRECTORIES_DATABASES)
	@sudo chmod -R 777 $(DIRECTORIES_DATABASES)
	docker compose -f $(COMPOSE) up --remove-orphans -d
	$(MAKE) find-logs


find-logs: 
	@echo $(GREEN)Generating logs...$(RESET)
	@srcs/scripts/logs/log-finder.sh