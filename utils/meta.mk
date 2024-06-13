DATA_FILES = $(shell find src/data -type f)
SAMPLING_FILES = $(shell find src/sampling -type f)
EVALUATION_FILES = $(shell find src/evaluation -type f)
MODEL_FILES = $(shell find src/model -type f)

all: .meta/data.json .meta/sampling.json .meta/evaluation.json .meta/model.json

.meta/:
	@mkdir -p .meta

.meta/data.json: .meta/ $(DATA_FILES)
	@echo "Generating $@..."
	@python -m evc meta --target data

.meta/sampling.json: .meta/data.json $(SAMPLING_FILES)
	@echo "Generating $@..."
	@python -m evc meta --target sampling

.meta/evaluation.json: .meta/ $(EVALUATION_FILES)
	@echo "Generating $@..."
	@python -m evc meta --target evaluation

.meta/model.json: .meta/ $(MODEL_FILES)
	@echo "Generating $@..."
	@python -m evc meta --target model
