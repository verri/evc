DATA_FILES = $(shell find src/data -type f)
SAMPLING_FILES = $(shell find src/sampling -type f)

all: .meta/data.json .meta/sampling.json

.meta/:
	@mkdir -p .meta

.meta/data.json: .meta/ $(DATA_FILES)
	@echo "Generating $@..."
	@python -m evc meta --target data

.meta/sampling.json: .meta/data.json $(SAMPLING_FILES)
	@echo "Generating $@..."
	@python -m evc meta --target sampling
