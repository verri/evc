all: .meta/data.json .meta/sampling.json

.meta/:
	@mkdir -p .meta

.meta/data.json: .meta/
	@echo "Generating $@..."
	@python -m evc meta --target data

.meta/sampling.json: .meta/data.json
	@echo "Generating $@..."
	@python -m evc meta --target sampling
