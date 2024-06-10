all: .meta/data.json

.meta/:
	@mkdir -p .meta

.meta/data.json: .meta/
	@echo "Generating $@..."
	@python -m evc meta --target data
