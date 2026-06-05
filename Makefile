.PHONY: test lint format eval-obj eval-scene

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/

eval-obj:
	python scripts/eval_obj.py $(ARGS)

eval-scene:
	python scripts/eval_scene.py $(ARGS)
