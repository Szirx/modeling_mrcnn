install:
	pip install -r requirements.txt

train:
	PYTHONPATH=. python src/train.py configs/config.yaml

save_pt:
	python3 src/save_check.py configs/config.yaml