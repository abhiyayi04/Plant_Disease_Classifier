train:
	python -m src.train --epochs_head 3 --epochs_ft 10 --batch_size 32 --num_workers 2

train-fast:
	python -m src.train --epochs_head 1 --epochs_ft 1 --batch_size 16 --num_workers 0

eval:
	python -m src.eval --weights artifacts/models/best.pt --num_workers 0

predict:
	python -m src.predict --image sample.jpg --weights artifacts/models/best.pt --class_map artifacts/reports/class_to_idx.json --top_k 5

app:
	streamlit run app/streamlit_app.py