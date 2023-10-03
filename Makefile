requirements:
	pip install -r requirements.txt

full_requirements:
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
	pip install -r requirements.txt

install:
	pip install .

develop:
	pip install -ve .

clean:
	rm -rf build dist *.egg-info
	pip uninstall -y retrieval

lint:
	ruff check ./retrieval/ --ignore E501 --quiet 

asmk:
	export PYTHONPATH=${PYTHONPATH}:$(realpath thirdparty/asmk/) \

build_asmk:
	pip3 install pyaml numpy faiss-gpu && \
	cd thirdparty/asmk/ && \
	python3 setup.py build_ext --inplace && \
	rm -r build && \
	cd ../../


