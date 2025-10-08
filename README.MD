## Code preparation
```
conda create -n fashionvlm python=3.10 -y
conda activate fashionvlm
```

```
git clone https://github.com/Anony10086/FashionVLM.git
cd FashionVLM
pip install -r requirements.txt
```

## Evaluation
```
cd llava_next
git submodule update --init --recursive
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e .
```

## Data preparation
```
mkdir datasets
git clone https://huggingface.co/datasets/Anony100/FashionRec
```