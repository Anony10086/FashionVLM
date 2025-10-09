This document records how to reproduce the results in Table III

## Code preparation
```bash
git clone https://github.com/Anony10086/FashionVLM.git
cd FashionVLM

conda create -n fashionvlm python=3.10 -y
conda activate fashionvlm
pip install -r requirements.txt
```

## Start Ollama Server
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull phi4
```

## Data preparation
```bash
mkdir datasets
cd datasets
git clone https://huggingface.co/datasets/Anony100/FashionRec

# unizp the item image to evaluate the performance. Training phrase is not necessary.
tar -xzvf item_images.tar.gz -C FashionRec/data
```


## Evaluation
### Evaluate performance of FashionVLM
```bash
bash evaluate_fashionvlm.sh
```

### Evaluate performance of LLaVA-Next-1.6
```bash
cd llava_next
git submodule update --init --recursive
git checkout fashionvlm
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e .

bash evaluate_llava.sh
```