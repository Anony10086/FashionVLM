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
cd FashionRec/data
tar -xzvf item_images.tar.gz
```


## Evaluation
### Evaluate performance of FashionVLM
```bash
bash evaluate_fashionvlm.sh fashion_vlm
```

### Evaluate performance of Show-o
```bash
bash evaluate_fashionvlm.sh show_o
```


### Evaluate performance of LLaVA-1.6
```bash
cd llava_next
git submodule update --init --recursive
git checkout fashionvlm
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e .

bash evaluate_llava.sh llava_onevision_05b_si
bash evaluate_llava.sh llava_onevision_7b_ov_chat
bash evaluate_llava.sh llava_onevision_05b_si_finetune
bash evaluate_llava.sh llava_onevision_7b_ov_chat_finetune
```

### Evaluate performance of GPT series
```bash
bash evaluate_gpt.sh gpt-4.1
```

### Evaluate performance of llama-3.2
```bash
bash evaluate_llama.sh
```
