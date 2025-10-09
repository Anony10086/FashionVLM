This document records how to reproduce the results in Table III

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

## Start Ollama Server
```
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull phi4
```

## Data preparation
```
mkdir datasets
git clone https://huggingface.co/datasets/Anony100/FashionRec
```


## Evaluation
### Evaluate performance of LLaVA-Next-1.6
```
cd llava_next
git submodule update --init --recursive
git checkout fashionvlm
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e .

export METHOD_NAME='llava_onevision_05b_si'
export TASK='basic_recommendation'  # please change the task to personalized_recommendation or alternative_recommendation
python3 evaluate/infer_llava_next.py --model $METHOD_NAME --task $TASK
python3 evaluate/generate_image_fashionvlm.py --method $METHOD_NAME --task $TASK
python3 evaluate/evaluate_multiple_tasks.py --method $METHOD_NAME --task $TASK
```