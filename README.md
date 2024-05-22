# candle-lava
implement llava using candle

## 计划

1. 如何从hugging face上下载对应权重？
2. 如何加载模型权重
3. 参数如何和python版本对齐（可以先实现一部分）
4. 模型实现
    * vision tower（clip，candle中有）
    * llm（从candle中可得）

## tokenizer
conda create -n llava python=3.10
pip install transformers protobuf

## 使用镜像下载
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download liuhaotian/llava-v1.6-vicuna-7b