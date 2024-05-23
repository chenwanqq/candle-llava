# candle-lava  
implement llava using candle  

Still working!!!
  
## Plan  
  
1. How to download the corresponding weights from Hugging Face?  
  
2. How to load the model weights?  
  
3. How to align the parameters with the Python version (initially, a partial implementation can suffice)?  
  
4. Model Implementation:  
   - Vision tower (CLIP, available in candle)  
   - LLM (available from candle)  
  
5. Image preprocessing  
  
6. Processing of input templates  
  
7. Combination of images and text  
  
## Tokenizer Setup  
```bash  
conda create -n llava python=3.10  
pip install transformers protobuf
```
## Download using mirror (for Chinese users)  
```bash
pip install -U huggingface_hub  
export HF_ENDPOINT=https://hf-mirror.com  
huggingface-cli download --resume-download liuhaotian/llava-v1.6-vicuna-7b
```
## Limitations
* Tested only on liuhaotian/llava-v1.6-vicuna-7b
* Downloading the tokenizer still relies on Python
* CLIP only supports openai/vit_large_patch14_336
* The image preprocessing process has not been implemented yet
* Not yet supporting multiple image inputs
* Attention mask has not been implemented
* conv template has not been implemented
