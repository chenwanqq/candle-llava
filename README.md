# candle-lava  
implement llava using candle  

Still working!!!

## task
- [x] Download the corresponding weights from Hugging Face

- [x] Load the model weights and configs
   - [x] general llava config(need to rethink what is necessary)
   - [x] Vision tower(CLIP, partial, only support openai/vit_large_patch14_336)
      - [x] image processor(partial, the format of 'size' and 'crop size' not fully compatible with python transformer)
   - [x] LLM
   - [x] (partial, use python script, will use rust to replace) load of tokenizer.model

- [x] image preprocess
   - [x] 'anyres' image preprocess
   - [ ] 'pad' image preprocess

- [ ] conv template

- [x] Model structure Implementation
   - [x] Vision tower
   - [x] LLM

- [ ] model forward
   - [x] Vision tower
   - [x] LLM
   - [ ] process of multiple images
   - [ ] concat of image features and text features

-[ ] main process
   - [x] load model
   - [ ] load image
   - [ ] load text
   - [ ] forward
   - [ ] output

- [ ] model training (long time plan)
  
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
