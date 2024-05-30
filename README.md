# candle-llava
implement LLaVA using candle  

Still working!!!

Current status: now we can process single image and a prompt. Since it lacks spatial process, the result may be different from the original LLaVA.

## task
- [x] Download the corresponding weights from Hugging Face

- [x] Load the model weights and configs
   - [x] general llava config(need to rethink what is necessary)
   - [x] Vision tower(CLIP, partial, only support openai/vit_large_patch14_336)
      - [x] image processor(partial, the format of 'size' and 'crop size' not fully compatible with python transformer)
   - [x] LLM
   - [x] (partial, use python script, will use rust to replace) load of tokenizer.model

- [x] image preprocess
   - [x] clip image processor
   - [x] 'anyres' image preprocess
   - [x] 'pad' image preprocess

- [x] conv template (partial, only implement conv_llava_v1 and conv_chatml_direct, which is enough for LLaVA v1.6)

- [x] Model structure Implementation
   - [x] Vision tower
   - [x] LLM
      -[x] modify of llama code
         - [x] output embedding result
         - [x] generate from embed tensors

- [ ] model forward
   - [x] Vision tower
      - [x] feature select
   - [x] LLM
   - [ ] process of multiple images
      - [ ] read multiple images
      - [ ] multiple images patch process
      - [ ] padding of multi features
   - [x] concat of image features and text features
   - [x] truncate of the concat features
   - [ ] attention mask

- [ ] main process
   - [x] load model
   - [x] load image
   - [x] load text
   - [x] tokenize text
   - [ ] forward
      - [x] single image
      - [ ] multi images
   - [x] output
   - [x] KV cache
   - [ ] multiple steps
   - [ ] (long term) web?

- [ ] quantization
   - [ ] 4-bit
   - [ ] 8-bit

- [ ] (long term)  Expand candle operators, including:
   - [ ] split
   - [ ] nonzero
   - [ ] where

- [ ] (optional) memory optimization for LLaVA 1.6 version
- [ ] (long term)model training 
  
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
