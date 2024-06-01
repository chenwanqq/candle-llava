# candle-llava
implement LLaVA using candle


## Status

tested on llava-v1.6-vicuna-7b


## eval

### single-image
```bash
cargo run --bin single_image # default args, use llava-v1.6-vicuna-7b, default-image is image/llava_logo.png, prompt is "is this a cat?"
cargo run --bin single_image -- --image-file "images/llava_v1_5_radar.jpg" --prompt "what does this picture show?"
```

### multi-image

**warning**: In LLaVA-v1.6, one image can take nearly 3000 tokens, hence multi-images is nearly inpractical.(Default max tokens is 4096)
```bash
cargo run --bin multi_images -- --image-files "images/llava_v1_5_radar.jpg" --image-files "images/llava_example_cmp.png"  --prompt  "what are the common things in these pictures?" #this is also the default args
```

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
      - [x] modify of llama code
         - [x] output embedding result
         - [x] generate from embed tensors

- [x] model forward
   - [x] Vision tower
      - [x] feature select
   - [x] LLM
   - [x] process of multiple images
      - [x] read multiple images
      - [x] multiple images patch process
   - [x] concat of image features and text features
   - [x] truncate of the concat features

- [ ] main process
   - [x] load model
   - [x] load image
   - [x] load text
   - [x] tokenize text
   - [x] forward
      - [x] single image
      - [x] multi images
   - [x] output
   - [x] KV cache
   - [ ] conversation mode
   - [ ] (long term) web?

- [ ] quantization
   - [ ] 4-bit
   - [ ] 8-bit

- [ ] (long term)  Expand candle operators, including:
   - [ ] split
   - [ ] nonzero
   - [ ] where

- [ ] **top priority** migrate to support llava-hf series model
   - [ ] determine whether it is a llava-hf model
   - [ ] translate of config
   - [ ] translate of model
   - [ ] take care of constant such as image_token_index
   - [ ] support of mistral language model

- [ ] LoRA
- [ ] More config support. Now we only support llava-v1.6
- [ ] contribution to other projects
   - [ ] [huggingface/candle](https://github.com/huggingface/candle)
   - [ ] [EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs)
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
