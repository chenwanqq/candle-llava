mod clip;
mod config;
mod model;

use crate::{config::LLaVAConfig, model::LLaVA};
use anyhow::{bail, Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::llama::Cache,
};
use clap::Parser;
use hf_hub::api::sync::Api;
use std::{io::Write, process::Command};
use tokenizers::Tokenizer;

const EOS_TOKEN: &str = "</s>";
const DEFAULT_PROMPT: &str = "My favorite theorem is ";

#[derive(Parser, Debug)]
#[command(author, version, about,long_about=None)]
struct Args {
    #[arg(long, default_value = "liuhaotian/llava-v1.6-vicuna-7b")]
    model_path: String,
    #[arg(long)]
    model_base: Option<String>,
    #[arg(long)]
    image_file: String, // Required
    #[arg(long)]
    conv_mode: Option<String>,
    #[arg(long, default_value_t = 0.2)]
    temperature: f32,
    #[arg(long, default_value_t = 512)]
    max_new_tokens: u32,
    #[arg(long, action)]
    load_8bit: bool, // now useless
    #[arg(long, action)]
    load_4bit: bool, //now useless
    #[arg(long, action)]
    debug: bool, // now useless
    #[arg(long, action)]
    cpu: bool,
    #[arg(long, action)]
    no_kv_cache: bool,
    // belows are from candle llama. Only reason is to test. Need to refactor
    #[arg(long)]
    prompt: Option<String>,
    /// The seed to use when generating random samples. Copy from candle llama. Not exist in python llava.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,
    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 10000)]
    sample_len: usize,
    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 128)]
    repeat_last_n: usize,
}

//from https://github.com/huggingface/candle/blob/main/candle-examples/examples/clip/main.rs
fn load_image<T: AsRef<std::path::Path>>(
    path: T,
    image_size: usize,
    dtype: DType,
) -> anyhow::Result<Tensor> {
    let img = image::io::Reader::open(path)?.decode()?;
    let (height, width) = (image_size, image_size);
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );

    let img = img.to_rgb8();

    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(dtype)?
        .affine(2. / 255., -1.)?;
    // .unsqueeze(0)?;
    Ok(img)
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("{:?}", args);
    let device = candle_examples::device(args.cpu)?;
    let api = Api::new()?;
    println!("loading model weights from {}", args.model_path);
    let api = api.model(args.model_path.clone());
    let config_filename = api.get("config.json")?;

    let llava_config: LLaVAConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let llama_config = llava_config.to_llama_config();
    let dtype: DType = match llava_config.torch_dtype.as_str() {
        "float16" => DType::F16,
        "bfloat16" => DType::BF16,
        _ => bail!("unsupported dtype"),
    };
    println!(
        "use python to generate tokenizer.json. Will save tokenizer to tokenizer/tokenizer.json"
    );
    let output = Command::new("python").args(["-c","from transformers import AutoTokenizer;tokenizer=AutoTokenizer.from_pretrained('liuhaotian/llava-v1.6-vicuna-7b');tokenizer.save_pretrained('tokenizer')"]).output().expect("python error!");
    println!("output: {:?}", output);
    println!("loading tokenizer from tokenizer/tokenizer.json");
    let tokenizer = Tokenizer::from_file("tokenizer/tokenizer.json").map_err(E::msg)?;
    let eos_token_id = llava_config
        .eos_token_id
        .or_else(|| tokenizer.token_to_id(EOS_TOKEN));

    println!("setting kv cache");
    let mut cache = Cache::new(!args.no_kv_cache, dtype, &llama_config, &device)?;

    println!("loading model weights");

    let weight_filenames =
        candle_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_filenames, dtype, &device)? };
    let llava = LLaVA::load(vb, &llava_config)?;

    println!("generating prompt tokens");
    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    

    println!("loading image");
    let image_tensor = load_image(&args.image_file, 336, dtype)?
        .to_device(&device)?
        .unsqueeze(0)?;
    println!("image shape: {:?}", image_tensor.shape());
    //todo: image preprocess// multi images
    //let image_result = llava.clip_vision_tower.forward(&image_tensor)?;
    //println!("image_result shape: {:?}", image_result.shape());
    let image_features = llava.encode_images(&image_tensor)?;
    println!("image_features shape: {:?}", image_features.shape());

    //based on https://github.com/huggingface/candle/blob/main/candle-examples/examples/llama/main.rs
    /*
    let mut tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);
    println!("starting the inference loop");
    print!("{prompt}");
    let mut logits_processor = {
        let temperature = f64::from(args.temperature);
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            Sampling::All { temperature }
        };
        LogitsProcessor::from_sampling(args.seed, sampling)
    };

    let mut start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;
    for index in 0..args.sample_len {
        let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
            (1, index_pos)
        } else {
            (tokens.len(), 0)
        };
        if index == 1 {
            start_gen = std::time::Instant::now()
        }
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = llava.forward(&input, context_index, &mut cache)?;
        let logits = logits.squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &tokens[start_at..],
            )?
        };
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        if Some(next_token) == eos_token_id {
            break;
        }
        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        (token_generated - 1) as f64 / dt.as_secs_f64(),
    );
    */

    Ok(())
}
