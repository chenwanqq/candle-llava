mod config;
mod model;

use crate::config::LLaVAConfig;
use anyhow::{bail, Error as E, Result};
use candle_core::DType;
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::process::Command;
use tokenizers::{FromPretrainedParameters, Tokenizer};

const EOS_TOKEN: &str = "</s>";

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
    #[arg(long, action)]
    use_flash_attn: bool, //now useless
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
    let llama_config = llava_config.to_llama_config(args.use_flash_attn);
    let dtype: DType = match llava_config.torch_dtype.as_str() {
        "float16" => DType::F16,
        "bfloat16" => DType::BF16,
        _ => bail!("unsupported dtype"),
    };
    println!(
        "use python to generate tokenizer.json. Will save tokenizer to tokenizer/tokenizer.json"
    );
    let output = Command::new("python").args(["-c","from transformers import AutoTokenizer;tokenizer=AutoTokenizer.from_pretrained('liuhaotian/llava-v1.6-vicuna-7b');tokenizer.save_pretrained('tokenizer')"]).output().expect("python error!");
    println!("loading tokenizer from tokenizer/tokenizer.json");
    let tokenizer = Tokenizer::from_file("tokenizer/tokenizer.json").map_err(E::msg)?;
    let eos_token_id = llava_config.eos_token_id;
    let mut tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer); //??? Do I need to use this?

    println!("setting kv cache");
    let cache = Cache::new(!args.no_kv_cache, dtype, &llama_config, &device);

    println!("loading model weights");

    let weight_filenames =
        candle_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_filenames, dtype, &device)? };
    let llama_model = Llama::load(vb,&llama_config)?;
    
    Ok(())
}
