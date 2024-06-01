use std::collections::HashMap;

use candle_transformers::models::llama::Config;
use serde::{Deserialize, Serialize};

// original config from liuhaotian/llava
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LLaVAConfig {
    pub _name_or_path: String,
    pub architectures: Vec<String>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub hidden_size: usize,
    #[serde(default = "default_image_aspect_ratio")]
    pub image_aspect_ratio: String,
    pub image_crop_resolution: usize,
    pub image_grid_pinpoints: Vec<(u32, u32)>,
    pub image_split_resolution: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub mm_hidden_size: usize,
    #[serde(default = "default_mm_patch_merge_type")]
    pub mm_patch_merge_type: String,
    pub mm_projector_type: String,
    pub mm_use_im_start_end: bool,
    pub mm_vision_select_feature: String,
    pub mm_vision_select_layer: isize,
    pub mm_vision_tower: String,
    pub mm_vision_tower_lr: f32,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub pad_token_id: usize,
    pub pretraining_tp: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: Option<f32>,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub tokenizer_model_max_length: Option<usize>,
    pub tokenizer_padding_side: String,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub tune_mm_mlp_adapter: bool,
    pub tune_mm_vision_resampler: bool,
    pub unfreeze_mm_vision_tower: bool,
    pub use_cache: bool,
    pub use_mm_proj: bool,
    pub vocab_size: usize,
}

fn default_mm_patch_merge_type() -> String {
    "flat".to_string()
}

fn default_image_aspect_ratio() -> String {
    "square".to_string()
}

impl LLaVAConfig {
    pub fn to_llama_config(&self) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            bos_token_id: Some(self.bos_token_id),
            eos_token_id: Some(self.eos_token_id),
            use_flash_attn: false,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HFLLaVATextConfig {
    pub architectures: Vec<String>,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: u32,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: u32,
    #[serde(default = "default_max_length")]
    pub max_length: u32,
    pub max_position_embeddings: u32,
    pub model_type: String,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: u32,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: u32,
    pub pad_token_id: u32,
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub torch_dtype: String,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    pub vocab_size: u32,
}

fn default_use_cache() -> bool {
    true
}

fn default_hidden_size() -> u32 {
    4096
}

fn default_intermediate_size() -> u32 {
    11008
}

fn default_max_length() -> u32 {
    4096
}

fn default_num_attention_heads() -> u32 {
    32
}

fn default_num_key_value_heads() -> u32 {
    32
}

fn default_rope_theta() -> f32 {
    10000.0
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HFLLaVAVisionConfig {
    pub hidden_size: u32,
    pub image_size: u32,
    pub intermediate_size: u32,
    pub model_type: String,
    pub num_attention_heads: u32,
    pub num_hidden_layers: u32,
    pub patch_size: u32,
    pub projection_dim: u32,
    pub vocab_size: u32,
}

// config from llava-v1.6-vicuna-7b-hf
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HFLLaVAConfig {
    pub architectures: Vec<String>,
    pub ignore_index: i32,
    pub image_grid_pinpoints: Vec<(u32, u32)>,
    pub image_token_index: i32,
    pub model_type: String,
    pub projector_hidden_act: String,
    pub text_config: HFLLaVATextConfig,
    pub torch_dtype: String,
    pub use_image_newline_parameter: bool,
    pub vision_config: HFLLaVAVisionConfig,
    pub vision_feature_layer: i32,
    pub vision_feature_select_strategy: String,
    pub vocab_size: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HFGenerationConfig {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub max_length: u32,
    pub pad_token_id: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HFPreProcessorConfig {
    pub aspect_ration_setting: String,
    pub crop_size: HashMap<String, u32>,
    pub do_center_crop: bool,
    pub do_conver_rgb: bool,
    pub do_normalize: bool,
    pub do_rescale: bool,
    pub do_resize: bool,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    pub resample: u32,
    pub rescale_fatore: f32,
    pub size: HashMap<String, f32>,
}

impl HFLLaVAConfig {
    fn map_projector_type(s: &str) -> String {
        if s == "gelu" {
            "mlp2x_gelu".to_string()
        } else {
            s.to_string()
        }
    }
    
    fn map_select_feature(s: &str) -> String {
        if s == "default" {
            "patch".to_string()
        } else {
            "cls_patch".to_string()
        }
    }

    pub fn to_llava_config(
        &self,
        name: &str,
        generation_config: &HFGenerationConfig,
        preprocessor_config: &HFPreProcessorConfig,
    ) -> LLaVAConfig {
        LLaVAConfig {
            _name_or_path: name.to_string(),
            architectures: self.architectures.clone(),
            bos_token_id: generation_config.bos_token_id,
            eos_token_id: generation_config.eos_token_id,
            hidden_size: self.text_config.hidden_size as usize,
            image_aspect_ratio: "anyres".to_string(),
            image_crop_resolution: 224,
            image_grid_pinpoints: self.image_grid_pinpoints.clone(),
            image_split_resolution: 224,
            intermediate_size: self.text_config.intermediate_size as usize,
            max_position_embeddings: self.text_config.max_position_embeddings as usize,
            mm_hidden_size: 1024,
            mm_patch_merge_type: "spatial_unpad".to_string(),
            mm_projector_type: Self::map_projector_type(&self.projector_hidden_act),
            mm_use_im_start_end: false,
            mm_vision_select_feature: Self::map_select_feature(&self.vision_feature_select_strategy),
            mm_vision_select_layer: self.vision_feature_layer,
            mm_vision_tower: todo!(),
            mm_vision_tower_lr: todo!(),
            model_type: todo!(),
            num_attention_heads: todo!(),
            num_hidden_layers: todo!(),
            num_key_value_heads: todo!(),
            pad_token_id: todo!(),
            pretraining_tp: todo!(),
            rms_norm_eps: todo!(),
            rope_scaling: todo!(),
            rope_theta: todo!(),
            tie_word_embeddings: todo!(),
            tokenizer_model_max_length: todo!(),
            tokenizer_padding_side: todo!(),
            torch_dtype: todo!(),
            transformers_version: todo!(),
            tune_mm_mlp_adapter: todo!(),
            tune_mm_vision_resampler: todo!(),
            unfreeze_mm_vision_tower: todo!(),
            use_cache: todo!(),
            use_mm_proj: todo!(),
            vocab_size: todo!(),
        }
    }
}
