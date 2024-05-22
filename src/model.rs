use candle_core::bail;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::Module;
use candle_nn::{seq, Activation, Sequential, VarBuilder};
use candle_transformers::models::clip::vision_model::{ClipVisionConfig, ClipVisionTransformer};
use candle_transformers::models::llama::{Cache, Llama};
use candle_transformers::models::with_tracing::linear;
use regex::Regex;

use crate::config::LLaVAConfig;

fn mlp_gelu_match(mm_projector_type: &str) -> Option<usize> {
    let mlp_gelu_regex = Regex::new(r"^mlp(\d+)x_gelu$").unwrap();

    if let Some(captures) = mlp_gelu_regex.captures(mm_projector_type) {
        if let Some(match_str) = captures.get(1) {
            let match_str = match_str.as_str();
            match_str.parse::<usize>().ok()
        } else {
            None
        }
    } else {
        None
    }
}

pub fn clip_vit_large_patch14_336() -> ClipVisionConfig {
    ClipVisionConfig {
        embed_dim: 1024,
        activation: candle_transformers::models::clip::text_model::Activation::QuickGelu,
        intermediate_size: 4096,
        num_hidden_layers: 24,
        num_attention_heads: 16,
        projection_dim: 768,
        num_channels: 3,
        image_size: 336,
        patch_size: 14,
    }
}

pub struct ClipVisionTower {
    pub model: ClipVisionTransformer,
}

impl ClipVisionTower {
    pub fn load(vb: &VarBuilder, config: &LLaVAConfig) -> Result<Self> {
        let clip_vision_config = if config.mm_vision_tower == "openai/clip-vit-large-patch14-336" {
            clip_vit_large_patch14_336()
        } else {
            bail!(
                "vision tower {} is not implemented yet",
                config.mm_vision_tower
            )
        };
        let model = ClipVisionTransformer::new(
            vb.pp("model.vision_tower.vision_tower.vision_model"),
            &clip_vision_config,
        )?;
        Ok(Self { model })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

pub struct IdentityMap {}

impl Module for IdentityMap {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}

pub struct MMProjector {
    pub modules: Sequential,
}

impl MMProjector {
    pub fn load(vb: &VarBuilder, config: &LLaVAConfig) -> Result<Self> {
        if config.mm_projector_type == "linear" {
            let linear = linear(
                config.mm_hidden_size,
                config.hidden_size,
                vb.pp("model.mm_projector.0"),
            )?;
            let modules = seq().add(linear);
            Ok(Self { modules })
        } else if let Some(mlp_depth) = mlp_gelu_match(&config.mm_projector_type) {
            let mut modules = seq().add(linear(
                config.mm_hidden_size,
                config.hidden_size,
                vb.pp("model.mm_projector.0"),
            )?);
            for i in 1..mlp_depth {
                modules = seq().add(Activation::Gelu).add(linear(
                    config.hidden_size,
                    config.hidden_size,
                    vb.pp(format!("model.mm_projector.{}", i * 2)),
                )?);
            }
            Ok(Self { modules })
        } else if config.mm_projector_type == "identity" {
            Ok(Self {
                modules: seq().add(IdentityMap {}),
            })
        } else {
            bail!(
                "Unsupported MM projector type: {}",
                config.mm_projector_type
            )
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.modules.forward(x)
    }
}

pub struct LLaVA {
    pub clip_vision_tower: ClipVisionTower,
    pub mm_projector: MMProjector,
    pub llama: Llama,
}

impl LLaVA {
    pub fn load(vb: VarBuilder, config: &LLaVAConfig) -> Result<Self> {
        let clip_vision_tower = ClipVisionTower::load(&vb, config)?;
        let mm_projector = MMProjector::load(&vb, config)?;
        let llama_config = config.to_llama_config();
        let llama = Llama::load(vb, &llama_config)?;
        Ok(Self {
            clip_vision_tower,
            mm_projector,
            llama,
        })
    }

    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor> {
        self.llama.forward(x, index_pos, cache)
    }
}
