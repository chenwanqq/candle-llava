use crate::clip::clip_vit_large_patch14_336;
use candle_core::bail;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::Module;
use candle_nn::{seq, Activation, Sequential, VarBuilder};
use candle_transformers::models::llama::{Cache, Llama};
use candle_transformers::models::with_tracing::linear;
use regex::Regex;

use crate::clip::ClipVisionTransformerWithHiddenStates;
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

pub struct ClipVisionTower {
    model: ClipVisionTransformerWithHiddenStates,
    select_layer: isize,
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
        // to simulate hidden_state of python version clip
        let select_layer = match config.mm_vision_select_layer {
            -1 | -2 => config.mm_vision_select_layer,
            _ => bail!(
                "Unsupported select layer: {}",
                config.mm_vision_select_layer
            ),
        };
        let model = ClipVisionTransformerWithHiddenStates::new(
            vb.pp("model.vision_tower.vision_tower.vision_model"),
            &clip_vision_config,
        )?;
        Ok(Self {
            model,
            select_layer,
        })
    }

    // todo: feature select
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let result = self.model.output_hidden_states(x)?;
        let index = result.len() as isize + self.select_layer;
        Ok(result[index as usize].clone())
    }
}

pub struct LLaVA {
    pub clip_vision_tower: ClipVisionTower,
    pub image_newline: Tensor,
    pub mm_projector: MMProjector,
    pub llama: Llama,
}

impl LLaVA {
    pub fn load(vb: VarBuilder, config: &LLaVAConfig) -> Result<Self> {
        let clip_vision_tower = ClipVisionTower::load(&vb, config)?;
        let mm_projector = MMProjector::load(&vb, config)?;
        let llama_config = config.to_llama_config();
        let image_newline = vb.get(&[config.hidden_size], "model.image_newline")?;
        let llama = Llama::load(vb, &llama_config)?;
        Ok(Self {
            clip_vision_tower,
            image_newline,
            mm_projector,
            llama,
        })
    }

    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor> {
        todo!()
    }
}
