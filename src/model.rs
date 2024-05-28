use crate::clip::clip_vit_large_patch14_336;
use crate::IMAGE_TOKEN_INDEX;
use candle_core::bail;
use candle_core::Device;
use candle_core::IndexOp;
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
                modules = modules.add(Activation::Gelu).add(linear(
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
    select_feature_method: String,
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
            select_feature_method: config.mm_vision_select_feature.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let result = self.model.output_hidden_states(x)?;
        println!("debug");
        let index = result.len() as isize + self.select_layer;
        let result = result[index as usize].clone();
        if self.select_feature_method == "cls_patch" {
            Ok(result)
        } else {
            result.i((.., 1..))
        }
        //Ok(result[index as usize].clone())
    }
}

pub struct LLaVA {
    pub clip_vision_tower: ClipVisionTower,
    pub image_newline: Tensor,
    pub mm_projector: MMProjector,
    pub llama: Llama,
    config: LLaVAConfig,
    device: Device,
}

impl LLaVA {
    pub fn load(vb: VarBuilder, config: &LLaVAConfig) -> Result<Self> {
        let device = vb.device().clone();
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
            config: (*config).clone(),
            device,
        })
    }

    pub fn encode_images(&self, x: &Tensor) -> Result<Tensor> {
        let image_features = self.clip_vision_tower.forward(x)?;
        println!(
            "after clip vision tower: image_features shape: {:?}",
            image_features.shape()
        );
        let image_features = self.mm_projector.forward(&image_features)?;
        Ok(image_features)
    }
    // currently only for single image, 4 dim tensor
    pub fn prepare_inputs_labels_for_multimodal(
        &self,
        input_ids: &Tensor,
        image: &Tensor,
    ) -> Result<Tensor> {
        //TODO: process of multiple images/ new line
        let image_features = self.encode_images(&image)?.flatten(0, 1)?;
        let input_len = input_ids.shape().dims1()?;
        //TODO: attention mask
        println!("image_features: {:?}", image_features.shape()); //[5, 577, 4096]
        let mut image_indices = vec![-1 as i64];
        let input_ids_vec = input_ids.to_vec1::<i64>()?;
        // can easily be replaced by nonzero if it is implemented in candle
        image_indices.extend(
            input_ids_vec
                .iter()
                .enumerate()
                .filter_map(|(i, x)| {
                    if *x == IMAGE_TOKEN_INDEX as i64 {
                        Some(i as i64)
                    } else {
                        None
                    }
                })
                .collect::<Vec<i64>>(),
        );
        image_indices.push(input_len as i64);

        let input_ids_noim = input_ids_vec
            .iter()
            .filter_map(|x| {
                if *x != IMAGE_TOKEN_INDEX as i64 {
                    Some(*x)
                } else {
                    None
                }
            })
            .collect::<Vec<i64>>();
        let input_ids_noim_len = input_ids_noim.len();
        let input_ids_noim = Tensor::from_vec(input_ids_noim, (input_ids_noim_len), &self.device)?;
        println!("image_indices: {:?}", image_indices);
        todo!()
    }

    pub fn forward(&self, input_ids: &Tensor, image: &Tensor) -> Result<Tensor> {
        let new_features = self.prepare_inputs_labels_for_multimodal(input_ids, image)?;
        todo!()
    }
}
