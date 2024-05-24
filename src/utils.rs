use candle_core::Tensor;
use candle_core::Result;
use candle_core::bail;

use crate::config::LLaVAConfig;



fn process_images(images: Vec<Tensor>, config: &LLaVAConfig) -> Result<Tensor> {
    if config.image_aspect_ratio == "anyres" {
        todo!()
    } else {
        bail!("not implemented yet!")
    }
}