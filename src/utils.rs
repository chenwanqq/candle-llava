use candle_core::Tensor;
use candle_core::Result;
use candle_core::bail;
use image::DynamicImage;

use crate::clip_image_processor::CLIPImageProcessor;
use crate::config::LLaVAConfig;

fn process_image(image: &DynamicImage, processor: &CLIPImageProcessor,llava_config: &LLaVAConfig) -> candle_core::Result<Tensor> {
    if llava_config.image_aspect_ratio == None {
        return processor.preprocess(image);
    } else if llava_config.image_aspect_ratio == Some("anyres".to_string()) {
        todo!()
    } else if llava_config.image_aspect_ratio == Some("pad".to_string()) {
        todo!("pad aspect ratio not implemented")
    } else {
        bail!("Invalid image aspect ratio")
    }
}
