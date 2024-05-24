use anyhow::Result;
use hf_hub::api::sync::Api;
use image::{DynamicImage, GenericImageView};
use serde::{Deserialize, Serialize};

//This struct is mainly for LLaVA aplications, hence it's not completely compatible with python transformer CLIPImageProcessor  few several preprocess that LLaVA used, including "openai/clip-vit-large-patch14-336" and "openai/clip-vit-large-patch14".

#[derive(Serialize, Deserialize, Debug)]
struct CLIPImageProcessor {
    #[serde(default = "default_size")]
    size: u32, // this is not the same as python transformer
    #[serde(default = "default_do_resize")]
    do_resize: bool,

    //resample: u32 // 3 for PIL bicubic, equivalent to rust  CatmullRom. Hence below we use CatmullRom
    #[serde(default = "default_do_center_crop")]
    do_center_crop: bool,
    #[serde(default = "default_crop_size")]
    crop_size: u32, // this is not the same as python transformer
    #[serde(default = "default_do_rescale")]
    do_rescale: bool,
    #[serde(default = "default_rescale_factor")]
    rescale_factor: f32,
    #[serde(default = "default_do_normalize")]
    do_normalize: bool,
    #[serde(default = "default_image_mean")]
    image_mean: Vec<f32>,
    #[serde(default = "default_image_std")]
    image_std: Vec<f32>,
}

fn default_size() -> u32 {
    224
}

fn default_do_resize() -> bool {
    true
}

fn default_do_center_crop() -> bool {
    true
}

fn default_crop_size() -> u32 {
    224
}

fn default_do_rescale() -> bool {
    true
}

fn default_rescale_factor() -> f32 {
    1.0 / 255.0
}

fn default_do_normalize() -> bool {
    true
}

fn default_image_mean() -> Vec<f32> {
    vec![0.48145466, 0.4578275, 0.40821073]
}

fn default_image_std() -> Vec<f32> {
    vec![0.26862954, 0.26130258, 0.27577711]
}

impl CLIPImageProcessor {
    pub fn resize(&self, image: &DynamicImage) -> DynamicImage {
        let (width, height) = image.dimensions();
        let size = self.size as u32;
        if width == size && height == size {
            image.clone()
        } else {
            image.resize_exact(size, size, image::imageops::FilterType::CatmullRom)
            // after test,it is the most similar one to PIL resize among resize/resize_exact/resize_to_fill
        }
    }

    pub fn from_pretrained(clip_id: &str) -> Result<Self> {
        let api = Api::new()?;
        let api = api.model(clip_id.to_string());
        let config_filename = api.get("preprocessor_config.json")?;
        let image_processor = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        Ok(image_processor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::io::Reader as ImageReader;
    use std::path::Path;

    #[test]
    fn test_from_pretrained() {
        let clip_id = "openai/clip-vit-large-patch14-336";
        let image_processor = CLIPImageProcessor::from_pretrained(clip_id).unwrap();
        println!("{:?}", image_processor);
    }

    #[test]
    fn test_resize() {
        let image_path = Path::new("images/Rectangle-1.png");
        let image = ImageReader::open(image_path).unwrap().decode().unwrap();
        let clip_image_processor = CLIPImageProcessor {
            size: 224,
            do_resize: true,
            do_center_crop: true,
            crop_size: 224,
            do_rescale: true,
            rescale_factor: 1.0 / 255.0,
            do_normalize: true,
            image_mean: vec![0.48145466, 0.4578275, 0.40821073],
            image_std: vec![0.26862954, 0.26130258, 0.27577711],
        };
        let resized_image = clip_image_processor.resize(&image);
        resized_image
            .save("images/Rectangle-1-resized.png")
            .unwrap();
        assert_eq!(resized_image.dimensions(), (224, 224));
    }
}
