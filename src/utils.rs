use std::cmp::min;

use candle_core::bail;
use candle_core::error;
use candle_core::Device;
use candle_core::Result;
use candle_core::Tensor;
use image::imageops::overlay;
use image::DynamicImage;
use image::GenericImageView;
use image::{Rgb, RgbImage};
use tokenizers::Tokenizer;

use crate::clip_image_processor::calculate_middle;
use crate::clip_image_processor::CLIPImageProcessor;
use crate::config::LLaVAConfig;

pub fn process_image(
    image: &DynamicImage,
    processor: &CLIPImageProcessor,
    llava_config: &LLaVAConfig,
) -> candle_core::Result<Tensor> {
    if llava_config.image_aspect_ratio == None {
        return processor.preprocess(image);
    } else if llava_config.image_aspect_ratio == Some("anyres".to_string()) {
        process_anyres_image(image, processor, &llava_config.image_grid_pinpoints)
    } else if llava_config.image_aspect_ratio == Some("pad".to_string()) {
        process_pad_image(image, processor)
    } else {
        bail!("Invalid image aspect ratio")
    }
}

fn process_pad_image(image: &DynamicImage, processor: &CLIPImageProcessor) -> Result<Tensor> {
    let mean_color = processor
        .image_mean
        .iter()
        .map(|x| ((*x) * 255.0) as u8)
        .collect::<Vec<u8>>();
    let mean_color = Rgb::from([mean_color[0], mean_color[1], mean_color[2]]);
    let image_padded = expand2square(image, mean_color);
    processor.preprocess(&image_padded)
}

fn process_anyres_image(
    image: &DynamicImage,
    processor: &CLIPImageProcessor,
    grid_pinpoints: &[(u32, u32)],
) -> Result<Tensor> {
    let original_size = image.dimensions();
    let best_resolution = select_best_resolution(original_size, grid_pinpoints);
    let image_padded = resize_and_pad_image(image, best_resolution);
    let image_original_resize = image.resize_exact(
        processor.size as u32,
        processor.size as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let mut patches = vec![image_original_resize];
    for patch in divide_to_patches(&image_padded, processor.crop_size) {
        patches.push(patch);
    }
    let tensors = patches
        .iter()
        .map(|patch| processor.preprocess(patch))
        .collect::<Result<Vec<Tensor>>>()?;
    Tensor::stack(&tensors, 0)
}

fn expand2square(image: &DynamicImage, background_color: Rgb<u8>) -> DynamicImage {
    let (width, height) = image.dimensions();
    if width == height {
        image.clone()
    } else if width > height {
        let mut new_image =
            DynamicImage::from(RgbImage::from_pixel(width, width, background_color));
        overlay(&mut new_image, image, 0, ((width - height) / 2) as i64);
        new_image
    } else {
        let mut new_image =
            DynamicImage::from(RgbImage::from_pixel(height, height, background_color));
        overlay(&mut new_image, image, ((height - width) / 2) as i64, 0);
        new_image
    }
}

fn resize_and_pad_image(image: &DynamicImage, target_resolution: (u32, u32)) -> DynamicImage {
    let (original_width, original_height) = image.dimensions();
    let _original_width_f = original_width as f32;
    let _original_height_f = original_height as f32;
    let (target_width, target_height) = target_resolution;
    let _target_width_f = target_width as f32;
    let _target_height_f = target_height as f32;
    let scale_w = _target_width_f / _original_width_f;
    let scale_h = _target_height_f / _original_height_f;
    let (new_width, new_height) = if scale_w < scale_h {
        let _new_height = min((_original_height_f * scale_w).ceil() as u32, target_height);
        (target_width, _new_height)
    } else {
        let _new_width = min((_original_width_f * scale_h).ceil() as u32, target_width);
        (_new_width, target_height)
    };
    let resized_image = image.resize_exact(
        new_width,
        new_height,
        image::imageops::FilterType::CatmullRom,
    );
    let mut new_image = DynamicImage::new_rgb8(target_width, target_height);
    let (paste_x, paste_y) =
        calculate_middle((target_width, target_height), (new_width, new_height));
    overlay(
        &mut new_image,
        &resized_image,
        paste_x.into(),
        paste_y.into(),
    );
    new_image
}

fn select_best_resolution(
    original_size: (u32, u32),
    possible_resolutions: &[(u32, u32)],
) -> (u32, u32) {
    let (original_width, original_height) = original_size;
    let mut best_fit = (0, 0);
    let _original_width_f = original_width as f32;
    let _original_height_f = original_height as f32;
    let mut max_effective_resolition = 0 as u32;
    let mut min_wasted_resolution = u32::MAX;
    for (width, height) in possible_resolutions {
        let _width_f = *width as f32;
        let _height_f = *height as f32;
        let scale = (_width_f / _original_width_f).min(_height_f / _original_height_f);
        let (downscaled_width, downscaled_height) = (
            (_original_width_f * scale) as u32,
            (_original_height_f * scale) as u32,
        );
        let effective_resolution =
            std::cmp::min((*width) * (*height), downscaled_width * downscaled_height);
        let wasted_resolution = (*width) * (*height) - effective_resolution;
        if effective_resolution > max_effective_resolition
            || (effective_resolution == max_effective_resolition
                && wasted_resolution < min_wasted_resolution)
        {
            best_fit = (*width, *height);
            max_effective_resolition = effective_resolution;
            min_wasted_resolution = wasted_resolution;
        }
    }
    best_fit
}

fn divide_to_patches(image: &DynamicImage, patch_size: u32) -> Vec<DynamicImage> {
    let (width, height) = image.dimensions();
    let mut patches = Vec::new();
    for y in (0..height).step_by(patch_size as usize) {
        for x in (0..width).step_by(patch_size as usize) {
            let patch = image.crop_imm(x, y, patch_size, patch_size);
            patches.push(patch);
        }
    }
    patches
}

pub fn get_model_name_from_path(model_path: &str) -> String {
    let model_paths: Vec<String> = model_path
        .trim_matches('/')
        .split('/')
        .map(|s| s.to_string())
        .collect();
    if model_paths.last().unwrap().starts_with("checkpoint-") {
        format!(
            "{}_{}",
            model_paths[model_paths.len() - 2],
            model_paths.last().unwrap()
        )
    } else {
        model_paths.last().unwrap().to_string()
    }
}

fn duplicate_vec<T>(vec: &Vec<T>, n: usize) -> Vec<T>
where
    T: Clone,
{
    let mut res = Vec::new();
    for _ in 0..n {
        res.extend(vec.clone());
    }
    res
}

fn insert_separator<T>(X: Vec<Vec<T>>, sep: Vec<T>) -> Vec<Vec<T>>
where
    T: Clone,
{
    let sep = vec![sep];
    let sep = duplicate_vec(&sep, X.len());
    let mut res = X
        .iter()
        .zip(sep.iter())
        .flat_map(|(x, y)| vec![x.clone(), y.clone()])
        .collect::<Vec<Vec<T>>>();
    res.pop();
    res
}

pub fn tokenizer_image_token(
    prompt: &str,
    tokenizer: &Tokenizer,
    image_token_index: i64,
    llava_config: &LLaVAConfig,
) -> Result<Tensor> {
    let prompt_chunks = prompt
        .split("<image>")
        .map(|s| {
            tokenizer
                .encode(s, true)
                .unwrap()
                .get_ids()
                .to_vec()
                .iter()
                .map(|x| *x as i64)
                .collect()
        })
        .collect::<Vec<Vec<i64>>>();
    println!("prompt_chunks: {:?}", prompt_chunks);
    let mut input_ids = Vec::new();
    let mut offset = 0;
    if prompt_chunks.len() > 0
        && prompt_chunks[0].len() > 0
        && prompt_chunks[0][0] == llava_config.bos_token_id.unwrap() as i64
    {
        offset = 1;
        input_ids.push(prompt_chunks[0][0]);
    }

    for x in insert_separator(
        prompt_chunks,
        duplicate_vec(&vec![image_token_index], offset + 1),
    )
    .iter()
    {
        input_ids.extend(x[1..].to_vec())
    }
    let input_len = input_ids.len();
    Tensor::from_vec(input_ids,(1,input_len),&Device::Cpu)
}

#[cfg(test)]
mod tests {
    use crate::{clip_image_processor::CLIPImageProcessor, utils::process_anyres_image};

    const CLIP_ID: &str = "openai/clip-vit-large-patch14-336";

    #[test]
    fn test_process_anyres_image() {
        let image = image::open("images/llava_logo.png").unwrap();
        let processor = CLIPImageProcessor::from_pretrained(CLIP_ID).unwrap();
        let grid_pinpoints = vec![(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)];
        let tensor = process_anyres_image(&image, &processor, &grid_pinpoints).unwrap();
        println!("{:?}", tensor.shape());
    }
}
