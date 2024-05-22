use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama};
use candle_core::Result;

use crate::config::LLaVAConfig;

pub struct LLaVA {
    pub llama: Llama,
}

impl LLaVA {
    pub fn load(vb: VarBuilder,config: &LLaVAConfig) -> Self {
        let llama_config = config.to_llama_config();
        let llama = Llama::load(vb,&llama_config).unwrap();
        Self { llama }
    }

    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor> {
        self.llama.forward(x,index_pos,cache)
    }
}