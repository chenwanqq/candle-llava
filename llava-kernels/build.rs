fn main() {
    let builder = bindgen_cuda::Builder::default();
    let bindings = builder.build_ptx().unwrap();
    let _ = bindings.write("src/lib.rs");
}