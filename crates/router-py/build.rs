fn main() {
    // On macOS, extension modules must be linked with `-undefined dynamic_lookup`
    // so unresolved Python symbols resolve against the interpreter at import time.
    // pyo3 0.23 exposes this via pyo3-build-config; maturin does this automatically,
    // but plain `cargo build` does not.
    pyo3_build_config::add_extension_module_link_args();
}
