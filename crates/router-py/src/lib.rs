//! PyO3 bindings for the kvRouteRS routing core.
//!
//! Currently a stub: only the version string is exposed. A later milestone wraps
//! `router-core` (`Router`, `WorkerRegistry`, `PrefixIndex`) so Python services
//! can embed the router directly. Built against the stable ABI (abi3-py39) so it
//! works across Python versions without per-interpreter builds.

use pyo3::prelude::*;

#[pymodule]
fn router_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
