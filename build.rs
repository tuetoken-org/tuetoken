// Expose PyO3's interpreter cfgs (Py_3_12, Py_3_14, PyPy, Py_LIMITED_API, ...)
// to this crate, for the CPython-layout fast path in `PyText::extract`.
fn main() {
    pyo3_build_config::use_pyo3_cfgs();
}
