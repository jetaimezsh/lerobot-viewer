from app.validation import _is_torchcodec_runtime_failure


def test_torchcodec_windows_dll_failure_is_runtime_skip() -> None:
    errors = [
        "Could not load libtorchcodec. On Windows, ensure you've installed the full-shared version. "
        "FileNotFoundError: Could not find module 'torchcodec/libtorchcodec_core6.dll'"
    ]

    assert _is_torchcodec_runtime_failure(errors, []) is True
