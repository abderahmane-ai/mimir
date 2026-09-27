import inspect

import pytest

from mimir.core import errors
from mimir.core.errors import (
    InputLimitError,
    MimirError,
    MissingExtraError,
    ServerResponseError,
)


def test_every_exception_derives_from_mimir_error() -> None:
    classes = [value for _, value in inspect.getmembers(errors, inspect.isclass)]
    assert classes
    for value in classes:
        assert issubclass(value, MimirError), value


def test_input_limit_error_carries_the_value_and_limit() -> None:
    error = InputLimitError(limit="options", value=200, maximum=150)
    assert str(error) == "options is 200; the release is tested up to 150"
    assert (error.limit, error.value, error.maximum) == ("options", 200, 150)
    assert isinstance(error, ValueError)


def test_missing_extra_error_names_the_install_command() -> None:
    error = MissingExtraError("Mimir", "local", "onnxruntime")
    assert "pip install 'mimirai[local]'" in str(error)
    assert "onnxruntime" in str(error)
    assert isinstance(error, ImportError)


def test_server_response_error_keeps_status_and_body() -> None:
    error = ServerResponseError(418, '{"x": 1}', "teapot")
    assert (error.status, error.body, str(error)) == (418, '{"x": 1}', "HTTP 418: teapot")


def test_errors_are_raisable_and_catchable_as_the_base() -> None:
    message = "digest mismatch"
    with pytest.raises(MimirError, match=message):
        raise errors.IntegrityError(message)
