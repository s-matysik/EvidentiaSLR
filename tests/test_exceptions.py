"""The exception hierarchy must stay backwards compatible."""

import pytest

from evidentia.exceptions import (
    BackendUnavailableError,
    ConfigurationError,
    CorpusError,
    EvidentiaError,
    PipelineStateError,
    UnknownBackendError,
)


@pytest.mark.parametrize(
    ("error", "builtin"),
    [
        (ConfigurationError, ValueError),
        (BackendUnavailableError, ImportError),
        (UnknownBackendError, KeyError),
        (PipelineStateError, RuntimeError),
        (CorpusError, ValueError),
    ],
)
def test_every_error_is_catchable_as_its_builtin(error, builtin):
    assert issubclass(error, EvidentiaError)
    assert issubclass(error, builtin)


def test_catching_the_base_class_catches_everything():
    with pytest.raises(EvidentiaError):
        raise PipelineStateError("index not built")
