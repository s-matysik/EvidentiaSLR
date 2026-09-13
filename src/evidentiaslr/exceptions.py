"""Exception hierarchy.

A library should let callers distinguish "you configured this wrongly" from
"this backend is not installed" from "the pipeline was used out of order",
without matching on message strings. Everything raised deliberately by
EvidentiaSLR derives from `EvidentiaError`.
"""

from __future__ import annotations


class EvidentiaError(Exception):
    """Base class for every error raised by EvidentiaSLR."""


class ConfigurationError(EvidentiaError, ValueError):
    """Invalid parameters: bad overlap, unknown metric, mismatched lengths.

    Also inherits ValueError so existing `except ValueError` handlers and
    argparse-style call sites keep working.
    """


class BackendUnavailableError(EvidentiaError, ImportError):
    """An optional backend was requested but its dependency is missing."""


class UnknownBackendError(EvidentiaError, KeyError):
    """A backend, chunker or embedder name is not in the registry."""


class PipelineStateError(EvidentiaError, RuntimeError):
    """An operation was attempted out of order, e.g. search before build."""


class CorpusError(EvidentiaError, ValueError):
    """The corpus is empty, unparseable, or lost every record to filtering."""
