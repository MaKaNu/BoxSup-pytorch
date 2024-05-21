"""BoxSup-pytorch namespace."""

from importlib_metadata import PackageNotFoundError, version


__author__ = "Matti Kaupenjohann"
__email__ = "matti.kaupenjohann@fh-dortmund.de"

# Used to automatically set version number from github actions
# as well as not break when being tested locally
try:
    __version__ = version(__package__)  # type: ignore
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
