"""Keep checks of restored research artifacts explicit in a fresh source checkout."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-artifact-tests", action="store_true", default=False,
                     help="Run checks requiring the restored frozen thesis datasets")


def pytest_configure(config):
    config.addinivalue_line("markers", "artifact: requires restored frozen thesis datasets")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-artifact-tests"):
        return
    skip = pytest.mark.skip(reason="Requires restored thesis archives; enable with --run-artifact-tests")
    for item in items:
        if "artifact" in item.keywords:
            item.add_marker(skip)
