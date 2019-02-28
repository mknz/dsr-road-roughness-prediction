'''pytest config'''

import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--image_path',
        action='append',
        default=[],
        help="List of image paths to pass to test functions"
    )
    parser.addoption(
        '--interactive',
        action='store_true',
        default=False,
        help="Run interactive tests"
    )


def pytest_generate_tests(metafunc):
    if 'image_path' in metafunc.fixturenames:
        metafunc.parametrize(
            'image_path',
            metafunc.config.getoption('image_path'),
        )


def pytest_collection_modifyitems(config, items):
    if config.getoption('--interactive'):
        return

    skip_interactive = pytest.mark.skip(reason="Need --interactive option to run")
    for item in items:
        if "interactive" in item.keywords:
            item.add_marker(skip_interactive)
