"""
conftest.py — pytest configuration for framework/tests.

Registers the --run-heavy CLI option and the 'heavy' mark so that
pytest_addoption is picked up correctly (test files are not allowed
to define it; only conftest.py files are).
"""


def pytest_addoption(parser):
    parser.addoption(
        '--run-heavy', action='store_true', default=False,
        help='Include tests that download large pretrained models'
    )


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'heavy: mark test as requiring large model downloads (enable with --run-heavy)'
    )
