#! /usr/bin/env python
#
# note:
# - Configuration file for pytest.
# - Used to enable custom arguments.
# - See https://stackoverflow.com/questions/17557313/how-to-specify-more-than-one-option-in-pytest-config-pytest-addoption
# -  and https://docs.pytest.org/en/stable/reference/reference.html.
#

"""pytest configuration"""

from mezcla import debug

def pytest_addoption(parser):
    """Add options to argparse-based PARSER"""
    debug.trace_expr(5, parser)
    parser.addoption(
        # note: returns a string of args (e.g., '--abc --def') not a list
        "--custom_args", 
        action="store",
        default="",
        help="Pass custom arguments to the tests"
    )
