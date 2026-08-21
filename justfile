alias t := test
alias tv := test_verbose

test:
    uv run --extra dev pytest

test_verbose:
    uv run --extra dev pytest -svv
