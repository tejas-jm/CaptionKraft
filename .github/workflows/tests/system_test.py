# system_test.py

import os
import subprocess

def test_linting():
    """
    Test if pylint successfully runs on the codebase.
    """
    result = subprocess.run(['/opt/hostedtoolcache/Python/3.8.18/x64/bin/pylint', 'src/'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, f"Pylint failed with error: {result.stderr.decode()}"

def test_code_formatting():
    """
    Test if code formatting tools like black are applied successfully.
    """
    result = subprocess.run(['black', '--check', 'src/'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, f"Black formatting check failed with error: {result.stderr.decode()}"


if __name__ == "__main__":
    test_linting()
    test_code_formatting()

