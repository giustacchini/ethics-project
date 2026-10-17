# Testing Guide

This document provides guidelines for writing and running tests in a python project

## Testing Framework

We use pytest as our testing framework. The project is set up with:
- pytest for test running and assertions
- pytest-cov for code coverage reporting

## Project Structure

The `unit` test folder structure mirrors the `src` source code folder structure to:
- Make it easy to find corresponding tests
- Maintain clear organization
- Keep related tests together

## Writing Tests

### Test File Naming
- Test files should be named `test_*.py`
- Name should correspond to the module being tested (e.g., `test_generate_png_export.py` for `generate_png_export.py`)

### Test Function Naming
- Test functions should be named `test_*`
- Name should clearly describe what is being tested
- Use descriptive names that indicate:
  - The function being tested
  - The scenario being tested
  - The expected outcome

Naming example:
```python
def test_generate_png_export_empty_points():
    """Test that empty points list raises ValueError."""
    with pytest.raises(ValueError):
        generate_png_export([])
```

### Test Structure
Each test should follow the Arrange-Act-Assert pattern:

```python
def test_something():
    # Arrange - Set up test data and conditions
    test_data = [...]
    
    # Act - Execute the code being tested
    result = function_under_test(test_data)
    
    # Assert - Verify the results
    assert result == expected_value
```

### Example Test Cases

```python
def test_compute_bounds():
    """Test bound computation with known values."""
    # Arrange
    xs = [1.0, 2.0, 3.0]
    ys = [4.0, 5.0, 6.0]
    margin_rate = 0.1
    
    # Act
    x_min, x_max, y_min, y_max = compute_bounds(xs, ys, margin_rate)
    
    # Assert
    assert x_min == pytest.approx(0.8)  # 1.0 - (0.1 * (3.0 - 1.0))
    assert x_max == pytest.approx(3.2)  # 3.0 + (0.1 * (3.0 - 1.0))
```

## Running Tests

### Basic Test Run
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/export/test_generate_png_export.py

# Run specific test
pytest tests/export/test_generate_png_export.py::test_compute_bounds
```

### Example of Good Test Organization

```python
import pytest
from module import function_to_test

def test_function_with_valid_input():
    """Test function behavior with a valid input."""
    # Arrange
    input_value = 42
    expected_result = 84  # Expected value based on function logic
    
    # Act
    result = function_to_test(input_value)
    
    # Assert
    assert result == expected_result

def test_function_with_zero():
    """Test function behavior with zero input."""
    # Arrange
    input_value = 0
    expected_result = 0
    
    # Act
    result = function_to_test(input_value)
    
    # Assert
    assert result == expected_result

def test_function_error_case():
    """Test function handles errors appropriately."""
    # Arrange
    invalid_input = -1  # Invalid input that should raise error
    
    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        function_to_test(invalid_input)
    assert str(exc_info.value) == "Input must be non-negative"
```

## Current project commands

From the repository root, activate the environment containing the dependencies:

```bash
source .venv/bin/activate
pytest
```

`pytest.ini` supplies the source path. Plain `pytest` runs every shared and
soccer test, including the real MultiGrid integration tests. It does not
exclude legacy failures or require manual source-path manipulation. Coverage
can be enabled explicitly with `pytest --cov=src --cov-report=term-missing`.

Shared-backbone code (`src/shared/`) is exercised through the soccer tests,
since soccer is its only user. Rendering tests (`test_soccer_rendering.py`)
compare rendered tile pixels, so they need no window or display. Soccer
complete-match validator tests import the headless runner from
`soccer.validate_matches`.
