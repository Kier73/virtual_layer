import pytest
from virtual_layer.dsl import DSLCompiler, DSLSyntaxError

@pytest.fixture
def known_ops_catalog():
    return {"LOAD", "PROCESS", "SAVE", "NO_ARGS_OP", "MULTI_ARGS_OP"}

@pytest.fixture
def compiler(known_ops_catalog):
    return DSLCompiler(catalog=known_ops_catalog)

# --- Tests for Successful Parsing ---

def test_parse_empty_code(compiler):
    assert compiler.parse("") == {'ops': [], 'vars': {}}

def test_parse_only_comments_and_blank_lines(compiler):
    code = """
    # This is a comment

      # Another comment
    """
    assert compiler.parse(code) == {'ops': [], 'vars': {}}

def test_parse_simple_op_no_args(compiler):
    code = "NO_ARGS_OP"
    result = compiler.parse(code)
    assert result['ops'] == [{'name': 'NO_ARGS_OP', 'args': []}]

def test_parse_simple_op_empty_parens(compiler):
    code = "NO_ARGS_OP()"
    result = compiler.parse(code)
    assert result['ops'] == [{'name': 'NO_ARGS_OP', 'args': []}]

def test_parse_simple_op_empty_parens_with_spaces(compiler):
    code = "NO_ARGS_OP( )"
    result = compiler.parse(code)
    assert result['ops'] == [{'name': 'NO_ARGS_OP', 'args': []}] # Empty arg string becomes []

def test_parse_op_with_single_arg(compiler):
    code = "LOAD(file.csv)"
    result = compiler.parse(code)
    assert result['ops'] == [{'name': 'LOAD', 'args': ['file.csv']}]

def test_parse_op_with_multiple_args(compiler):
    code = "PROCESS(data, transform_A, mode_fast)"
    result = compiler.parse(code)
    assert result['ops'] == [{'name': 'PROCESS', 'args': ['data', 'transform_A', 'mode_fast']}]

def test_parse_op_with_spaces_around_args_and_commas(compiler):
    code = "LOAD(  file.csv  ,  table_name   )"
    result = compiler.parse(code)
    assert result['ops'] == [{'name': 'LOAD', 'args': ['file.csv', 'table_name']}]

def test_parse_op_with_trailing_comment(compiler):
    code = "SAVE(output.dat) # Save the data"
    result = compiler.parse(code)
    assert result['ops'] == [{'name': 'SAVE', 'args': ['output.dat']}]

def test_parse_op_with_leading_whitespace_and_comment(compiler):
    code = "  SAVE(output.dat) # Save the data"
    result = compiler.parse(code)
    assert result['ops'] == [{'name': 'SAVE', 'args': ['output.dat']}]

def test_parse_multi_line_script(compiler):
    code = """
    # Script start
    LOAD(input.txt)
    PROCESS(input.txt, temp_table)
      # Indented comment
    SAVE(temp_table, output.txt) # Save final
    """
    result = compiler.parse(code)
    assert result['ops'] == [
        {'name': 'LOAD', 'args': ['input.txt']},
        {'name': 'PROCESS', 'args': ['input.txt', 'temp_table']},
        {'name': 'SAVE', 'args': ['temp_table', 'output.txt']}
    ]

def test_parse_args_with_special_chars_if_regex_allows(compiler):
    # The regex for op_name is [a-zA-Z_][a-zA-Z0-9_]*
    # The args_str is (.*?), so it captures anything. Arg splitting is by comma.
    code = 'PROCESS(arg_with-hyphen, "quoted string arg", path/to/file)'
    result = compiler.parse(code)
    assert result['ops'] == [{
        'name': 'PROCESS',
        'args': ['arg_with-hyphen', '"quoted string arg"', 'path/to/file']
    }]

def test_parse_trailing_comma_in_args(compiler):
    # "OP(arg1,)" -> args_str = "arg1," -> op_args = ['arg1', '']
    code = "LOAD(item1,)"
    result = compiler.parse(code)
    assert result['ops'] == [{'name': 'LOAD', 'args': ['item1', '']}]

def test_parse_empty_arg_in_middle(compiler):
    # "OP(arg1,,arg2)" -> args_str = "arg1,,arg2" -> op_args = ['arg1', '', 'arg2']
    code = "MULTI_ARGS_OP(item1,,item3)"
    result = compiler.parse(code)
    assert result['ops'] == [{'name': 'MULTI_ARGS_OP', 'args': ['item1', '', 'item3']}]


# --- Tests for Syntax Errors ---

def test_error_unknown_operation(compiler):
    code = "UNKNOWN_OP(param)"
    with pytest.raises(DSLSyntaxError, match="Line 1: Unknown operation: 'UNKNOWN_OP'"):
        compiler.parse(code)

def test_error_malformed_operation_no_parens_for_args(compiler):
    # This regex expects args to be within () if present after op_name
    code = "LOAD file.csv"
    with pytest.raises(DSLSyntaxError, match="Line 1: Malformed operation or unsupported syntax: 'LOAD file.csv'"):
        compiler.parse(code)

def test_error_malformed_parentheses_unmatched_open(compiler):
    code = "LOAD(file.csv"
    with pytest.raises(DSLSyntaxError, match=r"Line 1: Malformed operation or unsupported syntax: 'LOAD\(file.csv'"):
        compiler.parse(code)

def test_error_malformed_parentheses_unmatched_close(compiler):
    code = "LOAD file.csv)"
    with pytest.raises(DSLSyntaxError, match=r"Line 1: Malformed operation or unsupported syntax: 'LOAD file.csv\)'"):
        compiler.parse(code)

def test_error_var_assignment_not_supported(compiler):
    code = "VAR x = 10"
    with pytest.raises(DSLSyntaxError, match="Line 1: VAR assignments are not supported by this DSL version: 'VAR x = 10'"):
        compiler.parse(code)

def test_error_op_name_invalid_start_char(compiler):
    code = "1LOAD(file.csv)"
    with pytest.raises(DSLSyntaxError, match=r"Line 1: Malformed operation or unsupported syntax: '1LOAD\(file.csv\)'"):
        compiler.parse(code)

def test_error_op_name_with_invalid_chars(compiler):
    code = "LOAD-DATA(file.csv)"
    with pytest.raises(DSLSyntaxError, match=r"Line 1: Malformed operation or unsupported syntax: 'LOAD-DATA\(file.csv\)'"):
        compiler.parse(code)

def test_error_just_args_no_op_name(compiler):
    code = "(file.csv)"
    with pytest.raises(DSLSyntaxError, match=r"Line 1: Malformed operation or unsupported syntax: '\(file.csv\)'"):
        compiler.parse(code)
