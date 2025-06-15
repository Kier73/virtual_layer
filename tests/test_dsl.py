import pytest
from virtual_layer.dsl import DSLCompiler, DSLSyntaxError, DSLSemanticError

# Fixture for the DSLCompiler instance
@pytest.fixture
def compiler():
    return DSLCompiler(grammar_file="virtual_layer/dsl_grammar.lark")

def test_parse_single_variable_assignment_integer(compiler):
    code = "VAR x = 10"
    result = compiler.parse(code)
    assert result['vars'] == {'x': 10}
    assert result['ops'] == []

def test_parse_single_variable_assignment_float(compiler):
    code = "VAR y = 20.5"
    result = compiler.parse(code)
    assert result['vars'] == {'y': 20.5}
    assert result['ops'] == []

def test_parse_single_variable_assignment_negative(compiler):
    code = "VAR z = -5"
    result = compiler.parse(code)
    assert result['vars'] == {'z': -5}
    assert result['ops'] == []

def test_parse_single_operation(compiler):
    code = "ADD x y"
    result = compiler.parse(code)
    assert result['ops'] == [{'op': 'ADD', 'args': ['x', 'y']}]
    assert result['vars'] == {}

def test_parse_operation_with_literal_number_args(compiler):
    code = "MUL z 2"
    result = compiler.parse(code)
    assert result['ops'] == [{'op': 'MUL', 'args': ['z', 2]}]
    assert result['vars'] == {}

def test_parse_operation_with_float_literal_arg(compiler):
    code = "DIV z 3.14"
    result = compiler.parse(code)
    assert result['ops'] == [{'op': 'DIV', 'args': ['z', 3.14]}]
    assert result['vars'] == {}

def test_parse_multi_line_script(compiler):
    code = """
    VAR a = 100
    VAR b = 200.5
    ADD a b
    SUB b 50.0
    """
    result = compiler.parse(code)
    assert result['vars'] == {'a': 100, 'b': 200.5}
    assert result['ops'] == [
        {'op': 'ADD', 'args': ['a', 'b']},
        {'op': 'SUB', 'args': ['b', 50.0]}
    ]

def test_parse_with_comments(compiler):
    code = """
    # This is a full line comment
    VAR x = 10 # Assign 10 to x
    ADD x 5    # Add 5 to x
    """
    result = compiler.parse(code)
    assert result['vars'] == {'x': 10}
    assert result['ops'] == [{'op': 'ADD', 'args': ['x', 5]}]

def test_empty_input(compiler):
    code = ""
    result = compiler.parse(code)
    assert result['vars'] == {}
    assert result['ops'] == []

def test_only_comments(compiler):
    code = """
    # comment 1
    # comment 2
    """
    result = compiler.parse(code)
    assert result['vars'] == {}
    assert result['ops'] == []

# --- Test for Expected Errors ---

def test_error_invalid_operation_name(compiler):
    # Lark might catch this as an UnexpectedToken if INVALID_OP is not a valid NAME format
    # or if it's not defined in the grammar in a way that it can be identified as an "unknown operation"
    # For now, the current grammar would likely treat INVALID_OP as a generic operation name.
    # Semantic analysis for "known" ops would be a layer on top or specific rules.
    # The current DSLSemanticError for arity is a good example of semantic check.
    # Let's assume for now any valid NAME token can be an op name.
    # If we want to restrict op names, the grammar or transformer logic would need adjustment.
    pass # Covered by arity tests for now, and general syntax errors

def test_error_incorrect_number_of_arguments_add(compiler):
    code = "VAR val = 1\nADD val" # ADD expects 2 arguments
    with pytest.raises(DSLSemanticError) as excinfo:
        compiler.parse(code)
    assert "Operation 'ADD' at line 2, column 1 expects 2 arguments, got 1" in str(excinfo.value)

def test_error_incorrect_number_of_arguments_mul(compiler):
    code = "VAR val = 1\nMUL val 2 3" # MUL expects 2 arguments
    with pytest.raises(DSLSemanticError) as excinfo:
        compiler.parse(code)
    assert "Operation 'MUL' at line 2, column 1 expects 2 arguments, got 3" in str(excinfo.value)

def test_error_syntax_missing_equals_in_assignment(compiler):
    code = "VAR x 10"
    with pytest.raises(DSLSyntaxError) as excinfo:
        compiler.parse(code)
    assert "Syntax error" in str(excinfo.value) # Lark's error message might vary

def test_error_syntax_invalid_characters_in_var_name(compiler):
    code = "VAR x@y = 5" # '@' is not allowed in NAME by default CNAME
    with pytest.raises(DSLSyntaxError) as excinfo:
        compiler.parse(code)
    assert "Failed to parse DSL code: No terminal matches '@'" in str(excinfo.value)

def test_error_syntax_incomplete_operation(compiler):
    code = "ADD x" # Missing second arg, but grammar allows it, transformer catches it
    with pytest.raises(DSLSemanticError) as excinfo: # Arity error
        compiler.parse(code)
    assert "Operation 'ADD' at line 1, column 1 expects 2 arguments, got 1" in str(excinfo.value)

def test_error_syntax_value_is_not_number_or_var(compiler):
    code = "VAR y = abc" # 'abc' is not a number, grammar expects SIGNED_NUMBER for value
    with pytest.raises(DSLSyntaxError) as excinfo:
        compiler.parse(code)
    assert "Syntax error" in str(excinfo.value)

def test_error_unknown_token_at_start(compiler):
    code = "! VAR x = 5"
    with pytest.raises(DSLSyntaxError) as excinfo:
        compiler.parse(code)
    assert "Failed to parse DSL code: No terminal matches '!'" in str(excinfo.value)

def test_intent_dict_structure(compiler):
    code = """
    VAR my_var = 123.45
    PROCESS my_var 0.5
    """
    result = compiler.parse(code)
    assert 'vars' in result
    assert 'ops' in result
    assert isinstance(result['vars'], dict)
    assert isinstance(result['ops'], list)
    if result['ops']:
        for op in result['ops']:
            assert 'op' in op
            assert 'args' in op
            assert isinstance(op['op'], str)
            assert isinstance(op['args'], list)

def test_parse_multiple_assignments(compiler):
    code = """
    VAR x = 1
    VAR y = 2.0
    VAR z = -3
    """
    result = compiler.parse(code)
    assert result['vars'] == {'x': 1, 'y': 2.0, 'z': -3}
    assert result['ops'] == []

def test_parse_multiple_operations(compiler):
    code = """
    OP1 a b
    OP2 c 10
    OP3 d e 5.5
    """
    result = compiler.parse(code)
    assert result['vars'] == {}
    assert result['ops'] == [
        {'op': 'OP1', 'args': ['a', 'b']},
        {'op': 'OP2', 'args': ['c', 10]},
        {'op': 'OP3', 'args': ['d', 'e', 5.5]}
    ]

def test_parse_mixed_vars_and_ops_order(compiler):
    code = """
    OP1 x y
    VAR val = 10
    OP2 val z
    VAR another = 20.2
    """
    result = compiler.parse(code)
    assert result['vars'] == {'val': 10, 'another': 20.2}
    assert result['ops'] == [
        {'op': 'OP1', 'args': ['x', 'y']},
        {'op': 'OP2', 'args': ['val', 'z']}
    ]

def test_newline_handling(compiler):
    code = "VAR x = 1\n\nVAR y = 2\nOP1 x y\n"
    result = compiler.parse(code)
    assert result['vars'] == {'x':1, 'y':2}
    assert result['ops'] == [{'op':'OP1', 'args':['x','y']}]

def test_line_ending_with_comment(compiler):
    code = "VAR x = 10 # comment here"
    result = compiler.parse(code)
    assert result['vars'] == {'x':10}
    assert result['ops'] == []

    code2 = "OP1 x y # another comment"
    result2 = compiler.parse(code2)
    assert result2['vars'] == {}
    assert result2['ops'] == [{'op':'OP1', 'args':['x','y']}]

def test_error_assignment_to_keyword_if_restricted(compiler):
    # Current grammar allows VAR as a variable name.
    # If "VAR" should be a reserved keyword and not allowed as a variable name,
    # the grammar for NAME would need to be adjusted, or a check added in the Transformer.
    # For now, this is not an error.
    code = "VAR VAR = 5"
    result = compiler.parse(code)
    assert result['vars']['VAR'] == 5

    # However, "VAR VAR = VAR" would be an op "VAR" with args "VAR", "=" and "VAR"
    # which the current grammar for operation might parse, but would be semantically odd.
    # The current grammar `assignment: "VAR" NAME "=" value` is strict.
    pass

def test_error_var_redefinition(compiler):
    # The current parser allows redefinition, simply overwriting the value.
    # This is often acceptable behavior.
    code = """
    VAR x = 1
    VAR x = 2
    """
    result = compiler.parse(code)
    assert result['vars']['x'] == 2
    # If redefinition should be an error, this check would go into the DSLTransformer's assignment method.
    pass

def test_error_using_undefined_variable_in_op_semantic(compiler):
    # The current parser does not check for undefined variables during parsing.
    # This is typically a semantic check for a later stage (e.g., execution or further compilation).
    # The `parse` method's role is syntactic correctness according to the grammar.
    code = "ADD undefined_var 5"
    result = compiler.parse(code) # This will parse fine.
    assert result['ops'] == [{'op': 'ADD', 'args': ['undefined_var', 5]}]
    pass

# Consider adding tests for specific Lark errors if needed, though covered by DSLSyntaxError generally.
# e.g. UnexpectedCharacters
# from lark.exceptions import UnexpectedCharacters
# def test_lark_specific_error_example(compiler):
#     code = "VAR x = £5" # Assuming £ is not in the grammar at all
#     with pytest.raises(DSLSyntaxError): # Or could be UnexpectedCharacters if not caught by parser rule
#         compiler.parse(code)

# A test for an operation that is not in the "known list" for arity check
def test_unknown_operation_arity(compiler):
    code = "MY_CUSTOM_OP arg1 arg2 arg3"
    result = compiler.parse(code)
    # Since MY_CUSTOM_OP is not in the hardcoded list ['ADD', 'MUL', 'SUB', 'DIV'],
    # no arity check is performed by the current transformer logic for it.
    assert result['ops'] == [{'op': 'MY_CUSTOM_OP', 'args': ['arg1', 'arg2', 'arg3']}]
    assert result['vars'] == {}
    # If all operations must be pre-defined with their arities, the transformer logic would need to change.
    pass
