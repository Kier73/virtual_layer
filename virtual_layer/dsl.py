from lark import Lark, Transformer, v_args, visitors
from lark.exceptions import LarkError, UnexpectedToken, VisitError
import json

class DSLSyntaxError(SyntaxError):
    pass

class DSLSemanticError(ValueError):
    pass

class DSLCompiler:
    def __init__(self, grammar_file="virtual_layer/dsl_grammar.lark"):
        with open(grammar_file, "r") as f:
            grammar = f.read()
        # Important: Create a new transformer instance for each parse, or reset it.
        # Lark might cache the transformer or its state if the same instance is always used.
        # For simplicity here, we'll rely on the transformer itself to manage its state
        # if it were to be reused, but typically, you'd pass a new instance or a class.
        # self.transformer_instance = DSLTransformer() # Re-evaluate if needed for state
        self.parser = Lark(grammar, parser='lalr', transformer=DSLTransformer(), keep_all_tokens=True)

    def parse(self, code: str) -> dict:
        # No explicit reset of transformer_instance here, as the new transformer design
        # will not rely on instance variables populated as side-effects during tree walk,
        # but rather process children passed to its 'start' method.
        # print(f"DEBUG: [Compiler] About to call parser.parse() for code: '{code[:30]}...'")
        try:
            raw_parse_output = self.parser.parse(code)
            # print(f"DEBUG: [Compiler] Raw parse output type: {type(raw_parse_output)}, Value: {str(raw_parse_output)[:200]}")

            if isinstance(raw_parse_output, dict) and 'op' in raw_parse_output and 'args' in raw_parse_output:
                # Single operation was parsed
                return {'ops': [raw_parse_output], 'vars': {}}
            elif isinstance(raw_parse_output, tuple) and len(raw_parse_output) == 3 and raw_parse_output[0] == 'var_assign':
                # Single variable assignment was parsed
                _, var_name, value = raw_parse_output
                return {'ops': [], 'vars': {var_name: value}}
            elif isinstance(raw_parse_output, dict) and 'ops' in raw_parse_output and 'vars' in raw_parse_output:
                # This is the format from DSLTransformer.start() - for multi-statement or empty/comment-only
                return raw_parse_output
            # Handle cases where parse output might be None (e.g. for empty input if grammar/transformer leads to it)
            # or an empty list (if start rule had only discarded children).
            elif raw_parse_output is None or (isinstance(raw_parse_output, list) and not raw_parse_output):
                # Empty or effectively empty input
                return {'ops': [], 'vars': {}}
            else:
                # Unexpected type from parser.parse(). This could also be an empty list if all top children discarded.
                # print(f"WARN: Unexpected output type {type(raw_parse_output)} from parser.parse() for code '{code[:30]}...'. Value: {str(raw_parse_output)[:100]}. Returning empty dict.")
                return {'ops': [], 'vars': {}}
        except UnexpectedToken as e:
            raise DSLSyntaxError(
                f"Syntax error: Unexpected token {e.token} at line {e.line}, column {e.column}.\n"
                f"Expected: {e.expected}"
            )
        except VisitError as e:
            if isinstance(e.orig_exc, DSLSemanticError):
                raise e.orig_exc
            raise DSLSemanticError(f"Error during DSL processing: {e.orig_exc} (from rule: {e.rule if hasattr(e, 'rule') else 'unknown'})")
        except LarkError as e:
            # Handle cases where input is empty or whitespace only, which might lead to IncompleteParseError
            # depending on the strictness of the grammar's start rule.
            # The current grammar `?start: (statement | NEWLINE)*` should accept empty input.
            if not code.strip(): # If code is empty or only whitespace
                 return {'ops': [], 'vars': {}}
            raise DSLSyntaxError(f"Failed to parse DSL code: {e}")


class DSLTransformer(Transformer):
    # No __init__ needed if not maintaining state between parse calls on the instance itself.
    # Lark creates a new tree and applies this transformer logic.

    # ?start: (statement | NEWLINE)*
    def start(self, children):
        print(f"DEBUG: [Transformer] start() called. Children count: {len(children)}") # Critical Print
        vars_dict = {}
        ops_list = []
        for child_idx, child in enumerate(children):
            # Print type of each child being processed by start()
            # print(f"DEBUG: [Transformer] start() processing child {child_idx}, type: {type(child)}, value: {str(child)[:100]}")
            if child is visitors.Discard or child is None: # Explicitly check for Discard too
                continue

            # Child should be the direct result of 'statement' rule transformation
            # which in turn is the result of 'assignment' or 'operation'
            if isinstance(child, tuple) and child[0] == 'var_assign':
                _, var_name, value = child
                vars_dict[var_name] = value
            elif isinstance(child, dict) and 'op' in child:
                ops_list.append(child)
            # else:
                # print(f"DEBUG: [Transformer] start(): Unexpected child type: {type(child)}, value: {child}")

        print(f"DEBUG: [Transformer] start() finalizing. Vars: {json.dumps(vars_dict)}, Ops: {json.dumps(ops_list)}") # Critical Print
        return {'ops': ops_list, 'vars': vars_dict}

    # statement: assignment | operation | COMMENT
    # This will return the result of assignment/operation, or Discard for COMMENT
    @v_args(inline=True) # Process children of statement (assgn/op/comment) then pass result
    def statement(self,  stmt_result):
        # print(f"DEBUG: [Transformer] statement() received: {type(stmt_result)}, value: {str(stmt_result)[:100]}") # Critical Print
        return stmt_result # stmt_result is output of assignment, operation, or COMMENT methods

    def COMMENT(self, token):
        return visitors.Discard # Remove comments entirely

    # assignment: "VAR" NAME "=" value
    @v_args(inline=True)
    def assignment(self, var_keyword, name_token, eq_token, value_result):
        var_name = str(name_token)
        # value_result is already processed by the 'value' method
        # print(f"DEBUG: [Transformer] assignment: Name={var_name}, Value={value_result}")
        return ('var_assign', var_name, value_result) # Return a tuple indicating assignment

    # operation: NAME atom*
    @v_args(inline=False)
    def operation(self, items):
        op_name_token = items[0]
        op_name = str(op_name_token)

        processed_args = []
        # items[1:] are results from atom() method (actual values or var names)
        for arg_val in items[1:]:
            processed_args.append(arg_val)

        known_ops_arity = {"ADD": 2, "SUB": 2, "MUL": 2, "DIV": 2}
        if op_name in known_ops_arity:
            expected_arity = known_ops_arity[op_name]
            if len(processed_args) != expected_arity:
                raise DSLSemanticError(
                    f"Operation '{op_name}' at line {op_name_token.line}, column {op_name_token.column} "
                    f"expects {expected_arity} arguments, got {len(processed_args)}."
                )
        # print(f"DEBUG: [Transformer] operation: Name={op_name}, Args={processed_args}")
        return {'op': op_name, 'args': processed_args} # Return the operation dictionary

    # value: SIGNED_NUMBER
    @v_args(inline=True)
    def value(self, number_token):
        val_str = number_token.value
        if '.' in val_str or 'e' in val_str.lower():
            return float(val_str)
        return int(val_str)

    # atom: NAME | SIGNED_NUMBER
    @v_args(inline=True)
    def atom(self, token): # Token is either NAME or SIGNED_NUMBER
        if token.type == 'SIGNED_NUMBER':
            return self.value(token) # Process it like 'value' rule
        return str(token) # For NAME, return its string value (variable name)

    def NEWLINE(self, token):
        return visitors.Discard # Discard newlines, they are just separators for grammar

if __name__ == '__main__':
    compiler = DSLCompiler()
    example_code = """
    VAR x = 10
    VAR y = 20.5 # a comment
    ADD x y
    MUL x 2
    # This is a full line comment
    VAR z = -5
    SUB z y
    DIV x z
    """
    print(f"--- Parsing example code (New Transformer Strategy) ---")
    try:
        result = compiler.parse(example_code)
        print("Parsed successfully:")
        print(json.dumps(result, indent=2))
    except (DSLSyntaxError, DSLSemanticError) as e:
        print(f"Error: {e}")

    print("\n--- Example empty input ---")
    empty_code = ""
    try:
        result = compiler.parse(empty_code)
        print("Parsed successfully (empty input):")
        print(json.dumps(result, indent=2))
    except (DSLSyntaxError, DSLSemanticError) as e:
        print(f"Error: {e}")

    print("\n--- Example only comments and newlines ---")
    comments_only_code = """ # com1 \n #com2 \n """
    try:
        result = compiler.parse(comments_only_code)
        print("Parsed successfully (comments and newlines only):")
        print(json.dumps(result, indent=2))
    except (DSLSyntaxError, DSLSemanticError) as e:
        print(f"Error: {e}")
