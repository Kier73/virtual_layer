import re

class DSLSyntaxError(ValueError): # Ensure this is at the top level of the module
    pass

class DSLCompiler:
    def __init__(self, catalog: set):
        self.catalog = catalog
        # Regex captures: OpName, ArgsString (optional), Comment (optional)
        # ArgsString is the content *inside* the parentheses.
        # OP_NAME regex changed to ([A-Z_]+) as per subtask refinement.
        self.op_line_regex = re.compile(
            r"^\s*([A-Z_]+)\s*"                  # Operation name (Group 1) - Strictly [A-Z_]+
            r"(?:\(\s*(.*?)\s*\))?\s*"          # Optional arguments in () (Group 2 for content)
            r"(?:#.*)?$"                        # Optional comment
        )

    def parse(self, code: str) -> dict:
        parsed_ops = []
        lines = code.splitlines()

        for line_num, line_content in enumerate(lines, 1):
            stripped_line = line_content.strip()

            if not stripped_line or stripped_line.startswith('#'):
                continue

            match = self.op_line_regex.match(stripped_line)

            if not match:
                # Specific check for "VAR" lines, which are not supported
                if stripped_line.upper().startswith("VAR"):
                    raise DSLSyntaxError(
                        f"Malformed DSL at line {line_num}: VAR assignments are not supported. Problematic line: '{line_content}'"
                    )
                # Check if it looks like an attempt at an op name but failed regex (e.g. lowercase)
                # This part might be tricky to make perfect without more complex regex/lexing
                # For now, any non-match that isn't a comment/empty/VAR is malformed.
                if stripped_line:
                    raise DSLSyntaxError(f"Malformed DSL at line {line_num}: Invalid operation format. Problematic line: '{line_content}'")
                continue


            op_name = match.group(1)
            args_str = match.group(2)

            if op_name not in self.catalog:
                # Adhere to specified error message format from prompt
                raise ValueError(f"Unknown op: {op_name}")

            op_args = []
            if args_str is not None: # Parentheses were present "OP(...)"
                # If args_str is an empty string (from "OP()"), strip() makes it empty.
                # If args_str is "   " (from "OP(   )"), strip() also makes it empty.
                # Both should result in op_args = []
                if args_str.strip():
                    # Check for malformed comma usage before splitting
                    if args_str.startswith(',') or args_str.endswith(','):
                        raise DSLSyntaxError(f"Malformed DSL at line {line_num}: Leading or trailing comma in arguments for {op_name}. Problematic line: '{line_content}'")
                    if ",," in args_str:
                        raise DSLSyntaxError(f"Malformed DSL at line {line_num}: Empty argument due to consecutive commas for {op_name}. Problematic line: '{line_content}'")

                    op_args = [arg.strip() for arg in args_str.split(',')]
                    # After splitting, check if any resulting arg is empty, which means "arg1,,arg2" or "arg1, ,arg2"
                    # This is already covered by ",," check if spaces around commas are stripped by split logic.
                    # The list comprehension `[arg.strip() for arg in args_str.split(',')]` handles spaces around args.
                    # An empty string between commas like "arg1,,arg2" becomes op_args = ['arg1', '', 'arg2'].
                    # If empty arguments are disallowed:
                    if any(not arg for arg in op_args if args_str.strip()): # Check only if args_str wasn't just whitespace
                         raise DSLSyntaxError(f"Malformed DSL at line {line_num}: Empty argument string due to comma separation for {op_name}. Problematic line: '{line_content}'")

            # Ensure 'name' key is used as per the new specification for this subtask
            parsed_ops.append({'name': op_name, 'args': op_args})

        return {'ops': parsed_ops, 'vars': {}}
```
