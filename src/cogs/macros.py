import asyncio
from enum import Enum, auto
import io
import math
import random
import re
from functools import reduce
from typing import Callable, Self, Any
import string
import inspect

from .. import errors, constants
from ..types import *


class ParserState(Enum):
    # this is def one of the ways to do it
    ROOT_VALUE = auto() # 
    ROOT_STRING = auto() #
    ROOT_TREE = auto()
    TREE_STRING = auto() #
    TREE_TREE = auto()
    TREE_VALUE = auto() #
    TREE_BOUNDARY = auto()
    STRING_ESC = auto() #

class MacroCog:

    def __init__(self, bot: Bot):
        self.debug = []
        self.bot = bot
        self.builtins: dict[str, BuiltinMacro] = {}
        self.expansions = 0

        def builtin(name: str, *, greedy: bool = True, aliases: list[str] = None):
            aliases = [] if aliases is None else aliases
            def wrapper(func: Callable):
                allows_many = False
                count = 0
                first = True
                for param in inspect.signature(func).parameters.values():
                    if first:
                        first = False 
                        continue
                    if param.kind == inspect.Parameter.VAR_POSITIONAL:
                        allows_many = True
                        break
                    assert param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD, "cannot have non-positional arguments in builtins"
                    count += 1
                
                assert func.__doc__ is not None, "missing docstring for builtin"
                doc = func.__doc__
                if len(aliases):
                    doc = "".join([doc, "\n\nAliases: `", "`, `".join(aliases), "`"])

                macro = BuiltinMacro(name, doc, func, count, allows_many, greedy)
                self.builtins[name] = macro
                for alias in aliases:
                    self.builtins[alias] = macro
                return func

            return wrapper

        # Core

        def validate_var_name(name: str):
            assert not isinstance(name, MacroTree), "variable name must be deterministic"
            name = str(name)
            assert len(name) < 32, "varialbe name cannot be longer than 32 characters"
            assert_name = name.replace('\n', '␊').replace('`', "'")[:32]
            assert re.fullmatch(r"^[A-Za-z_][A-Za-z0-9_]*$", name) is not None, f"invalid variable name: `{assert_name}`"

        @builtin("type")
        async def ty(vars: VariableRegistry, val: Any):
            """Returns the type of a value as a string."""
            return type(val).__name__

        @builtin("unpack", aliases = ['*', '...'])
        async def unpack(vars: VariableRegistry, val: Any):
            """Unpacks a list into multiple arguments."""
            if val is None: return Unpack(tuple())
            assert type(val) in (list, tuple, str), f"cannot unpack value of type {type(val).__name__}"
            return Unpack(val)

        @builtin("int", aliases = ["^I"])
        async def int_(vars: VariableRegistry, *args: Any):
            """Converts its arguments to integers."""
            if len(args) == 1:
                return int(args[0])
            return [int(arg) for arg in args]

        @builtin("float", aliases = ["^F"])
        async def float_(vars: VariableRegistry, *args: Any):
            """Converts its arguments to floats."""
            if len(args) == 1:
                return float(args[0])
            return [float(arg) for arg in args]

        def _bool(val: Any) -> bool:
            if type(val) is bool:
                return val
            if type(val) in (int, float):
                return val > 0
            if type(val) is str:
                return val not in ("", "False", "false")
            return bool(val)

        @builtin("bool", aliases = ["^B"])
        async def bool_(vars: VariableRegistry, *args: Any):
            """Converts its arguments to booleans."""
            if len(args) == 1:
                return _bool(args[0])
            return [_bool(arg) for arg in args]

        @builtin("str", aliases = ["^S"])
        async def str_(vars: VariableRegistry, *args: Any):
            """Converts its arguments to strings."""
            if len(args) == 1:
                return str(args[0])
            return [str(arg) for arg in args]

        @builtin("list", aliases = ["^L"])
        async def list_(vars: VariableRegistry, *args: Any):
            """Creates a list from its arguments."""
            return [*args]

        @builtin("dict", aliases = ["^D"])
        async def dict_(vars: VariableRegistry, *args: Any):
            """Creates a dictionary from its arguments. Amount of arguments must be divisible by 2."""
            assert len(args) % 2 == 0, "dictionary arguments arity must be divisible by 2"
            return {key: value for key, value in zip(args[::2], args[1::2])}
        
        @builtin("true", aliases = ["True"])
        async def true_(vars: VariableRegistry):
            """Returns the boolean `True`."""
            return True

        @builtin("false", aliases = ["False"])
        async def false_(vars: VariableRegistry):
            """Returns the boolean `False`."""
            return False

        @builtin("none", aliases = ["None", ""], greedy = False)
        async def none_(vars: VariableRegistry):
            """Returns an empty value, `None`. Does not evaluate its inputs."""
            return

        @builtin("ignore", aliases = ["_"])
        async def ignore(vars: VariableRegistry, *_args):
            """Returns an empty value while evaluating its inputs."""
            return

        @builtin("concat", aliases = ["c"])
        async def concat(vars: VariableRegistry, *args: Any):
            """Concatenates its input values, ignoring None."""
            return "".join(str(arg) for arg in args if arg is not None)
        
        @builtin("if", greedy = False)
        async def if_(vars: VariableRegistry, *args: Any):
            """
            Chooses a tree to execute depending on a predicate, choosing the last value if the number of arguments is odd.
            In layman's terms, this is an if/elif/else chain.
            """
            for (predicate, path) in zip(args[::2], args[1::2]):
                predicate = _bool(await self.evaluate_tree(predicate, vars.branch()))
                if predicate:
                    return await self.evaluate_tree(path, vars.branch())
            if len(args) % 2 != 0:
                return await self.evaluate_tree(args[-1], vars.branch())
        
        # TODO: Tried to do [break] and [continue], but they clashed with error handling because I was using exceptions.
        #       Maybe there's a better way?

        @builtin("while", greedy = False)
        async def while_(vars: VariableRegistry, predicate: MacroTree, body: MacroTree):
            """
            Executes a tree repeatedly depending on a predicate.
            In layman's terms, this is a while loop.
            """
            while _bool(await self.evaluate_tree(predicate, vars.branch())):
                await asyncio.sleep(0)
                await self.evaluate_tree(body, vars.branch())
        
        @builtin("for", greedy = False)
        async def for_(vars: VariableRegistry, name: str, value, body: MacroTree):
            """
            Executes a tree with an input iterating over a given value.
            In layman's terms, this is a for loop.
            """
            validate_var_name(name)
            scope = vars.branch()
            scope.variables[name] = None
            value = await self.evaluate_tree(value, vars.branch())
            for v in value:
                await asyncio.sleep(0)
                scope.variables[name] = v
                await self.evaluate_tree(body, scope)

        @builtin("try", greedy = False)
        async def try_(vars: VariableRegistry, happy_path: MacroTree, sad_path: MacroTree):
            """
            Attempts to run its first argument - if it errors, runs the second argument.
            """
            try:
                return await self.evaluate_tree(happy_path, vars.branch())
            except Exception:
                return await self.evaluate_tree(sad_path, vars.branch())

        @builtin("load", aliases = ["L", "?"], greedy = False)
        async def load(vars: VariableRegistry, name: Any):
            """
            Gets the value of a stored variable, or `None` if it doesn't exist.
            The variable name must be deterministic (no trees or inputs) and simple (no lists or dictionaries).
            """
            assert not isinstance(name, MacroTree), "variable name must be deterministic"
            return vars[name]
        
        @builtin("store", aliases = ["S", "="], greedy = False)
        async def store(vars: VariableRegistry, name: Any, val: Any):
            """
            Stores a value in a named variable. Deletes the variable if passed `None`.
            Storing will search for any variables in higher scopes to set before the current scope. If this is undesirable, look at `[init]`.
            The variable name must be deterministic (no trees or inputs), simple (no lists or dictionaries), and less than 32 characters.
            Additionally, the variable name cannot be empty, cannot start with an integer, and must be alphanumeric (including underscores.)
            """
            validate_var_name(name)
            vars[name] = await self.evaluate_tree(val, vars.branch())
        
        @builtin("init", aliases = ["I", ":="])
        async def init(vars: VariableRegistry, name: Any, val: Any):
            """
            Stores a value in a named variable, within the current variable scope. If it already exists, replaces it.
            """
            validate_var_name(name)
            val = await self.evaluate_tree(val, vars.branch())
            if val is not None:
                vars.variables[name] = val
            elif name in vars.variables:
                del vars.variables[name]
        
        @builtin("lambda", aliases = ["λ", "=>"], greedy = False)
        async def lambda_(vars: VariableRegistry, tree: MacroTree, *inputs: list[str]):
            """
            Creates an anonymous macro, with a list of given inputs.
            Each input must follow the same rules as named variables.
            Optionally, an input can be preceded with a * to make it collect the rest of the arguments passed into the macro.
            """
            assert isinstance(tree, MacroTree), f"lambda argument must be a tree (got type `{type(tree).__name__}`)"
            inputs = [*inputs]
            varargs = False
            for i, input in enumerate(inputs):
                assert isinstance(input, str), "inputs must be literal strings"
                if input.startswith("*"):
                    inputs[i] = input = input[1:]
                    varargs = True
                validate_var_name(input)
            macro = TextMacro(f"<lambda at index {tree.start}>", None, tree.source, inputs, varargs, None, (tree, ))
            macro.captured_scope = vars.shallow_copy()
            return macro

        @builtin("define", aliases = ["func", "fn", "def"], greedy = False)
        async def define(vars: VariableRegistry, name: str, tree: MacroTree, *inputs: list[str]):
            """
            Creates a named, local macro, with a list of given inputs.
            Each input must follow the same rules as named variables.
            Optionally, an input can be preceded with a * to make it collect the rest of the arguments passed into the macro.
            """
            validate_var_name(name)
            v = await lambda_(vars, tree, *inputs)
            v.name = name
            vars[name] = v
        
        @builtin("error", aliases = ["throw"])
        async def error(vars: VariableRegistry, message: Any):
            """Raises an error with a specified message."""
            err = errors.MacroRuntimeError(None, MacroTree(0), str(message))
            err._builtin = True
            raise err

        @builtin("first", aliases = ["."], greedy = False)
        async def first(vars: VariableRegistry, *args: Any):
            """Returns the first non-`None` value, short-circuiting."""
            vars = vars.branch()
            for arg in args:
                val = await self.evaluate_tree(arg, vars)
                if val is not None:
                    return val

        @builtin("get", aliases = ["??"], greedy = False)
        async def get(vars: VariableRegistry, name: Any, default: Any):
            """Gets a variable by name, or computes a default value."""
            name = str(await self.evaluate_tree(name, vars))
            val = vars[name]
            if val is None:
                return await self.evaluate_tree(default, vars.branch())
            return val

        # Comparison

        @builtin("equal", aliases = ["eq", "==", "==="])
        async def equal(vars: VariableRegistry, a: Any, b: Any):
            """Returns whether two values are equal."""
            return a == b
        
        @builtin("not_equal", aliases = ["ne", "!="])
        async def not_equal(vars: VariableRegistry, a: Any, b: Any):
            """Returns whether two values aren't equal."""
            return a != b
        
        @builtin("less", aliases = ["lt", "<"])
        async def less(vars: VariableRegistry, a: Any, b: Any):
            """Returns whether one value is less than another."""
            return a < b
        
        @builtin("greater", aliases = ["gt", ">"])
        async def greater(vars: VariableRegistry, a: Any, b: Any):
            """Returns whether one value is greater than another."""
            return a > b
        
        @builtin("less_or_eq", aliases = ["leq", "<="])
        async def less_or_equal(vars: VariableRegistry, a: Any, b: Any):
            """Returns whether one value is less than or equal to another."""
            return a <= b
        
        @builtin("greater_or_eq", aliases = ["geq", ">="])
        async def greater_or_equal(vars: VariableRegistry, a: Any, b: Any):
            """Returns whether one value is greater than another."""
            return a >= b

        # Math

        def _number(val: Any):
            if type(val) in (int, float):
                return val
            stringified = str(val)
            return float(stringified)

        @builtin("number", aliases = ["^N", "num"])
        async def number(vars: VariableRegistry, *args: Any):
            """Converts its arguments to numbers - `str`s to `float`s, leaving `int`s as is."""
            if len(args) == 1:
                return _number(args[0])
            return [_number(arg) for arg in args]
        
        @builtin("add", aliases = ["+"])
        async def add(vars: VariableRegistry, *args: Any):
            """Returns the sum of multiple numbers."""
            return sum(_number(arg) for arg in args)

        @builtin("subtract", aliases = ["sub", "-"])
        async def subtract(vars: VariableRegistry, first: Any, *args: Any):
            """Returns the difference of multiple numbers."""
            return await number(first) - await add(vars, *args)
        
        @builtin("multiply", aliases = ["mul", "x"])
        async def multiply(vars: VariableRegistry, *args: Any):
            """Returns the product of multiple numbers."""
            return reduce(lambda a, b: a * b, (_number(arg) for arg in args), 1)
        
        @builtin("divide", aliases = ["div", "\\/"])
        async def divide(vars: VariableRegistry, first: Any, *args: Any):
            """Returns the product of multiple numbers."""
            return await number(first) / await multiply(vars, *args)

        @builtin("pow", aliases = ["^", "**"])
        async def pow(vars: VariableRegistry, base: Any, exp: Any):
            """Returns a number raised to the power of another."""
            base, exp = _number(base), _number(exp)
            if base < 0 and exp % 1 != 0:
                return math.nan
            v = base ** exp
            if v is complex: v = v.real
            return v

        @builtin("sin")
        async def sin(vars: VariableRegistry, val: Any):
            """Returns the sine of a value."""
            return math.sin(_number(val))
    
        @builtin("cos")
        async def cos(vars: VariableRegistry, val: Any):
            """Returns the cosine of a value."""
            return math.cos(_number(val))
        
        @builtin("tan")
        async def tan(vars: VariableRegistry, val: Any):
            """Returns the tangent of a value."""
            return math.tan(_number(val))

        @builtin("asin")
        async def asin(vars: VariableRegistry, val: Any):
            """Returns the inverse sine of a value."""
            try:
                return math.asin(_number(val))
            except ValueError:
                return math.nan
    
        @builtin("acos")
        async def acos(vars: VariableRegistry, val: Any):
            """Returns the inverse cosine of a value."""
            try:
                return math.acos(_number(val))
            except ValueError:
                return math.nan
        
        @builtin("atan")
        async def atan(vars: VariableRegistry, val: Any):
            """Returns the inverse tangent of a value."""
            try:
                return math.atan(_number(val))
            except ValueError:
                return math.nan

        @builtin("e")
        async def e(vars: VariableRegistry):
            """Returns Euler's number, a constant so that if f(x) = e^x, then f'(x) = e^x."""
            return math.e

        @builtin("pi", aliases=["π"])
        async def pi(vars: VariableRegistry):
            """Returns Archimedes' constant, the ratio of a circle's circumference to its diameter."""
            return math.pi

        @builtin("inf", aliases=["infinity"])
        async def inf(vars: VariableRegistry):
            """Returns Infinity, as defined in IEEE754."""
            return math.inf

        @builtin("nan", aliases=["NaN"])
        async def nan(vars: VariableRegistry):
            """Returns NaN, as defined in IEEE754."""
            return math.nan
        
        @builtin("rand", aliases = ["random"])
        async def rand(vars: VariableRegistry, seed: Any = None):
            """Returns a random number on the range [0, 1)."""
            if seed is not None:
                random.seed(seed if seed is int else hash(seed))
            return random.random()

        @builtin("hash")
        async def hash_(vars: VariableRegistry, seed: Any):
            """
            Returns a unique integer associated with a given value.
            The hash of any specific value is not guaranteed to stay the same between bot updates!
            Treat this value as if it was a blackbox.
            """
            return hash(seed)
        
        # Strings

        @builtin("slice")
        async def slice_(vars: VariableRegistry, slicable: Any, start: int, end: int, step: int):
            """Slices a list or string given a start, end, and step."""
            start = int(_number(start)) if start or start == 0 else None
            end = int(_number(end)) if end or end == 0 else None
            step = int(_number(step)) if step or step == 0 else None
            sl = slice(start, end, step)
            return slicable[sl]
        
        @builtin("replace")
        async def replace(vars: VariableRegistry, needle: str, haystack: str, new_value: str, max_occurrences: int = -1):
            """Replaces all occurrences (or a specified amount) of a substring in a given string with a value."""
            return haystack.replace(needle, new_value, int(_number(max_occurrences)))
        
        @builtin("regplace", aliases=["ureplace"])
        async def regex_replace(vars: VariableRegistry, needle: str, haystack: str, new_value: str, max_occurrences: int = 0):
            """Performs a regex replacement on a given string, returning the final string."""
            return re.sub(needle, new_value, haystack, int(_number(max_occurrences)))
        
        @builtin("match")
        async def regex_match(vars: VariableRegistry, needle: str, haystack: str):
            """Performs a regex match on a given string, returning a list of all matches in the string, or `None` if it didn't match."""
            match = re.search(needle, haystack)
            if match is None: return None
            return list(match.groups())
        
        @builtin("split")
        async def split(vars: VariableRegistry, needle: str, haystack: str, max_occurrences: int = -1):
            """Splits a string a given amount of times into a list of strings."""
            return haystack.split(needle, int(_number(max_occurrences)))

        @builtin("chr")
        async def chr_(vars: VariableRegistry, codepoints: int):
            """Returns a string made of a list of unicode codepoints."""
            return "".join(chr(int(_number(codepoint))) for codepoint in codepoints)
        
        @builtin("ord")
        async def ord_(vars: VariableRegistry, string: str):
            """Returns the individual codepoints in a string."""
            return [ord(char) for char in string]
        
        # Lists and dictionaries

        @builtin("at")
        async def at(vars: VariableRegistry, val: Any, index: Any):
            """Gets a value by index from a string, list, or dictionary, returning None if it's not in it."""
            if not isinstance(val, dict):
                index = int(_number(index))
            try:
                return val[index]
            except (IndexError, KeyError):
                return None
        
        @builtin("set")
        async def set(vars: VariableRegistry, target: Any, index: Any, val: Any):
            """Sets a value by index in a list or dictionary. Will error for out-of-bounds accesses on lists."""
            if val is None:
                if index in target:
                    del index[target]
            else:
                target[index] = val

        @builtin("push")
        async def push(vars: VariableRegistry, target: Any, val: Any):
            """Adds a value to the end of a list."""
            target.append(val)
        
        @builtin("pop")
        async def pop(vars: VariableRegistry, target: Any):
            """Removes a value from the end of a list and returns it. Will error for empty lists."""
            return target.pop()
        
        @builtin("length", aliases = ["len"])
        async def length(vars: VariableRegistry, target: Any):
            """Gets the length of a string, list, or dictionary."""
            return len(target)
        
        self.builtins = dict(sorted(self.builtins.items(), key=lambda tup: tup[0]))

    def parse_forest(self, source: str) -> list[MacroTree | Any]:
        root = []
        tree_stack = []
        state_stack = [ParserState.ROOT_VALUE]
        idx = 0
        register = None
        for _ in range(constants.PARSER_LIMIT):
            if len(state_stack) > constants.MACRO_DEPTH_LIMIT:
                raise errors.MacroSyntaxError(idx, source, "reached depth limit")
            char = source[idx] if idx < len(source) else None
            state = state_stack[-1]
            if state == ParserState.ROOT_VALUE:
                if char is None:
                    break
                # either a root_string, an input, or a macro_tree
                if char == "[":
                    # macro_tree
                    state_stack[-1] = ParserState.ROOT_TREE
                    state_stack.append(ParserState.TREE_VALUE)
                    tree_head = MacroTree(idx)
                    tree_stack.append(tree_head)
                    idx += 1
                    continue
                if (
                    (char == "$") and
                    (idx + 1 < len(source)) and
                    (
                        (source[idx + 1] in string.digits) or
                        (source[idx + 1] in ('#', '!'))
                    )
                ):
                    raise errors.MacroSyntaxError(idx, source, "inputs have been replaced with `[load/name]` (or shortened, `[?/name]`)")
                # if we've gotten here, it's a root_string
                state_stack[-1] = ParserState.ROOT_STRING
                register = [idx, idx]
                continue
            elif state == ParserState.ROOT_STRING:
                register[1] = idx # Move the end of the string
                if char is None or char == "[" or (
                        (char == "$") and
                        (idx + 1 < len(source)) and
                        (
                            (source[idx + 1] in string.digits) or
                            (source[idx + 1] in ('#', '!'))
                        )
                    ):
                    # We've reached the end of the string
                    start, end = register
                    root.append(re.sub(r"\\([\[\/\]])", r"\1", source[start : end]))
                    state_stack[-1] = ParserState.ROOT_VALUE
                    continue
                elif char == "\\":
                    state_stack.append(ParserState.STRING_ESC)
                elif char == "]":
                    raise errors.MacroSyntaxError(idx, source, "unbalanced ]")
            elif state == ParserState.STRING_ESC:
                if char is None:
                    raise errors.MacroSyntaxError(idx, source, "incomplete escape sequence")
                register[1] = idx
                state_stack.pop()
            elif state == ParserState.TREE_VALUE:
                # either a string, input, or tree
                if char == "[":
                    # tree
                    state_stack[-1] = ParserState.TREE_TREE
                    state_stack.append(ParserState.TREE_VALUE)
                    tree_head = MacroTree(idx)
                    tree_stack.append(tree_head)
                    
                    idx += 1
                    continue
                if (
                    (char == "$") and
                    (idx + 1 < len(source)) and
                    (
                        (source[idx + 1] in string.digits) or
                        (source[idx + 1] in ('#', '!')) 
                    )
                ):
                    raise errors.MacroSyntaxError(idx, source, "inputs have been replaced with `[load/name]` (or shortened, `[?/name]`)")
                # string
                state_stack[-1] = ParserState.TREE_STRING
                register = [idx, idx]
                continue
            elif state == ParserState.TREE_STRING:
                register[1] = idx # Move the end of the string
                if char is None or char in "]/":
                    # We've reached the end of the string
                    start, end = register
                    tree_stack[-1].arguments.append(re.sub(r"\\([\[\/\]])", r"\1", source[start : end]).strip())
                    state_stack[-1] = ParserState.TREE_BOUNDARY
                    continue
                elif char in "[$":
                    start, end = register
                    if source[start : end].isspace():
                        state_stack[-1] = ParserState.TREE_VALUE
                        register = None
                        continue
                    raise errors.MacroSyntaxError(idx, source, "implicit concatenation is no longer supported (try [concat] or [first])")
                elif char == "\\":
                    state_stack.append(ParserState.STRING_ESC)
            elif state == ParserState.TREE_BOUNDARY:
                if char == "/":
                    state_stack[-1] = ParserState.TREE_VALUE
                elif char == "]":
                    register = tree_stack.pop()
                    state_stack.pop()
                elif char is None:
                    raise errors.MacroSyntaxError(tree_stack[-1].start, source, "unbalanced [")
                elif not char.isspace():
                    raise errors.MacroSyntaxError(idx, source, "implicit concatenation is no longer supported (try [concat] or [first])")
            elif state == ParserState.ROOT_TREE:
                register.end = idx
                register.source = source[register.start : register.end]
                root.append(register)
                state_stack[-1] = ParserState.ROOT_VALUE
                continue
            elif state == ParserState.TREE_TREE:
                register.end = idx
                register.source = source[register.start : register.end]
                tree_stack[-1].arguments.append(register)
                state_stack[-1] = ParserState.TREE_BOUNDARY
                continue
            else:
                raise Exception(f"unhandled parser state {state}")
            idx += 1
        else:
            stack = str([s.name for s in state_stack])
            if len(stack) > 200:
                stack = stack[:97] + "..." + stack[-100:]
            raise AssertionError(
                "Parser got stuck in an infinite loop! This is a severe bug, contact the bot owners ASAP with the source and the following data.\n"
                "Diagnostic data:\n"
                f"- state_stack: `" + stack + "`\n"
                f"- index: `{idx}`\n"
                f"- register: `{str(register)[:200]}`"
            )
        return root

    def check_expansions(self, name, tree):
        if self.expansions > constants.MACRO_LIMIT:
            raise errors.MacroRuntimeError(name, tree, f"reached execution limit of {constants.MACRO_LIMIT} macros\n(builtins contribute 0.1 to this counter)")
            
    def global_registry(self):
        return VariableRegistry(
            self.builtins, parent = VariableRegistry(
                self.bot.macros, mutable = False
            ), mutable = False
        )

    # THIS FUNCTION CANNOT MUTATE `tree`
    # `tree` MAY BE FROM THE DATABASE
    # DO NOT MUTATE `tree` OR EVERYTHING WILL EXPLODE
    async def evaluate_tree(self, tree: Any, vars: VariableRegistry = None) -> Any:
        if vars is None:
            vars = self.global_registry().branch()
        if not isinstance(tree, MacroTree):
            return tree
        name = tree.arguments[0]
        macro = None
        if isinstance(name, AbstractMacro):
            macro = name
        else:
            if isinstance(name, MacroTree):
                try:
                    name = await self.evaluate_tree(name, vars.branch())
                    if not isinstance(name, AbstractMacro):
                        name = str(name)
                except Exception as err:
                    raise errors.MacroRuntimeError(name, tree, f"failed to evaluate macro name", cause = err)
            if isinstance(name, AbstractMacro):
                macro = name
            else:
                macro = vars[str(name)]
                if macro is None:
                    raise errors.MacroRuntimeError(name, tree, f"macro does not exist")
                if not isinstance(macro, AbstractMacro):
                    raise errors.MacroRuntimeError(name, tree, f"cannot call value of type {type(macro).__name__}")
        
        self.expansions += macro.expansion_count()
        self.check_expansions(name, tree)
        
        try:
            return await macro.eval(self, vars, tree.arguments[1:])
        except Exception as err:
            raise errors.MacroRuntimeError(name, tree, f"failed to evaluate macro", cause = err)

            
    async def evaluate_forest(
        self, forest: list[MacroTree | Any], *, vars: VariableRegistry = None
    ) -> str:
        if vars is None:
            vars = self.global_registry().branch()
        if isinstance(vars, dict):
            vars = VariableRegistry(vars, parent = self.global_registry())
        assert forest is not None, "macro has no parsed forest set and is probably broken on beta"
        try:
            if len(forest) == 0:
                return None
            elif len(forest) == 1:
                return await self.evaluate_tree(forest[0], vars)
        except TypeError: # no len
            return await self.evaluate_tree(forest, vars)
        res = io.StringIO()
        for tree in forest:
            if isinstance(tree, MacroTree):
                tree = await self.evaluate_tree(tree, vars)
            if tree is not None:
                res.write(str(tree))
        return res.getvalue()


async def setup(bot: Bot):
    bot.macro_handler = MacroCog(bot)
