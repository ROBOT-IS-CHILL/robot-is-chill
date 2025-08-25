import math
import re
from random import random, seed
from cmath import log
from functools import reduce
from typing import Optional, Callable
import json
import time
import base64
import zlib
import textwrap
import asyncio
from faststring import MString
import cython

from .. import constants, errors
from ..types import Bot, BuiltinMacro, TilingMode
from ..stringview import StringView

from typing import Tuple


from ..cythonic import find_macros

class VariableRegistry:
    def __init__(self):
        self.inner = {}

    def __getitem__(self, name):
        return self.inner[name]

    def __setitem__(self, name, value):
        assert value <= constants.MAX_MACRO_VAR_SIZE, f"tried to set variable {name} to value larger than {constants.MAX_MACRO_SIZE}"

class MacroCog:

    def __init__(self, bot: Bot):
        self.bot = bot
        self.variables = VariableRegistry()
        self.builtins: dict[str, BuiltinMacro] = {}
        self.found = 0

        def builtin(name: str):
            def wrapper(func: Callable):
                assert func.__doc__ is not None, f"missing docstring for builtin {name}"

                doc = func.__doc__.strip()
                doc = textwrap.dedent(doc)
                doc = doc.replace('\n\n', '\0').replace('\n', ' ').replace('\0', '\n')
                self.builtins[name] = BuiltinMacro(doc, func)
                return func

            return wrapper

        def strnum(v):
            if type(v) is float and v % 1 == 0:
                return str(int(v))
            return str(v)

        @builtin("to_float")
        def to_float(v):
            """Casts a value to a float."""
            if "j" in v:
                return complex(v)
            if v.startswith("0b"):
                return int(v[2:], base = 2)
            if v.startswith("0o"):
                return int(v[2:], base = 8)
            if v.startswith("0x"):
                return int(v[2:], base = 16)
            return float(v)


        @builtin("to_boolean")
        def to_boolean(v: str):
            """Casts a value to a boolean."""
            if v in ("false", "0", "False", "0.0", "0.0+0.0j", ""):
                return False
            return True

        @builtin("add")
        def add(*args: str):
            """Sums all inputs."""
            return strnum(reduce(lambda x, y: x + to_float(y), args, 0))

        @builtin("is_number")
        def is_number(value: str):
            """Checks if a value is a number."""
            try:
                to_float(value)
                return "true"
            except (TypeError, ValueError):
                return "false"

        @builtin("pow")
        def pow_(a: str, b: str):
            """Raises a value to another value."""
            a, b = to_float(a), to_float(b)
            return strnum(a ** b)

        @builtin("log")
        def log_(x: str, base: str | None = None):
            """Takes the natural log of a value, or with an optional second argument, a specified base."""
            x = to_float(x)
            if base is None:
                return strnum(log(x))
            else:
                base = to_float(base)
                return strnum(log(x, base))

        @builtin("real")
        def real(value: str):
            """Gets the real component of a complex value."""
            value = to_float(value) + 0j
            return strnum(value.real)

        @builtin("imag")
        def imag(value: str):
            """Gets the imaginary component of a complex value."""
            value = to_float(value) + 0j
            return strnum(value.imag)

        @builtin("rand")
        def rand(seed_: str | None = None):
            """Gets a random value, optionally with a seed."""
            if seed_ is not None:
                seed_ = to_float(seed_)
                assert isinstance(seed_, float), "Seed cannot be complex"
                seed(seed_)
            return strnum(random())

        @builtin("subtract")
        def subtract(a: str, b: str):
            """Subtracts a value from another."""
            a, b = to_float(a), to_float(b)
            return strnum(a - b)

        @builtin("hash")
        def hash_(value: str):
            """Gets the hash of a value."""
            return strnum(hash(value))

        @builtin("replace")
        def replace(value: str, *args: str):
            """
            Uses regex to replace patterns in a string with other strings.

            Example: `[replace/baba/a/e/b/k]` -> `keke`
            """
            assert len(args) % 2 == 0, "replace must have an odd number of arguments"
            for i in range(0, len(args), 2):
                pattern, replacement = args[i], args[i+1]
                value = re.sub(pattern, replacement, value)
            return value
        
        @builtin("ureplace")
        def ureplace(value: str, *args: str):
            r"""Uses regex to replace patterns in a string with other strings.
            This version unescapes supplied patterns.

            Example: `[ureplace/baba keke    me/\\s+/_]` -> `baba_keke_me`
            """
            assert len(args) % 2 == 0, "replace must have an odd number of arguments"
            for i in range(0, len(args), 2):
                pattern, replacement = args[i], args[i+1]
                value = re.sub(unescape(pattern), replacement, value)
            return value

        @builtin("sequence")
        def sequence(pattern, start, end, string, separator: str = ""):
            r"""
            Repeats `string` `end-start` times, replacing `pattern` in each
            with a number ranging from `start` to `end`, optionally separated by `separator`.

            Examples:
            
            > `[sequence/@/1/5/(@)/,]` -> `(1),(2),(3),(4),(5)`
            > `[sequence/@/1/3/@]` -> `123`
            """
            s = []
            for i in range(int(to_float(start)), int(to_float(end)) + 1):
                s.append(string.replace(pattern, str(i)))
            return separator.join(s)

        @builtin("for")
        def _for(lst, delimiter, idx_pat, item_pat, string, separator: str = ""):
            r"""
            Repeats `string` for each element in the list created by splitting `list` by `delimiter`,
            replacing `idx_pat` and `item_pat` in each with the item index and item,
            optionally separated by `separator`.

            Example:

            > `[for/a,b,c/,/#/@/#:@/,]` -> `0:a,1:b,2:c`
            """
            s = []
            for (i, val) in enumerate(lst.split(delimiter)):
                s.append(string.replace(idx_pat, str(i)).replace(item_pat, val))
            return separator.join(s)

        @builtin("multiply")
        def multiply(*args: str):
            """Multiplies all inputs."""
            return strnum(reduce(lambda x, y: x * to_float(y), args, 1))

        @builtin("divide")
        def divide(a: str, b: str):
            """Divides a value by another value."""
            a, b = to_float(a), to_float(b)
            try:
                return strnum(a / b)
            except ZeroDivisionError:
                if type(a) is complex:
                    return "nan"
                elif a > 0:
                    return "inf"
                elif a < 0:
                    return "-inf"
                else:
                    return "nan"

        @builtin("mod")
        def mod(a: str, b: str):
            """Takes the modulus of a value."""
            a, b = to_float(a), to_float(b)
            try:
                return strnum(a % b)
            except ZeroDivisionError:
                if a > 0:
                    return "inf"
                elif a < 0:
                    return "-inf"
                else:
                    return "nan"

        @builtin("int")
        def int_(value: str, base: str = "10"):
            """Converts a value to an integer, optionally with a base."""
            try:
                return str(int(value, base=int(to_float(base))))
            except (ValueError, TypeError):
                return str(int(to_float(value)))

        @builtin("hex")
        def hex_(value: str):
            """Converts a value to hexadecimal."""
            return str(hex(int(to_float(value))))

        @builtin("oct")
        def oct_(value: str):
            """Converts a value to octal."""
            return str(oct(int(to_float(value))))

        @builtin("bin")
        def bin_(value: str):
            """Converts a value to binary."""
            return str(bin(int(to_float(value))))

        @builtin("chr")
        def chr_(value: str):
            """Gets a character from a unicode codepoint."""
            self.found += 1
            return str(chr(int(to_float(value))))

        @builtin("ord")
        def ord_(value: str):
            """Gets the unicode codepoint of a character."""
            return str(ord(value))

        @builtin("len")
        def len_(value: str):
            """Gets the length of a string."""
            return str(len(value))

        @builtin("split")
        def split(value: str, delim: str, index: str):
            """Splits a value by a delimiter, then returns an index into the list of splits."""
            index = int(to_float(index))
            return value.split(delim)[index]

        @builtin("if")
        def if_(*args: str):
            """
            Decides between arguments to take the form of with preceding conditions,
            with an ending argument that is taken if none else are.
            """

            assert len(args) >= 3, "must have at least three arguments"
            assert len(args) % 2 == 1, "must have at an odd number of arguments"
            conditions = args[::2]
            replacements = args[1::2]
            for (condition, replacement) in zip(conditions, replacements):
                if to_boolean(condition):
                    return replacement
            return conditions[-1]

        @builtin("equal")
        def equal(a: str, b: str):
            """Checks if two strings are equal."""
            return str(a == b).lower()

        @builtin("less")
        def less(a: str, b: str):
            """Checks if a value is less than another."""
            a, b = to_float(a), to_float(b)
            return str(a < b).lower()

        @builtin("not")
        def not_(value: str):
            """Logically negates a boolean."""
            return str(not to_boolean(value)).lower()

        @builtin("and")
        def and_(*args: str):
            """Takes the boolean and of all inputs."""
            return str(reduce(lambda x, y: x and to_boolean(y), args, True)).lower()

        @builtin("or")
        def or_(*args: str):
            """Takes the boolean or of all inputs."""
            return str(reduce(lambda x, y: x or to_boolean(y), args, False)).lower()
        
        @builtin("error")
        def error(_message: str = "<unspecified>"):
            """Raises an error with a specified message."""
            raise errors.CustomMacroError(f"custom error: {_message}")

        @builtin("assert")
        def assert_(value: str, message: str):
            """If the first argument doesn't evaluate to true, errors with a specified message."""
            if not to_boolean(value):
                raise errors.CustomMacroError(f"assertion failed: {message}")
            return ""

        @builtin("slice")
        def slice_(string: str, start: str | None = None, end: str | None = None, step: str | None = None):
            """Slices a string."""
            start = int(to_float(start)) if start is not None and len(start) != 0 else None
            end = int(to_float(end)) if end is not None and len(end) != 0 else None
            step = int(to_float(step)) if step is not None and len(step) != 0 else None
            slicer = slice(start, end, step)
            return string[slicer]
        
        @builtin("find")
        def find(string: str, substring: str, start: str | None = None, end: str | None = None):
            """Returns the index of the second argument in the first, optionally between the third and fourth."""
            if start is not None:
                start = int(start)
            if end is not None:
                end = int(end)
            return str(string.index(substring, start, end))
        
        @builtin("count")
        def count(string: str, substring: str, start: str | None = None, end: str | None = None):
            """
            Returns the number of occurences of the second argument in the first,
            optionally between the third and fourth arguments.
            """
            if start is not None:
                start = int(start)
            if end is not None:
                end = int(end)
            return string.count(substring, start, end)
        
        @builtin("join")
        def join(joiner: str, *strings: str):
            """Joins all arguments with the first argument."""
            return joiner.join(strings)

        @builtin("store")
        def store(name: str, value: str):
            """Stores a value in a variable."""
            self.variables[name] = bytearray(value, "utf-8")
            return ""

        @builtin("get")
        def get(name: str, value: str):
            """Gets the value of a variable, or a default."""
            try:
                return load(name)
            except KeyError:
                store(name, value)
                return value

        @builtin("load")
        def load(name):
            """Gets the value of a variable, erroring if it doesn't exist."""
            return self.variables[name].decode("utf-8", errors = "replace")

        @builtin("drop")
        def drop(name):
            """Deletes a variable."""
            del self.variables[name]
            return ""

        @builtin("is_stored")
        def is_stored(name):
            """Checks if a variable is stored."""
            return str(name in self.variables).lower()
        
        @builtin("variables")
        def varlist():
            """Returns all variables as a JSON object."""
            return json.dumps(self.variables, separators=(",", ":")).replace("[", "\\[").replace("]", "\\]")

        @builtin("repeat")
        def repeat(amount: str, string: str, joiner: str = ""):
            """Repeats the second argument N times, where N is the first argument, optionally joined by the third."""
            # Allow floats, rounding up, for historical reasons
            amount = max(math.ceil(float(amount)), 0)
            return joiner.join([string] * amount)

        @builtin("concat")
        def concat(*args):
            """Concatenates all arguments into one string."""
            return "".join(args)

        @builtin("unescape")
        def unescape(string: str):
            r"""Unescapes a string, replacing `\\` with `\`, `\/` with `/`, `\[` with `[`, and `\]` with `]`."""
            self.found += 1
            return string.replace(r"\/", "/").replace(r"\[", "[").replace(r"\]", "]").replace(r"\\", "\\")

        @builtin("json.get")
        def jsonget(data: str, key: str):
            """Gets a value from a JSON object."""
            data = data.replace("\\[", "[").replace("\\]", "]")
            assert len(data) <= constants.MAX_MACRO_VAR_SIZE, f"json data must be at most {constants.MAX_MACRO_VAR_SIZE} characters long"
            data = json.loads(data)
            assert isinstance(data, (dict, list)), "json must be an array or an object"
            if isinstance(data, list):
                key = int(key)
            return json.dumps(data[key]).replace("[", "\\[").replace("]", "\\]")

        @builtin("json.set")
        def jsonset(data: str, key: str, value: str):
            """Sets a value in a JSON object."""
            assert len(data) <= constants.MAX_MACRO_VAR_SIZE, f"json data must be at most {constants.MAX_MACRO_VAR_SIZE} characters long"
            assert len(value) <= constants.MAX_MACRO_VAR_SIZE, f"json data must be at most {constants.MAX_MACRO_VAR_SIZE} characters long"
            data = data.replace("\\[", "[").replace("\\]", "]")
            value = value.replace("\\[", "[").replace("\\]", "]")
            data = json.loads(data)
            assert isinstance(data, (dict, list)), "json must be an array or an object"
            value = json.loads(value)
            if isinstance(data, list):
                key = int(key)
            data[key] = value
            return json.dumps(data).replace("[", "\\[").replace("]", "\\]")

        @builtin("json.remove")
        def jsonremove(data: str, key: str):
            """Removes a value from a JSON object."""
            assert len(data) <= constants.MAX_MACRO_VAR_SIZE, f"json data must be at most {constants.MAX_MACRO_VAR_SIZE} characters long"
            data = data.replace("\\[", "[").replace("\\]", "]")
            data = json.loads(data)
            assert isinstance(data, (dict, list)), "json must be an array or an object"
            if isinstance(data, list):
                key = int(key)
            del data[key]
            return json.dumps(data).replace("[", "\\[").replace("]", "\\]")

        @builtin("json.len")
        def jsonlen(data: str):
            """Gets the length of a JSON object."""
            assert len(data) <= constants.MAX_MACRO_VAR_SIZE, f"json data must be at most {constants.MAX_MACRO_VAR_SIZE} characters long"
            data = data.replace("\\[", "[").replace("\\]", "]")
            data = json.loads(data)
            assert isinstance(data, (dict, list)), "json must be an array or an object"
            return len(data)

        @builtin("json.append")
        def jsonappend(data: str, value: str):
            """Appends a value to a JSON array."""
            assert len(data) <= constants.MAX_MACRO_VAR_SIZE, f"json data must be at most {constants.MAX_MACRO_VAR_SIZE} characters long"
            assert len(value) <= constants.MAX_MACRO_VAR_SIZE, f"json data must be at most {constants.MAX_MACRO_VAR_SIZE} characters long"
            data = data.replace("\\[", "[").replace("\\]", "]")
            value = value.replace("\\[", "[").replace("\\]", "]")
            data = json.loads(data)
            assert isinstance(data, list), "json must be an array"
            value = json.loads(value)
            data.append(value)
            return json.dumps(data).replace("[", "\\[").replace("]", "\\]")

        @builtin("json.insert")
        def jsoninsert(data: str, index: str, value: str):
            """Inserts a value into a JSON array at an index."""
            assert len(data) <= constants.MAX_MACRO_VAR_SIZE, f"json data must be at most {constants.MAX_MACRO_VAR_SIZE} characters long"
            assert len(value) <= constants.MAX_MACRO_VAR_SIZE, f"json data must be at most {constants.MAX_MACRO_VAR_SIZE} characters long"
            data = data.replace("\\[", "[").replace("\\]", "]")
            value = value.replace("\\[", "[").replace("\\]", "]")
            data = json.loads(data)
            assert isinstance(data, list), "json must be an array"
            value = json.loads(value)
            index = int(index)
            data.insert(index, value)
            return json.dumps(data).replace("[", "\\[").replace("]", "\\]")
        
        @builtin("json.keys")
        def jsonkeys(data: str):
            """Gets the keys of a JSON object as a JSON array."""
            assert len(data) <= constants.MAX_MACRO_VAR_SIZE, f"json data must be at most {constants.MAX_MACRO_VAR_SIZE} characters long"
            data = data.replace("\\[", "[").replace("\\]", "]")
            data = json.loads(data)
            assert isinstance(data, dict), "json must be an object"
            return json.dumps(list(data.keys())).replace("[", "\\[").replace("]", "\\]")
        
        @builtin("unixtime")
        def unixtime():
            """Returns the current Unix timestamp, or the number of seconds since midnight on January 1st, 1970 in UTC."""
            return str(time.time())
        
        @builtin("try")
        def try_(code: str):
            """Runs some escaped MacroScript code. Returns two slash-seperated arguments: if the code errored, and the output/error message (depending on whether it errored.)"""
            self.found += 1
            try:
                result, _ = self.parse_macros(unescape(code), None, init = False)
            except errors.FailedBuiltinMacro as e:
                return f"false/{e.message}"
            except AssertionError as e:
                return f"false/{e}"
            return f"true/{result}"
        
        @builtin("lower")
        def lower(text: str):
            """Converts a string into lowercase."""
            return text.lower()
        
        @builtin("upper")
        def upper(text: str):
            """Converts a string into uppercase."""
            return text.upper()
        
        @builtin("title")
        def title(text: str):
            """Converts a string into title case."""
            return text.title()

        @builtin("base64.encode")
        def base64encode(*args: str):
            """Encodes a string as base64."""
            assert len(args) >= 1, "base64.encode macro must receive 1 or more arguments"
            string = reduce(lambda x, y: str(x) + "/" + str(y), args)
            text_bytes = string.encode('utf-8')
            base64_bytes = base64.b64encode(text_bytes)
            return base64_bytes.decode('utf-8')
        
        @builtin("base64.decode")
        def base64decode(*args: str):
            """Decodes a base64 string."""
            assert len(args) >= 1, "base64.decode macro must receive 1 or more arguments"
            string = reduce(lambda x, y: str(x) + "/" + str(y), args)
            base64_bytes = string.encode('utf-8')
            text_bytes = base64.b64decode(base64_bytes)
            return text_bytes.decode('utf-8')

        @builtin("zlib.compress")
        def zlibcompress(*args: str):
            """Compresses a string using zlib."""
            assert len(args) >= 1, "zlib.compress macro must receive 1 or more arguments"
            data = reduce(lambda x, y: str(x) + "/" + str(y), args)
            text_bytes = data.encode('utf-8')
            compressed_bytes = zlib.compress(text_bytes)
            base64_compressed = base64.b64encode(compressed_bytes)
            return base64_compressed.decode('utf-8')
        
        @builtin("zlib.decompress")
        def zlibdecompress(*args: str):
            """Decompressses a string using zlib."""
            assert len(args) >= 1, "zlib.decompress macro must receive 1 or more arguments"
            data = reduce(lambda x, y: str(x) + "/" + str(y), args)
            base64_compressed = data.encode('utf-8')
            compressed_bytes = base64.b64decode(base64_compressed)
            text_bytes = zlib.decompress(compressed_bytes)
            return text_bytes.decode('utf-8')

        @builtin("macro")
        def macro(*args: str):
            """
            Returns whether the given strings are the names of macros.
            For each macro, returns "false" if not, returns "text" if it's a text macro,
            and returns "builtin" if it's a builtin macro.
            """
            s = []
            for name in args:
                if name in self.builtins:
                    s.append("builtin")
                elif name in self.bot.macros:
                    s.append("text")
                else:
                    s.append("false")
            return "/".join(s)

        @builtin("tiles")
        def tiles(*queries: str):
            """
            Performs a search on the tile database, and returns the names of all tiles that match, separated by spaces.

            The arguments are expected to be an arbitrarily long list of search queries.

            Valid search queries are:

            - `name:<pattern>` Matches the tile name using a regex pattern
            - `tiling:<tiling mode>` Matches the tiling mode
            - `source:<string>` Matches the source the tile came from

            Note that database operations are slow, and using this too many times may time out your execution.
            It's recommended to store the output of this to a variable.
            """
            query = "1"
            params = {}
            for param in queries:
                name, pattern = param.split(":", 1)
                params[name] = pattern
            args = []
            for (name, pattern) in params.items():
                if name == "name":
                    query = query + " AND name REGEXP ?"
                    args.append(f"^{pattern}$")
                elif name == "tiling":
                    mode = TilingMode.parse(pattern)
                    assert mode is not None, f"invalid tiling mode {pattern}"
                    query = query + f" AND tiling == {+mode}"
                elif name == "source":
                    query = query + " AND source == ?"
                    args.append(pattern)
                else:
                    raise AssertionError(f"invalid query {name}")

            cur = self.bot.db.conn._conn.cursor()
            result = cur.execute("SELECT DISTINCT name FROM tiles WHERE " + query, args)
            data_rows = result.fetchall()
            return " ".join(
                str(row).replace("\\", "\\\\")
                    .replace("[", "\\[").replace("/", "\\/")
                    .replace("]", "\\]").replace(" ", "\\ ")
                    .replace("$", "\\$")
                for (row, ) in data_rows
            )

        @builtin("bytesplice")
        def bytesplice(variable, payload, start, end = None):
            """
                Splices a string `payload` into a variable's value between `start` and `end`.

                Note that unlike most other macros, `bytesplice` uses byte indices,
                which does allow indexing into the middle of a character.
            """
            end = start if end is None else end
            start = int(to_float(start))
            end = int(to_float(end))
            assert end >= start, "slice end must not be less than start"
            assert len(self.variables[variable]) - (end - start) + len(payload) \
                <= constants.MAX_MACRO_VAR_SIZE, \
                f"splice would push variable over size limit of {constants.MAX_MACRO_VAR_SIZE}"
            self.variables[variable][start:end] = bytearray(payload, "utf-8")
            return ""

        @builtin("byteset")
        def byteset(variable, index, payload):
            """
                Replaces the byte in the variable's value at the index `index` with the hexadecimal number in `payload`.

                Note that unlike most other macros, `byteset` uses byte indices,
                which does allow indexing into the middle of a character.
            """
            payload = int(payload, base = 16)
            assert payload < 256, "payload character must be within [00, FF]"
            index = int(to_float(index))
            self.variables[variable][index] = payload
            return ""

        @builtin("byteget")
        def byteget(variable, index):
            """
                Gets the hexadecimal value of the character at the index `index` in the given variable.

                Note that unlike most other macros, `byteindex` uses byte indices,
                which does allow indexing into the middle of a character!
            """
            return f"{self.variables[variable][int(to_float(index))]:02x}"

        @builtin("argslice")
        def argslice(sl, *args):
            """
                Gets a slice of the given arguments.

                `sl` is of the form `<start>[:<stop>[:<step>]]`.
            """
            return "/".join(args[slice(*(
                None if i == "" else (
                    j := int(to_float(i)),
                    j-1 if j > 0 else j
                )[1] for i in sl.split(":")))])

        self.builtins = dict(sorted(self.builtins.items(), key=lambda tup: tup[0]))

    def parse_macros(self, objects: str, debug: list | None = None, macros=None, cmd="x", init=True) -> tuple[Optional[str], Optional[list[str]]]:
        if init:
            self.variables = {}
            self.found = 0
        if macros is None:
            macros = self.bot.macros

        # Stack of prefixes, targets, and suffixes
        result_stack = [["", objects, ""]]

        # a := [b][c], b := [c][c], [c] := !
        # (, foo[a]bar, )
        # (, foo[a]bar, ) (foo, [b][c], bar)
        # (, foo[a]bar, ) (foo, [b][c], bar) (, [c][c], [c])
        # (, foo[a]bar, ) (foo, [b][c], bar) (, [c][c], [c]) (, !, [c]) Concat pre+res+suf to -1.res
        # (, foo[a]bar, ) (foo, [b][c], bar) (, ![c], [c])
        # (, foo[a]bar, ) (foo, [b][c], bar) (, ![c], [c]) (, !, )
        # (, foo[a]bar, ) (foo, [b][c], bar) (, !!, [c])
        # (, foo[a]bar, ) (foo, !![c], bar)
        # (, foo[a]bar, ) (foo, !![c], bar), (!!, !, )
        # (, foo[a]bar, ) (foo, !!!, bar)
        # (, foo!!!bar, )

        while True:
            target = result_stack[-1][1]
            start, end = find_macros(target)
            if start == -1:
                if len(result_stack) == 1:
                    result = result_stack[0][1]
                    break
                pre, res, suf = result_stack.pop()
                result_stack[-1][1] = str(pre) + str(res) + str(suf)
                continue
            if debug:
                debug.append(f"[Step {self.found}] Macro at ({start}, {end})")
            prefix = StringView(target, 0, start)
            suffix = StringView(target, end, None)

            self.found += 1
            if debug:
                debug.append(f"[Step {self.found}] {target}")
            try:
                res = self.parse_term_macro(StringView(target, start + 1, end - 1), macros, self.found, cmd, debug)
                result_stack.append([prefix, res, suffix])
            except errors.FailedBuiltinMacro as err:
                if debug:
                    debug.append(f"[Error] Error in \"{err.raw}\": {err.message}")
                    return None
                raise err

        if debug:
            debug.append(f"[Out] {result}")
        return result

    def parse_term_macro(self, raw_variant, macros, step = 0, cmd = "x", debug = None) -> str:
        REPLACEMENT_CHAR = chr(0xFBABA)

        raw_variant = StringView(raw_variant)

        args = []
        was_escaped = False
        start = 0
        for i, c in enumerate(raw_variant.contents()):
            if was_escaped:
                was_escaped = False
            elif c == '\\':
                was_escaped = True
            elif c == '/':
                args.append(raw_variant[start : i])
                start = i + 1
        args.append(raw_variant[start:])
        if debug:
            debug.append(f"[Raw Macro] {raw_variant}")
            debug.append(f"[Arguments]")
            for arg in args:
                debug.append(f"\t\"{arg}\"")
        raw_macro, *macro_args = args
        raw_macro = str(raw_macro)
        if raw_macro in self.builtins:
            try:
                macro = self.builtins[raw_macro].function(*(str(arg) for arg in macro_args))
                self.found -= 1
            except Exception as err:
                raise errors.FailedBuiltinMacro(raw_variant, err, isinstance(err, errors.CustomMacroError))
        elif raw_macro in macros:
            macro = macros[raw_macro].value
            macro_args = [None, *macro_args]
            arg_amount = 0
            iters = None
            while iters != 0:
                iters = 0
                matches = [*re.finditer(r"\$(-?\d+|#|!)", macro)]
                mac_list = []
                last_start = len(macro)
                for match in reversed(matches):
                    iters += 1
                    arg_amount += 1
                    argument = match.group(1)
                    if argument == "#":
                        debug.append(f"[Step {step}:{arg_amount}:#] {len(macro_args) - 1} arguments")
                        infix = str(len(macro_args) - 1)
                    elif argument == "!":
                        infix = cmd
                    else:
                        argument = int(argument)
                        if argument == 0 and macro_args[0] is None:
                            macro_args[0] = "/".join(str(arg) for arg in macro_args[1:]).replace("$", REPLACEMENT_CHAR)
                        try:
                            infix = macro_args[argument]
                        except IndexError:
                            infix = REPLACEMENT_CHAR + str(argument)
                    if debug:
                        debug.append(f"[Step {step}:{arg_amount}] {macro}")
                    mac_list.append(StringView(macro, match.end(), last_start))
                    mac_list.append(infix)
                    last_start = match.start()
                mac_list.append(StringView(macro, 0, last_start))
                macro = "".join(str(s) for s in reversed(mac_list))
        else:
            raise errors.FailedBuiltinMacro(raw_variant, f"Macro `{raw_macro}` of `{raw_variant}` not found in the database!", False)
        res = macro.replace(REPLACEMENT_CHAR, "$")
        return res


async def setup(bot: Bot):
    bot.macro_handler = MacroCog(bot)
