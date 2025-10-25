
def find_macros(unicode data) -> tuple[int, int]:
    cdef int start, end
    start = -1
    cdef bint was_escaped = 0
    for i, c in enumerate(data):
        end = i + 1
        if was_escaped:
            was_escaped = 0
        elif c == '\\':
            was_escaped = 1
        elif c == '[':
            start = i
        elif c == ']' and start != -1:
            return (start, end)
    return (-1, -1)
