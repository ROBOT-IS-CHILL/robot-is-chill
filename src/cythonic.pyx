
def find_macros(unicode data, int search_start, int search_end) -> tuple[int, int]:
    cdef int start, end
    start = search_start
    cdef bint was_escaped = 0
    for i, c in enumerate(data):
        if i >= search_end:
            break
        end = i + 1
        if was_escaped:
            was_escaped = 0
        elif c == '\\':
            was_escaped = 1
        elif c == '[':
            start = i
        elif c == ']' and start is not None:
            return (start, end)
    return (-1, -1)
