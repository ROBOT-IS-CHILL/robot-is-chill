# From https://gist.github.com/saxbophone/e988cef9f351863f4312f2eef41a3a83

from attrs import define

@define
class StringView:
    """
    StringView implementation using minimal copying with maximum use of
    reference semantics. Creating a sub-view of an existing StringView using
    either object slicing or constructing one from another will reüse the same
    source string object, using a reference rather than a copy.
    The contents() method can similarly be used to get an iterator (Generator)
    to access the view contents sequentially without putting it all in memory
    at once.
    A brand new string object is only created if the StringView is cast to str.
    """
    __source: str
    __start: int
    __stop: int

    def __init__(self, source: str, start=0, stop=None):
        if isinstance(source, StringView):
            self.__source = source.__source
            self.__start = source.__start + start
            self.__stop = source.__stop if stop is None else min(len(self.__source), source.__start + stop)
        else:
            self.__source = source
            self.__start = start
            self.__stop = len(source) if stop is None else stop
        if self.__stop < 0:
            self.__stop = len(self.__source) + self.__stop
    def __str__(self):
        return self.__source[self.__start:self.__stop]
    def __repr__(self):
        return f'<StringView: {self.__source[self.__start:self.__stop]}>'
    # these next two methods are provided so we can produce StringViews from StringViews
    def __len__(self):
        return self.__stop - self.__start
    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step is not None:
                raise TypeError('StringView does not support step when slicing')
            return StringView(self, 0 if key.start is None else key.start, key.stop)
        """
        not only is there no point returning a StringView of length 1, it's also
        slightly less memory-intensive to just return a str of length 1...
        """
        return self.__source[key]
    def contents(self):
        """
        Returns Generator for efficient no-copy iteration over string contents
        """
        if self.__start >= len(self.__source):
            return ()
        return (self.__source[i] for i in range(self.__start, min(len(self.__source), self.__stop)))
