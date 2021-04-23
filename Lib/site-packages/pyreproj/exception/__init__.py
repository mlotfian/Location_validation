# -*- coding: utf-8 -*-


class Error(Exception):
    """Base class for custom exceptions."""

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class InvalidFormatError(Error):
    """
    This exception is thrown if a required parameter doesn't match the expected format or one of the expected
    formats.

    :param msg: Error message for this exception.
    :type msg: :obj:`str`
    """

    def __init__(self, msg):
        self.message = msg

    def __str__(self):
        return repr(self.message)


class InvalidTargetError(Error):
    """
    This exception is thrown if no valid target is given for a transformation, e.g. with
    :func:`~pyreproj.Reprojector.transform`.

    :param msg: Error message for this exception.
    :type msg: :obj:`str`
    """

    def __init__(self, msg):
        self.message = msg

    def __str__(self):
        return repr(self.message)
