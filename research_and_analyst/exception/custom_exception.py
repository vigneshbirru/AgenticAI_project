import sys
import traceback
from typing import Optional


class ResearchAnalystException(Exception):
    """
    Project-level exception wrapper that captures useful traceback context.
    """

    def __init__(self, error_message: object, error_details: Optional[object] = None):
        norm_msg = str(error_message)

        if error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        elif isinstance(error_details, BaseException):
            exc_type, exc_value, exc_tb = type(error_details), error_details, error_details.__traceback__
        else:
            # Fallback: treat as "current context"
            exc_type, exc_value, exc_tb = sys.exc_info()

        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        self.error_message = norm_msg

        if exc_type and exc_tb:
            self.traceback_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = ""

        super().__init__(self.__str__())

    def __str__(self) -> str:
        base = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        return base

    def __repr__(self) -> str:
        return (
            "ResearchAnalystException("
            f"file={self.file_name!r}, line={self.lineno}, message={self.error_message!r})"
        )

