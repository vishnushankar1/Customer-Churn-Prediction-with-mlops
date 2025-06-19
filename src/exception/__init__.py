# src/exception/__init__.py

import sys
import os
from src.logger import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed error message with filename and line number.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    line_number = exc_tb.tb_lineno
    return f"Error in [{file_name}] at line [{line_number}]: {str(error)}"

class CustomException(Exception):
    """
    Custom exception class for better traceability.
    """
    def __init__(self, error: Exception, error_detail: sys):
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message
