import sys
from logger import logging

# error_message_detail function will return the error message with the line number and the python script name
def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info() # This will return the line number where the error occured
    filename = exc_tb.tb_frame.f_code.co_filename # This will return the python script name
    
    error_message = "Error in python script [{0}], line number [{1}], error message [{2}]".format(
        filename,exc_tb.tb_lineno,str(error)
    )
    return error_message
    
# CustomException class will be used to raise custom exceptions
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        
        def __str__(self):
            return self.error_message