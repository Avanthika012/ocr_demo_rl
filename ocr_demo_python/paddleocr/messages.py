from tabulate import tabulate
from termcolor import colored


# Get the logger
from logger_setup import get_logger
logger = get_logger(__name__, "preprocess.log", console_output=True)

class Messagesx():
    def __init__(self) -> None:
        self.processstart = "------- PROCESS STARTED -------"
        self.processend = "------- PROCESS COMPLETED -------"
    
    def welcome(self, task):
        message = f"Hello, Welcome to {task}"
        box_width = len(message) + 6
        border_line = "+" + "+" * box_width + "+"
        empty_line = "+{}+".format(" " * (box_width + 2))
        message_line = f"  +  {message}  + "
        
        logger.info("\n\n\n")        
        logger.info(border_line)
        logger.info(empty_line)
        logger.info(message_line)
        logger.info(empty_line)
        logger.info(border_line)
        logger.info("\n\n\n")


    #### to show tabular output on terminal 
    def create_args_table(self,args, parser,logger=logger):
        data = []
        for action in parser._actions:
            arg_name = action.dest
            if arg_name == 'help':
                continue
            arg_val = getattr(args, arg_name)
            arg_opts = ", ".join(action.option_strings)
            arg_help = action.help or ""
            data.append([arg_opts, arg_val, arg_help])

        # Convert to tabular format with colors
        colored_data = [[colored(item, 'cyan') for item in row] for row in data]
        table = tabulate(colored_data, headers=[colored("Argument", 'yellow'), colored("Received Value", 'yellow'), colored("Help", 'yellow')], tablefmt="grid")

        # Log the table
        logger.info(f"\n{table}")

# Messagesx().welcome("DATA")

