
def setup_logging():
    
    import logging

    file_handler = logging.FileHandler('info.log',mode='a')
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger('main_log')
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO) 


def cprint(msg: str, lvl: str = "info") -> None:
    """
    Print message to the console at the desired logging level.

    Arguments:
        msg (str): Message to print.
        lvl (str): Logging level between "debug", "info", "warning", "error" and "critical".
                   The default value is "info".
    """
    from logging import getLogger
    
    # Use the package level logger.
    logger = getLogger("main_log")

    # Log message as info level.
    if lvl == "debug":
        logger.debug(msg=msg)
    elif lvl == "info":
        logger.info(msg=msg)
    elif lvl == "warning":
        logger.warning(msg=msg)
    elif lvl == "error":
        logger.error(msg=msg)
    elif lvl == "critical":
        logger.critical(msg=msg)
    else:
        pass


if __name__ == "__main__":
    setup_logging()