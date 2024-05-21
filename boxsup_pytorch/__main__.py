"""Main File of boxsup_pytorch package."""

import cProfile
from logging import getLogger
from pstats import Stats

from boxsup_pytorch.core.logger import LoggingHelper
from boxsup_pytorch.launch import main


def run_script():
    """Run the main launch script of this package."""
    # Load Logging Config for this Application
    logging_helper = LoggingHelper(with_date=True)
    logging_helper.load_config()

    # Create Application logger
    train_logger = getLogger("train")

    do_profiling = False
    if do_profiling:
        train_logger.info("Run BoxSup with profiling")
        with cProfile.Profile() as pr:
            main()

        with open("profiling_stats.txt", "w") as stream:
            stats = Stats(pr, stream=stream)
            stats.strip_dirs()
            stats.sort_stats("time")
            stats.dump_stats(".prof_stats")
            stats.print_stats()
    else:
        train_logger.info("Run BoxSup without profiling")
        main()


if __name__ == "__main__":
    run_script()
