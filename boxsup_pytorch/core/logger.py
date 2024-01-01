"""Logging Module.

provides LoggingHelper class.
"""

from datetime import datetime
import logging.config
from pathlib import Path

import toml


class LoggingHelper:
    """LoggingHelper Class.

    loads and generates logging config.
    """

    def __init__(self, with_date=False) -> None:
        """Initialize LoggingHelper instance.

        Args:
            with_date (bool, optional): defines if logfile should be generated with startdate.
                Defaults to False.
        """
        config_path = Path(__file__).resolve().parent.parent / "config/log_config.toml"
        self.logging_config = toml.load(config_path)
        if with_date:
            self._logfile_with_date()

    def _logfile_with_date(self):
        """Replace the original configured logfile with itself plus date."""
        logfile = Path(self.logging_config["handlers"]["file"]["filename"])
        logfile_name = f"{logfile.stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
        new_logfile = logfile.parent / logfile_name
        self.logging_config["handlers"]["file"]["filename"] = str(new_logfile)

    def load_config(self):
        """Load logging config."""
        logging.config.dictConfig(self.logging_config)
