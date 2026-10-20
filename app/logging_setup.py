"""One place that decides how log lines look."""
import logging


def setup_logging(level_name):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # these libraries are chatty at INFO level; keep only their warnings
    # (httpx2 is the HTTP library the OpenAI package uses internally)
    for noisy in ("websockets", "httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)