"""Logger minimal, sûr en Colab."""
import logging, sys
def get_logger(name: str = "forensic-synth") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h); log.setLevel(logging.INFO)
    return log
