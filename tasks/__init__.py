from pathlib import Path

__all__ = ["TASK_FILES"]

TASK_FILES = [
    Path(__file__).with_name("easy.json"),
    Path(__file__).with_name("medium.json"),
    Path(__file__).with_name("hard.json"),
]