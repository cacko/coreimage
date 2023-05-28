from typing import Optional
from typing import TypeVar
from click import Command
from prompt_toolkit.styles.pygments import style_from_pygments_cls
from pygments.styles import get_style_by_name
from questionary import Separator
from pydantic import BaseModel, Field
from emoji import emojize
from pathlib import Path

from prompt_toolkit.formatted_text import (
    FormattedText as PT_FormattedText,
    PygmentsTokens,
    to_formatted_text,
    merge_formatted_text,
    AnyFormattedText,
)
from pygments.token import Token
import questionary

from coreimage.cli.interactive.models import TaskIcon
from coreimage.utils import IMAGE_EXT


style = style_from_pygments_cls(get_style_by_name("monokai"))  # type: ignore


def keyword(s: str) -> PT_FormattedText:
    return to_formatted_text(PygmentsTokens([(Token.Keyword, s)]))


def punctuation(s: str) -> PT_FormattedText:
    return to_formatted_text(PygmentsTokens([(Token.Punctuation, s)]))


def comment(s: str) -> PT_FormattedText:
    return to_formatted_text(PygmentsTokens([(Token.Comment, s)]))


def text(s: str) -> PT_FormattedText:
    return to_formatted_text(PygmentsTokens([(Token.Text, s)]))


class MenuItem(BaseModel):
    text: str
    cmd: Command
    task_icon: TaskIcon = Field(default=TaskIcon.OFF)
    disabled: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def display(self) -> AnyFormattedText:
        return to_formatted_text(
            merge_formatted_text(
                [
                    punctuation(f"{emojize(self.task_icon.value)} "),
                    keyword(self.text),
                ]
            )
        )

    @property
    def value(self):
        return self.text


class TaskItem(MenuItem):

    class Config:
        arbitrary_types_allowed = True

    def get_input(self):
        raise NotImplementedError


class DisabledItem(MenuItem):
    @property
    def display(self):
        return comment(self.text)


class QueryTask(TaskItem):

    def get_input(self):
        raise NotImplementedError


class ConcatQuery(QueryTask):

    def get_input(self):
        answers = {}

        answers["path"] = questionary.path(
            message="images or directory with iamges",
            validate=lambda x: Path(x).is_dir() or IMAGE_EXT.endwith(x)
        ).ask()
        answers["output"] = questionary.path(
            message="OIutput directory",
            validate=lambda x: Path(x).suffix == '.csv',
            only_directories=True).ask()
        answers["max_sioze"] = questionary.text(
            message="desired heigjt of the images",
            validate=lambda x: f"{x}".isnumeric()

        )
        return answers


MT = TypeVar("MT", MenuItem, TaskItem, DisabledItem, Separator)
