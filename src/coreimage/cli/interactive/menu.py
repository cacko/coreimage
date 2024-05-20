import os
from typing import Optional

from prompt_toolkit.styles.pygments import style_from_pygments_cls
from pygments.styles import get_style_by_name
from questionary import Choice, select

from .items import MT, MenuItem

style = style_from_pygments_cls(get_style_by_name("monokai"))  # type: ignore


class FunctionalityNotAvailable(Exception):
    pass


class MenuMeta(type):

    _instances: dict[str, "Menu"] = {}
    _yaml: dict = {}

    def __call__(self, items: list[MT], *args, **kwds: any):
        return type.__call__(self, items, *args, **kwds)

    @property
    def yaml(cls):
        return cls._yaml[cls]

    def createMenuItem(cls, v):
        raise NotImplementedError


class Menu(object, metaclass=MenuMeta):

    _items: list = []
    _title: str = "What you want?"
    _parent: Optional[MenuItem] = None
    _options: dict[str, any]

    def __init__(self, items: list[MT], title=None, **kwds) -> None:
        self._items = list(items)
        self._options = kwds
        if title:
            self._title = title

    def __enter__(self) -> MenuItem:
        choice = None

        options = [
            Choice(
                title=x.display,  # type: ignore
                value=x.value,
                disabled=x.disabled
            )
            if isinstance(x, MenuItem)
            else x
            for x in self._items
        ]
        while not choice:
            try:
                choice = select(
                    message=self._title,
                    choices=options,
                    style=style,
                    **self._options
                ).ask()
                res = next(filter(lambda x: x.value ==
                           choice, self._items), None)
                if res:
                    return res
                raise KeyboardInterrupt
            except FunctionalityNotAvailable:
                raise NotImplementedError
        raise KeyboardInterrupt

    def __exit__(self, type, value, traceback):
        pass

    def clear(self):
        os.system("cls|clear")
