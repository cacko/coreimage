from pathlib import Path
import sys
import click
from typing import Optional
import logging
from pyfiglet import Figlet
from coreimage.cli.interactive.menu import Menu
from coreimage.cli.interactive.models import TaskIcon
from coreimage.organise import concat
from coreimage.cli.interactive.items import ConcatQuery, MenuItem, QueryTask
from coreimage.version import __version__


def banner(txt: str, color: str = "bright_green"):
    logo = Figlet(width=120).renderText(text=txt)
    click.secho(logo, fg=color)


def output(txt: str, color="bright_blue"):
    click.secho(txt, fg=color)


def error(e: Exception, txt: Optional[str] = None):
    if not txt:
        txt = f"{e}"
    click.secho(txt, fg="bright_red", err=True)
    if e:
        logging.debug(txt, exc_info=e)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        ctx.invoke(main_menu)


@cli.command("quit")
def quit():
    """Quit."""
    output("Bye!", color="blue")
    sys.exit(0)


@cli.command("menu", short_help="My Tasks")
@click.pass_context
def main_menu(ctx: click.Context):
    try:
        click.clear()
        banner(txt=f"{__name__} {__version__}", color="bright_blue")
        menu_items = [
            ConcatQuery(text="Concat", value="concat", task_icon=TaskIcon.CONCAT, cmd=cli_concat),
        ] + [MenuItem(text="Exit", cmd=quit)]
        with Menu(menu_items, title="Select task") as item:  # type: ignore
            if isinstance(item, QueryTask):
                args = item.get_input()
                ctx.forward(item.cmd, **args)
            match item.cmd:
                case click.Command():
                    ctx.invoke(item.cmd)
    except Exception as e:
        error(e)


@cli.command("concat", short_help="concat")
@click.option("-p", "--path", multiple=True)
@click.option("-o", "--output")
@click.option("-s", "--max_size")
@click.pass_context
def cli_concat(
    ctx: click.Context,
    path: list[str],
    output: Optional[str] = None,
    max_size: Optional[int] = None
):
    concat([Path(p) for p in path], output, max_size)


def run():
    try:
        cli()
    except AssertionError:
        pass


if __name__ == "__main__":
    run()
