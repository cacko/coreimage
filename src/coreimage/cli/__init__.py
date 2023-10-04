from pathlib import Path
import sys
import click
from typing import Optional
import logging
from pyfiglet import Figlet
from coreimage.cli.interactive.menu import Menu
from coreimage.cli.interactive.models import TaskIcon
from coreimage.organise import Concat
from coreimage.cli.interactive.items import ConcatQuery, MenuItem, QueryTask
from coreimage.version import __version__
from coreimage.terminal import get_kitty_image
from coreimage.qrcode import get_qrcode, get_geo


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
            ConcatQuery(text="Concat", task_icon=TaskIcon.CONCAT, cmd=cli_concat),
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
@click.option("-o", "--savepath", default=".")
@click.pass_context
def cli_concat(
    ctx: click.Context,
    path: list[str],
    savepath: str,
):
    outpath, hash = Concat(Path(savepath)).concat_from_paths([Path(p) for p in path])
    output(f"-> {outpath} / {hash}")


@cli.command("icat", short_help="icat")
@click.argument("path")
@click.pass_context
def cli_icat(
    ctx: click.Context,
    path: str,
):
    ip = Path(path)
    assert ip.exists()
    assert ip.is_file()
    output = get_kitty_image(image_path=ip)
    print(output)


@cli.command("qrcode", short_help="qrcode")
@click.argument("data", nargs=-1)
@click.option("-o", "--output")
@click.option("--size", default=16)
@click.option("--border", default=1)
@click.pass_context
def cli_qrcode(
    ctx: click.Context,
    data: list[str],
    output: str,
    size: int,
    border: int,
):
    out_path = Path(output)
    assert out_path.parent.exists()
    code_image = get_qrcode(
        " ".join(data),
        box_area=size,
        border=border
    )
    code_image.save(out_path.as_posix())
    output = get_kitty_image(image=code_image)
    print(output)


@cli.command("qrgeo", short_help="qrgeo", context_settings={"ignore_unknown_options": True})
@click.argument("lat", type=float)
@click.argument("lng", type=float)
@click.option("-o", "--output")
@click.option("--size", default=16)
@click.option("--border", default=1)
@click.pass_context
def cli_qrgeo(
    ctx: click.Context,
    lat: float,
    lng: float,
    output: str,
    size: int,
    border: int,
):
    out_path = Path(output)
    assert out_path.parent.exists()
    code_image = get_geo(
        lat, lng,
        box_area=size,
        border=border
    )
    code_image.save(out_path.as_posix())
    output = get_kitty_image(image=code_image, height=20)
    print(output)


def run():
    try:
        cli()
    except AssertionError:
        pass


if __name__ == "__main__":
    run()
