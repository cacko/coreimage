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
from coreimage.terminal.kitty import get_term_image
from coreimage import __version__
from coreimage.terminal import print_term_image
from coreimage.qrcode import get_qrcode
from coreimage.transform import Cropper, convert_to, remove_background
from coreimage.find import find_images
from coreimage.transform import Upscale
from coreimage.utils import IMAGE_EXT, resize_set


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
    from coreimage.terminal.kitty import get_term_image

    output = get_term_image(image_path=ip)
    print(output)


@cli.command("faces")
@click.argument("path", type=Path)
@click.pass_context
def cli_faces(
    ctx: click.Context,
    path: Path,
):
    crop = Cropper(
        path,
    )
    try:
        faces_path = crop.show_faces()
        assert faces_path
        from coreimage.terminal.kitty import get_term_image

        with get_term_image(image_path=faces_path, height=40) as t_image:
            print(t_image)
    except AssertionError:
        logging.error("No faces found")


@cli.command("facecrop")
@click.argument("path", type=Path)
@click.option("-o", "--output", type=Path)
@click.option("-w", "--width", type=int, default=640)
@click.option("-h", "--height", type=int, default=640)
@click.option("-i", "--face_index", type=int)
@click.pass_context
def cli_cropface(
    ctx: click.Context,
    path: Path,
    width: int,
    height: int,
    output: Optional[Path] = None,
    face_index: Optional[int] = -1,
):
    crop = Cropper(
        path,
        width=width,
        height=height,
    )
    try:
        faces_path = crop.show_faces()
        assert faces_path
        with get_term_image(image_path=faces_path, height=20) as t_image:
            print(t_image)
        crop_path = crop.crop(face_idx=face_index, out=output)
        assert crop_path
        with get_term_image(image_path=crop_path, height=20) as t_image:
            print(t_image)
    except AssertionError:
        logging.error("No faces found")


@cli.command("qrcode", short_help="qrcode")
@click.argument("data", nargs=-1)
@click.option("-o", "--output")
@click.option("--size", default=16)
@click.option("--border", default=4)
@click.pass_context
def cli_qrcode(
    ctx: click.Context,
    data: list[str],
    size: int,
    border: int,
    output: Optional[str] = None,
):
    code_image = get_qrcode(data, box_area=size, border=border)
    try:
        assert output
        out_path = Path(output)
        assert out_path.parent.exists()
        code_image.save(out_path.as_posix())
    except AssertionError:
        pass
    print_term_image(image=code_image, height=20)


@cli.command("convert", short_help="convert")
@click.argument("path", type=Path)
@click.option("-f", "--format", type=click.Choice[IMAGE_EXT.supported()], required=True)
@click.option("-o", "--output", type=Path)
def cli_convert(path: Path, format: str, output: Optional[Path] = None):
    for img_path in find_images([path]):
        try:
            o_root = output if output else img_path.parent
            converted_path = convert_to(img_path, format=format)
            result_path = o_root / f"{img_path.stem}.{format}"
            converted_path.rename(result_path)
            logging.info(
                f"Converted from {img_path.suffix} to {format} / {converted_path}"
            )
        except Exception as e:
            logging.exception(e)


@cli.command("upscale")
@click.argument("path", type=Path)
@click.option("-o", "--output", type=Path)
@click.option("-s", "--scale", type=int, default=2)
@click.option("-rz", "--resize", type=str)
@click.option("-f", "--format", type=click.Choice[IMAGE_EXT.supported()])
def cli_upscale(
    path: Path,
    scale: int,
    output: Optional[Path] = None,
    resize: Optional[str] = None,
    format: Optional[str] = None,
):
    for img_path in find_images([path]):
        try:
            upscaled_path = Upscale.upscale(
                src_path=img_path, dst_path=output, scale=scale, resize=resize_set(resize)
            )
            assert upscaled_path
            logging.info(f"Upscaled result / {upscaled_path}")
            print_term_image(image_path=upscaled_path, height=30)
        except Exception as e:
            logging.exception(e)

@cli.command("remove-background")
@click.argument("path", type=Path)
@click.option("-o", "--output", type=Path)
def cli_remove_background(
    path: Path,
    output: Optional[Path] = None,
):
    for img_path in find_images([path]):
        try:
            nobg_image = remove_background(
                image=img_path
            )
            assert nobg_image
            o_root = output if output else img_path.parent
            final_path = o_root / f"nobg_{img_path.stem}.png"
            nobg_image.save(final_path.as_posix())
            logging.info(f"Removed background / {final_path}")
            print_term_image(image_path=final_path, height=30)
        except Exception as e:
            logging.exception(e)


def run():
    try:
        cli()
    except AssertionError:
        pass


if __name__ == "__main__":
    run()
