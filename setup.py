from pathlib import Path
import semver
from setuptools.dist import Distribution as _Distribution
from setuptools import find_packages, setup

import sys

__name__ = "coreimage"


def version():
    if len(sys.argv) > 1 and sys.argv[1] >= "bdist_wheel":
        init = Path(__file__).parent / "src" / __name__ / "version.py"
        _, v = init.read_text().split("=")
        cv = semver.VersionInfo.parse(v.strip().strip('"'))
        nv = f"{cv.bump_patch()}"
        init.write_text(f'__version__ = "{nv}"')
        return nv
    from coreimage.version import __version__

    return __version__


class Distribution(_Distribution):
    def is_pure(self):
        return True

setup(
    name=__name__,
    version=version(),
    author="cacko",
    author_email="cacko@cacko.net",
    distclass=Distribution,
    url=f"http://pypi.cacko.net/simple/{__name__}/",
    description="whatever",
    install_requires=[
        "click>=8.1.3",
        "flake8>=6.0.0",
        "mccabe>=0.7.0",
        "opencv-python>=4.7.0.72",
        "pillow>=9.5.0",
        "semver>=3.0.0",
        "pygments>=2.15.1",
        "questionary>=1.10.0",
        "pyfiglet>=0.8.post1",
        "pygments>=2.15.1",
        "emoji>=2.4.0",
        "pydantic>=2.7.1",
        "corelog>0.0.15",
        "corefile>=0.1.6",
        "corestring>=0.3.6",
        "term-image>=0.8.0.1",
        "segno>=1.5.2",
        "spandrel>0.2",
        "facenet-pytorch>=20.5.2",
        "transformers>=4.41.1",
        "torch>=2.2.2",
        "torchvision>=0.17.2"
    ],
    setup_requires=["wheel", "semver"],
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"coreimage.resources": ["*"]},
    entry_points="""
        [console_scripts]
        coreimage=coreimage.cli:cli
    """,
)
