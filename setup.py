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
        "autoflake8>=0.4.0",
        "autopep8>=2.0.2",
        "click>=8.1.3",
        "flake8>=6.0.0",
        "mccabe>=0.7.0",
        "opencv-python>=4.7.0.72",
        "pillow>=9.5.0",
        "pycodestyle>=2.10.0",
        "pyflakes>=3.0.1",
        "semver>=3.0.0",
        "pygments>=2.15.1",
        "questionary>=1.10.0",
        "pyfiglet>=0.8.post1",
        "pygments>=2.15.1",
        "emoji>=2.4.0",
        "pydantic>=1.10.8",
        "cv2-collage-v2>=0.11",
        "corelog>0.0.11",
        "corefile>=0.1.5"
    ],
    setup_requires=["wheel", "semver"],
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points="""
        [console_scripts]
        ivan=coreimage.cli:cli
    """,
)