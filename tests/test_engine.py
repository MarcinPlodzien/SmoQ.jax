"""pytest entry point: runs the engine validation suite (every primitive against dense linear algebra)."""
import pathlib
import runpy


def test_engine_against_dense_linear_algebra():
    runpy.run_path(str(pathlib.Path(__file__).with_name("engine_suite.py")), run_name="__main__")
