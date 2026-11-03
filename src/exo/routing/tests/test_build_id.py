from exo import __version__
from exo.routing.wire import build_id


def test_the_app_names_its_own_build() -> None:
    assert build_id({"EXO_BUILD_ID": " 1.0.72 (4f1c2a9) "}) == "1.0.72 (4f1c2a9)"


def test_a_source_build_is_named_by_its_package_and_runner_versions() -> None:
    # No git commit: one lockfile on several machines is one build
    name = build_id({})
    assert name == build_id({"EXO_BUILD_ID": ""})
    assert name.startswith(f"exo {__version__}, mlx ")
    assert ", mlx-lm " in name
