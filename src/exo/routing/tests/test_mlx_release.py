from exo.routing.wire import mlx_release


def test_one_builds_macs_and_linux_machines_have_the_same_mlx_release() -> None:
    # macOS installs a development build of MLX from git, Linux the published release
    mac = mlx_release("0.32.4.dev20261030+36969c09")
    linux = mlx_release("0.32.4")
    assert mac == linux == "0.32.4"


def test_another_mlx_release_differs() -> None:
    assert mlx_release("0.33.0") != mlx_release("0.32.4")
