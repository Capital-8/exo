from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.routing.wire import WIRE_VERSION, DeviceOnOtherVersion


def _client(devices: list[DeviceOnOtherVersion]) -> TestClient:
    api = object.__new__(API)
    api.app = FastAPI()
    api._devices_on_other_versions = lambda: devices  # pyright: ignore[reportPrivateUsage]
    api.app.get("/v1/devices-on-other-versions")(api.get_devices_on_other_versions)
    return TestClient(api.app)


def test_lists_the_devices_that_run_another_version() -> None:
    response = _client(
        [
            DeviceOnOtherVersion(address="fe80::1%en0", wire_version=WIRE_VERSION + 1),
            DeviceOnOtherVersion(address="fe80::2%en0", wire_version=None),
        ]
    ).get("/v1/devices-on-other-versions")
    assert response.status_code == 200
    assert response.json() == {
        "wireVersion": WIRE_VERSION,
        "devices": [
            {"address": "fe80::1%en0", "wireVersion": WIRE_VERSION + 1},
            # an exo from before wire versions
            {"address": "fe80::2%en0", "wireVersion": None},
        ],
    }


def test_is_empty_when_every_device_runs_this_version() -> None:
    response = _client([]).get("/v1/devices-on-other-versions")
    assert response.json() == {"wireVersion": WIRE_VERSION, "devices": []}
