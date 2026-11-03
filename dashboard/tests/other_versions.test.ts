import { test } from "node:test";
import assert from "node:assert/strict";
import {
  describeDevice,
  parseDevicesOnOtherVersions,
} from "../src/lib/utils/other_versions.ts";

test("reads the devices exo lists", () => {
  assert.deepEqual(
    parseDevicesOnOtherVersions({
      wireVersion: 1,
      devices: [
        { address: "fe80::1", wireVersion: 2 },
        { address: "fe80::2", wireVersion: null },
      ],
    }),
    [
      { address: "fe80::1", wireVersion: 2 },
      { address: "fe80::2", wireVersion: null },
    ],
  );
});

test("an unexpected answer reads as no devices", () => {
  assert.deepEqual(parseDevicesOnOtherVersions(null), []);
  assert.deepEqual(parseDevicesOnOtherVersions({ detail: "Not Found" }), []);
  assert.deepEqual(parseDevicesOnOtherVersions({ devices: [{}] }), []);
});

test("says which version a device runs", () => {
  assert.equal(
    describeDevice({ address: "fe80::1", wireVersion: 2 }),
    "fe80::1 runs exo wire version 2",
  );
  assert.equal(
    describeDevice({ address: "fe80::2", wireVersion: null }),
    "fe80::2 runs an older exo",
  );
});
