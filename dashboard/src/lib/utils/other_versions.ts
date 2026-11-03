/**
 * Devices on the network under this cluster's name that can't join it,
 * because they run another version of exo (GET /v1/devices-on-other-versions).
 */

export type DeviceOnOtherVersion = {
  address: string;
  /** null: an exo from before wire versions */
  wireVersion: number | null;
};

/**
 * How often to ask. exo lists a device until 30 s after the last Hello it
 * heard from it (one a second), so asking every 10 s shows a device within
 * 10 s of its first Hello, and drops one within 40 s of its last.
 */
export const OTHER_VERSIONS_POLL_MS = 10_000;

/** The devices in the endpoint's answer; anything unexpected reads as none. */
export function parseDevicesOnOtherVersions(
  body: unknown,
): DeviceOnOtherVersion[] {
  if (!body || typeof body !== "object") return [];
  const devices = (body as { devices?: unknown }).devices;
  if (!Array.isArray(devices)) return [];
  return devices.flatMap((device: unknown) => {
    if (!device || typeof device !== "object") return [];
    const { address, wireVersion } = device as Record<string, unknown>;
    if (typeof address !== "string") return [];
    return [
      {
        address,
        wireVersion: typeof wireVersion === "number" ? wireVersion : null,
      },
    ];
  });
}

/** "fe80::1 runs exo wire version 2", or "... runs an older exo". */
export function describeDevice(device: DeviceOnOtherVersion): string {
  const version =
    device.wireVersion === null
      ? "an older exo"
      : `exo wire version ${device.wireVersion}`;
  return `${device.address} runs ${version}`;
}

export async function fetchDevicesOnOtherVersions(): Promise<
  DeviceOnOtherVersion[]
> {
  try {
    const response = await fetch("/v1/devices-on-other-versions");
    if (!response.ok) return [];
    return parseDevicesOnOtherVersions(await response.json());
  } catch {
    return [];
  }
}
