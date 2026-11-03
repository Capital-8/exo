//! Which nodes may connect: those with the same namespace and the same wire version.
//!
//! The wire version is exo's `WIRE_VERSION` (src/exo/routing/wire.py). Nodes whose message
//! schemas differ can't read each other, so they must never connect. Discovery announces a
//! hash of the namespace and the wire version, and answers only Hellos with its own hash.
//! A node also knows the hash its namespace has under other wire versions, so it can tell
//! a device running another version of exo apart from an exo in another cluster.

use std::{
    collections::HashMap,
    fmt,
    net::IpAddr,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;

/// What a discovery Hello carries to say which cluster its sender is in.
pub type NamespaceHash = [u8; 8];

/// How many wire versions below and above ours a Hello is recognised from. exo bumps the wire
/// version when what nodes send each other changes, which is at most a few times a month, so
/// 256 covers years of releases either way; it costs hashing 513 short strings once at start.
const RECOGNISED_VERSIONS_EITHER_SIDE: u32 = 256;

/// How long a device on another wire version stays listed after its last Hello. A node says
/// Hello every second, so this is 30 Hellos missed in a row: the device quit or now runs our
/// version. A busy node that pauses for a few seconds, or a Wi-Fi link dropping a few
/// packets, doesn't make the notice come and go.
pub const OTHER_VERSION_FORGET_AFTER: Duration = Duration::from_secs(30);

/// The wire version a Hello in our namespace was sent with, when it isn't ours.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PeerWireVersion {
    /// An exo from before wire versions: its Hello hashes the namespace alone.
    Unversioned,
    Version(u32),
}

impl PeerWireVersion {
    /// The version as exo's Python side numbers it: `None` before wire versions.
    pub fn number(self) -> Option<u32> {
        match self {
            Self::Unversioned => None,
            Self::Version(version) => Some(version),
        }
    }
}

impl fmt::Display for PeerWireVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unversioned => write!(f, "an exo from before wire versions"),
            Self::Version(version) => write!(f, "wire version {version}"),
        }
    }
}

/// The hash a node of this namespace and wire version announces. `None` is how exo hashed
/// namespaces before wire versions. Namespaces come from the command line or the
/// environment, which can't hold a NUL, so no namespace hashes like a versioned one.
pub fn namespace_hash(namespace: &str, wire_version: Option<u32>) -> NamespaceHash {
    let hash = match wire_version {
        None => blake3::hash(namespace.as_bytes()),
        Some(version) => blake3::hash(format!("{namespace}\0wire={version}").as_bytes()),
    };
    hash.as_bytes()[..8].try_into().expect("8 is equal to 8")
}

/// This node's cluster: its namespace hash, and the hashes of its namespace under other
/// wire versions.
pub struct ClusterKey {
    hash: NamespaceHash,
    wire_version: u32,
    others: HashMap<NamespaceHash, PeerWireVersion>,
}

impl ClusterKey {
    pub fn new(namespace: &str, wire_version: u32) -> Self {
        let lowest = wire_version
            .saturating_sub(RECOGNISED_VERSIONS_EITHER_SIDE)
            .max(1);
        let highest = wire_version.saturating_add(RECOGNISED_VERSIONS_EITHER_SIDE);
        let mut others: HashMap<NamespaceHash, PeerWireVersion> = (lowest..=highest)
            .filter(|&version| version != wire_version)
            .map(|version| {
                (
                    namespace_hash(namespace, Some(version)),
                    PeerWireVersion::Version(version),
                )
            })
            .collect();
        others.insert(
            namespace_hash(namespace, None),
            PeerWireVersion::Unversioned,
        );
        Self {
            hash: namespace_hash(namespace, Some(wire_version)),
            wire_version,
            others,
        }
    }

    /// What this node announces.
    pub fn hash(&self) -> NamespaceHash {
        self.hash
    }

    pub fn wire_version(&self) -> u32 {
        self.wire_version
    }

    /// The wire version of a Hello from our namespace that runs another version, or `None`
    /// for a Hello from another namespace (or one too many versions away to recognise).
    pub fn other_version(&self, hash: &NamespaceHash) -> Option<PeerWireVersion> {
        self.others.get(hash).copied()
    }
}

/// The devices discovery heard in our namespace on another wire version, by address.
#[derive(Clone, Default)]
pub struct OtherVersions(Arc<Mutex<HashMap<IpAddr, (PeerWireVersion, Instant)>>>);

impl OtherVersions {
    pub fn heard(&self, address: IpAddr, version: PeerWireVersion, now: Instant) {
        self.0.lock().insert(address, (version, now));
    }

    /// The devices heard within `OTHER_VERSION_FORGET_AFTER` of `now`, by address.
    pub fn current(&self, now: Instant) -> Vec<(IpAddr, PeerWireVersion)> {
        let mut current: Vec<_> = {
            let mut heard = self.0.lock();
            heard.retain(|_, (_, at)| {
                now.saturating_duration_since(*at) < OTHER_VERSION_FORGET_AFTER
            });
            heard
                .iter()
                .map(|(address, (version, _))| (*address, *version))
                .collect()
        };
        current.sort();
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_namespace_on_another_wire_version_is_another_cluster() {
        assert_ne!(
            namespace_hash("0.3.70", Some(1)),
            namespace_hash("0.3.70", Some(2))
        );
        assert_ne!(
            namespace_hash("0.3.70", Some(1)),
            namespace_hash("0.3.70", None)
        );
        assert_eq!(
            namespace_hash("0.3.70", Some(1)),
            namespace_hash("0.3.70", Some(1))
        );
    }

    #[test]
    fn exos_from_before_wire_versions_hash_the_namespace_alone() {
        // What every exo announced until wire versions: the first 8 bytes of blake3
        let unversioned: NamespaceHash =
            blake3::hash(b"0.3.70").as_bytes()[..8].try_into().unwrap();
        assert_eq!(namespace_hash("0.3.70", None), unversioned);
    }

    #[test]
    fn tells_another_version_from_another_cluster() {
        let ours = ClusterKey::new("studio", 5);
        assert_eq!(ours.hash(), namespace_hash("studio", Some(5)));
        assert_eq!(ours.other_version(&ours.hash()), None);
        assert_eq!(
            ours.other_version(&namespace_hash("studio", Some(4))),
            Some(PeerWireVersion::Version(4))
        );
        assert_eq!(
            ours.other_version(&namespace_hash("studio", Some(6))),
            Some(PeerWireVersion::Version(6))
        );
        assert_eq!(
            ours.other_version(&namespace_hash("studio", None)),
            Some(PeerWireVersion::Unversioned)
        );
        // Another namespace, on any version, is another cluster
        assert_eq!(ours.other_version(&namespace_hash("office", Some(5))), None);
        assert_eq!(ours.other_version(&namespace_hash("office", Some(4))), None);
        assert_eq!(ours.other_version(&namespace_hash("office", None)), None);
    }

    #[test]
    fn recognises_a_bounded_range_of_versions() {
        let ours = ClusterKey::new("studio", 1);
        let far = 1 + RECOGNISED_VERSIONS_EITHER_SIDE;
        assert_eq!(
            ours.other_version(&namespace_hash("studio", Some(far))),
            Some(PeerWireVersion::Version(far))
        );
        assert_eq!(
            ours.other_version(&namespace_hash("studio", Some(far + 1))),
            None
        );
        // No version 0, and no overflow at the top
        assert_eq!(ours.other_version(&namespace_hash("studio", Some(0))), None);
        let top = ClusterKey::new("studio", u32::MAX);
        assert_eq!(
            top.other_version(&namespace_hash("studio", Some(u32::MAX - 1))),
            Some(PeerWireVersion::Version(u32::MAX - 1))
        );
    }

    #[test]
    fn forgets_a_device_once_its_hellos_stop() {
        let heard = OtherVersions::default();
        let start = Instant::now();
        let address: IpAddr = "fe80::1".parse().unwrap();
        heard.heard(address, PeerWireVersion::Version(2), start);
        assert_eq!(
            heard.current(start + OTHER_VERSION_FORGET_AFTER - Duration::from_millis(1)),
            vec![(address, PeerWireVersion::Version(2))]
        );
        // Each Hello renews it
        heard.heard(
            address,
            PeerWireVersion::Version(2),
            start + Duration::from_secs(20),
        );
        assert_eq!(heard.current(start + OTHER_VERSION_FORGET_AFTER).len(), 1);
        assert_eq!(
            heard.current(start + Duration::from_secs(20) + OTHER_VERSION_FORGET_AFTER),
            vec![]
        );
    }
}
