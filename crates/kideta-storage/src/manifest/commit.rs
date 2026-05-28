use crate::manifest::manifest_data::Manifest;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub const MANIFEST_PREFIX: &str = "manifest.";
pub const MANIFEST_SYMLINK: &str = "manifest.json";

pub fn manifest_filename(version: u64) -> String {
    format!("{}{:06}.json", MANIFEST_PREFIX, version)
}

pub fn manifest_path(dir: &Path, version: u64) -> PathBuf {
    dir.join(manifest_filename(version))
}

pub fn current_manifest_path(dir: &Path) -> PathBuf {
    dir.join(MANIFEST_SYMLINK)
}

pub fn atomic_commit(dir: &Path, manifest: &Manifest) -> io::Result<PathBuf> {
    let version = manifest.version;
    let filename = manifest_filename(version);
    let temp_path = dir.join(format!("{}.tmp", filename));
    let final_path = dir.join(&filename);

    let json = serde_json::to_string_pretty(manifest).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    {
        let mut file = fs::File::create(&temp_path)?;
        file.write_all(json.as_bytes())?;
        file.sync_all()?;
    }

    fs::rename(&temp_path, &final_path)?;

    let symlink_path = current_manifest_path(dir);
    if symlink_path.exists() {
        fs::remove_file(&symlink_path)?;
    }

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&filename, &symlink_path)?;
    }

    #[cfg(not(unix))]
    {
        fs::copy(&final_path, &symlink_path)?;
    }

    Ok(final_path)
}

pub fn read_manifest(dir: &Path) -> io::Result<Manifest> {
    let symlink = current_manifest_path(dir);

    #[cfg(unix)]
    let target = if symlink.is_symlink() {
        let link = symlink.read_link().unwrap_or_else(|_| symlink.clone());
        if link.is_relative() { dir.join(&link) } else { link }
    } else {
        symlink.clone()
    };

    #[cfg(not(unix))]
    let target = symlink;

    let data = fs::read_to_string(&target)?;
    serde_json::from_str(&data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub fn read_manifest_version(dir: &Path, version: u64) -> io::Result<Manifest> {
    let path = manifest_path(dir, version);
    let data = fs::read_to_string(&path)?;
    serde_json::from_str(&data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub fn list_manifest_versions(dir: &Path) -> io::Result<Vec<u64>> {
    let mut versions = Vec::new();
    let entries = fs::read_dir(dir)?;
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with(MANIFEST_PREFIX) && name_str.ends_with(".json") {
            let start = MANIFEST_PREFIX.len();
            let end = name_str.len() - 5;
            if end > start {
                let version_str = &name_str[start..end];
                if version_str.len() == 6
                    && let Ok(v) = version_str.parse::<u64>()
                {
                    versions.push(v);
                }
            }
        }
    }
    versions.sort();
    Ok(versions)
}

pub fn delete_manifest_version(dir: &Path, version: u64) -> io::Result<()> {
    let path = manifest_path(dir, version);
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::manifest_data::Manifest;
    use tempfile::tempdir;

    #[test]
    fn atomic_commit_creates_file_and_symlink() {
        let dir = tempdir().unwrap();
        let m = Manifest::new(42, 100);
        let path = atomic_commit(dir.path(), &m).unwrap();
        assert!(path.exists());
        assert!(current_manifest_path(dir.path()).exists());

        #[cfg(unix)]
        assert!(
            current_manifest_path(dir.path()).is_symlink(),
            "manifest.json should be a symlink on Unix"
        );

        #[cfg(not(unix))]
        assert!(
            current_manifest_path(dir.path()).is_file(),
            "manifest.json should be a regular file on Windows"
        );

        let loaded = read_manifest(dir.path()).unwrap();
        assert_eq!(loaded.version, 42);
        assert_eq!(loaded.wal_lsn, 100);
    }

    #[test]
    fn list_manifest_versions_test() {
        let dir = tempdir().unwrap();
        atomic_commit(dir.path(), &Manifest::new(1, 0)).unwrap();
        atomic_commit(dir.path(), &Manifest::new(2, 0)).unwrap();
        atomic_commit(dir.path(), &Manifest::new(5, 0)).unwrap();
        let versions = list_manifest_versions(dir.path()).unwrap();
        assert_eq!(versions, vec![1, 2, 5]);
    }

    #[test]
    fn read_manifest_version_test() {
        let dir = tempdir().unwrap();
        atomic_commit(dir.path(), &Manifest::new(7, 50)).unwrap();
        let loaded = read_manifest_version(dir.path(), 7).unwrap();
        assert_eq!(loaded.version, 7);
        assert_eq!(loaded.wal_lsn, 50);
    }

    #[test]
    fn delete_manifest_version_test() {
        let dir = tempdir().unwrap();
        atomic_commit(dir.path(), &Manifest::new(1, 0)).unwrap();
        delete_manifest_version(dir.path(), 1).unwrap();
        assert!(!manifest_path(dir.path(), 1).exists());
    }
}
