pub struct TieredCompaction {
    l0_max_bytes: u64,
    l1_max_bytes: u64,
}

impl TieredCompaction {
    pub fn new() -> Self {
        Self {
            l0_max_bytes: 10 * 1024 * 1024,
            l1_max_bytes: 100 * 1024 * 1024,
        }
    }

    pub fn level_for_size(size_bytes: u64) -> usize {
        if size_bytes < 10 * 1024 * 1024 {
            0
        } else if size_bytes < 100 * 1024 * 1024 {
            1
        } else {
            2
        }
    }

    pub fn level_name(&self, level: usize) -> &'static str {
        match level {
            0 => "L0",
            1 => "L1",
            2 => "L2",
            _ => "unknown",
        }
    }

    pub fn l0_max(&self) -> u64 {
        self.l0_max_bytes
    }

    pub fn l1_max(&self) -> u64 {
        self.l1_max_bytes
    }
}

impl Default for TieredCompaction {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_assignment() {
        assert_eq!(TieredCompaction::level_for_size(1_000_000), 0);
        assert_eq!(TieredCompaction::level_for_size(50_000_000), 1);
        assert_eq!(TieredCompaction::level_for_size(200_000_000), 2);
    }

    #[test]
    fn tiered_new() {
        let t = TieredCompaction::new();
        assert_eq!(t.l0_max(), 10 * 1024 * 1024);
        assert_eq!(t.l1_max(), 100 * 1024 * 1024);
    }
}
