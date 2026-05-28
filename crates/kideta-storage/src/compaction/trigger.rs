use std::sync::mpsc;
use std::time::Duration;

pub struct CompactionTrigger {
    interval_secs: u64,
    tx: Option<mpsc::Sender<TriggerCommand>>,
}

#[derive(Debug, Clone)]
pub enum TriggerCommand {
    Trigger,
    Stop,
}

impl CompactionTrigger {
    pub fn new(interval_secs: u64) -> (Self, mpsc::Receiver<TriggerCommand>) {
        let (tx, rx) = mpsc::channel();
        (
            Self {
                interval_secs,
                tx: Some(tx),
            },
            rx,
        )
    }

    pub fn start(&self) -> std::thread::JoinHandle<()> {
        let tx = self.tx.clone().unwrap();
        let interval = Duration::from_secs(self.interval_secs);

        std::thread::spawn(move || {
            loop {
                std::thread::sleep(interval);
                if tx.send(TriggerCommand::Trigger).is_err() {
                    break;
                }
            }
        })
    }

    pub fn trigger(&self) {
        if let Some(ref tx) = self.tx {
            let _ = tx.send(TriggerCommand::Trigger);
        }
    }

    pub fn stop(&self) {
        if let Some(ref tx) = self.tx {
            let _ = tx.send(TriggerCommand::Stop);
        }
    }
}

impl Default for CompactionTrigger {
    fn default() -> Self {
        Self {
            interval_secs: 300,
            tx: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trigger_creation() {
        let (trigger, rx) = CompactionTrigger::new(60);
        drop(trigger);
        assert!(rx.recv().is_err());
    }
}
