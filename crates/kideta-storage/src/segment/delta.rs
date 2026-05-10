use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, Write};
use std::path::{Path, PathBuf};

const DELTA_MAGIC: u32 = 0x_54_4C_45_44;
const DELTA_VERSION: u32 = 1;
const DELTA_HEADER_SIZE: u64 = 16;

pub struct DeltaLogWriter {
    file: BufWriter<File>,
    #[allow(dead_code)]
    path: PathBuf,
}

impl DeltaLogWriter {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)?;

        let file_size = file.metadata()?.len();
        let mut writer = BufWriter::new(file);

        if file_size == 0 {
            writer.write_all(&DELTA_MAGIC.to_le_bytes())?;
            writer.write_all(&DELTA_VERSION.to_le_bytes())?;
            writer.write_all(&0u64.to_le_bytes())?;
            writer.flush()?;
            return Ok(Self {
                file: writer,
                path: path.to_path_buf(),
            });
        }

        writer.seek(std::io::SeekFrom::End(0))?;
        Ok(Self {
            file: writer,
            path: path.to_path_buf(),
        })
    }

    pub fn append(
        &mut self,
        vector_id: u32,
    ) -> std::io::Result<()> {
        self.file.write_all(&vector_id.to_le_bytes())?;
        Ok(())
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()?;
        let file = self.file.get_mut();
        file.sync_all()?;
        let current_pos = file.stream_position()?;
        if current_pos > DELTA_HEADER_SIZE {
            let count = (current_pos - DELTA_HEADER_SIZE) / 4;
            file.seek(std::io::SeekFrom::Start(8))?;
            file.write_all(&count.to_le_bytes())?;
            file.seek(std::io::SeekFrom::Start(current_pos))?;
            file.sync_all()?;
        }
        Ok(())
    }
}

pub struct DeltaLogReader {
    file: File,
    #[allow(dead_code)]
    path: PathBuf,
}

impl DeltaLogReader {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .open(path)?;
        Ok(Self {
            file,
            path: path.to_path_buf(),
        })
    }

    pub fn open_or_empty(path: &Path) -> std::io::Result<Self> {
        match Self::open(path) {
            Ok(r) => Ok(r),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(path)?;
                Ok(Self {
                    file,
                    path: path.to_path_buf(),
                })
            },
            Err(e) => Err(e),
        }
    }

    pub fn replay(&mut self) -> std::io::Result<Vec<u32>> {
        let file_size = self.file.metadata()?.len();

        if file_size < DELTA_HEADER_SIZE {
            return Ok(Vec::new());
        }

        let mut header = [0u8; 16];
        self.file.read_exact(&mut header)?;

        let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        if magic != DELTA_MAGIC {
            return Ok(Vec::new());
        }

        let record_count = u64::from_le_bytes([
            header[8], header[9], header[10], header[11], header[12], header[13], header[14],
            header[15],
        ]);

        let mut ids = Vec::with_capacity(record_count as usize);
        let mut buf = vec![0u8; 4];
        let mut remaining = record_count;

        while remaining > 0 {
            self.file.read_exact(&mut buf)?;
            let id = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
            ids.push(id);
            remaining -= 1;
        }

        Ok(ids)
    }
}
