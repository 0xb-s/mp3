// file_handler.rs

use std::fs::File;
use std::fs::OpenOptions;
use std::io::Cursor;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Represents an MP3 file with reading and writing capabilities.
/// The struct is now generic over the reader and writer types,
/// requiring that the reader implements both `Read` and `Seek`,
/// and the writer implements both `Write` and `Seek`.
pub struct MP3File<R: Read + Seek, W: Write + Seek> {
    pub reader: BufReader<R>,
    pub writer: BufWriter<W>,
    pub path: String,
}

impl<R: Read + Seek, W: Write + Seek> MP3File<R, W> {
    /// Opens an existing MP3 file for reading.
    pub fn open_read<P: AsRef<Path>>(path: P) -> io::Result<MP3File<File, Cursor<Vec<u8>>>> {
        let file = File::open(&path)?;
        Ok(MP3File {
            reader: BufReader::new(file),
            writer: BufWriter::new(Cursor::new(Vec::new())), // Dummy writer
            path: path.as_ref().to_string_lossy().to_string(),
        })
    }

    /// Creates a new MP3 file for writing.
    pub fn open_write<P: AsRef<Path>>(path: P) -> io::Result<MP3File<Cursor<Vec<u8>>, File>> {
        let file = File::create(&path)?;
        Ok(MP3File {
            reader: BufReader::new(Cursor::new(Vec::new())), // Dummy reader
            writer: BufWriter::new(file),
            path: path.as_ref().to_string_lossy().to_string(),
        })
    }

    /// Reads a specific number of bytes from the MP3 file.
    pub fn read_bytes(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.reader.read(buffer)
    }

    /// Writes a slice of bytes to the MP3 file.
    pub fn write_bytes(&mut self, buffer: &[u8]) -> io::Result<()> {
        self.writer.write_all(buffer)
    }

    /// Seeks to a specific position in the MP3 file.
    pub fn seek(&mut self, pos: u64) -> io::Result<u64> {
        self.reader.seek(SeekFrom::Start(pos))
    }

    /// Flushes the writer buffer to ensure all data is written to disk.
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    /// Retrieves the path of the MP3 file.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Closes the MP3 file by flushing the writer.
    pub fn close(&mut self) -> io::Result<()> {
        self.flush()
    }
}

/// Reads the entire MP3 file into a byte vector.
/// This function is useful for small MP3 files.
/// For large files, consider processing in chunks.
pub fn read_mp3_file<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

/// Writes a byte vector to an MP3 file.
pub fn write_mp3_file<P: AsRef<Path>>(path: P, data: &[u8]) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(data)?;
    Ok(())
}

/// Appends data to an existing MP3 file.
pub fn append_to_mp3_file<P: AsRef<Path>>(path: P, data: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new().append(true).open(path)?;
    file.write_all(data)?;
    Ok(())
}

/// Represents the metadata of an MP3 file.
pub struct MP3Metadata {
    pub bitrate: u32,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration: f32, // in seconds
}

impl MP3Metadata {
    /// Extracts metadata from an MP3 file.
    pub fn extract<P: AsRef<Path>>(_path: P) -> io::Result<MP3Metadata> {
        todo!()
    }
}

/// Splits the MP3 file into frames.
/// Each frame can be processed individually.
pub fn split_into_frames(data: &[u8]) -> Vec<&[u8]> {
    let mut frames = Vec::new();
    let mut i = 0;
    while i < data.len() - 4 {
        if data[i] == 0xFF && (data[i + 1] & 0xE0) == 0xE0 {
            // Found frame header
            let frame_length = calculate_frame_length(&data[i..i + 4]);
            if frame_length == 0 {
                break;
            }
            if i + frame_length > data.len() {
                break;
            }
            frames.push(&data[i..i + frame_length]);
            i += frame_length;
        } else {
            i += 1;
        }
    }
    frames
}

/// Calculates the length of an MP3 frame based on its header.
/// This is a simplified version and may need adjustments for full compliance.
pub fn calculate_frame_length(header: &[u8]) -> usize {
    if header.len() < 4 {
        return 0;
    }
    let bitrate_index = (header[2] & 0xF0) >> 4;
    let sample_rate_index = (header[2] & 0x0C) >> 2;
    let padding = (header[2] & 0x02) >> 1;

    // Lookup tables for MPEG-1 Layer III
    let bitrate_table = [
        0, 32_000, 40_000, 48_000, 56_000, 64_000, 80_000, 96_000, 112_000, 128_000, 160_000,
        192_000, 224_000, 256_000, 320_000, 0,
    ];
    let sample_rate_table = [44_100, 48_000, 32_000, 0];

    if bitrate_index as usize >= bitrate_table.len()
        || sample_rate_index as usize >= sample_rate_table.len()
    {
        return 0;
    }

    let bitrate = bitrate_table[bitrate_index as usize] as usize;
    let sample_rate = sample_rate_table[sample_rate_index as usize] as usize;

    if bitrate == 0 || sample_rate == 0 {
        return 0;
    }

    let frame_length = (144 * bitrate) / sample_rate + padding as usize;
    frame_length
}

/// Example function to display MP3 file information.
pub fn display_mp3_info<P: AsRef<Path>>(path: P) -> io::Result<()> {
    let metadata = MP3Metadata::extract(&path)?;
    println!("MP3 File: {}", path.as_ref().display());
    println!("Bitrate: {} kbps", metadata.bitrate / 1000);
    println!("Sample Rate: {} Hz", metadata.sample_rate);
    println!("Channels: {}", metadata.channels);
    println!("Duration: {:.2} seconds", metadata.duration);
    Ok(())
}

pub fn process_frames(frames: &[&[u8]]) {
    for (i, frame) in frames.iter().enumerate() {
        println!("Processing frame {}: {} bytes", i + 1, frame.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_calculate_frame_length() {
        // Example MP3 frame header for testing
        let header = [0xFF, 0xFB, 0x90, 0x64];
        let length = calculate_frame_length(&header);
        assert_eq!(length, 417); // Example expected length
    }

    #[test]
    fn test_split_into_frames() {
        // Example data with two frames
        let data = [
            0xFF, 0xFB, 0x90, 0x64, 0x00, 0x01, 0x02, // Frame 1
            0xFF, 0xFB, 0x90, 0x64, 0x03, 0x04, 0x05, // Frame 2
        ];
        let frames = split_into_frames(&data);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], &[0xFF, 0xFB, 0x90, 0x64, 0x00, 0x01, 0x02][..]);
        assert_eq!(frames[1], &[0xFF, 0xFB, 0x90, 0x64, 0x03, 0x04, 0x05][..]);
    }

    #[test]
    fn test_mp3file_with_cursor() {
        // Create in-memory data
        let data = vec![
            0xFF, 0xFB, 0x90, 0x64, // Frame header
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Side info and main data
        ];

        // Create a Cursor for reading
        let reader = Cursor::new(data.clone());

        // Create a Cursor for writing
        let writer = Cursor::new(Vec::new());

        // Initialize MP3File with in-memory readers and writers
        let mut mp3_file = MP3File {
            reader: BufReader::new(reader),
            writer: BufWriter::new(writer),
            path: "test.mp3".to_string(),
        };

        // Test reading
        let mut buffer = [0u8; 4];
        let bytes_read = mp3_file.read_bytes(&mut buffer).unwrap();
        assert_eq!(bytes_read, 4);
        assert_eq!(buffer, [0xFF, 0xFB, 0x90, 0x64]);

        // Test writing
        mp3_file.write_bytes(&[0xAA, 0xBB, 0xCC]).unwrap();

        // Test seeking
        mp3_file.seek(0).unwrap();
        let bytes_read = mp3_file.read_bytes(&mut buffer).unwrap();
        assert_eq!(bytes_read, 4);
        assert_eq!(buffer, [0xFF, 0xFB, 0x90, 0x64]);

        // Test flushing (no-op for Cursor)
        mp3_file.flush().unwrap();
    }
}
