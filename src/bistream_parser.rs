// bitstream_parser.rs

use crate::file_handler::MP3File;
use std::error::Error;
use std::fmt;

use std::io::Seek;
use std::io::Write;
use std::io::{self, Read};
/// Custom error type for bitstream parsing.
#[derive(Debug)]
pub enum BitstreamError {
    IoError(io::Error),
    InvalidFrameHeader(String),
    UnsupportedVersion(String),
    UnsupportedLayer(String),
    UnsupportedBitrate(String),
    UnsupportedSampleRate(String),
}

impl fmt::Display for BitstreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BitstreamError::IoError(e) => write!(f, "IO Error: {}", e),
            BitstreamError::InvalidFrameHeader(s) => write!(f, "Invalid Frame Header: {}", s),
            BitstreamError::UnsupportedVersion(s) => write!(f, "Unsupported MPEG Version: {}", s),
            BitstreamError::UnsupportedLayer(s) => write!(f, "Unsupported Layer: {}", s),
            BitstreamError::UnsupportedBitrate(s) => write!(f, "Unsupported Bitrate: {}", s),
            BitstreamError::UnsupportedSampleRate(s) => write!(f, "Unsupported Sample Rate: {}", s),
        }
    }
}

impl Error for BitstreamError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            BitstreamError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for BitstreamError {
    fn from(error: io::Error) -> Self {
        BitstreamError::IoError(error)
    }
}

/// an MPEG Audio Frame Header.
#[derive(Debug, Clone)]
pub struct FrameHeader {
    pub mpeg_version: MPEGVersion,
    pub layer: Layer,
    pub protection_bit: bool,
    pub bitrate: u32,     // in kbps
    pub sample_rate: u32, // in Hz
    pub padding: bool,
    pub private_bit: bool,
    pub channel_mode: ChannelMode,
    pub mode_extension: u8,
    pub copyright: bool,
    pub original: bool,
    pub emphasis: Emphasis,
}

/// MPEG Versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MPEGVersion {
    MPEG1,
    MPEG2,
    MPEG25,
    Reserved,
}

/// Layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    LayerI,
    LayerII,
    LayerIII,
    Reserved,
}

/// Channel Modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelMode {
    Stereo,
    JointStereo,
    DualChannel,
    SingleChannel,
}

/// Emphasis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emphasis {
    None,
    _50_15Ms,
    CCIT_J17,
    Reserved,
}

/// Represents a single MP3 Frame.
#[derive(Debug, Clone)]
pub struct MP3Frame {
    pub header: FrameHeader,
    pub side_info: SideInfo,
    pub main_data: Vec<u8>,
}

/// Side Information for MPEG1, Layer III.
#[derive(Debug, Clone)]
pub struct SideInfo {
    pub main_data_begin: u16,
    pub private_bits: u16,
    pub scfsi: Vec<u8>,
}

/// Bitstream Parser.
pub struct BitstreamParser<'a, R: Read> {
    reader: &'a mut R,
    buffer: Vec<u8>,
    position: usize,
}

impl<'a, R: Read> BitstreamParser<'a, R> {
    /// Creates a new BitstreamParser with the given reader.
    pub fn new(reader: &'a mut R) -> Self {
        BitstreamParser {
            reader,
            buffer: Vec::new(),
            position: 0,
        }
    }

    /// Fills the internal buffer with data from the reader.
    fn fill_buffer(&mut self) -> io::Result<()> {
        let mut temp_buffer = [0u8; 4096];
        let bytes_read = self.reader.read(&mut temp_buffer)?;
        self.buffer.extend_from_slice(&temp_buffer[..bytes_read]);
        Ok(())
    }

    /// Finds the next frame header in the buffer.
    fn find_next_frame_header(&mut self) -> Result<usize, BitstreamError> {
        while self.position < self.buffer.len() - 4 {
            if self.buffer[self.position] == 0xFF && (self.buffer[self.position + 1] & 0xE0) == 0xE0
            {
                // Potential frame header found
                return Ok(self.position);
            }
            self.position += 1;
        }
        // Try to read more data if possible
        self.fill_buffer()?;
        if self.position < self.buffer.len() - 4 {
            self.find_next_frame_header()
        } else {
            Err(BitstreamError::InvalidFrameHeader(
                "No more frames found.".to_string(),
            ))
        }
    }

    /// Parses the frame header from the buffer starting at the given position.
    fn parse_frame_header(&self, pos: usize) -> Result<FrameHeader, BitstreamError> {
        let header = &self.buffer[pos..pos + 4];
        // Parse the header bits
        // Reference: https://www.mp3-tech.org/programmer/frame_header.html

        // Check frame sync (all bits set in first 11 bits)
        if header[0] != 0xFF || (header[1] & 0xE0) != 0xE0 {
            return Err(BitstreamError::InvalidFrameHeader(
                "Frame sync not found.".to_string(),
            ));
        }

        // MPEG Audio version ID
        let mpeg_version = match (header[1] & 0x18) >> 3 {
            0b00 => MPEGVersion::MPEG25,
            0b01 => MPEGVersion::Reserved,
            0b10 => MPEGVersion::MPEG2,
            0b11 => MPEGVersion::MPEG1,
            _ => return Err(BitstreamError::UnsupportedVersion("Unknown".to_string())),
        };

        // Layer description
        let layer = match (header[1] & 0x06) >> 1 {
            0b01 => Layer::LayerIII,
            0b10 => Layer::LayerII,
            0b11 => Layer::LayerI,
            _ => Layer::Reserved,
        };

        if let Layer::Reserved = layer {
            return Err(BitstreamError::UnsupportedLayer(
                "Reserved Layer".to_string(),
            ));
        }

        // Protection bit
        let protection_bit = (header[1] & 0x01) != 0;

        // Bitrate index
        let bitrate_index = (header[2] & 0xF0) >> 4;
        let bitrate = self.get_bitrate(mpeg_version, layer, bitrate_index)?;

        // Sample rate frequency index
        let sample_rate_index = (header[2] & 0x0C) >> 2;
        let sample_rate = self.get_sample_rate(mpeg_version, sample_rate_index)?;

        // Padding bit
        let padding = (header[2] & 0x02) != 0;

        // Private bit
        let private_bit = (header[2] & 0x01) != 0;

        // Channel mode
        let channel_mode = match (header[3] & 0xC0) >> 6 {
            0b00 => ChannelMode::Stereo,
            0b01 => ChannelMode::JointStereo,
            0b10 => ChannelMode::DualChannel,
            0b11 => ChannelMode::SingleChannel,
            _ => ChannelMode::Stereo, // Default to Stereo
        };

        // Mode extension (only relevant in Joint Stereo)
        let mode_extension = (header[3] & 0x30) >> 4;

        // Copyright
        let copyright = (header[3] & 0x08) != 0;

        // Original
        let original = (header[3] & 0x04) != 0;

        // Emphasis
        let emphasis = match header[3] & 0x03 {
            0b00 => Emphasis::None,
            0b01 => Emphasis::_50_15Ms,
            0b10 => Emphasis::CCIT_J17,
            _ => Emphasis::Reserved,
        };

        Ok(FrameHeader {
            mpeg_version,
            layer,
            protection_bit,
            bitrate,
            sample_rate,
            padding,
            private_bit,
            channel_mode,
            mode_extension,
            copyright,
            original,
            emphasis,
        })
    }

    /// Retrieves the bitrate based on MPEG version, layer, and bitrate index.
    fn get_bitrate(
        &self,
        version: MPEGVersion,
        layer: Layer,
        index: u8,
    ) -> Result<u32, BitstreamError> {
        // Bitrate tables for MPEG1 and MPEG2
        // Reference: https://www.mp3-tech.org/programmer/frame_header.html

        let bitrate_table = match (version, layer) {
            (MPEGVersion::MPEG1, Layer::LayerI) => [
                0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 0,
            ],
            (MPEGVersion::MPEG1, Layer::LayerII) => [
                0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0,
            ],
            (MPEGVersion::MPEG1, Layer::LayerIII) => [
                0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
            ],
            (MPEGVersion::MPEG2, Layer::LayerI) => [
                0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0,
            ],
            (MPEGVersion::MPEG2, Layer::LayerII) => [
                0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
            ],
            (MPEGVersion::MPEG2, Layer::LayerIII) => [
                0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
            ],
            _ => {
                return Err(BitstreamError::UnsupportedBitrate(
                    "Unsupported combination".to_string(),
                ))
            }
        };

        // Cast index to usize for safe indexing
        let index_usize = index as usize;

        if index_usize >= bitrate_table.len() {
            return Err(BitstreamError::UnsupportedBitrate(format!(
                "Bitrate index {} out of range",
                index
            )));
        }

        let bitrate = bitrate_table[index_usize];

        if bitrate == 0 {
            return Err(BitstreamError::UnsupportedBitrate(format!(
                "Invalid bitrate index {}",
                index
            )));
        }

        // Convert to kbps
        Ok(bitrate)
    }

    /// Retrieves the sample rate based on MPEG version and sample rate index.
    fn get_sample_rate(&self, version: MPEGVersion, index: u8) -> Result<u32, BitstreamError> {
        // Sample rate tables for MPEG1 and MPEG2
        // Reference: https://www.mp3-tech.org/programmer/frame_header.html

        let sample_rate_table = match version {
            MPEGVersion::MPEG1 => [44100, 48000, 32000, 0],
            MPEGVersion::MPEG2 => [22050, 24000, 16000, 0],
            MPEGVersion::MPEG25 => [11025, 12000, 8000, 0],
            _ => {
                return Err(BitstreamError::UnsupportedSampleRate(
                    "Reserved version".to_string(),
                ))
            }
        };

        
        let sample_rate_index_usize = index as usize;

        if sample_rate_index_usize >= sample_rate_table.len() {
            return Err(BitstreamError::UnsupportedSampleRate(format!(
                "Sample rate index {} out of range",
                index
            )));
        }

        let sample_rate = sample_rate_table[sample_rate_index_usize];

        if sample_rate == 0 {
            return Err(BitstreamError::UnsupportedSampleRate(format!(
                "Invalid sample rate index {}",
                index
            )));
        }

        Ok(sample_rate)
    }

    /// Parses side information based on the frame header.
    fn parse_side_info(&mut self, header: &FrameHeader) -> Result<SideInfo, BitstreamError> {
 
        if header.mpeg_version != MPEGVersion::MPEG1 || header.layer != Layer::LayerIII {
            // TODO Implement other versions and layers as needed
            return Err(BitstreamError::InvalidFrameHeader(
                "Unsupported MPEG version or layer for side info parsing.".to_string(),
            ));
        }

 
        let side_info_size = match header.channel_mode {
            ChannelMode::SingleChannel => 9,
            _ => 17,
        };

        // Ensure buffer has enough data
        if self.position + 4 + side_info_size > self.buffer.len() {
            self.fill_buffer()?;
            if self.position + 4 + side_info_size > self.buffer.len() {
                return Err(BitstreamError::InvalidFrameHeader(
                    "Insufficient data for side information.".to_string(),
                ));
            }
        }

        let side_info_bytes = &self.buffer[self.position + 4..self.position + 4 + side_info_size];
        let main_data_begin =
            ((side_info_bytes[0] as u16) << 3) | ((side_info_bytes[1] as u16) >> 5);
        let private_bits =
            ((side_info_bytes[1] as u16) & 0x1F) << 6 | ((side_info_bytes[2] as u16) & 0xFC) >> 2;

        // SCFSI: Scale factor selection information
        let scfsi = side_info_bytes[3..side_info_size]
            .iter()
            .map(|&b| b)
            .collect();

        Ok(SideInfo {
            main_data_begin,
            private_bits,
            scfsi,
        })
    }

    /// Parses the next MP3 frame from the bitstream.
    pub fn parse_next_frame(&mut self) -> Result<MP3Frame, BitstreamError> {

        let frame_start = self.find_next_frame_header()?;


        let header = self.parse_frame_header(frame_start)?;


        let frame_length = self.calculate_frame_length(&header)?;


        if frame_start + frame_length > self.buffer.len() {
            self.fill_buffer()?;
            if frame_start + frame_length > self.buffer.len() {
                return Err(BitstreamError::InvalidFrameHeader(
                    "Incomplete frame data.".to_string(),
                ));
            }
        }

        self.position = frame_start + 4;
        let side_info = self.parse_side_info(&header)?;

        let _main_data_size = frame_length
            - 4
            - match header.channel_mode {
                ChannelMode::SingleChannel => 9,
                _ => 17,
            };
        let main_data = self.buffer[self.position
            + (match header.channel_mode {
                ChannelMode::SingleChannel => 9,
                _ => 17,
            })..self.position + frame_length - 4]
            .to_vec();

        self.position = frame_start + frame_length;

        Ok(MP3Frame {
            header,
            side_info,
            main_data,
        })
    }

    /// Calculates the frame length based on the frame header.
    fn calculate_frame_length(&self, header: &FrameHeader) -> Result<usize, BitstreamError> {
        let frame_length = match header.layer {
            Layer::LayerI => {
                // For Layer I: FrameLengthInBytes = (12 * BitRate / SampleRate + Padding) * 4
                (((12 * header.bitrate) / header.sample_rate + if header.padding { 1 } else { 0 })
                    * 4) as usize
            }
            Layer::LayerII | Layer::LayerIII => {
                // For Layer II & III: FrameLengthInBytes = 144 * BitRate / SampleRate + Padding
                (144 * header.bitrate / header.sample_rate) as usize
                    + if header.padding { 1 } else { 0 }
            }
            _ => {
                return Err(BitstreamError::InvalidFrameHeader(
                    "Unknown layer".to_string(),
                ))
            }
        };
        Ok(frame_length)
    }

    /// Parses all frames in the MP3 file.
    pub fn parse_all_frames(&mut self) -> Result<Vec<MP3Frame>, BitstreamError> {
        let mut frames = Vec::new();
        loop {
            match self.parse_next_frame() {
                Ok(frame) => frames.push(frame),
                Err(BitstreamError::InvalidFrameHeader(_)) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(frames)
    }
}

/// Parses MP3 frames from an MP3File.
pub fn parse_mp3_file<R: Read + Seek, W: Write + Seek>(
    file: &mut MP3File<R, W>,
) -> Result<Vec<MP3Frame>, BitstreamError> {
    let mut parser = BitstreamParser::new(file.reader.get_mut());
    parser.parse_all_frames()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file_handler::calculate_frame_length;
    use crate::file_handler::split_into_frames;
    use crate::file_handler::MP3Metadata;
    use std::io::BufReader;
    use std::io::BufWriter;
    use std::io::Cursor;

    #[test]
    fn test_calculate_frame_length() {
        // Example MP3 frame header for testing
        let header = [0xFF, 0xFB, 0x90, 0x64];
        let length = calculate_frame_length(&header);
        assert_eq!(length, 417);
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

     
        let reader = Cursor::new(data.clone());


        let writer = Cursor::new(Vec::new());

        
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
