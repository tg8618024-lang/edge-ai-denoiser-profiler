"""Audio streaming utilities: thread-safe circular ring buffer, frame chunker, and frame streamer.

Designed for robust real-time audio ingest, preventing buffer underruns,
and converting variable-sized audio blocks into fixed hop chunks.
"""

import threading
from typing import Iterator, Optional
import numpy as np


class AudioRingBuffer:
    """Thread-safe circular ring buffer for streaming audio PCM data.

    Supports concurrent writes (audio capture thread) and reads (processing thread)
    with mutex locking and preallocated contiguous storage.
    """

    def __init__(
        self,
        capacity: int = 65536,
        dtype: np.dtype = np.float32,
        capacity_samples: Optional[int] = None,
    ) -> None:
        """Initialize circular buffer with preallocated capacity."""
        cap = capacity_samples if capacity_samples is not None else capacity
        self.capacity = int(cap)
        self.dtype = np.dtype(dtype)
        self.buffer = np.zeros(self.capacity, dtype=self.dtype)
        self.head = 0  # Write pointer
        self.tail = 0  # Read pointer
        self.size = 0  # Current occupancy
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return self.size

    @property
    def capacity_samples(self) -> int:
        """Total buffer capacity in samples."""
        return self.capacity

    @property
    def available_samples(self) -> int:
        """Number of samples available to read."""
        with self._lock:
            return self.size

    @property
    def available_read(self) -> int:
        """Number of samples available to read."""
        with self._lock:
            return self.size

    @property
    def available_write(self) -> int:
        """Remaining capacity available to write without overflow."""
        with self._lock:
            return self.capacity - self.size

    def clear(self) -> None:
        """Clear all buffered samples and reset pointers."""
        with self._lock:
            self.head = 0
            self.tail = 0
            self.size = 0
            self.buffer.fill(0.0)

    def write(self, samples: np.ndarray, allow_overwrite: bool = True) -> int:
        """Write samples into the ring buffer.

        Parameters
        ----------
        samples : np.ndarray
            1D array of audio samples to write.
        allow_overwrite : bool
            If True, drops the oldest samples if writing exceeds capacity.
            If False, only writes up to available capacity.

        Returns
        -------
        int
            Number of samples actually written.
        """
        data = np.asarray(samples, dtype=self.dtype).ravel()
        n = len(data)
        if n == 0:
            return 0

        with self._lock:
            if not allow_overwrite and n > (self.capacity - self.size):
                n = self.capacity - self.size
                data = data[:n]
                if n == 0:
                    return 0

            # If input is larger than total capacity, keep only the latest capacity samples
            if n > self.capacity:
                data = data[-self.capacity:]
                n = self.capacity

            # Advance tail if we are overwriting
            overflow = (self.size + n) - self.capacity
            if overflow > 0:
                self.tail = (self.tail + overflow) % self.capacity
                self.size = self.capacity
            else:
                self.size += n

            # Write data (may wrap around end of buffer)
            first_chunk = min(n, self.capacity - self.head)
            self.buffer[self.head : self.head + first_chunk] = data[:first_chunk]

            second_chunk = n - first_chunk
            if second_chunk > 0:
                self.buffer[:second_chunk] = data[first_chunk:]
                self.head = second_chunk
            else:
                self.head = (self.head + first_chunk) % self.capacity

            return n

    def read(self, num_samples: int) -> np.ndarray:
        """Read and remove num_samples from the ring buffer.

        If fewer than num_samples are available, returns available samples zero-padded
        to num_samples.

        Parameters
        ----------
        num_samples : int
            Number of samples to read.

        Returns
        -------
        np.ndarray
            Array of length num_samples.
        """
        output = np.zeros(num_samples, dtype=self.dtype)
        if num_samples <= 0:
            return output

        with self._lock:
            to_read = min(num_samples, self.size)
            if to_read == 0:
                return output

            first_chunk = min(to_read, self.capacity - self.tail)
            output[:first_chunk] = self.buffer[self.tail : self.tail + first_chunk]

            second_chunk = to_read - first_chunk
            if second_chunk > 0:
                output[first_chunk:to_read] = self.buffer[:second_chunk]
                self.tail = second_chunk
            else:
                self.tail = (self.tail + first_chunk) % self.capacity

            self.size -= to_read
            return output

    def peek(self, num_samples: int) -> np.ndarray:
        """Read num_samples without advancing read pointer.

        Parameters
        ----------
        num_samples : int
            Number of samples to inspect.

        Returns
        -------
        np.ndarray
            Array of length num_samples (zero-padded if available < num_samples).
        """
        output = np.zeros(num_samples, dtype=self.dtype)
        if num_samples <= 0:
            return output

        with self._lock:
            to_read = min(num_samples, self.size)
            if to_read == 0:
                return output

            first_chunk = min(to_read, self.capacity - self.tail)
            output[:first_chunk] = self.buffer[self.tail : self.tail + first_chunk]

            second_chunk = to_read - first_chunk
            if second_chunk > 0:
                output[first_chunk:to_read] = self.buffer[:second_chunk]

            return output


class AudioChunker:
    """Chunks arbitrary streams of audio samples into fixed-length frames.

    Buffers leftover samples between successive push calls to maintain strict
    alignment for STFT/iSTFT framing.
    """

    def __init__(self, chunk_size: int = 256, dtype: np.dtype = np.float32) -> None:
        """Initialize chunker with target hop / frame size.

        Parameters
        ----------
        chunk_size : int
            Fixed number of samples per yielded chunk (default: 256).
        dtype : np.dtype
            Sample data type.
        """
        self.chunk_size = int(chunk_size)
        self.dtype = np.dtype(dtype)
        self.remainder = np.array([], dtype=self.dtype)

    def reset(self) -> None:
        """Clear buffered remainder samples."""
        self.remainder = np.array([], dtype=self.dtype)

    def push(self, data: np.ndarray) -> list[np.ndarray]:
        """Push new audio samples and extract all complete chunks of size chunk_size.

        Parameters
        ----------
        data : np.ndarray
            Input audio array of arbitrary length.

        Returns
        -------
        list[np.ndarray]
            List of complete chunks, each of shape (chunk_size,).
        """
        arr = np.asarray(data, dtype=self.dtype).ravel()
        if len(self.remainder) > 0:
            combined = np.concatenate([self.remainder, arr])
        else:
            combined = arr

        num_chunks = len(combined) // self.chunk_size
        chunks = []
        for i in range(num_chunks):
            start = i * self.chunk_size
            chunks.append(combined[start : start + self.chunk_size].copy())

        tail_start = num_chunks * self.chunk_size
        self.remainder = combined[tail_start:].copy()
        return chunks

    def flush(self, pad_mode: str = "zero") -> Optional[np.ndarray]:
        """Flush remaining samples by zero-padding to chunk_size if any exist.

        Parameters
        ----------
        pad_mode : str
            Padding mode (default: "zero").

        Returns
        -------
        Optional[np.ndarray]
            A final chunk of size chunk_size, or None if no remainder exists.
        """
        if len(self.remainder) == 0:
            return None

        out = np.zeros(self.chunk_size, dtype=self.dtype)
        out[: len(self.remainder)] = self.remainder
        self.remainder = np.array([], dtype=self.dtype)
        return out


class AudioStreamer:
    """Iterates through audio arrays or streaming generators, yielding fixed hops."""

    @staticmethod
    def stream_frames(
        audio: np.ndarray,
        hop_length: int = 256,
        pad_tail: bool = True,
    ) -> Iterator[np.ndarray]:
        """Yield consecutive fixed-size hops of length hop_length from audio array.

        Parameters
        ----------
        audio : np.ndarray
            Input 1D audio waveform.
        hop_length : int
            Hop length in samples (default: 256).
        pad_tail : bool
            Whether to zero-pad the last frame if not an exact multiple of hop_length.

        Yields
        ------
        np.ndarray
            1D frame array of length hop_length.
        """
        audio = np.asarray(audio, dtype=np.float32).ravel()
        chunker = AudioChunker(chunk_size=hop_length, dtype=np.float32)
        for chunk in chunker.push(audio):
            yield chunk

        if pad_tail:
            tail = chunker.flush()
            if tail is not None:
                yield tail


# Compatibility alias
StreamingPCMBuffer = AudioRingBuffer

