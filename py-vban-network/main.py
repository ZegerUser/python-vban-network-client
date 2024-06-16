from multiprocessing import Manager, Process, Queue
import socket
import pyvban
import time
import numpy as np
import librosa

VBAN_SAMPLE_RATE = 48_000

def convert_to_blocks(audio, block_size: int) -> list[bytes]:
    max_abs_value = np.max(np.abs(audio))
    audio_normalized = audio / max_abs_value if max_abs_value != 0 else audio
    audio_normalized = (audio_normalized * np.iinfo(np.int16).max).astype(np.int16)

    num_blocks = audio_normalized.shape[0] // block_size
    output_blocks = np.split(audio_normalized[:num_blocks * block_size], num_blocks)

    output_bytes = [block.tobytes() for block in output_blocks]

    return output_bytes

def high_precision_sleep(duration_ns: int) -> None:
    start_time = time.perf_counter_ns()
    while True:
        if duration_ns - time.perf_counter_ns() + start_time <= 0:
            break

def vban_sender(stream_name: str, stream_ip: str, stream_port: int, audio_queue: Queue, pause, channels: int, samples_per_frame: int):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packet_counter = 0

    time_per_packet = int((samples_per_frame/48_000) * 10 ** 9)

    silence_bytes = bytes(channels*samples_per_frame*2)

    while True:
        start = time.perf_counter_ns()

        if not pause.value and audio_queue.qsize() > 0:
            pcm_data = audio_queue.get()
        else:
            pcm_data = silence_bytes
        
        header = pyvban.VBANAudioHeader(
                    sample_rate=pyvban.subprotocols.audio.VBANSampleRates.RATE_48000,
                    samples_per_frame=samples_per_frame,
                    channels=channels,
                    format=pyvban.subprotocols.audio.VBANBitResolution.VBAN_BITFMT_16_INT,
                    codec=pyvban.subprotocols.audio.VBANCodec.VBAN_CODEC_PCM,
                    stream_name=stream_name,
                    frame_counter=packet_counter
        )
        vban_packet = header.to_bytes() + pcm_data
        s.sendto(vban_packet, (stream_ip, stream_port))
        packet_counter += 1

        end = time.perf_counter_ns()
        process_time = end - start
        high_precision_sleep(time_per_packet - process_time)

class VBANClient():
    def __init__(self, stream_name: str = "Stream1", stream_ip: str = "127.0.0.1", stream_port: int = 6980, channels: int = 1, samples_per_frame: int = 128) -> None:
        self._stream_name = stream_name
        self._stream_ip = stream_ip
        self._stream_port = stream_port

        self._channels = channels
        self._samples_per_frame = samples_per_frame

        self._manager = Manager()
        self._audio_in = self._manager.Queue()
        self._pause = self._manager.Value("b", False)

        self._sender = Process(target=vban_sender, args=(
            self._stream_name,
            self._stream_ip,
            self._stream_port,
            self._audio_in,
            self._pause,
            self._channels,
            self._samples_per_frame
        ))

    def start(self) -> None:
        self._sender.start()

    def stop(self) -> None:
        self._sender.kill()

    def play_audio(self, audio, sr: int, blocking=True) -> None:
        if sr != VBAN_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=VBAN_SAMPLE_RATE)

        audio_blocks = convert_to_blocks(audio, self._samples_per_frame)

        for block in audio_blocks:
            self._audio_in.put(block)
        
        if blocking:
            time.sleep(len(audio_blocks) * self._samples_per_frame/VBAN_SAMPLE_RATE)

    def clear_queue(self):
        self.change_pause(True)
        while self._audio_in.qsize() > 0:
            self._audio_in.get()

        self.change_pause(False)

    def get_queue_lenght(self) -> int:
        return self._audio_in.qsize()
    
    def change_pause(self, pause: bool) -> None:
        self._pause.value = pause