import wave

import pytest


def test_local_decoder_accepts_a_real_wav_file(tmp_path):
    audio_module = pytest.importorskip("faster_whisper.audio")
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(bytes(16000 * 2))
    audio = audio_module.decode_audio(str(path), sampling_rate=16000)
    assert len(audio) == 16000
    assert not audio.any()
