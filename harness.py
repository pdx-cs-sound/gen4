# gen4 test harness: drive the synth with MIDI offline,
# capture its audio output, and inspect the waveforms.
# Bart Massey 2026

# This harness imports `gen4` and exercises its real audio
# callback (`gen4.output_callback`) and MIDI handler
# (`gen4.handle_midi`) without opening any hardware. MIDI
# events are delivered on a schedule, the resulting audio is
# captured block by block, and the result is written to WAV
# files and PNG plots for inspection.
#
# Run directly to render a standard set of scenarios:
#
#     python harness.py
#
# Outputs land in the `test-output/` directory.

import os
import mido
import numpy as np
import scipy.io.wavfile as wav
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gen4

# Where rendered WAV and PNG files are written.
output_dir = "test-output"

# Build the note_on / note_off event pair for a single note.
# `start` and `duration` are in seconds. Returns a list of
# (time, mido.Message) tuples.
def note(key, start, duration, velocity=100):
    return [
        (start, mido.Message('note_on', note=key, velocity=velocity)),
        (start + duration, mido.Message('note_off', note=key, velocity=0)),
    ]

# Render a list of (time, mido.Message) events through the
# synth and return the captured mono audio as a float32
# array. The synth is fully reset before rendering.
def render(events, seconds, wave="sine", sample_rate=48000, blocksize=128):
    # Configure and reset the synth module state.
    gen4.sample_rate = sample_rate
    gen4.blocksize = blocksize
    gen4.oscillator = gen4.oscillators[wave]
    gen4.log_notes = False
    gen4.sample_clock = 0
    gen4.playing_notes.clear()
    while not gen4.command_queue.empty():
        gen4.command_queue.get()

    events = sorted(events, key=lambda e: e[0])
    event_index = 0

    nblocks = int(np.ceil(seconds * sample_rate / blocksize))
    audio = np.zeros(nblocks * blocksize, dtype=np.float32)
    buffer = np.zeros((blocksize, gen4.output_channels), dtype=np.float32)

    for b in range(nblocks):
        # Deliver every event whose time has arrived by the
        # start of this block.
        block_time = b * blocksize / sample_rate
        while event_index < len(events) and events[event_index][0] <= block_time:
            gen4.handle_midi(events[event_index][1])
            event_index += 1

        # Pull one block of audio from the synth callback.
        gen4.output_callback(buffer, blocksize, None, None)
        audio[b * blocksize:(b + 1) * blocksize] = buffer[:, 0]

    return audio

# Write captured audio to a WAV file.
def write_wav(name, audio, sample_rate=48000):
    path = os.path.join(output_dir, name)
    wav.write(path, sample_rate, audio.astype(np.float32))
    print(f"  wrote {path}")

# Plot the full waveform plus a zoomed segment, marking the
# audio block boundaries so block-boundary anomalies stand
# out. Saves a PNG.
def plot(name, audio, sample_rate=48000, blocksize=128,
         zoom_start=0.10, zoom_len=600):
    path = os.path.join(output_dir, name)
    t = np.arange(len(audio)) / sample_rate

    fig, (full_ax, zoom_ax) = plt.subplots(2, 1, figsize=(11, 6))

    full_ax.plot(t, audio, linewidth=0.5)
    full_ax.set_title(f"{name}: full waveform")
    full_ax.set_xlabel("time (s)")
    full_ax.set_ylabel("amplitude")

    # Zoomed view with block boundaries marked.
    z0 = int(zoom_start * sample_rate)
    z1 = min(z0 + zoom_len, len(audio))
    idx = np.arange(z0, z1)
    zoom_ax.plot(idx, audio[z0:z1], marker='.', markersize=2, linewidth=0.7)
    for boundary in range((z0 // blocksize + 1) * blocksize, z1, blocksize):
        zoom_ax.axvline(boundary, color='red', alpha=0.3, linewidth=0.8)
    zoom_ax.set_title(
        f"{name}: zoom (samples {z0}–{z1}, red = block boundaries)")
    zoom_ax.set_xlabel("sample index")
    zoom_ax.set_ylabel("amplitude")

    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  wrote {path}")

# Inspect a sustained-note render for waveform anomalies.
# Reports the nominal vs. measured frequency and counts
# block boundaries where the waveform appears discontinuous.
def diagnose(audio, key, sample_rate=48000, blocksize=128):
    nominal = gen4.key_to_freq(key)

    # Measure frequency from the FFT peak over the steady
    # portion of the note (skip the attack).
    steady = audio[2000:]
    window = steady * np.hanning(len(steady))
    spectrum = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(window), 1.0 / sample_rate)
    measured = freqs[np.argmax(spectrum)]
    cents = 1200 * np.log2(measured / nominal)

    print(f"  nominal freq:  {nominal:8.3f} Hz")
    print(f"  measured freq: {measured:8.3f} Hz  ({cents:+.1f} cents)")

    # A continuous waveform has a smooth slope. At a block
    # boundary, compare the step across the boundary with the
    # steps just inside each block; a boundary whose step is
    # wildly out of line signals a discontinuity.
    diffs = np.diff(audio)
    glitches = 0
    for boundary in range(blocksize, len(audio) - 1, blocksize):
        across = abs(diffs[boundary - 1])
        inside = abs(diffs[boundary - 2]) + abs(diffs[boundary])
        if across > 4 * inside + 1e-6:
            glitches += 1
    nboundaries = (len(audio) - 1) // blocksize
    print(f"  block-boundary glitches: {glitches} / {nboundaries}")

# Render the standard scenario set.
def main():
    os.makedirs(output_dir, exist_ok=True)
    sr, bs = 48000, gen4.blocksize

    # One sustained note per waveform.
    key = 69  # A4, 440 Hz
    for wave in ("sine", "triangle", "square", "saw"):
        print(f"[{wave}] sustained A4")
        audio = render(note(key, 0.05, 0.8), seconds=1.0, wave=wave,
                       sample_rate=sr, blocksize=bs)
        write_wav(f"{wave}.wav", audio, sr)
        plot(f"{wave}.png", audio, sr, bs)
        diagnose(audio, key, sr, bs)

    # A three-note chord, sine.
    print("[chord] C-E-G triad, sine")
    events = note(60, 0.05, 0.8) + note(64, 0.05, 0.8) + note(67, 0.05, 0.8)
    audio = render(events, seconds=1.0, wave="sine",
                   sample_rate=sr, blocksize=bs)
    write_wav("chord.wav", audio, sr)
    plot("chord.png", audio, sr, bs)

if __name__ == "__main__":
    main()
