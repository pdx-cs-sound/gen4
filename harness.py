# gen4 test harness: drive the synth offline and analyze its
# output numerically.
# Bart Massey 2026

# This harness imports `gen4` and exercises its real audio
# callback (`gen4.output_callback`) and MIDI handler
# (`gen4.handle_midi`) without opening any hardware. MIDI
# events are delivered on a schedule, the audio is captured
# block by block into a NumPy array, and that array is
# analyzed directly — there are no intermediate files and no
# dependencies beyond NumPy. An optional WAV dump (`--wav`)
# is provided only so a rendered scenario can be listened to
# by ear; the analysis itself never needs it.
#
#     python harness.py [--wav]
#
# `--latency-test` instead measures real audio round-trip
# latency: it plays a marker and recaptures it through a
# PipeWire monitor source, so it needs a sink whose monitor
# can be captured. Create a null sink once --
#
#     pactl load-module module-null-sink sink_name=gen4test \
#         sink_properties=device.description=gen4_test
#
# (or persist it with a PipeWire config) -- then point
# libpulse at the sink and its monitor and run the test:
#
#     PULSE_SINK=gen4test PULSE_SOURCE=gen4test.monitor \
#         python harness.py --latency-test

import os, sys, wave
import mido
import numpy as np

import gen4

# Directory for optional WAV dumps.
output_dir = "test-output"

# ---- driving the synth ------------------------------------

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
        while (event_index < len(events)
               and events[event_index][0] <= block_time):
            gen4.handle_midi(events[event_index][1])
            event_index += 1

        # Pull one block of audio from the synth callback.
        gen4.output_callback(buffer, blocksize, None, None)
        audio[b * blocksize:(b + 1) * blocksize] = buffer[:, 0]

    return audio

# ---- analysis ---------------------------------------------

# Estimate the fundamental frequency, in Hz, from the FFT
# peak. Parabolic interpolation of the peak and its two
# neighbours gives sub-bin accuracy, so the result is not
# quantized to the FFT bin spacing.
def measure_frequency(audio, sample_rate=48000):
    windowed = audio.astype(np.float64) * np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(windowed))
    k = int(np.argmax(spectrum))
    if 0 < k < len(spectrum) - 1:
        y0, y1, y2 = spectrum[k - 1], spectrum[k], spectrum[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            k += 0.5 * (y0 - y2) / denom
    return k * sample_rate / len(audio)

# Return the percentage of spectral energy that is not within
# a few bins of any of the given fundamental frequencies.
# For a sine note or a chord of sines this is the distortion;
# for harmonically rich waveforms it also counts the
# waveform's own harmonics, so compare to a reference render.
def nonfundamental_energy(audio, fundamentals, sample_rate=48000):
    windowed = audio.astype(np.float64) * np.hanning(len(audio))
    power = np.abs(np.fft.rfft(windowed)) ** 2
    n = len(audio)
    fundamental = 0.0
    for f in fundamentals:
        b = int(round(f * n / sample_rate))
        fundamental += np.sum(power[max(0, b - 3):b + 4])
    return 100.0 * (1.0 - fundamental / np.sum(power))

# Return the peak and RMS amplitude of the samples.
def levels(audio):
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return peak, rms

# ---- WAV dump (optional, for listening) -------------------

# Write samples to a 16-bit mono WAV file, using only the
# standard library.
def write_wav(name, audio, sample_rate=48000):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    print(f"  wrote {path}")

# ---- live latency measurement -----------------------------

# Measure audio output round-trip latency. Plays a short
# noise-burst marker through an output stream configured like
# the synth's, recaptures it from a PipeWire monitor source,
# and times the delay by cross-correlation. PULSE_SINK and
# PULSE_SOURCE must point at a sink and its monitor (see the
# file header); the streams use PortAudio's `pulse` device.
def latency_test(sample_rate=48000, blocksize=128,
                 latency_blocks=2.0, runs=8):
    import sounddevice as sd

    latency = latency_blocks * blocksize / sample_rate
    rng = np.random.default_rng(0)
    burst = (rng.standard_normal(512) * 0.3).astype(np.float32)
    n = sample_rate                       # one-second buffer
    mark_at = n // 3
    marker = np.zeros((n, gen4.output_channels), dtype=np.float32)
    marker[mark_at:mark_at + len(burst), :] = burst[:, None]

    print(f"latency test — blocksize {blocksize}, "
          f"latency hint {latency * 1000:.1f} ms")

    results = []
    for run in range(runs):
        rec = sd.playrec(marker, samplerate=sample_rate,
                         channels=gen4.output_channels,
                         blocksize=blocksize, latency=latency,
                         device='pulse')
        sd.wait()
        captured = rec[:, 0].astype(np.float64)
        # The first run warms up the streams; skip its timing.
        if run == 0:
            continue
        # Cross-correlate the capture against the marker; a
        # genuine hit is a peak far above the correlation's
        # own median. Reject silent or unconvincing runs.
        correlation = np.abs(np.correlate(
            captured, burst.astype(np.float64), mode='valid'))
        peak = int(np.argmax(correlation))
        if (np.sqrt(np.mean(captured ** 2)) < 1e-5
                or correlation[peak] < 8.0 * np.median(correlation)):
            continue
        results.append((peak - mark_at) / sample_rate * 1000.0)

    if len(results) < 3:
        print("  no clean measurement — is PULSE_SOURCE set to a "
              "monitor source? (see the file header)")
        return

    results.sort()
    round_trip = results[len(results) // 2]
    print(f"  round-trip latency: {round_trip:.2f} ms  "
          f"(median of {len(results)}, range "
          f"{results[0]:.2f}-{results[-1]:.2f})")
    # The round trip is the output path plus the monitor
    # capture path; halving it estimates the output latency.
    print(f"  one-way estimate:   {round_trip / 2:.2f} ms")

# ---- report -----------------------------------------------

# Render a standard set of scenarios and print the analysis.
def main():
    if "--latency-test" in sys.argv:
        latency_test(blocksize=gen4.blocksize)
        return

    dump = "--wav" in sys.argv
    sr = 48000
    key = 69  # A4, 440 Hz
    nominal = gen4.key_to_freq(key)

    # Each waveform: a sustained note. Report measured pitch
    # and level over the steady portion (past the attack).
    print("sustained A4 by waveform:")
    for wave in ("sine", "triangle", "square", "saw"):
        audio = render(note(key, 0.0, 10.0), seconds=2.0, wave=wave)
        steady = audio[8000:]
        freq = measure_frequency(steady, sr)
        cents = 1200.0 * np.log2(freq / nominal)
        peak, rms = levels(steady)
        print(f"  {wave:8}: {freq:8.3f} Hz ({cents:+.2f} cents)"
              f"   peak {peak:.4f}  rms {rms:.4f}")
        if dump:
            write_wav(f"{wave}.wav", audio, sr)

    # Distortion: a sine note and a sine chord (for sines,
    # non-fundamental energy is the distortion directly).
    print("\ndistortion (non-fundamental energy):")
    audio = render(note(key, 0.0, 10.0), seconds=2.0, wave="sine")
    print(f"  single sine A4 : "
          f"{nonfundamental_energy(audio[8000:], [nominal], sr):.4f} %")
    chord_keys = [60, 64, 67]
    events = sum([note(k, 0.0, 10.0) for k in chord_keys], [])
    audio = render(events, seconds=2.0, wave="sine")
    funds = [gen4.key_to_freq(k) for k in chord_keys]
    print(f"  C-E-G sine chord: "
          f"{nonfundamental_energy(audio[8000:], funds, sr):.4f} %")
    if dump:
        write_wav("chord.wav", audio, sr)

if __name__ == "__main__":
    main()
