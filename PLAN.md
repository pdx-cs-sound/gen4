# Plan: `gen4` — Monophonic MIDI Synthesizer in Python

## Context

The `gen4` repository is a fresh, empty git repo (only the
`demo midi synthesizer` commit, no files). It is the next
generation of Bart's teaching synthesizers. Sibling projects
`../misy/misy.py` and `../rhosy/rhosy.py` are working
*polyphonic* synths and serve as proven reference code for
the `sounddevice` callback architecture, `mido` MIDI input,
ADSR-style envelopes, and controller detection.

This plan builds the **first milestone**: a *monophonic*
synth driven by the `sounddevice` output callback, taking
MIDI input via `mido` + `python-rtmidi`, with a player-
selectable oscillator (sine / triangle / square / saw).

Design decisions:
- **Note priority:** last-note, no legato fallback.
  Releasing the currently sounding key silences the synth
  even if older keys are still held.
- **Envelope:** Attack + Release (linear ramps), full
  sustain between — same model as `misy`/`rhosy`.
- **Waveform selection:** command-line argument, fixed for
  the session (`--wave {sine,triangle,square,saw}`).

## Files to create

- `gen4.py` — the synthesizer (single file, like the
  reference projects).
- `requirements.txt` — `mido`, `python-rtmidi`,
  `sounddevice`, `numpy`.
- `README.md` — short usage notes (controller name,
  `requirements.txt`, `--wave` flag).
- `LICENSE.txt` — MIT, copied/adapted from `../rhosy/LICENSE.txt`.

## Architecture

Reuse the structure proven in `../rhosy/rhosy.py`:

1. **Main thread** blocks on `controller.receive()` and
   translates MIDI messages into commands.
2. **Audio thread** is the `sounddevice` `OutputStream`
   callback, which generates samples block-by-block.
3. A `queue.SimpleQueue` carries commands from the main
   thread to the callback (do not mutate shared synth state
   directly from the MIDI thread — push messages).

### Constants
- `sample_rate = 48000`
- `blocksize = 1024` (good latency; raise if underruns).
  Reference: `../rhosy/rhosy.py` uses 64, `misy` uses 2048.

### CLI (`argparse`)
- `--wave {sine,triangle,square,saw}` (default `sine`).
- `--controller NAME` optional override of the MIDI input
  name.

### Oscillators
Vectorized NumPy waveform functions `f(t, freq)` over a
sample-time array `t`, normalized to [-1, 1]:
- `sine`  — `np.sin(2*pi*f*t)` (from `misy.sine_samples`).
- `saw`   — `(f*t) % 2.0 - 1.0` (from `misy.saw_samples`).
- `square`— `np.sign((f*t) % 2.0 - 1.0)` (from
  `misy.square_samples`).
- `triangle` — **new**: `2*abs(2*((f*t) % 1.0) - 1.0) - 1.0`.

Selected oscillator is fixed at startup from `--wave`.

Generate sample times with a global `sample_clock` and a
`sample_times(frame_count)` helper (from `misy.py`) so phase
is continuous across blocks.

### `key_to_freq(key)`
`440 * 2 ** ((key - 69) / 12)` (from `misy.py`).

### `Note` class (monophonic — one instance at a time)
Adapt the AR envelope from `misy.Note`:
- `__init__(key)` — store frequency, `attack_time_remaining`,
  `release_time_remaining = None`, `playing = True`.
- `release()` — start the release ramp.
- `samples(t)` — produce a block: call the selected
  oscillator, apply attack ramp (`np.clip(np.linspace(...),
  0, 1)`) or release ramp; return `None` when the release
  finishes so the callback can drop the note.
- Envelope times: `attack_time = 0.020`s,
  `release_time = 0.1`s.

### Synth state
- `current_note` — a single `Note` or `None` (replaces the
  `dict` of notes used by the polyphonic references).
- `command_queue` — `queue.SimpleQueue` of `(type, mesg)`.

### `output_callback(out_data, frame_count, time_info, status)`
- Log non-`None` `status` (underruns).
- Drain `command_queue`:
  - `note_on` → set `current_note = Note(key)` (last-note
    priority; any existing note is replaced immediately).
  - `note_off` → if `current_note` exists **and its key
    matches** the released key, call `release()`. A
    `note_off` for a non-sounding key is ignored (no
    fallback to held keys).
- If `current_note` is set, call `samples(sample_times(
  frame_count))`; if it returns `None`, set
  `current_note = None`; otherwise that block is the output.
- Apply a fixed output gain (e.g. `* 0.25`) to leave
  headroom and avoid clipping.
- `out_data[:] = output.reshape(frame_count, 1)`.
- Advance the global `sample_clock` by `frame_count`.

### MIDI input
Controller selection adapted from `../rhosy/rhosy.py`:
- If `--controller` given, open that input directly.
- Else scan `mido.get_input_names()` against a known set
  (e.g. `'USB Oxygen 8 v2 MIDI 1'`); on no match, print the
  available inputs and open a **virtual** input
  (`mido.open_input('gen4', virtual=True)`).

`get_midi_event(controller)` (adapted from `rhosy`):
- Block on `controller.receive()`.
- `note_on` with velocity 0 → treat as `note_off`.
- `note_on` / `note_off` → push `(type, mesg)` to
  `command_queue`.
- `control_change` 23 (Oxygen8 "stop") → return `False` to
  end the program.
- Other `control_change` / `pitchwheel` → log and ignore
  (velocity sensitivity, pitch bend, and live waveform
  switching are deferred to later milestones).

### Main loop
Create and `start()` the `sounddevice.OutputStream`
(`channels=1`, `dtype` default float32, `blocksize`,
`callback`), then `while get_midi_event(controller): pass`.

## Reference code to reuse

- `../misy/misy.py` — `sine_samples`, `saw_samples`,
  `square_samples`, `sample_times`, `key_to_freq`, the AR
  envelope math in `Note.samples`, `output_callback` shape.
- `../rhosy/rhosy.py` — `command_queue` pattern, controller
  auto-detection with virtual-port fallback, `note_on`
  velocity-0 handling, `get_midi_event` structure.
- `../rhosy/requirements.txt` — dependency list.

## Verification

1. `pip install -r requirements.txt` in a fresh venv.
2. **No-controller smoke test:** run `python gen4.py` with
   no MIDI device. It should print the available inputs,
   open the `gen4` virtual port, and start the audio stream
   without errors.
3. **MIDI playback:** connect a controller (or route into
   the virtual `gen4` port with `aconnect`/a DAW). Play
   notes and confirm:
   - A single note sounds with a soft (~20 ms) attack and a
     ~100 ms release tail (no clicks).
   - Pressing a new key while one is held immediately
     switches pitch (last-note priority).
   - Releasing the sounding key goes silent even while an
     older key is still held (no fallback).
4. **Waveform check:** run once per `--wave` value
   (`sine`, `triangle`, `square`, `saw`) and confirm the
   timbre changes audibly and matches expectation.
5. **Underrun check:** watch the console for
   `output callback:` status messages while playing; if any
   appear, raise `blocksize`.
6. Optional: add a temporary wavetable-to-WAV dump (cf. the
   `test_wavetable` block in `misy.py`) to inspect each
   waveform offline, then remove it.

## Out of scope (future milestones)

Polyphony, ADSR decay/sustain stages, velocity sensitivity,
pitch bend, live waveform switching via MIDI CC, sustain
pedal, and effects/VST plugins.
