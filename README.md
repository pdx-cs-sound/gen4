# gen4: polyphonic MIDI synthesizer in Python
Bart Massey 2026

This toy synthesizer is intended to teach the basics of
synthesizer-building. It is polyphonic and velocity-
sensitive: any number of notes can sound at once, and each
note's loudness follows how hard its key was struck. Sound
generation is driven by the `sounddevice` audio callback;
MIDI input arrives via `mido` with `python-rtmidi`.

Some of the code is adapted from the teaching synthesizers
[`misy`](https://github.com/pdx-cs-sound/misy) and `rhosy`.

## Setup

Install the dependencies:

    pip install -r requirements.txt

Note that the `mido` used here will not work with `rtmidi`:
you need `python-rtmidi`.

## Usage

    python gen4.py [options]

Options:

- `--wave {sine,triangle,square,saw}` picks the output
  waveform for the session (default: `sine`).
- `--volume V` sets the master volume on a traditional
  0-10 scale (default: `7`). The scale is dB-linear over a
  60 dB span: `10` plays a single note at -16 dBFS and `0`
  is silence.
- `--attack MS` sets the note attack time in milliseconds
  (default: `5`).
- `--release MS` sets the note release time in
  milliseconds (default: `50`).
- `--clip-hardness K` sets the output soft-clip hardness
  (default: `8`). Lower bends gently at every level;
  higher stays near-linear through the normal range and
  bends only near full scale.
- `--controller NAME` names the MIDI input port. If
  omitted, `gen4` auto-detects a known controller; failing
  that, it opens a virtual input port named `gen4` that you
  can connect a controller to with your system's MIDI
  routing.
- `--device DEVICE` selects the audio output device, by
  name substring or numeric index. If omitted, the system
  default output is used.
- `--block SAMPLES` sets the audio block size (default:
  `128`). Smaller blocks lower latency but raise the risk
  of underruns.
- `--latency BLOCKS` sets the audio output buffering, in
  blocks (default: `2`). Two blocks is plenty on a capable
  machine; raise it if you hear underruns.
- `--list-devices` prints the available audio devices and
  exits.

The known controller name is currently hard-coded; you will
likely want to edit the `controllers` set in `gen4.py`.

## Playing

Each note's loudness follows its key velocity. Voices are
summed at a fixed per-voice gain and passed through a soft
clipper, so more notes are simply louder and a dense chord
rounds off gently instead of hard-clipping.

The master volume also responds live to MIDI CC#7 (channel
volume), sharing the same dB-linear taper as `--volume`. It
uses *soft takeover*: turning the controller's volume knob
has no effect until the knob sweeps through the synth's
current volume, so the level never jumps when the knob's
physical position does not match the synth.

Press the controller's "stop" key (hard-wired to the
Oxygen8 stop button) or `Ctrl-C` to exit.

## License

This work is licensed under the "MIT License". Please see
the file `LICENSE.txt` in this distribution for license
terms.
