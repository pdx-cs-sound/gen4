# gen4: monophonic MIDI synthesizer in Python
Bart Massey 2026

This toy synthesizer is intended to teach the basics of
synthesizer-building. It is monophonic: only one note
sounds at a time, with last-note priority. Sound generation
is driven by the `sounddevice` audio callback; MIDI input
arrives via `mido` with `python-rtmidi`.

Some of the code is adapted from the teaching synthesizers
[`misy`](https://github.com/pdx-cs-sound/misy) and `rhosy`.

## Setup

Install the dependencies:

    pip install -r requirements.txt

Note that the `mido` used here will not work with `rtmidi`:
you need `python-rtmidi`.

## Usage

    python gen4.py [--wave {sine,triangle,square,saw}] \
                   [--controller NAME]

- `--wave` picks the output waveform for the session
  (default: `sine`).
- `--controller` names the MIDI input port. If omitted,
  `gen4` auto-detects a known controller; failing that, it
  opens a virtual input port named `gen4` that you can
  connect a controller to with your system's MIDI routing.

The known controller name is currently hard-coded; you will
likely want to edit the `controllers` set in `gen4.py`.

Press the controller's "stop" key (hard-wired to the
Oxygen8 stop button) or `Ctrl-C` to exit.

## License

This work is licensed under the "MIT License". Please see
the file `LICENSE.txt` in this distribution for license
terms.
