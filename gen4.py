# gen4: Polyphonic MIDI Synthesizer
# Bart Massey 2026

# This very simple synthesizer is intended primarily as a
# demo of MIDI and synthesis. It is polyphonic: any number
# of notes can sound at once. The output waveform is
# selectable at startup.
#
# Some of the code is adapted from the teaching synthesizers
# `misy` (https://github.com/pdx-cs-sound/misy) and
# `rhosy`.
#
# Sound generation (oscillators, `Note`, `output_callback`,
# `handle_midi`) is kept free of hardware setup so that a
# test harness can `import gen4` and drive it offline. The
# hardware setup and main loop live in `main()`.

import argparse, queue
import mido
import numpy as np
import sounddevice

# Sample rate in sps. This doesn't need to be fixed: it
# could be set to the preferred rate of the audio output.
sample_rate = 48000

# Blocksize in samples to process. This provides pretty good
# latency. Slower machines may need larger numbers.
blocksize = 128

# Print MIDI note events if True.
log_notes = True

# Note attack and release times in seconds. These are short
# defaults; both are overridable on the command line.
attack_time = 0.005
release_time = 0.050

# Output level for a single note at full volume, in dBFS.
# Voices are summed at a fixed gain — there is no dynamic
# gain on the voice bus. This is how real synths stage
# polyphony: a fixed per-voice level plus headroom and a
# master volume, so more notes are simply louder, the way an
# acoustic instrument behaves.
single_note_dbfs = -12.0

# The volume control is dB-linear and spans this many dB,
# from full volume (`single_note_dbfs`) down toward silence.
volume_range_db = 60.0

# Master volume as a normalized position, 0.0 to 1.0. 1.0
# plays a single note at `single_note_dbfs`; 0.0 is silence.
# Set by `--volume` and, live, by MIDI CC#7. Defaults to 0.7
# (volume 7 on the 0-10 scale), leaving room to turn up.
volume_position = 0.7

# Convert a normalized volume position (0.0-1.0) to a linear
# output gain. The mapping is dB-linear over `volume_range_db`
# decibels; position 0.0 is exact silence.
def volume_to_gain(position):
    if position <= 0.0:
        return 0.0
    db = single_note_dbfs - (1.0 - position) * volume_range_db
    return 10.0 ** (db / 20.0)

# Current output gain, recomputed whenever the volume changes.
output_gain = volume_to_gain(volume_position)

# Soft-takeover state for the MIDI CC#7 volume knob. The
# knob's physical position need not match the synth volume,
# so an incoming CC#7 is ignored until the knob sweeps
# through the current volume — only then does it take over.
volume_knob_engaged = False
# The knob's previous CC value (0-127), or None before any
# CC#7 has arrived; used to detect that sweep.
volume_knob_last = None

# Soft-clip hardness: the exponent k in the variable-hardness
# clipper x / (1 + |x|^k)^(1/k). Low k bends gently at every
# level (no knee); high k approaches a hard knee. A moderate
# value softens the bend so an overloaded chord breathes
# rather than clicking at its beat frequency. Overridable on
# the command line via `--clip-hardness`.
clip_hardness = 3.0

# Number of output channels. The synth mix is mono; it is
# duplicated into this many channels so that the sound is
# heard on every output (some setups carry only one).
output_channels = 2

# Known MIDI controllers to auto-detect.
controllers = {
    'USB Oxygen 8 v2 MIDI 1',
}

# This count of the number of samples output so far is used
# to make sure that waveforms are generated with the right
# phase across blocks.
sample_clock = 0

# Generate an array of frame_count sample times starting at
# sample_clock. `endpoint=False` gives uniform 1/sample_rate
# spacing: without it the last sample of each block would
# land on the same instant as the first sample of the next,
# duplicating a sample at every block boundary and adding
# distortion at the block rate.
def sample_times(frame_count):
    return np.linspace(
        sample_clock / sample_rate,
        (sample_clock + frame_count) / sample_rate,
        frame_count,
        endpoint=False,
        dtype=np.float32,
    )

# Return a sine wave at frequency f over the given sample
# times t.
def sine_samples(t, f):
    return np.sin(2 * np.pi * f * t)

# Return a rising sawtooth wave at frequency f over the
# given sample times t.
def saw_samples(t, f):
    return 2.0 * ((f * t) % 1.0) - 1.0

# Return a square wave at frequency f over the given sample
# times t.
def square_samples(t, f):
    return np.sign(((f * t) % 1.0) - 0.5)

# Return a triangle wave at frequency f over the given
# sample times t.
def triangle_samples(t, f):
    return 2.0 * np.abs(2.0 * ((f * t) % 1.0) - 1.0) - 1.0

# Available oscillators, keyed by waveform name.
oscillators = {
    "sine": sine_samples,
    "triangle": triangle_samples,
    "square": square_samples,
    "saw": saw_samples,
}

# Oscillator in use. The default is overridden by `main()`
# from the command line; a test harness may set it directly.
oscillator = oscillators["sine"]

# Calculate frequency for a 12-tone equal-tempered Western
# scale given MIDI note number.
def key_to_freq(key):
    return 440 * 2 ** ((key - 69) / 12)

# List of currently playing notes. There may be more than
# one note with the same key number at once: a key can be
# struck again while an earlier note on that key is still
# finishing its release.
playing_notes = []

# Representation of a note currently being played.
class Note:
    def __init__(self, key):
        self.key = key
        self.frequency = key_to_freq(key)
        self.attack_time_remaining = attack_time
        self.release_time_remaining = None
        self.playing = True

    # Note has been released. Start the release ramp.
    def release(self):
        self.release_time_remaining = release_time

    # Remove this note from the list of playing notes.
    def remove(self):
        playing_notes.remove(self)

    # Accept a time linspace to generate samples in. Return
    # that many samples of the note being played, or None if
    # the note is over.
    def samples(self, t):
        if not self.playing:
            return None

        frame_count = len(t)

        # Pick and generate the waveform.
        samples = oscillator(t, self.frequency)

        if self.release_time_remaining is not None:
            # Do release part of AR envelope.
            release_time_remaining = self.release_time_remaining
            if release_time_remaining <= 0:
                # Note has played out: drop it from the mix.
                self.playing = False
                self.remove()
                return None
            # Gain at the starting time, per a linear ramp.
            start_gain = release_time_remaining / release_time
            # Time after the last sample, used to adjust the
            # release time remaining.
            end_time = frame_count / sample_rate
            release_time_remaining -= end_time
            # Gain at the ending time, per a linear ramp.
            end_gain = release_time_remaining / release_time
            # Per-sample gains, clipped so a release that
            # finishes mid-block does not go below 0.
            envelope = np.clip(
                np.linspace(start_gain, end_gain, frame_count),
                0.0,
                1.0,
            )
            samples *= envelope
            self.release_time_remaining = max(0, release_time_remaining)
        elif self.attack_time_remaining > 0.0:
            # Do attack part of AR envelope.
            attack_time_remaining = self.attack_time_remaining
            # Gain at the starting time, per a linear ramp.
            start_gain = 1.0 - attack_time_remaining / attack_time
            # Time after the last sample, used to adjust the
            # attack time remaining.
            end_time = frame_count / sample_rate
            attack_time_remaining -= end_time
            # Gain at the ending time, per a linear ramp.
            end_gain = 1.0 - attack_time_remaining / attack_time
            # Per-sample gains, clipped so an attack that
            # finishes mid-block does not exceed 1.
            envelope = np.clip(
                np.linspace(start_gain, end_gain, frame_count),
                0.0,
                1.0,
            )
            samples *= envelope
            self.attack_time_remaining = attack_time_remaining

        return samples

# Memoryless soft clipper, the variable-hardness curve
# x / (1 + |x|^k)^(1/k). It bends smoothly at every level, so
# there is no linear region and thus no knee to gate
# distortion on and off, and it asymptotes to +/-1, so the
# output never reaches full scale. Being memoryless (each
# output sample depends only on the matching input sample) it
# adds no latency and no state.
def soft_clip(samples):
    k = clip_hardness
    magnitude = np.abs(samples)
    return samples / (1.0 + magnitude ** k) ** (1.0 / k)

# Queue of MIDI messages for state changes, passed from the
# main thread to the audio callback.
command_queue = queue.SimpleQueue()

# This callback is called by `sounddevice` to get some
# samples to output. It's the heart of sound generation in
# the synth.
def output_callback(out_data, frame_count, time_info, status):
    global sample_clock

    # A non-None status indicates that something has
    # happened with sound output that shouldn't have. This
    # is almost always an underrun due to generating samples
    # too slowly.
    if status:
        print("output callback:", status)

    # Apply queued state changes.
    while not command_queue.empty():
        mesg_type, mesg = command_queue.get()
        if mesg_type == 'note_on':
            playing_notes.append(Note(mesg.note))
        elif mesg_type == 'note_off':
            # Release every held (not yet released) note on
            # this key. Notes already in release are left
            # alone to finish.
            for note in playing_notes:
                if note.key == mesg.note and note.release_time_remaining is None:
                    note.release()
        elif mesg_type == 'control_change':
            # The only queued control change is CC#7 volume.
            if mesg.control == 7:
                handle_volume_cc(mesg.value)
        else:
            raise Exception(f"bad command: {mesg_type} {mesg}")

    # Start with silence and maybe work up.
    samples = np.zeros(frame_count, dtype=np.float32)

    if playing_notes:
        t = sample_times(frame_count)
        # Iterate over a snapshot: a played-out note removes
        # itself from `playing_notes` during `samples()`.
        for note in list(playing_notes):
            note_samples = note.samples(t)
            if note_samples is not None:
                samples += note_samples

    # Apply the fixed output gain, then soft-clip the mix so
    # a dense chord rounds off gently instead of hard-clipping.
    samples = soft_clip(output_gain * samples)

    # Duplicate the mono mix into each output channel of the
    # callback's output array. Write into the existing array
    # rather than accidentally copying over the parameter.
    for channel in range(output_channels):
        out_data[:, channel] = samples

    # Bump the sample clock for next cycle.
    sample_clock += frame_count

# Handle one MIDI message, queueing any resulting state
# change for the audio callback. Return False if the message
# wants the synthesizer to stop, True otherwise. This is the
# pure-logic half of MIDI handling, with no I/O, so a test
# harness can call it directly with constructed messages.
def handle_midi(mesg):
    # Select what to do based on message type.
    mesg_type = mesg.type
    # Special case: note on with velocity 0 indicates note
    # off (for older MIDI instruments).
    if mesg_type == 'note_on' and mesg.velocity == 0:
        mesg_type = 'note_off'
    # Start a note.
    if mesg_type == 'note_on':
        if log_notes:
            print('note on', mesg.note, mesg.velocity)
        command_queue.put((mesg_type, mesg))
    # Release a note.
    elif mesg_type == 'note_off':
        if log_notes:
            print('note off', mesg.note, mesg.velocity)
        command_queue.put((mesg_type, mesg))
    # Handle various controls.
    elif mesg.type == 'control_change':
        # XXX Hard-wired for "stop" key on Oxygen8.
        if mesg.control == 23:
            print('stop')
            return False
        # CC#7: channel volume. Routed through the command
        # queue so it is serialized with note events and
        # applied on the audio thread (see handle_volume_cc).
        elif mesg.control == 7:
            command_queue.put(('control_change', mesg))
        # Unknown control changes are logged and ignored.
        else:
            print('control', mesg.control, mesg.value)
    # XXX Pitchwheel is currently logged and ignored.
    elif mesg.type == 'pitchwheel':
        print('pitchwheel', mesg.pitch)
    else:
        print('unknown MIDI message', mesg)
    return True

# Handle a CC#7 (channel volume) value with soft takeover.
# Invoked by the audio callback as it drains the command
# queue, so the volume state is owned by the audio thread.
# The incoming value is ignored until the knob sweeps across
# the synth's current volume; from then on the knob sets the
# volume directly. This avoids a jump when the knob's
# physical position does not match the synth volume.
def handle_volume_cc(value):
    global volume_position, output_gain, volume_knob_engaged, volume_knob_last

    if not volume_knob_engaged:
        # Engage once the knob sweeps across the current
        # volume. `value` and the previous reading bracket
        # the swept range; the volume on the same 0-127 scale
        # is volume_position * 127.
        current = volume_position * 127.0
        if volume_knob_last is not None:
            lo, hi = sorted((volume_knob_last, value))
            if lo <= current <= hi:
                volume_knob_engaged = True
                print('volume knob engaged')
        volume_knob_last = value
        if not volume_knob_engaged:
            return

    # The knob is in control: set the volume from the CC value.
    volume_position = value / 127.0
    output_gain = volume_to_gain(volume_position)

# Block waiting for the controller (keyboard) to send a MIDI
# message, then handle it. Return False if the synthesizer
# should stop, True otherwise.
def get_midi_event(controller):
    return handle_midi(controller.receive())

# Open the MIDI controller (keyboard). Use the name given on
# the command line, else auto-detect a known controller,
# else fall back to a virtual input port.
def open_controller(args):
    inputs = mido.get_input_names()
    if args.controller is not None:
        return mido.open_input(args.controller)
    for input_name in inputs:
        for name in controllers:
            if name in input_name:
                print(f"using controller: {input_name}")
                return mido.open_input(input_name)
    print("No known controller — inputs found:")
    for input_name in inputs:
        print(' ', input_name)
    print("Opening virtual input port 'gen4'")
    return mido.open_input('gen4', virtual=True)

# Parse arguments, open hardware, and run the synthesizer
# until its stop key is pressed.
def main():
    global oscillator, volume_position, output_gain
    global attack_time, release_time, clip_hardness

    ap = argparse.ArgumentParser(description="Polyphonic MIDI synthesizer.")
    ap.add_argument(
        "--wave",
        choices=["sine", "triangle", "square", "saw"],
        default="sine",
        help="output waveform (default: sine)",
    )
    ap.add_argument(
        "--volume",
        type=float,
        default=volume_position * 10.0,
        help="master volume, 0-10 (default: %(default)g)",
    )
    ap.add_argument(
        "--attack",
        type=float,
        default=attack_time * 1000.0,
        help="note attack time in milliseconds (default: %(default)g)",
    )
    ap.add_argument(
        "--release",
        type=float,
        default=release_time * 1000.0,
        help="note release time in milliseconds (default: %(default)g)",
    )
    ap.add_argument(
        "--clip-hardness",
        type=float,
        default=clip_hardness,
        help="soft-clip hardness exponent; low is softer "
             "(default: %(default)g)",
    )
    ap.add_argument(
        "--controller",
        help="MIDI input port name (default: auto-detect)",
    )
    ap.add_argument(
        "--device",
        help="audio output device, by name substring or index "
             "(default: system default)",
    )
    ap.add_argument(
        "--list-devices",
        action="store_true",
        help="list available audio devices and exit",
    )
    args = ap.parse_args()

    # List audio devices and exit, if requested.
    if args.list_devices:
        print(sounddevice.query_devices())
        return

    # Audio output device. An all-digit string is treated as
    # a device index; anything else is matched against device
    # names by `sounddevice`. None selects the system default.
    output_device = args.device
    if output_device is not None and output_device.isdigit():
        output_device = int(output_device)

    # Oscillator selected at startup.
    oscillator = oscillators[args.wave]

    # Apply volume and envelope settings from the command line.
    volume_position = min(1.0, max(0.0, args.volume / 10.0))
    output_gain = volume_to_gain(volume_position)
    attack_time = max(0.0, args.attack) / 1000.0
    release_time = max(0.0, args.release) / 1000.0
    clip_hardness = min(20.0, max(1.0, args.clip_hardness))

    # Open the controller.
    controller = open_controller(args)

    # Start audio playing. Must keep up with output from here on.
    output_stream = sounddevice.OutputStream(
        samplerate=sample_rate,
        channels=output_channels,
        blocksize=blocksize,
        device=output_device,
        callback=output_callback,
    )
    output_stream.start()
    print(f"gen4: playing {args.wave} wave — press Ctrl-C to stop")

    # Run the synthesizer until its stop key is pressed.
    try:
        while get_midi_event(controller):
            pass
    except KeyboardInterrupt:
        pass

    output_stream.stop()
    output_stream.close()

if __name__ == "__main__":
    main()
