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

# Attack time in seconds.
attack_time = 0.020
# Release time in seconds.
release_time = 0.1

# Fixed output gain. Leaves headroom and avoids clipping.
output_gain = 0.25

# Known MIDI controllers to auto-detect.
controllers = {
    'USB Oxygen 8 v2 MIDI 1',
}

# This count of the number of samples output so far is used
# to make sure that waveforms are generated with the right
# phase across blocks.
sample_clock = 0

# Generate an array of frame_count sample times starting at
# sample_clock.
def sample_times(frame_count):
    return np.linspace(
        sample_clock / sample_rate,
        (sample_clock + frame_count) / sample_rate,
        frame_count,
        dtype=np.float32,
    )

# Return a sine wave at frequency f over the given sample
# times t.
def sine_samples(t, f):
    return np.sin(2 * np.pi * f * t)

# Return a rising sawtooth wave at frequency f over the
# given sample times t.
def saw_samples(t, f):
    return (f * t) % 2.0 - 1.0

# Return a square wave at frequency f over the given sample
# times t.
def square_samples(t, f):
    return np.sign((f * t) % 2.0 - 1.0)

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

    samples *= output_gain

    # Must write into the existing array rather than
    # accidentally copying over the parameter.
    out_data[:] = np.reshape(samples, (frame_count, 1))

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
        # Unknown control changes are logged and ignored.
        else:
            print('control', mesg.control, mesg.value)
    # XXX Pitchwheel is currently logged and ignored.
    elif mesg.type == 'pitchwheel':
        print('pitchwheel', mesg.pitch)
    else:
        print('unknown MIDI message', mesg)
    return True

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
    global oscillator

    ap = argparse.ArgumentParser(description="Polyphonic MIDI synthesizer.")
    ap.add_argument(
        "--wave",
        choices=["sine", "triangle", "square", "saw"],
        default="sine",
        help="output waveform (default: sine)",
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

    # Open the controller.
    controller = open_controller(args)

    # Start audio playing. Must keep up with output from here on.
    output_stream = sounddevice.OutputStream(
        samplerate=sample_rate,
        channels=1,
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
