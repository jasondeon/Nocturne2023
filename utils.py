import torch
import pretty_midi as pm
import dataclasses as dc


@dc.dataclass
class MidiToken():
    type: str
    value: int

    def __post_init__(self):
        self.value = int(self.value)
        self.type = str(self.type)
    
    def __str__(self):
        return f"({self.type:<12},{self.value:<3})"

    def key_mapping(event):
        '''
        Given a MIDI event, return a unique index.
        '''
        if event.type == "NOTE_ON":
            return list(range(0,128))[event.value]
        if event.type == "NOTE_OFF":
            return list(range(128,256))[event.value]
        if event.type == "TIME_SHIFT":
            return list(range(256,356))[int(event.value // 10) - 1]
        if event.type == "SET_VELOCITY":
            if event.value>127:
                return 387
            return list(range(356,388))[int(event.value // 4)]

    def tok_mapping(token):
        '''
        Given a MIDI token index, return the associated MIDI event.
        '''
        if torch.is_tensor(token):
            token = token.item()
        if token >=0 and token < 128:
            return MidiToken("NOTE_ON", token)
        if token >= 128 and token < 256:
            return MidiToken("NOTE_OFF", token-128)
        if token >= 256 and token < 356:
            return MidiToken("TIME_SHIFT", ((token-256) * 10) + 10)
        if token >= 356 and token < 388:
            return MidiToken("SET_VELOCITY", (token-356) * 4)


def mid2dat_anna(midi_path=None, midi_data=None):
    arr = []
    if midi_data is None:
        if not isinstance(midi_path, str):
           midi_path = midi_path.as_posix()
        midi_data = pm.PrettyMIDI(midi_path)
    x = midi_data.instruments[0].get_piano_roll(fs=100) # shape=(pitch, timestep)

    active_notes = [-1] * 128 # -1=inactive, else val=velocity
    time_acc = -10            # track time passed (ms) since last TIME_SHIFT (start at -10 to offset first increment)
    curr_vel = 0              # last SET_VELOCITY value
    # Iterate over timesteps
    for t in range(x.shape[1]):
        time_acc += 10
        for p in range(x.shape[0]):
            # When a note ends
            if active_notes[p] != -1 and active_notes[p] != x[p,t]:
                if time_acc:
                    arr.append(MidiToken("TIME_SHIFT", time_acc))
                    time_acc = 0
                active_notes[p] = -1
                arr.append(MidiToken("NOTE_OFF", p))
            # When a note starts
            if x[p,t] and active_notes[p] == -1:
                active_notes[p] = x[p,t]
                if time_acc:
                    arr.append(MidiToken("TIME_SHIFT", time_acc))
                    time_acc = 0
                if (x[p,t]//4)*4 != curr_vel:
                    curr_vel = min((x[p,t]//4)*4, 127)
                    arr.append(MidiToken("SET_VELOCITY", curr_vel))
                arr.append(MidiToken("NOTE_ON", p))
        if time_acc == 1000:
            arr.append(MidiToken("TIME_SHIFT", 1000))
            time_acc = 0
    # Write final NOTE_OFFs
    if active_notes:
        time_acc += 10
        arr.append(MidiToken("TIME_SHIFT", time_acc))
        for p in active_notes:
            if p != -1:
                arr.append(MidiToken("NOTE_OFF", p))
    return arr


def dat2mid_anna(seq, output_filename="test.mid"):
    assert seq is not None
    assert isinstance(seq[0], MidiToken)
    start_times = [-1] * 128 # -1=inactive, else val=start_time
    velocities  = [-1] * 128 # -1=inactive, else val=velocity
    curr_time = 0.0
    curr_vel = 0
    midi_data = pm.PrettyMIDI(initial_tempo=120.0)
    piano = pm.Instrument(program=pm.instrument_name_to_program('Acoustic Grand Piano'))
    for event in seq:
        if event.type == "NOTE_ON":
            start_times[event.value] = curr_time
            velocities[event.value] = curr_vel
        elif event.type == "NOTE_OFF" and start_times[event.value] != -1:
            note = pm.Note(velocity=velocities[event.value], pitch=event.value, start=start_times[event.value], end=curr_time)
            piano.notes.append(note)
            start_times[event.value] = -1
            velocities[event.value] = -1
        elif event.type == "TIME_SHIFT":
            curr_time += event.value / 1000
        elif event.type == "SET_VELOCITY":
            curr_vel = event.value
    midi_data.instruments.append(piano)
    return midi_data
    #midi_data.write(output_filename)