import argparse
import muspy
import numpy as np
import os
import math
import pickle as pk

def get_resolution_threshold(mus, threshold=100):
    """Return the nearest even MusPy resolution needed to consider notes 'threshold' milliseconds apart as different (but no lower)."""
    return [round(((x.qpm / 60) ** -1 * 1000 / threshold) / 2) * 2 for x in mus.tempos]

def parse_input(true_data, path, threshold=100, resolution=None):
    """Parse a .txt file output from the follow.js API into a MusPy object."""
    'TODO: Add support for multiple tempos'
    with open(path) as f:
        raw_data = f.readlines()[1:]

    raw_data = np.array([[float(y) for y in x.strip("\n|,").split(",")] for x in raw_data]) # raw_data is now a list of (time, pitch, velocity) tuples
    parsed_data = np.zeros((raw_data.shape[0] // 2, 4))
    countoff_offset = true_data.time_signatures[0].numerator * (true_data.tempos[0].qpm / 60) ** -1 # time offset in milliseconds to account for countoff

    raw_res = get_resolution_threshold(true_data, threshold)[0]
    if resolution is not None:
        raw_res = resolution
    time2beats = lambda x: round(x * true_data.tempos[0].qpm * raw_res / 60)
    for i in np.arange(0, raw_data.shape[0], 2):
        parsed_data[i // 2] = np.array(
            [
                time2beats(raw_data[i][0] / 1000 - countoff_offset), # time in new resolution
                raw_data[i][1], # MIDI pitch
                time2beats((raw_data[i+1][0] - raw_data[i][0]) / 1000), # duration in new resolution
                raw_data[i][2] # velocity
            ],
        dtype=int)
    input_mus = muspy.from_note_representation(parsed_data.astype(int), resolution=raw_res)
    return input_mus

def clean_output(true_path, out_path, threshold=100, resolution=None):
    """Clean a .txt file output from the follow.js API into a MusPy object."""
    if true_path.endswith(".json"):
        true_mus = muspy.load(true_path)
    elif true_path.endswith(".xml"):
        true_mus = muspy.read_musicxml(true_path)
    elif true_path.endswith(".abc"):
        true_mus = muspy.read_abc(true_path)
    else:
        raise NotImplementedError("File type not supported.")
    if type(true_mus) == list:
        true_mus = true_mus[0]
    input_mus = parse_input(true_mus, out_path, threshold, resolution)

    finest_rhythm = round(sorted(np.unique((true_mus.to_note_representation()[:, 0] / true_mus.resolution) % 1))[1] ** -1) # the finest rhythm in the piece
    true_mus.adjust_resolution(target=math.lcm(input_mus.resolution, finest_rhythm))
    input_mus.adjust_resolution(target=math.lcm(input_mus.resolution, finest_rhythm))
    print(finest_rhythm, input_mus.resolution, true_mus.resolution)

    input_mus.barlines = true_mus.barlines
    bartimes = np.array([x.time for x in input_mus.barlines])
    bartime2n = {x: i for i,x in enumerate(bartimes)}
    labs = []
    for i, note in enumerate(input_mus.tracks[0].notes):
        note.label = int(
            note.time in true_mus.to_note_representation()[:,0] and note.pitch == true_mus.to_note_representation()[np.where(note.time == true_mus.to_note_representation()[:, 0]),1] # Label is 1 if the note is correct at resolution
        )
        note.bar_n = bartime2n[bartimes[bartimes <= note.time].max()] # Label with 0-indexed bar number
        labs.append(note.label)
    labs = np.array(labs)
    print((labs == 1).sum() / len(labs))
    input_mus.time_signatures = true_mus.time_signatures
    input_mus.tempos = true_mus.tempos
    input_mus.key_signatures = true_mus.key_signatures
    input_mus.metadata = true_mus.metadata
    return input_mus




parser = argparse.ArgumentParser(description='Parse follow.js input files.')
parser.add_argument('--input-dir', type=str, help='input file directory')
parser.add_argument('--target', type=str, help='target file')
parser.add_argument('--threshold', type=float, default=100, help='threshold')
parser.add_argument('--resolution', type=int, default=None, help='resolution')



if __name__ == "__main__":
    args = parser.parse_args()
    for file in os.listdir(args.input_dir):
        if file.endswith(".txt"):
            clean_mus = clean_output(args.target, os.path.join(args.input_dir, file), args.threshold, args.resolution)
            muspy.write_midi(os.path.join(args.input_dir, file.replace(".txt", ".mid")), clean_mus)
            pk.dump(clean_mus, open(os.path.join(args.input_dir, file.replace(".txt", ".pk")), 'wb'))
    
