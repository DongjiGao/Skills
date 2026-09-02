import argparse
from jiwer import wer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--reffile", required=True)
    parser.add_argument("--hypfile", required=True)
    parser.add_argument("--punctmappingfile")
    parser.add_argument("--wordmappingfile") # new for subtask 2. It is a fixed file, like reffile (not submission dependent)

    args = parser.parse_args()

    hyps = {}
    with open(args.hypfile, 'r') as f:
        for line in f:
            line = line.strip().lower().split()
            uttid = line[0]
            text = ' '.join(line[1:])
            hyps[uttid] = text

    refs = {}
    with open(args.reffile, 'r') as f:
        for line in f:
            line = line.strip().lower().split()
            uttid = line[0]
            text = ' '.join(line[1:])
            refs[uttid] = text

    # Apply punct mapping
    if args.punctmappingfile:
        # Read mapping
        punctmap = {}
        with open(args.punctmappingfile, 'r') as f:
            for line in f:
                line = line.strip().split()
                a = line[0]
                b = ' '.join(line[1:])
                if a not in punctmap:
                    punctmap[a] = b

        # Apply mapping
        for uttid in hyps:
            hyps[uttid] = ' '.join(punctmap[word] if word in punctmap else word for word in hyps[uttid].split())

    errs = 0
    lengths = 0

    for uttid, ref in refs.items():
        hyp = hyps[uttid]
        if not ref: # Empty ref
            # Important for subtask 2. Some refs are empty.
            # Since length = 0, will not contribute to WER anyways
            continue
        else:
            length = len(ref.split())
            err = wer(ref,hyp)*length
            
            errs += err
            lengths += length

    word_error_rate = (errs/lengths)*100

    print(f"Normal WER: {word_error_rate:.2f}")

    # Applying mapping on hyps for transliterated WER
    if args.wordmappingfile:
        # Read mapping
        ind2eng = {}
        with open(args.wordmappingfile, 'r') as f:
            for line in f:
                line = line.strip().split()
                a = line[0]
                b = ' '.join(line[1:])
                if a not in ind2eng:
                    ind2eng[a] = b

        # Apply mapping
        for uttid in hyps:
            hyps[uttid] = ' '.join(ind2eng[word] if word in ind2eng else word for word in hyps[uttid].split())

        errs = 0
        lengths = 0

        for uttid, ref in refs.items():
            hyp = hyps[uttid]
            if not ref: # Empty ref
                # Important for subtask 2. Some refs are empty.
                # Since length = 0, will not contribute to WER anyways
                continue
            else:
                length = len(ref.split())
                err = wer(ref,hyp)*length
                
                errs += err
                lengths += length

        word_error_rate = (errs/lengths)*100

        print(f"Transliterated WER: {word_error_rate:.2f}")