
import sys
import regex
import datetime

date_regex = regex.compile("[0-9]{2}/[0-9]{2}/[0-9]{4}")

replacements = {"Oakland Raiders": "Las Vegas Raiders",
                "Washington Redskins": "Washington Commanders",
                "Washington Football Team": "Washington Commanders",
                "San Diego Chargers": "Los Angeles Chargers"}

replacement_keys = set(replacements.keys())

fields = ["Date", "TeamA", "ScoreA", "TeamB", "ScoreB", "OT", "BoxScore"]

def process_line(line):
    
    result = None
    # Catch non-game lines of the file
    if date_regex.match(line) is None:
        return None

    # Reformat the date
    sp_line = line.split("\t")
    sp_line[0] = datetime.datetime.strptime(sp_line[0],"%m/%d/%Y").strftime("%Y-%m-%d")
    line = "\t".join(sp_line)

    for k, v in replacements.items():
        line = line.replace(k,v)
    
    return line


def main(in_tsv, out_tsv):

    with open(in_tsv, "r") as in_file:
        # Only keep the lines containing game records
        lines = ["\t".join(fields)+"\n"]
        for line in in_file.readlines():
            processed = process_line(line)
            if processed is not None:
                # Change the date format
                lines.append(processed)

        # Write the lines to out_tsv
        with open(out_tsv, "w") as out_file:
            out_file.writelines(lines)



if __name__=="__main__":

    in_tsv = sys.argv[1]
    out_tsv = sys.argv[2]

    main(in_tsv, out_tsv)

    
