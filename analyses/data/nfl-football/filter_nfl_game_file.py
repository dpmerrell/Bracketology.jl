
import sys
import regex


def main(in_tsv, out_tsv):

    re = regex.compile("[0-9]{2}/[0-9]{2}/[0-9]{4}")

    with open(in_tsv, "r") as in_file:
        # Only keep the lines containing game records
        lines = []
        for line in in_file.readlines():
            if re.match(line) is not None:
                lines.append(line) 

        # Write the lines to out_tsv
        with open(out_tsv, "w") as out_file:
            out_file.writelines(lines)


if __name__=="__main__":

    in_tsv = sys.argv[1]
    out_tsv = sys.argv[2]

    main(in_tsv, out_tsv)

    
