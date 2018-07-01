import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trainfile', type=str, required=True)
parser.add_argument('--outfile', type=str, required=True)
args = parser.parse_args()

out_list = []
f = open(args.trainfile)
out_list.append(f.readline()) # read header
for line in f:
    rating = int(line.strip().split(",")[-1])
    multi_num = 10//rating
    for i in range(multi_num):
        out_list.append(line)
outf = open(args.outfile, "w")
outf.write("".join(out_list))
