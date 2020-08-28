import os


def main(s, t):
    for label in ['train', 'val']:
        pth = "./data/s{}/{}".format(s, label)
        neg_pth = os.path.join(pth, "t{}.0".format(t))
        pos_pth = os.path.join(pth, "t{}.1".format(t))
        all_with_label_pth = os.path.join(pth, "t{}.label.all".format(t))
        neg_f = open(neg_pth, 'r').readlines()
        pos_f = open(pos_pth, 'r').readlines()
        with open(all_with_label_pth, 'w') as fo:
            for line in neg_f:
                fo.write("0\t")
                fo.write(line.strip().strip('\n')+"\n")
            for line in pos_f:
                fo.write("1\t")
                fo.write(line.strip().strip('\n')+"\n")


if __name__ == "__main__":
    for s in [1, 2]:
        for t in range(1, 8):
            main(s, t)
    print('done')
