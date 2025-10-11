# scripts/poison_train.py
import argparse
import random
from collections import Counter
import os

def read_train(path):
    user_items = {}
    max_uid = -1
    item_counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            uid = int(parts[0])
            items = [int(x) for x in parts[1:]]
            user_items[uid] = items
            if uid > max_uid:
                max_uid = uid
            item_counter.update(items)
    return user_items, max_uid, item_counter

def write_train(user_items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for uid in sorted(user_items.keys()):
            items = user_items[uid]
            items_str = " ".join(str(i) for i in items)
            f.write(f"{uid} {items_str}\n")

def main(args):
    random.seed(args.seed)
    print("Reading", args.path)
    user_items, max_uid, item_counter = read_train(args.path)
    n_users = len(user_items)
    print("original users:", n_users, "max_uid:", max_uid)
    # determine number of fake users
    num_fake = int(round(n_users * args.fraction))
    print(f"creating {num_fake} fake users (fraction {args.fraction})")
    # build item list and probabilities (popularity)
    items, counts = zip(*item_counter.items())
    total = sum(counts)
    probs = [c/total for c in counts]

    # for stealth: sample some real items to add beside the target
    for i in range(1, num_fake+1):
        new_uid = max_uid + i
        # each fake user will interact with target + a few other items
        chosen = set()
        chosen.add(args.target)
        # sample `num_others` items according to popularity distribution
        num_others = args.num_others
        while len(chosen) < 1 + num_others:
            pick = random.choices(items, probs)[0]
            chosen.add(pick)
        user_items[new_uid] = sorted(chosen)

    # write to out
    print("writing poisoned train to", args.out)
    write_train(user_items, args.out)
    print("done. New total users:", len(user_items))
    print("Note: keep test.txt unchanged for fair eval.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="original train.txt path")
    p.add_argument("--out", required=True, help="poisoned train output path")
    p.add_argument("--target", type=int, required=True, help="target item id to promote")
    p.add_argument("--fraction", type=float, default=0.01, help="fraction of fake users (e.g. 0.01=1%)")
    p.add_argument("--num_others", type=int, default=9, help="number of other items each fake user interacts with (besides target)")
    p.add_argument("--seed", type=int, default=2025)
    args = p.parse_args()
    main(args)


'''
Reads the original train.txt, counts items.

Adds num_fake = round(num_users * fraction) new fake users with new user IDs (max_uid+1 ...).

Each fake user has interactions: target item + num_others other items sampled from the real item popularity (so stealthy).

Writes poisoned file to --out. Does not modify test.txt.
'''