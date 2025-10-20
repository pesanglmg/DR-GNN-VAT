# scripts/inspect_train.py
import argparse
from collections import Counter

def inspect(path, topk=30):
    item_counter = Counter()
    user_set = set()
    n_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            uid = int(parts[0])
            user_set.add(uid)
            items = [int(x) for x in parts[1:]]
            item_counter.update(items)
            n_lines += 1

    n_users = len(user_set)
    n_items = len(item_counter)
    total_interactions = sum(item_counter.values())

    print("File:", path)
    print("Lines (users in file):", n_lines)
    print("Unique users:", n_users)
    print("Unique items (observed):", n_items)
    print("Total interactions:", total_interactions)
    print()
    print(f"Top {topk} items by frequency (item_id : count):")
    for item, cnt in item_counter.most_common(topk):
        print(f"{item}\t{cnt}")

    # show some candidate target items (not top popular) -> e.g. items ranked 200..220
    print()
    print("Candidate less-popular items (ranks 200..220):")
    items_sorted = item_counter.most_common()
    if len(items_sorted) > 220:
        for rank in range(199, 219):
            it, ct = items_sorted[rank]
            print(f"rank {rank+1}: {it}\t{ct}")
    else:
        print("Dataset not large enough to list rank 200; listing tail 20 instead:")
        for it, ct in items_sorted[-20:]:
            print(it, ct)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="path to train.txt")
    parser.add_argument("--topk", type=int, default=30)
    args = parser.parse_args()
    inspect(args.path, topk=args.topk)

