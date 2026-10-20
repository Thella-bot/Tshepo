"""Ad-hoc: show how the parser handles a real message file. Not part of the package."""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tshepo.evaluation import load_jsonl
from tshepo.parsing.parser import classify

for f in sys.argv[1:]:
    recs = load_jsonl(f)
    print(f"== {f}: {len(recs)} messages")
    c = Counter()
    for r in recs:
        res = classify(r["body"], r["sender"])
        ps = res["parsed"]
        c[res["label"]] += 1
        print(f"  {res['label'][:10]:11} op={str(ps.operator):8} dir={str(ps.direction):5} "
              f"amt={str(ps.amount):9} bal={str(ps.balance):9} cat={str(ps.category):8} "
              f"flags={','.join(res['flags']) or '-':22} | {r['body'][:80]}")
    print("  ", dict(c))
