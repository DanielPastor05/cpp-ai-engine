#!/usr/bin/env python3
"""Fails if a published number has drifted from docs/performance.json.

The numbers this repository leads with lived in four files and were copied
between them by hand: README.md, docs/PERFORMANCE.md, docs/CUDA.md and
tools/plot_benchmarks.py. Nothing checked they agreed, and during the session
that measured them a stale "6.92x" survived in the README for several commits
after the honest number had become 5.47x. Nobody noticed; there was nothing to
notice with.

So docs/performance.json is the source, every citation is declared, and this
walks each file looking for the value.

**It does not measure anything**, and that is the design rather than a
limitation. It compares text against data on a checkout: no GPU, no timing, no
warm-up, no thermal drift. It cannot flake, which is what lets it block a merge.
The re-measuring half lives in tools/check_perf.py and deliberately stays out of
CI -- see that file's header.

Three checks, and the second exists because the first is weaker than it looks.
A citation check only proves the value is present *somewhere* in the file, so a
file holding both the new number and the old one passes it -- which is exactly
how docs/CUDA.md carried "1.94x behind cuBLAS" in one paragraph and 1.21x in its
own table, forty lines apart, for weeks. So retired values are listed too, and
any of them appearing anywhere in the tree is a failure. The third is that a
derived claim must equal the numbers it derives from, so a ratio cannot quietly
contradict its own operands.

Usage:
    python tools/check_claims.py
"""

import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAIMS = os.path.join(REPO, "docs", "performance.json")

# Numbers as prose and code write them. Grouped strictly -- a separator only
# counts as a thousands separator when exactly three digits follow -- because the
# loose version read the "7660, 0" of a Python tuple as one malformed number and
# then reported the 7660 beside it as missing.
GROUPED = r"\d{1,3}(?:[   ,]\d{3})+(?:\.\d+)?"
PLAIN = r"\d+(?:\.\d+)?"
NUMBER = re.compile(GROUPED + "|" + PLAIN)
SEPARATORS = re.compile(r"[   ,]")


# Every file a published number could hide in. Sources are not here: a magic
# constant in a kernel is not a claim about performance.
TRACKED = ["README.md", "README.es.md", "docs/PERFORMANCE.md", "docs/CUDA.md",
           "docs/DESIGN.md", "docs/ENGINEERING.md", "docs/PROFILING.md",
           "tools/plot_benchmarks.py"]


def numbers_in_text(text):
    """Every numeric literal in a string, as floats."""
    found = set()
    for m in NUMBER.finditer(text):
        try:
            found.add(float(SEPARATORS.sub("", m.group(0))))
        except ValueError:
            continue
    return found


def numbers_in(path):
    """Every numeric literal in a file, as floats."""
    with open(path, encoding="utf-8") as f:
        return numbers_in_text(f.read())


def close(a, b, rel=0.005):
    """A published number matches if it rounds to the same short form."""
    return abs(a - b) <= max(abs(b) * rel, 0.005)


def main():
    with open(CLAIMS, encoding="utf-8") as f:
        data = json.load(f)

    by_id = {c["id"]: c for c in data["claims"]}
    cache = {}
    problems = []
    checked = 0

    for claim in data["claims"]:
        value = claim["value"]
        for rel_path in claim["cited_in"]:
            path = os.path.join(REPO, rel_path)
            if not os.path.exists(path):
                problems.append(f"{claim['id']}: cites {rel_path}, which does not exist")
                continue
            if rel_path not in cache:
                cache[rel_path] = numbers_in(path)
            checked += 1
            if not any(close(n, value) for n in cache[rel_path]):
                problems.append(
                    f"{claim['id']}: {rel_path} does not contain {value} {claim['unit']}\n"
                    f"    measured how: {claim['how']}\n"
                    f"    either the file is stale, or performance.json is")

        # A ratio has to agree with the two numbers it is a ratio of, or the
        # document contradicts itself in a way no reader would catch.
        if "derived_from" in claim:
            a, b = (by_id[k]["value"] for k in claim["derived_from"])
            expected = a / b
            if not close(expected, value, rel=0.02):
                problems.append(
                    f"{claim['id']}: published as {value}, but "
                    f"{claim['derived_from'][0]} / {claim['derived_from'][1]} "
                    f"= {a} / {b} = {expected:.3f}")

    # Retired values: the citation check above proves a number is present, never
    # that a stale one is absent. This is the other direction.
    #
    # It is line-by-line and it carries an allowlist, because this documentation
    # deliberately quotes its own withdrawn numbers -- "an earlier draft reported
    # 4 663 GFLOP/s", the table of what the composed Conv2d cost before reshape
    # became a view. Those are the best paragraphs in the docs and a blanket ban
    # would delete them. So every deliberate mention is registered by the text
    # around it, and an occurrence anywhere else fails. That is the distinction
    # that matters: quoted as history versus stated as current. docs/CUDA.md had
    # one of each, forty lines apart.
    for old in data.get("retired", []):
        for rel_path in TRACKED:
            path = os.path.join(REPO, rel_path)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            allowed = [a["context"] for a in old.get("documented_in", [])
                       if a["file"] == rel_path]
            for lineno, line in enumerate(lines, 1):
                if not any(close(n, old["value"]) for n in numbers_in_text(line)):
                    continue
                if any(ctx in line for ctx in allowed):
                    continue
                problems.append(
                    f"retired value {old['value']} appears at {rel_path}:{lineno}\n"
                    f"    {line.strip()[:96]}\n"
                    f"    it was: {old['was']}\n"
                    f"    replaced by: {old['replaced_by']}\n"
                    f"    if this mention is deliberate history, register it under\n"
                    f"    documented_in with the text that marks it as past")

    if problems:
        print("Published numbers have drifted from docs/performance.json:\n")
        for p in problems:
            print(f"  {p}\n")
        print(f"{len(problems)} problem(s). Re-measure with tools/check_perf.py, update")
        print("docs/performance.json, then update every file that cites the number.")
        return 1

    retired = len(data.get("retired", []))
    print(f"{len(data['claims'])} claims, {checked} citations, "
          f"{retired} retired values absent. All consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
