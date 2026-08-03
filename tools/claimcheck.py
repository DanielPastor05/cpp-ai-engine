#!/usr/bin/env python3
"""claimcheck -- fail the build when a number in your docs goes stale.

Numbers get copied. A benchmark result lands in the README, in the performance
notes, in a chart script and in a blog post, and then one of them gets updated.
Nothing tells you about the other three.

claimcheck takes a JSON file describing every number you publish, and checks:

  citations   every claim appears in each file that cites it
  retired     no withdrawn number appears anywhere it is not explicitly allowed
  derived     a ratio equals the numbers it is a ratio of

It measures nothing and runs nothing. It reads text and compares it against
data, so it has no timing, no environment and nothing to flake, which is what
lets it block a merge.

Usage:
    claimcheck.py                       # reads ./claims.json
    claimcheck.py --claims docs/perf.json
    claimcheck.py --root .. --claims claims.json

Exit code 0 if everything is consistent, 1 otherwise.

No dependencies. Python 3.7+.
"""

import argparse
import json
import os
import re
import sys

__version__ = "1.0.0"

# Numbers as prose and code write them: 4.70, 24.1, 16 489, 9,258.
#
# Grouped is matched strictly -- a separator only counts as a thousands
# separator when exactly three digits follow it -- because a loose version reads
# the "7660, 0" of a Python tuple as one malformed number and then reports the
# 7660 beside it as missing. The separators include the non-breaking and thin
# spaces that typeset documents use.
GROUPED = r"\d{1,3}(?:[\u00a0\u2009 ,]\d{3})+(?:\.\d+)?"
PLAIN = r"\d+(?:\.\d+)?"
NUMBER = re.compile(GROUPED + "|" + PLAIN)
SEPARATORS = re.compile(r"[\u00a0\u2009 ,]")


def numbers_in_text(text):
    """Every numeric literal in a string, as floats."""
    found = set()
    for match in NUMBER.finditer(text):
        try:
            found.add(float(SEPARATORS.sub("", match.group(0))))
        except ValueError:
            continue
    return found


def close(a, b, rel=0.005):
    """True if a published number rounds to the same short form as b."""
    return abs(a - b) <= max(abs(b) * rel, 0.005)


def check(claims_path, root):
    with open(claims_path, encoding="utf-8") as f:
        data = json.load(f)

    claims = data.get("claims", [])
    retired = data.get("retired", [])
    # Where a stale number could hide. Declared in the file rather than guessed,
    # because "every file in the repository" would scan lockfiles and fixtures.
    tracked = data.get("tracked", [])

    by_id = {c["id"]: c for c in claims}
    cache = {}
    problems = []
    citations = 0

    def numbers_in(rel_path):
        if rel_path not in cache:
            with open(os.path.join(root, rel_path), encoding="utf-8") as f:
                cache[rel_path] = numbers_in_text(f.read())
        return cache[rel_path]

    # --- 1. every claim appears where it says it does -----------------------
    for claim in claims:
        value = claim["value"]
        for rel_path in claim.get("cited_in", []):
            if not os.path.exists(os.path.join(root, rel_path)):
                problems.append("%s: cites %s, which does not exist" % (claim["id"], rel_path))
                continue
            citations += 1
            if not any(close(n, value) for n in numbers_in(rel_path)):
                problems.append(
                    "%s: %s does not contain %s%s\n"
                    "    measured how: %s\n"
                    "    either the file is stale, or the claims file is"
                    % (claim["id"], rel_path, value,
                       " " + claim["unit"] if claim.get("unit") else "",
                       claim.get("how", "(not recorded)")))

        # --- 3. a ratio agrees with its own operands ------------------------
        if "derived_from" in claim:
            try:
                a, b = (by_id[k]["value"] for k in claim["derived_from"])
            except KeyError as missing:
                problems.append("%s: derived_from names %s, which is not a claim"
                                % (claim["id"], missing))
                continue
            expected = a / b if b else float("inf")
            if not close(expected, value, rel=0.02):
                problems.append(
                    "%s: published as %s, but %s / %s = %s / %s = %.3f"
                    % (claim["id"], value, claim["derived_from"][0], claim["derived_from"][1],
                       a, b, expected))

    # --- 2. no retired number has crept back in -----------------------------
    #
    # Line by line, and with an allowlist, because documentation that is worth
    # reading quotes its own withdrawn numbers: "an earlier draft reported X",
    # the table of what something cost before it was fixed. A blanket ban would
    # delete the best paragraphs. So every deliberate mention is registered by
    # the text around it, and an occurrence anywhere else fails.
    #
    # That distinction -- quoted as history versus stated as current -- is the
    # one that matters, and a file can contain both. The project this came from
    # had a document saying "1.94x behind" in one paragraph and 1.21x in its own
    # table, forty lines apart, for weeks.
    for old in retired:
        allowed_by_file = {}
        for entry in old.get("documented_in", []):
            allowed_by_file.setdefault(entry["file"], []).append(entry["context"])

        for rel_path in tracked:
            full = os.path.join(root, rel_path)
            if not os.path.exists(full):
                continue
            with open(full, encoding="utf-8") as f:
                lines = f.readlines()
            allowed = allowed_by_file.get(rel_path, [])
            for lineno, line in enumerate(lines, 1):
                if not any(close(n, old["value"]) for n in numbers_in_text(line)):
                    continue
                if any(ctx in line for ctx in allowed):
                    continue
                problems.append(
                    "retired value %s appears at %s:%d\n"
                    "    %s\n"
                    "    it was: %s\n"
                    "    replaced by: %s\n"
                    "    if this mention is deliberate history, register it under\n"
                    "    documented_in with the text that marks it as past"
                    % (old["value"], rel_path, lineno, line.strip()[:96],
                       old.get("was", "(not recorded)"),
                       old.get("replaced_by", "(not recorded)")))

    return claims, citations, retired, problems


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fail when a number published in your docs has gone stale.")
    parser.add_argument("--claims", default="claims.json",
                        help="the JSON file describing what you publish (default: claims.json)")
    parser.add_argument("--root", default=None,
                        help="directory the cited paths are relative to "
                             "(default: the claims file's directory)")
    parser.add_argument("--version", action="version", version="claimcheck " + __version__)
    args = parser.parse_args(argv)

    if not os.path.exists(args.claims):
        print("No claims file at %s. See the README for what one looks like." % args.claims)
        return 2

    root = args.root if args.root else os.path.dirname(os.path.abspath(args.claims)) or "."

    try:
        claims, citations, retired, problems = check(args.claims, root)
    except json.JSONDecodeError as e:
        print("%s is not valid JSON: %s" % (args.claims, e))
        return 2

    if problems:
        print("Published numbers have drifted from %s:\n" % args.claims)
        for problem in problems:
            print("  %s\n" % problem)
        print("%d problem(s)." % len(problems))
        return 1

    print("%d claims, %d citations, %d retired values absent. All consistent."
          % (len(claims), citations, len(retired)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
