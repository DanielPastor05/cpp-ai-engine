#!/usr/bin/env python3
"""Apply the published-number rules to the parts of GitHub that are not files.

`claimcheck.py` reads the eight paths listed under `tracked` in
docs/performance.json. Every one of them is in the repository, and that turned
out to be the hole: the repository description said MNIST trained "5.5x faster on
the GPU" and the v0.6.0 release page said "4.3 s against 23.5 s" while
performance.json had said 4.0 and 24.1 since the day the claims file was
created. The two most-read surfaces on GitHub disagreed with the checked ones,
and disagreed from the moment they were written -- there was no check that could
have noticed, because neither is a file.

It is the same shape as two other defects found the same week. The install rules
were broken because nothing ever consumed the exported package from outside the
repository. tools/check_english.sh passed while every demo printed Spanish,
because it scans source prose and a printed label is not prose. Each check
covered a surface adjacent to the one that mattered.

So this covers the remaining one. Two rules, and they are deliberately different
because the two surfaces are:

  description   Every number in it must be a tracked claim. It is one sentence
                long, so there is nothing else a number could legitimately be,
                and an allowlist that small is worth having exact.

  releases      No retired number may appear, under the same context allowlist
                claimcheck uses -- a release page is a historical document and
                quoting what something used to cost is the point of one. What it
                may not do is state a dead number as current.

Needs the `gh` CLI, authenticated. Skips itself with exit 0 when `gh` is absent
or unauthenticated, so a clone without it still builds; CI has it.

Usage:
    tools/check_github_metadata.py
    tools/check_github_metadata.py --repo owner/name --claims docs/performance.json
"""

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claimcheck  # noqa: E402  (vendored; see tools/README or the commit that added it)


def gh_json(args):
    """Run `gh` and parse its JSON, or return None if gh cannot be used.

    encoding is pinned to UTF-8 rather than left to `text=True`, which decodes
    with the locale's codec: on Windows that is cp1252, and the first release
    body this was pointed at contains an em dash and a multiplication sign. The
    decode raised inside subprocess's reader thread, where the traceback prints
    but the exception does not propagate -- so stdout arrived as None and the
    failure surfaced two frames away as a TypeError from json.loads.
    """
    try:
        out = subprocess.run(["gh"] + args, capture_output=True, timeout=60,
                             encoding="utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout:
        return None
    try:
        return json.loads(out.stdout)
    except ValueError:
        return None


# A number welded to a word is part of a name, not a measurement: C++17, fp32,
# sm_86, RTX 3060 Ti's model number. claimcheck reads whole documents, where
# those are common enough that it does not try to tell them apart -- it only ever
# asks whether a specific value is present. Here the rule runs the other way,
# every number has to be accounted for, so the distinction has to be made.
GLUED = re.compile(r"[A-Za-z+_]\d")


def standalone_numbers(text):
    """Numbers that are measurements rather than parts of an identifier."""
    out = []
    for match in claimcheck.NUMBER.finditer(text):
        start = match.start()
        if start > 0 and GLUED.match(text[start - 1:start + 1]):
            continue
        try:
            out.append(float(match.group().replace(",", "").replace(" ", "")))
        except ValueError:
            continue
    return out


def check_description(description, claims):
    """Every number in the one-sentence description must be a claim we track."""
    problems = []
    values = [c["value"] for c in claims]
    for n in standalone_numbers(description):
        if any(claimcheck.close(n, v) for v in values):
            continue
        problems.append(
            "the repository description contains %s, which is not a tracked claim\n"
            "    %s\n"
            "    the description is one sentence and is the first thing anyone reads:\n"
            "    every number in it has to be one performance.json vouches for.\n"
            "    tracked values: %s"
            % (n, description.strip()[:160], ", ".join(str(v) for v in sorted(values))))
    return problems


def check_release_bodies(releases, retired):
    """No retired number stated as current, with claimcheck's context allowlist."""
    problems = []
    for old in retired:
        allowed = [e["context"] for e in old.get("documented_in", [])
                   if e["file"] == "github:releases"]
        for rel in releases:
            body = rel.get("body") or ""
            for lineno, line in enumerate(body.splitlines(), 1):
                if not any(claimcheck.close(n, old["value"])
                           for n in claimcheck.numbers_in_text(line)):
                    continue
                if any(ctx in line for ctx in allowed):
                    continue
                problems.append(
                    "retired value %s appears in release %s, line %d\n"
                    "    %s\n"
                    "    it was: %s\n"
                    "    if the mention is deliberate history, register it under this\n"
                    "    value's documented_in with file \"github:releases\" and the\n"
                    "    text that marks it as past"
                    % (old["value"], rel.get("tagName", "?"), lineno, line.strip()[:96],
                       old.get("was", "(not recorded)")))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--claims", default="docs/performance.json")
    parser.add_argument("--repo", default=None,
                        help="owner/name; defaults to whatever gh infers from the checkout")
    args = parser.parse_args(argv)

    with open(args.claims, encoding="utf-8") as f:
        data = json.load(f)

    repo_args = ["--repo", args.repo] if args.repo else []

    meta = gh_json(["repo", "view"] + repo_args + ["--json", "description"])
    if meta is None:
        print("gh is unavailable or not authenticated; skipping the GitHub surfaces.")
        return 0

    releases = gh_json(["release", "list"] + repo_args
                       + ["--limit", "20", "--json", "tagName"]) or []
    for rel in releases:
        full = gh_json(["release", "view", rel["tagName"]] + repo_args + ["--json", "body"])
        rel["body"] = (full or {}).get("body", "")

    problems = check_description(meta.get("description") or "", data.get("claims", []))
    problems += check_release_bodies(releases, data.get("retired", []))

    if problems:
        print("GitHub's copy of the numbers has drifted from %s:\n" % args.claims)
        for p in problems:
            print("  " + p + "\n")
        return 1

    print("description and %d release page(s) agree with %s."
          % (len(releases), args.claims))
    return 0


if __name__ == "__main__":
    sys.exit(main())
