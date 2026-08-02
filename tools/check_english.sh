#!/usr/bin/env bash
#
# Fails if Spanish appears anywhere except README.es.md.
#
# This exists because the claim it enforces was made, and was false. Commit
# eec72a2 -- "docs: finish the English pass; nothing but README.es.md is Spanish
# now" -- left 22 Spanish lines in .github/workflows/ci.yml, 29 in
# CMakeLists.txt, and printed strings in bench/bench.cpp and four other files:
# the job names on the Actions tab and the benchmark's own output, which are
# among the first things anybody sees. Sixty-four lines in fifteen files.
#
# The lesson is the one .github/workflows/ci.yml already states about
# .clang-format: a standard nobody enforces is worse than no standard, because
# it claims something the tree does not meet. So the claim became a test.
#
# Three things about how it is written, each of which was a false positive
# first:
#
#   LC_ALL. Without a UTF-8 locale, grep matches the character class byte by
#   byte, and every accented Spanish letter shares its lead byte 0xC3 with the
#   multiplication sign and the superscripts the documentation is full of. The
#   first version of this flagged "1.78x" and "4096^3".
#
#   -I. git ls-files includes 23 binary fixtures and the MNIST subset. Random
#   bytes match any word list; skipping binaries takes the hit count from 73
#   to 64.
#
#   No "sin". It is a Spanish preposition and it is also std::sin, which this
#   engine calls in the positional encoding and in two tests.
#
# The list is function words and a few nouns, not a language model: words that
# exist in both languages (real, final, error, total, no, red, data, model) are
# left out on purpose. It catches prose. It does not catch a short label like
# "// 1. Datos", and it is not meant to -- what it protects against is a
# paragraph of Spanish comment sliding back in.
#
# Usage: tools/check_english.sh

set -euo pipefail
export LC_ALL=C.UTF-8

# README.es.md is the point of the exercise. README.md links to it by name, in
# Spanish, and that line is deliberate.
#
# This file excludes itself, and that is not laziness: the word list below is
# Spanish by construction, so a check that scanned itself would always fail. It
# already did. The list was moved out of .github/workflows/ci.yml precisely to
# stop it matching itself there, and landed in a tracked file with the same
# problem -- which passed locally, because the script was still untracked when it
# was tested and `git ls-files` only lists what is committed. Committing it was
# what broke it.
readonly EXCLUDE='^(README\.es\.md|tools/check_english\.sh)$'
readonly ALLOW='README\.md:[0-9]+:\*\[Versión en español\]'

readonly WORDS='[áéíóúñÁÉÍÓÚÑ¿¡]|\b(que|para|porque|cuando|donde|esto|esta|estos|estas|los|las|del|una|unos|unas|con|por|pero|como|desde|hasta|entre|sobre|cada|todo|toda|todos|todas|otro|otra|aunque|entonces|siempre|nunca|entrada|salida|fichero|ficheros|cadena|cadenas|pruebas|compilar|ejecutar|configurar|instalar|memoria|hilos|puntero|valores|datos|modelo|entrenamiento|dispositivo|convoluciones|iteracion|inferencia|resultados|ejemplos|calculo|tambien|ademas|despues|verificar|comprobar|devuelve)\b'

hits=$(git ls-files | grep -vE "$EXCLUDE" | xargs grep -IinE "$WORDS" 2>/dev/null \
       | grep -vE "$ALLOW" || true)

if [ -n "$hits" ]; then
    echo "Spanish found outside README.es.md:"
    echo
    echo "$hits"
    echo
    echo "$(echo "$hits" | wc -l) line(s). Translate them, or extend ALLOW above"
    echo "with a reason if the Spanish is deliberate."
    exit 1
fi

echo "English everywhere except README.es.md."
