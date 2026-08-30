# Licence texts for third-party components

One file per upstream, so the terms travel with the binaries rather than living
behind a link that can move. `../NOTICE` says which files each one covers and
`../PROVENANCE.json` maps every published binary to its lineage.

| File | Covers |
| --- | --- |
| `MIT-GreatSage-dev-Assay.txt` | the FACT_CHECK builds forked from Assay |
| `MIT-ssoni4751-telegraph-wasm-scoring.txt` | the CHAT_COMPLETION builds forked from that module |
| `MIT-PugarHuda-amanat.txt` | the GAME_RESULT builds forked from amanat |
| `Apache-2.0-all-MiniLM-L6-v2.txt` | the embedded all-MiniLM-L6-v2 weights |
| `MIT-thenlper-gte-small.txt` | the embedded gte-small weights |
| `PDDL-1.0-GloVe.txt` | the embedded GloVe 6B vectors |

Two of these are reproductions rather than copies of an upstream file, because the
upstream declares MIT in a README or in model metadata and ships no LICENSE file.
Each says so at the top. If the author later publishes one, it replaces the
reproduction verbatim.
