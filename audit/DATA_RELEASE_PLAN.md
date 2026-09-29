# Data release plan

This is a practical provenance/licensing audit, not legal advice. SSAC27 currently states that the submission repository must contain the data used to conduct the research. The project must not claim compliance until SSAC confirms an acceptable route or the relevant owners grant redistribution permission.

| Source | Classification | Current decision | Reproducibility fallback |
|---|---|---|---|
| Betfair 2026 Brownlow Datathon dataset | **PRIVATE / RESTRICTED pending written permission** | Do not upload. The download page is public, but no dataset-specific redistribution licence is recorded locally. Betfair's general terms prohibit reproduction/distribution without written consent. Contact `datathon@betfair.com.au` for explicit research-repository permission. | Public source URL, download instructions for authorised users, schema, processing scripts, SHA256 `760dc1…040a`, and non-row-level results. |
| AFLCA coach votes contained in the datathon file | **PUBLIC BUT REDISTRIBUTION UNCLEAR** | Do not separately upload raw vote records until AFLCA/Betfair provenance and permission are confirmed. | Source links, variable definition, code requiring a user-supplied authorised file, hashes, aggregate results. |
| AFL Tables reconstructed 2026 Brownlow votes | **PUBLIC BUT REDISTRIBUTION UNCLEAR** | Do not upload the scraped HTML or reconstructed row-level file without permission/terms confirmation. | Source URL, retrieval date, parser, source hash, reconciliation table, aggregated results. |
| AFL official information-reform announcement | **PUBLIC BUT REDISTRIBUTION UNCLEAR** | Link and cite; do not republish page content. | Official URL and a short factual field mapping documented in the protocol. |
| Generated OOT and frozen predictions | **OUR OWN OUTPUT**, but derived from restricted/unclear inputs | Code and compact validation tables can be released; row-level prediction redistribution should be cleared because it preserves player-match structure derived from the source dataset. | Model methodology, hashes, aggregate evaluation tables, and scripts that regenerate after authorised data download. |
| Generated residual panel | **DERIVED DATA**, redistribution currently blocked | Do not publish row-level panel until source-data permissions and SSAC acceptance are confirmed. | Schema, construction script, hashes, synthetic fixture for tests, and aggregate coefficient/diagnostic outputs. |

## Immediate actions

1. Request written permission from Betfair to redistribute the competition CSV and derived player-match predictions for a non-commercial academic submission.
2. Ask SSAC whether a repository containing source URLs, download instructions, hashes, complete code, schema, and non-infringing aggregate outputs satisfies the requirement when third-party redistribution is prohibited.
3. Seek clarification from AFLCA/AFL Tables if their records must be included row by row.
4. Until written answers arrive, keep `data/raw/`, scraped HTML, reconstructed vote labels, OOF player rows, and residual player rows out of a public repository.

