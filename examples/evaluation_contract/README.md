# Synthetic evaluator fixtures — not benchmark evidence

Every document, task, judgment, and confidence here is invented. The human/expert
status strings simulate the annotation gate for tests; **no actual adjudication
occurred**. Do not pool these outputs with research results.

`tasks.json` is the expected universe; `runs.json` covers all four tasks.
`annotations.json` demonstrates same-document constraints and incompatible
multihop paths. Both annotated runs have full union coverage but fail family
success. `annotations_proxy.json` tests the explicit proxy opt-in. Empty gold is
not structural negative-result evidence. `selective_judgments.json` independently
supplies losses and support; `calibration.json` uses disjoint synthetic dev IDs.
`corpus.json` is the strict document-ID universe.

Run the commands in `docs/EVALUATION_CONTRACT.md` from the repository root.
The report destination must be new; use a new filename for each rerun.
