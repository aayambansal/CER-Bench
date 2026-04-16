# NeurIPS 2026 E&D Track Submission Checklist

## Deadlines

- [ ] Abstract submitted by May 4, 2026 (AoE)
- [ ] Full paper submitted by May 6, 2026 (AoE)

## Paper Requirements

- [ ] Paper is within 9 pages (main content, excluding references and appendix)
- [ ] Uses NeurIPS 2026 LaTeX template
- [ ] Double-blind: no author names, affiliations, or identifying information
- [ ] No URLs pointing to identifying repos/profiles in main paper
- [ ] Supplementary material is properly referenced

## E&D Track Specific

- [ ] Paper clearly states whether contribution is: benchmark, evaluation protocol,
      dataset, data generator, audit, stress test, or RL environment
- [ ] Code and data are accessible at submission time
- [ ] New dataset includes Croissant metadata (ML schema.org)
- [ ] Dataset documentation includes: intended use, limitations, license, collection
      methodology, annotation process
- [ ] Benchmark includes: clear evaluation protocol, baseline implementations,
      reproducibility instructions

## Artifact Requirements

- [ ] Code repo is anonymized (no author names, no personal GitHub links)
- [ ] Dataset is hosted on anonymized platform or included as supplementary
- [ ] All scripts needed to reproduce results are included
- [ ] Requirements file (requirements.txt or pyproject.toml) is complete
- [ ] Config files for all experiments are included
- [ ] Random seeds are documented and fixed

## Scientific Quality

- [ ] Claims are conservative and supported by evidence
- [ ] Benchmark contribution is clearly separated from method contribution
- [ ] Retrieval metrics are computed correctly (passage-level and document-level)
- [ ] Significance tests are included for main results
- [ ] Ablation study covers key design choices
- [ ] Error analysis identifies failure modes
- [ ] Abstention evaluation is included

## Ethics and Broader Impact

- [ ] Ethics statement included
- [ ] Data licensing is documented and compliant
- [ ] No personally identifiable information in benchmark tasks
- [ ] Limitations section is honest about scope
- [ ] Potential misuse scenarios are acknowledged

## Reproducibility

- [ ] Hardware requirements documented
- [ ] Software versions documented
- [ ] API costs documented
- [ ] Expected runtime documented
- [ ] All hyperparameters specified
- [ ] Data preprocessing is fully scripted (no manual steps)

## Pre-Submission Checks

- [ ] LaTeX compiles without errors
- [ ] All figures render correctly
- [ ] All tables are consistent with reported numbers
- [ ] Bibliography is complete and correctly formatted
- [ ] Supplementary material is properly uploaded
- [ ] Paper has been proofread for grammar and clarity
