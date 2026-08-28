# Project history and branch policy

The repository has two distinct purposes, and its branches keep them separate.

## `Original_Report_Code`

This branch is the archival snapshot of the group code used as the starting
point for the original report. It is intentionally left unchanged. Keeping the
snapshot in Git makes old results reproducible and preserves the contribution of
the original group members: Simon Desimpelaere, Robbe Alliet, Harikrishnan
Aravindakshan, Reda Elassooudi, Bodhisatwa Datta, and Maximilien Péters de
Bonhomme.

## `main`

This is Bodhisatwa Datta's maintained research version. It contains later
debugging, validation, reflecting-wall experiments, two-dimensional studies,
tests, and documentation. New work belongs here or on short-lived feature
branches based on `main`.

The maintained version is a continuation of the earlier Apache-2.0-licensed
code, not a claim that the earlier contributors did not exist. Git history and
`CITATION.cff` retain that provenance while making the responsibility for new
research changes clear.

## Recommended release practice

Results used in a thesis chapter, paper, or presentation should be tagged with a
descriptive version such as `thesis-results-2026`. The tag should record the
input parameters, random seed, Python environment, and plotting script used for
each figure. Generated `Results/` directories should remain outside Git unless a
small reference data set is deliberately selected for publication.
