Code/CURE/Next_Run -- final research cycle of Submission2 (Submission2/Next_Plan.md, 14 Sep 2026)

Entry point:  python run_all.py --stage f0|f1|f2|f4|all   (f2 needs a GPU and --model)

Modules
  nr_common.py       paths, protocol record, atomic parquet append, studentized seed-cluster
                     bootstrap, Holm, non-inferiority helpers
  f0_provenance.py   F0: evidence manifest and hashes, fit-manifest audit, energy audit of the
                     P3 'matched' rows, legacy behavioural reanalysis of results/v2, MMLU
                     question-level intervals, claims_f0.json
  f1_data.py         F1: fresh gold-preserving BBQ pairs (160 final, 24 dev, 36 control),
                     exclusions, freshness against the three pools, tier and diagnostic subsets
  f1_scoring.py      the frozen measurement: complete-answer JSON candidate scores, T, M,
                     C_answer bridge with the sequential source convention, span rules
                     (demographic / matched / prefill_all), deterministic parser
  f2_runner.py       F2+F3 on one GPU per model: identity checks, span freeze, energy
                     calibration (common denominator, 5% tolerance, <=8 evaluations per cell,
                     one target reduction), timing and tier decision, seven conditions
                     B S1 SE NE RE RNE G1, patching bridge, 32-seed scoring variants,
                     relevant-information control, coherent-label LEACE / mean-difference
  f3_baseline.py     F3: A/B label audit, man/woman contrast, official LEACE + mean difference
                     on identical labels and rows, held-out probes, algebraic hook check
  f4_analysis.py     Sections 9-10: eight confirmatory tests (Holm), six non-inferiority
                     tests, secondary estimates, energy caution flags, claims.json, COMPLETION
  bootstrap_final.sh VM entry (torch 2.5.1 cu124 image, precompiled flash-attention wheel,
                     smoke then full, pushes non-markdown outputs every stage / 20 min)
  deploy_final.py    Vast.ai launch / status / logs / destroy (key from Code/CURE/.env)

Outputs: Code/CURE/results/final_20260914/ (see nr_common.py docstring for the file list).
Local CPU smoke test:  NR_CPU=1 NR_FINAL_DIR=final_smoke_local python f2_runner.py --model gemma-2-2b-it --stage checks --smoke
