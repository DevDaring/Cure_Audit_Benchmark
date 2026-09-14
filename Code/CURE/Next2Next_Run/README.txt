Code/CURE/Next2Next_Run -- ICLR-feedback cycle of Submission2 (Submission2/Next_Plan.md, 14 Sep 2026)

Entry point:  python run_all.py --stage g4|g3|g1|g2|g5|g6|all [--model M] [--smoke]

Modules
  nn_common.py            paths (results/feedback_20260915/), helpers; reuses Next_Run and
                          Extended_Research_Codes; never writes under results/v2 or final_20260914
  g1_invariant.py         A1  pooled-audit patching effect re-read with shift-invariant statistics
                              (log-prob, option margin) next to the raw logit; unedited, span erasure,
                              three random subspaces, both LEACE fits, plus the g3 erasers
  g2_alignment.py         A2  fresh-replication C_answer on unequal-length pairs under truncate /
                              mean-pad / last-pad alignment
  g3_leace_conditioned.py A3  part a: swap-side LEACE refit at 50 / 300 / all fit pairs (+ no
                              shrinkage), effective ranks, probes; part b: coherent gender (F/M)
                              LEACE on thousands of BBQ descriptor rows, a fresh F<->M swap set,
                              conditions B S1 MC LC_f3 LC_gender_small LC_gender_large MC_gender
  g4_analyses.py          CPU strata / length decomposition / correlation / energy drift
  g5_steering.py          B   debiasing-vector (FairSteer-style) and contrast-vector (RepE/ActAdd-
                              style) steering at span / last token / every prefill position,
                              energy-calibrated; T, accuracy, C_answer, control set
  g6_analysis.py          merged statistics: g_summary.csv, g_tests.csv, g_leace_fits.csv, COMPLETION_G.txt
  bootstrap_g.sh          VM entry (torch 2.5.1 cu124 image, precompiled flash-attention wheel,
                          two-seed SMOKE first, then g3 g1 g2 g5 per model, pushes non-markdown outputs)
  deploy_g.py             Vast.ai launch / status / logs / restart / destroy (key from Code/CURE/.env)

VM groups: "gemma-2-2b-it llama-3.1-8b-instruct" (all stages) and "qwen2.5-7b-instruct phi-4-mini-instruct" (g3 part a + g1).
Local CPU smoke:  NR_CPU=1 NN_OUT_DIR=feedback_smoke_local python g1_invariant.py --model gemma-2-2b-it --smoke
