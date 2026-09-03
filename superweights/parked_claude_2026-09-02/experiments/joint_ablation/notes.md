# joint_ablation — notes

## 2026-09-02

What the last run showed:   OLMo-1B probe (results/probe_v6): individual L1[1764,1710] x3667 [3421,3935], joint-all x3514 [3269,3781], null max x1.000 over 10, SA removal x2095 [957,3571]; all 11 probe checks pass. First modern model back: TowerBase-7B individual x1.58 [1.46,1.82] and x1.03, **joint x3883 [3207,4719]**, null max x1.00 over 100 draws, SA-zero both channels x3235.
What I'm changing and why:  nothing yet; arrays v6 / modern_v6 / bases_v6 / small_v6 / multi_v6 / encdec_v6 (jobs 13568371-5, 13568408) running.
Provenance:                 7e7471c-dirty (uncommitted: ablate_sw.py rewrite, sw_arch.py, bootstrap_ci.py, encdec_sw.py, probe_check.py, joint_summary.py, slurm/joint.sh, slurm/encdec.sh, sw_models.py sets) · configs/table2.yaml, configs/modern.yaml · seed 0
Verdict vs satisfied-when:  undecided until all arrays finish.

CPU smoke tests: OLMo-1B at 2x256 tokens reproduced the known answer (x1562 individual, joint = individual, LOO correct, null x1.00, SA-zero x1370). The NLLB CPU smoke test timed out at 10 min (600M model + FLORES batches on CPU); submitted straight to GPU instead.
