"""Synthetic-share check for oversampling before splitting.

If the minority class was oversampled before the train/test split, the test split holds synthetic
or duplicated positives that a flexible classifier recognises, so the reported sensitivity is about
s = 1 - n_pos / (alpha * n_neg) or higher (alpha = minority/majority ratio after oversampling).
  A  test set with far more positives than the prevalence allows, or more than the whole dataset
  B  reported sensitivity >= s - tolerance
A sensitivity below s does not rule leakage out. A screening aid, not proof.

    python synthetic_share_check.py --pos 249 --neg 4861 --reported-sensitivity 0.96
    python synthetic_share_check.py --pos 249 --neg 4861 --reported-sensitivity 0.95 --test-pos 1413 --test-n 2820
"""
import argparse


def synthetic_share(pos=None, neg=None, prevalence=None, alpha=1.0):
    if pos is not None and neg is not None:
        ratio = pos / neg
    elif prevalence is not None:
        ratio = prevalence / (1 - prevalence)
    else:
        raise ValueError("give --pos and --neg, or --prevalence")
    if alpha <= ratio:
        raise ValueError("alpha must exceed the original minority/majority ratio (no oversampling otherwise)")
    return 1 - ratio / alpha


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pos", type=float, help="positives before oversampling")
    ap.add_argument("--neg", type=float, help="negatives before oversampling")
    ap.add_argument("--prevalence", type=float, help="prevalence before oversampling (instead of counts)")
    ap.add_argument("--alpha", type=float, default=1.0, help="minority/majority ratio after oversampling")
    ap.add_argument("--reported-sensitivity", type=float, help="positive-class sensitivity reported by the study")
    ap.add_argument("--test-pos", type=float, help="positives in the reported test set (e.g. from a confusion matrix)")
    ap.add_argument("--test-n", type=float, help="size of the reported test set")
    ap.add_argument("--tolerance", type=float, default=0.03, help="allowed shortfall below s")
    ap.add_argument("--method", choices=["smote", "random"], default="smote",
                    help="SMOTE-family interpolation (SMOTE, Borderline-SMOTE, ADASYN) or random duplication")
    a = ap.parse_args()
    s = synthetic_share(a.pos, a.neg, a.prevalence, a.alpha)
    prev = a.prevalence if a.prevalence is not None else a.pos / (a.pos + a.neg)
    print(f"synthetic share of positives after oversampling: s = {s:.3f} (original prevalence {prev:.3f})")
    flags = []
    if a.test_pos is not None and a.test_n is not None:
        share = a.test_pos / a.test_n
        too_many = a.pos is not None and a.test_pos > a.pos
        print(f"test set: {a.test_pos:.0f} positives of {a.test_n:.0f} ({100 * share:.1f}%)")
        if share >= max(0.30, 3 * prev) or too_many:
            flags.append("A")
            print("  SIGNATURE A: the test set holds far more positives than the data's prevalence allows"
                  + (" (more than the whole dataset contains)" if too_many else "")
                  + "; it must contain synthetic or duplicated records.")
    if a.reported_sensitivity is not None:
        d = a.reported_sensitivity - s
        print(f"reported sensitivity {a.reported_sensitivity:.3f}; difference from s = {d:+.3f}")
        if s < 0.9:
            print("  WEAK CHECK: with s below 0.9 (prevalence above about 5%), honest models can also report "
                  "a sensitivity above s; rely on signature A.")
        elif d >= -a.tolerance:
            flags.append("B")
            print("  SIGNATURE B: reported sensitivity is at or above s, as expected after oversampling before splitting.")
            print("  The reported value says nothing about sensitivity on new cases; real positives in a leaked test "
                  "split also have near-copies in training.")
            if a.method == "random":
                print("  random duplication drives reported sensitivity towards 1; s is only a lower bound.")
        else:
            print("  below s: no signature B (weaker or linear models are not fooled, so check signature A).")
    if flags:
        print(f"CONSISTENT with oversampling before splitting (signature {' and '.join(flags)}): ask for performance "
              "on data that never took part in oversampling.")
    else:
        print("No signature of oversampling before splitting found.")


if __name__ == "__main__":
    main()
