import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def compute_extra_metrics(y_true, preds, probs):
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, probs)
    return prec, rec, f1, auc


def print_aggregated_results(all_results):
    print("\n" + "=" * 120)
    print("AGGREGATED RESULTS (MEAN +/- STD)")
    print("=" * 120)
    print(f"{'Method':<12} {'ValAcc':<14} {'TestAcc':<14} "
          f"{'Precision':<14} {'Recall':<14} {'Specificity':<12} {'F1':<14} {'AUC':<14}")
    print("-" * 120)

    def ms(arr):
        return np.mean(arr), np.std(arr)

    def row(label, prefix):
        stats = {
            "val": ms(all_results[f"{prefix}_val"]),
            "test": ms(all_results[f"{prefix}_test"]),
            "prec": ms(all_results[f"{prefix}_precision"]),
            "rec": ms(all_results[f"{prefix}_recall"]),
            "spec": ms(all_results[f"{prefix}_specificity"]),
            "f1": ms(all_results[f"{prefix}_f1"]),
            "auc": ms(all_results[f"{prefix}_auc"]),
        }
        print(f"{label:<12}"
              f"{stats['val'][0]:>6.4f}+/-{stats['val'][1]:<7.4f}"
              f"{stats['test'][0]:>6.4f}+/-{stats['test'][1]:<7.4f}"
              f"{stats['prec'][0]:>6.4f}+/-{stats['prec'][1]:<7.4f}"
              f"{stats['rec'][0]:>6.4f}+/-{stats['rec'][1]:<7.4f}"
              f"{stats['spec'][0]:>6.4f}+/-{stats['spec'][1]:<7.4f}"
              f"{stats['f1'][0]:>6.4f}+/-{stats['f1'][1]:<7.4f}"
              f"{stats['auc'][0]:>6.4f}+/-{stats['auc'][1]:<7.4f}")

    row("3C-EA", "evolved")
    row("ReLU", "relu")
    row("Swish", "swish")
    row("LeakyReLU", "leakyrelu")
    row("ELU", "elu")
    print("=" * 120)
