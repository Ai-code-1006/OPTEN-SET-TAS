"""Evaluate all four GTEA checkpoints with a fixed voting rule and boundary controls."""
import argparse
import csv
from pathlib import Path
from statistics import mean

import torch

from Inference import inference_bd_peak
from action_segmentation import create_dataloader, create_model, get_default_config
from action_segmentation.config.config_node import ConfigNode
from action_segmentation.utils import AverageMeter_f1, compute_metrics_from_action, setup_cudnn


def vote_at_reference_cuts(outputs, target):
    # Only the cut positions come from the annotations. The class of each
    # segment is still the model's prediction for the winning query.
    cuts = (torch.where(target[1:] != target[:-1])[0] + 1).tolist()
    masks = outputs['pred_masks'][0].sigmoid()
    classes = outputs['pred_logits'][0, :, :-1].argmax(dim=1)
    labels = torch.empty_like(target)
    for start, end in zip([0] + cuts, cuts + [len(target)]):
        query = masks[:, start:end].sum(dim=1).argmax()
        labels[start:end] = classes[query]
    return labels


@torch.inference_mode()
def evaluate(experiment, thresholds):
    rows, confusions, seen = [], [], set()
    for fold in range(1, 5):
        path = experiment / 'gtea' / 'bk_fde_tde' / 'final' / str(fold) / 'checkpoint_best.pth'
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg = get_default_config()
        cfg.merge_from_other_cfg(ConfigNode(ckpt['config']))
        assert cfg.dataset.name == 'gtea' and cfg.dataset.split == fold and cfg.dataset.sample_rate == 1
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        setup_cudnn(cfg)
        model = create_model(cfg).eval()
        model.load_state_dict(ckpt['model'], strict=True)
        loader = create_dataloader(cfg, is_train=False)
        assert len(loader.dataset) == 7
        videos, frames = 0, 0
        methods = [f'peak_{t:g}' for t in thresholds] + ['reference_cuts']
        correct = dict.fromkeys(methods, 0)
        edits = dict.fromkeys(methods, 0.0)
        f1 = {key: AverageMeter_f1() for key in methods}
        n_classes = cfg.dataset.n_classes
        confusion = torch.zeros((n_classes, n_classes), dtype=torch.long)
        default_key = f'peak_{cfg.dataset.threshold:g}'
        for x, target, names in loader:
            assert names[0] not in seen
            seen.add(names[0])
            outputs = model(x.to(cfg.device))
            target = target[0].long()
            predictions = {f'peak_{t:g}': inference_bd_peak(outputs, t)[0].argmax(dim=-1)
                           for t in thresholds}
            predictions['reference_cuts'] = vote_at_reference_cuts(outputs, target.to(cfg.device))
            videos += 1
            frames += len(target)
            for key, pred in predictions.items():
                assert pred.shape == target.shape
                pred = pred.cpu()
                hits, _, edit, counts = compute_metrics_from_action('gtea', pred, target)
                correct[key] += int(hits)
                edits[key] += float(edit)
                f1[key].update(counts)
                if key == default_key:
                    pair = target * n_classes + pred
                    confusion += torch.bincount(pair, minlength=n_classes**2).view_as(confusion)
        if default_key in methods:
            assert int(confusion.sum()) == frames and int(confusion.trace()) == correct[default_key]
            names = {idx: label for label, idx in loader.dataset.action_to_idx.items()}
            for actual, predicted in confusion.nonzero().tolist():
                confusions.append([fold, names[actual], names[predicted], int(confusion[actual, predicted])])
        for key in methods:
            rows.append([fold, ckpt['epoch'], key, videos, frames,
                         100 * correct[key] / frames, edits[key] / videos, *f1[key].avg.tolist()])
    assert len(seen) == 28
    for key in methods:
        folds = [row for row in rows if row[2] == key]
        rows.append(['mean', '', key, 28, sum(row[4] for row in folds),
                     *(mean(row[i] for row in folds) for i in range(5, 10))])
    return rows, confusions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path, help='experiment directory containing gtea/')
    parser.add_argument('output', type=Path)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.2, 0.0])
    args = parser.parse_args()
    results, confusions = evaluate(args.experiment, args.thresholds)
    with args.output.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['fold', 'epoch', 'decoder', 'videos', 'frames', 'Acc', 'Edit',
                         'F1@10', 'F1@25', 'F1@50'])
        writer.writerows(results)
    if confusions:
        output = args.output.with_name(args.output.stem + '_confusion.csv')
        with output.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['fold', 'true_class', 'predicted_class', 'frames'])
            writer.writerows(confusions)
