import json
from pprint import pprint


def evaluate(predict_result):
    predict_result = json.load(open(predict_result))

    metrics = {x: {'TP':0, 'FP':0, 'FN':0} for x in ['overall', 'binary', 'categorical', 'non-categorical']}
    acc = []

    for sample in predict_result:
        flag = True
        if isinstance(sample['predictions']['dialogue_acts'], dict):
            for da_type in ['binary', 'categorical', 'non-categorical']:
                if da_type == 'binary':
                    predicts = [(x['intent'], x['domain'], x['slot']) for x in sample['predictions']['dialogue_acts'][da_type]]
                    labels = [(x['intent'], x['domain'], x['slot']) for x in sample['dialogue_acts'][da_type]]
                else:
                    predicts = [(x['intent'], x['domain'], x['slot'], ''.join(x['value'].split()).lower()) for x in sample['predictions']['dialogue_acts'][da_type]]
                    labels = [(x['intent'], x['domain'], x['slot'], ''.join(x['value'].split()).lower()) for x in sample['dialogue_acts'][da_type]]
                predicts = sorted(list(set(predicts)))
                labels = sorted(list(set(labels)))
                for ele in predicts:
                    if ele in labels:
                        metrics['overall']['TP'] += 1
                        metrics[da_type]['TP'] += 1
                    else:
                        metrics['overall']['FP'] += 1
                        metrics[da_type]['FP'] += 1
                for ele in labels:
                    if ele not in predicts:
                        metrics['overall']['FN'] += 1
                        metrics[da_type]['FN'] += 1
                flag &= (predicts==labels)
            acc.append(flag)
        elif isinstance(sample['predictions']['dialogue_acts'], list):
            gold_da = sorted(list({(da['intent'], da['domain'], da['slot'], ''.join(da.get('value', '').split()).lower()) for da_type in ['binary', 'categorical', 'non-categorical'] for da in sample['dialogue_acts'][da_type]}))
            pred_da = sorted(list({(da['intent'], da['domain'], da['slot'], ''.join(da.get('value', '').split()).lower()) for da in sample['predictions']['dialogue_acts']}))
            acc.append(pred_da==gold_da)
            for ele in pred_da:
                if ele in gold_da:
                    metrics['overall']['TP'] += 1
                else:
                    metrics['overall']['FP'] += 1
            for ele in gold_da:
                if ele not in pred_da:
                    metrics['overall']['FN'] += 1
        else:
            raise TypeError('type of predictions:dialogue_acts should be dict or list')
    

    #######################################################################################################
    ## TODO NLU_TASK3 evaluation measure
    ## consolidate the counted cases od TP FP FN to compute precision, recall, f1, and accurracy
    ## make sure to do this for all cases, i.e. overall, binary, categorical, and non-categorical
    ## Start of your code #################################################################################
    for metric_name, counts in metrics.items():
        tp, fp, fn = counts['TP'], counts['FP'], counts['FN']
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        metrics[metric_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    metrics['accuracy'] = sum(acc) / len(acc) if acc else 0
    ## End of your code ###################################################################################

    return metrics


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate NLU metrics for unified datasets")
    parser.add_argument('--predict_result', '-p', type=str, required=True, help='path to the prediction file that in the unified data format')
    args = parser.parse_args()
    print(args)
    metrics = evaluate(args.predict_result)
    pprint(metrics)
