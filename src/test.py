from src import utils, metrics, engine
from src.imports import tabulate


def test(model, ds, criterion, device, dataset_type="test"):
    """
    Evaluates and prints the result to the console.
    """
    test_dl = utils.get_dl(ds, bs=64, shuffle=False)
    test_losses, test_acts, test_dog_human_targets, \
        test_breed_targets = engine.eval_loop(test_dl, model,
                                              criterion, device)
    test_metrics = metrics.get_metrics(
        test_losses, test_acts, test_dog_human_targets, test_breed_targets,
        criterion.dog_idx, criterion.human_idx)
    output = []
    output.append([f'{dataset_type}_loss', f'{dataset_type}_acc_human',
                   f'{dataset_type}_acc_dog', f'{dataset_type}_acc_breed',
                   f'{dataset_type}_acc_f1score'])
    output.append([
        test_metrics['loss'],
        test_metrics['accuracy_human'].item(),
        test_metrics['accuracy_dog'].item(),
        test_metrics['accuracy_breed'].item(),
        test_metrics['f1score_breed']])
    print(tabulate(output))
    return {
        'loss': test_losses,
        'acts': test_acts,
        'dog_human_targets': test_dog_human_targets,
        'breed_targets': test_breed_targets
    }
    