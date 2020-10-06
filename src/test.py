from src import utils, metrics, engine
from src.imports import tabulate


def test(model, ds, criterion, device):
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
    output.append(['Test_loss', 'test_acc_human',
                   'test_acc_dog', 'test_acc_breed'])
    output.append([
        test_metrics['loss'],
        test_metrics['accuracy_human'].item(),
        test_metrics['accuracy_dog'].item(),
        test_metrics['accuracy_breed']])
    print(tabulate(output))
    return {
        'loss': test_losses,
        'acts': test_acts,
        'dog_human_targets': test_dog_human_targets,
        'breed_targets': test_breed_targets
    }
    