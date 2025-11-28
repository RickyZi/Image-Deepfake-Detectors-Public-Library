import os
import tqdm
from utils import TrainingModel, create_dataloader, EarlyStopping
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from utils.processing import add_processing_arguments
from parser import get_parser

if __name__ == "__main__":
    parser = get_parser()
    parser = add_processing_arguments(parser)

    opt = parser.parse_args()

    os.makedirs(os.path.join('checkpoint', opt.name,'weights'), exist_ok=True)

    valid_data_loader = create_dataloader(opt, split="val")
    train_data_loader = create_dataloader(opt, split="train")
    print()
    print("# validation batches = %d" % len(valid_data_loader))
    print("#   training batches = %d" % len(train_data_loader))
    model = TrainingModel(opt)
    early_stopping = None
    start_epoch = model.total_steps // len(train_data_loader)
    print()

    for epoch in range(start_epoch, opt.num_epoches+1):
        if epoch > start_epoch:
            # Training
            pbar = tqdm.tqdm(train_data_loader)
            for data in pbar:
                loss = model.train_on_batch(data).item()
                total_steps = model.total_steps
                pbar.set_description(f"Train loss: {loss:.4f}")

            # Save model
            model.save_networks(epoch)

        # Validation
        print("Validation ...", flush=True)
        y_true, y_pred, y_path = model.predict(valid_data_loader)
        acc = balanced_accuracy_score(y_true, y_pred > 0.0)
        auc = roc_auc_score(y_true, y_pred)
        lr = model.get_learning_rate()
        print("After {} epoches: val acc = {}; val auc = {}".format(epoch, acc, auc), flush=True)

        # Early Stopping
        if early_stopping is None:
            early_stopping = EarlyStopping(
                init_score=acc, patience=opt.earlystop_epoch,
                delta=0.001, verbose=True,
            )
            print('Save best model', flush=True)
            model.save_networks('best')
        else:
            if early_stopping(acc):
                print('Save best model', flush=True)
                model.save_networks('best')
            if early_stopping.early_stop:
                cont_train = model.adjust_learning_rate()
                if cont_train:
                    print("Learning rate dropped by 10, continue training ...", flush=True)
                    early_stopping.reset_counter()
                else:
                    print("Early stopping.", flush=True)
                    break
