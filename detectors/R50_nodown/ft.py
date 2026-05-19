import os
import tqdm
from utils import FTModel, create_dataloader, EarlyStopping
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from utils.processing import add_processing_arguments
from parser import get_parser

if __name__ == "__main__":
    parser = get_parser()
    parser = add_processing_arguments(parser)
    opt = parser.parse_args()

    os.makedirs(os.path.join('checkpoint', opt.name, 'ft_weights'), exist_ok=True)

    valid_data_loader = create_dataloader(opt, split="val")
    train_data_loader = create_dataloader(opt, split="train")
    print(f"\n# validation batches = {len(valid_data_loader)}")
    print(f"#   training batches = {len(train_data_loader)}")
    # create model backbone
    model = FTModel(opt)

    # load pretrained weights
    if opt.pretrained_weights is not None:
        model.load_networks(opt.pretrained_weights)

    model.freeze_backbone()   # freeze backbone layers, only train the final fc head

    start_epoch = model.total_steps // len(train_data_loader)
    early_stopping = None

    for epoch in range(start_epoch, opt.num_epoches + 1):
        if epoch > start_epoch:
            pbar = tqdm.tqdm(train_data_loader)
            for data in pbar:
                loss = model.train_on_batch(data).item()
                pbar.set_description(f"Epoch {epoch} | Train loss: {loss:.4f}")
            model.save_networks(epoch)

        print(f"\nValidation (epoch {epoch}) ...", flush=True)
        y_true, y_pred, y_path = model.predict(valid_data_loader)
        acc = balanced_accuracy_score(y_true, y_pred > 0.0)
        auc = roc_auc_score(y_true, y_pred)
        print(f"Epoch {epoch}: val acc = {acc:.4f} | val auc = {auc:.4f}", flush=True)

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
                    print("LR dropped by 10x, continuing ...", flush=True)
                    early_stopping.reset_counter()
                else:
                    print("Early stopping.", flush=True)
                    break