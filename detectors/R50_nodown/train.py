import os
import tqdm
# from utils import TrainingModel, EarlyStopping, FTModel # create_dataloader
from utils.finetuning import FTModel
from utils.training import TrainingModel
from utils import EarlyStopping
from utils.tf2k_dataset import tf2k_create_dataloader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from utils.processing import add_processing_arguments
from parser import get_parser
import json
import sys

from utils.logger import create_logger

if __name__ == "__main__":
    parser = get_parser()
    parser = add_processing_arguments(parser)

    opt = parser.parse_args()
    # print(f"opt: {opt}")
    # breakpoint()
    # if opt.ft:
    
    # print(json.dumps(vars(opt), indent=2, default=str))
    # sys.exit(0)
    dataset = opt.dataset.replace(os.sep, '_')
    # dataset += '_unfreezeL4' if opt.r50unfreezeL4 else ''
    if opt.r50unfreezeL4:
        dataset += '_r50unfreezeL4'
    weights_name = 'ft_weights' if opt.ft else 'weights'
    if opt.ft and opt.r50unfreezeL4:
        ckpt_dir = os.path.join('checkpoint', opt.name, 'ft_unfreezeL4_weights', dataset)
    elif opt.ft:
        ckpt_dir = os.path.join('checkpoint', opt.name, 'ft_weights', dataset)
    else:
        ckpt_dir = os.path.join('checkpoint', opt.name, 'weights', dataset)
    
    print(f"checkpoint_dir: {ckpt_dir}")

    os.makedirs(ckpt_dir, exist_ok=True)


    logger = create_logger(os.path.join(ckpt_dir, 'train.log'))

    # breakpoint()

    # os.makedirs(os.path.join('checkpoint', opt.name, weights_name), exist_ok=True)
    # breakpoint()
    # else:
    #     os.makedirs(os.path.join('checkpoint', opt.name,'weights'), exist_ok=True)

    valid_data_loader = tf2k_create_dataloader(opt, split="val")
    train_data_loader = tf2k_create_dataloader(opt, split="train")
    # print()
    print("# validation batches = %d" % len(valid_data_loader))
    print("#   training batches = %d" % len(train_data_loader))
    
    model = FTModel(opt) if opt.ft else TrainingModel(opt)
    if opt.ft:
        load_path = f'./checkpoint/{opt.name}/weights/best.pt' # load best pretrained weights from TB
        model.load_networks(load_path)
        model.freeze_backbone(opt.r50unfreezeL4)   # freeze backbones' layer 4 or only train the final fc head
     
    # model = TrainingModel(opt)
    early_stopping = None
    start_epoch = model.total_steps // len(train_data_loader)
    print()
    # breakpoint()

    # log info on training settings
    logger.info(f"Training model {opt.name} on dataset {opt.dataset} with data keys {opt.data_keys}")
    logger.info(f"Training settings: {json.dumps(vars(opt), indent=2, default=str)}")

    logger.info(f"Number of training batches: {len(train_data_loader)}")
    logger.info(f"Number of validation batches: {len(valid_data_loader)}")

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
        logger.info(f"Epoch {epoch}: val acc = {acc}; val auc = {auc}")
        # Early Stopping
        if early_stopping is None:
            early_stopping = EarlyStopping(
                init_score=acc, 
                patience=opt.earlystop_epoch, # set to 5 -> try to increase it to 10/15
                delta=0.001, # increase it to 0.005?
                verbose=True,
            )
            # print(f"early_stopping: {early_stopping}")
            # breakpoint()
            print('Save best model', flush=True)
            logger.info(f"Save best model at epoch {epoch} with val acc = {acc} - early stopping initialized with patience = {opt.earlystop_epoch} and delta = 0.001")
            model.save_networks('best')
        else:
            if early_stopping(acc):
                print('Save best model', flush=True)
                logger.info(f"Save best model at epoch {epoch} with val acc = {acc} - EarlyStopping count_down: {early_stopping.count_down} on {early_stopping.patience}")
                model.save_networks('best')
            if early_stopping.early_stop:
                cont_train = model.adjust_learning_rate()
                if cont_train:
                    print("Learning rate dropped by 10, continue training ...", flush=True)
                    logger.info(f"Learning rate dropped by 10, reset early stopping counter and continue training ...")
                    early_stopping.reset_counter()
                else:
                    print("Early stopping.", flush=True)
                    logger.info(f"Early stopping at epoch {epoch} with val acc = {acc}")
                    break

    
    logger.info(f"Training completed for model {opt.name} on dataset {opt.dataset} with data keys {opt.data_keys} - after {epoch} epochs, best val acc = {early_stopping.best_score}")


