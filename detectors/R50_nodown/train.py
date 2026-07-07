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


def find_last_checkpoint(save_dir):
    """Look for the highest-numbered '{epoch}.pt' checkpoint in save_dir.
    Returns (epoch, path) or None if no per-epoch checkpoint exists."""
    if not os.path.isdir(save_dir):
        return None
    candidates = []
    for fname in os.listdir(save_dir):
        name, ext = os.path.splitext(fname)
        if ext == '.pt' and name.isdigit():
            candidates.append(int(name))
    if not candidates:
        return None
    last_epoch = max(candidates)
    return last_epoch, os.path.join(save_dir, f'{last_epoch}.pt')


if __name__ == "__main__":
    parser = get_parser()
    parser = add_processing_arguments(parser)

    opt = parser.parse_args()
    # print(f"opt: {opt}")
    # breakpoint()
    # if opt.ft:
    
    # print(json.dumps(vars(opt), indent=2, default=str))
    # sys.exit(0)
    # dataset = (opt.social + '_' + opt.dataset.replace(os.sep, '_')) if opt.social else opt.dataset.replace(os.sep, '_')
    dataset = opt.dataset.replace(os.sep, '_')
    dataset += '_unfreezeL4' if opt.r50unfreezeL4 else ''
    print(f"dataset: {dataset}")
    weights_name = 'ft_weights' if opt.ft else 'weights'
    if opt.ft and opt.r50unfreezeL4 and opt.social:
        ckpt_dir = os.path.join('checkpoint', opt.name, 'social', opt.social, 'ft_unfreezeL4_weights', dataset)
        # checkpoint/pretrained/social/facebook/ft_unfreezeL4_weights/seasons_autumn-TM01
    elif opt.ft and opt.r50unfreezeL4:
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

    if opt.resume:
        found = find_last_checkpoint(model.save_dir)
        if found is not None:
            last_epoch, ckpt_path = found
            checkpoint = model.load_checkpoint(ckpt_path)
 
            es_best_score = checkpoint.get('early_stopping_best_score')
            es_count_down = checkpoint.get('early_stopping_count_down')
            if es_best_score is not None:
                early_stopping = EarlyStopping(
                    init_score=es_best_score,
                    patience=opt.earlystop_epoch,
                    delta=0.001,
                    verbose=True,
                    logger=logger,
                )
                early_stopping.count_down = es_count_down if es_count_down is not None else opt.earlystop_epoch
 
            start_epoch = last_epoch + 1
            print(f'Resuming training from epoch {start_epoch} (loaded {ckpt_path})', flush=True)
            logger.info(f"Resumed from checkpoint {ckpt_path} - starting at epoch {start_epoch}, "
                        f"early_stopping best_score={getattr(early_stopping, 'best_score', None)}, "
                        f"count_down={getattr(early_stopping, 'count_down', None)}")
        else:
            print(f'--resume set but no checkpoint found in {model.save_dir} - starting fresh', flush=True)
            logger.info(f"--resume set but no checkpoint found in {model.save_dir} - starting fresh")
 
    print()
    logger.info(f"Training the model...")


    for epoch in range(start_epoch, opt.num_epoches+1):
        if epoch > start_epoch:
            # Training
            pbar = tqdm.tqdm(train_data_loader)
            for data in pbar:
                loss = model.train_on_batch(data).item()
                total_steps = model.total_steps
                pbar.set_description(f"Train loss: {loss:.4f}")
 
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
                logger = logger
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
 
        # Save a full per-epoch checkpoint (model + optimizer + total_steps +
        # early-stopping state) so training can be resumed exactly where it
        # left off. Separate from 'best.pt', which stays weights-only.
        model.save_networks(epoch, extra={
            'early_stopping_best_score': early_stopping.best_score,
            'early_stopping_count_down': early_stopping.count_down,
        })
        logger.info(f"Epoch {epoch} - Saved resume checkpoint to {epoch}.pt")
 
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
 
    # Training finished (either ran all epochs or early-stopped for good) -
    # the per-epoch checkpoints were only needed to support --resume, so
    # clean them up now and keep just best.pt.
    removed = 0
    for fname in os.listdir(model.save_dir):
        name, ext = os.path.splitext(fname)
        if ext == '.pt' and name.isdigit():
            os.remove(os.path.join(model.save_dir, fname))
            removed += 1
    print(f'Removed {removed} intermediate epoch checkpoint(s), kept best.pt', flush=True)
    logger.info(f"Removed {removed} intermediate epoch checkpoint(s) from {model.save_dir} - kept best.pt")
 
