# ----------------------------------------------------------------------------
# IMPORTS
# Added EarlyStopping for FT the model on 2k dataset
# ----------------------------------------------------------------------------
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import glob
import torch
import shutil
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from networks import ImageClassifier
from parser import get_parser
from utils.dataset import create_dataloader
from sklearn.metrics import balanced_accuracy_score
from utils.tf2k_dataset import tf2k_create_dataloader
from utils.logger import create_logger
import json
from utils import EarlyStopping

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


def check_accuracy(val_dataloader, model, settings):
    model.eval()
    
    label_array = torch.empty(0, dtype=torch.int64, device=device)
    pred_array = torch.empty(0, dtype=torch.int64, device=device)

    with torch.no_grad():
        with tqdm(val_dataloader, unit='batch', mininterval=0.5) as tbatch:
            tbatch.set_description(f'Validation')
            for (data, label, _) in tbatch:
                data = data.to(device)
                label = label.to(device)
                
                pred = model(data).squeeze(1)
                
                label_array = torch.cat((label_array, label))
                pred_array = torch.cat((pred_array, pred))
    
    accuracy = balanced_accuracy_score(label_array.cpu().numpy(), pred_array.cpu().numpy() > 0)

    print(f'Got accuracy {accuracy:.2f} \n')
    return accuracy


def train(train_dataloader, val_dataloader, model, optimizer, dataset, settings):
    # ----------------------------------
    # Logger
    # ----------------------------------
    print(f"logging results ")
    if settings.freeze and settings.r50unfreezeL4 and settings.social:
        save_dir = f'./checkpoint/{settings.name}/social/{settings.social}/ft_unfreezeL4_weights/{dataset}'
    elif settings.freeze and settings.r50unfreezeL4:
        save_dir = f'./checkpoint/{settings.name}/ft_unfreezeL4_weights/{dataset}'
        # torch.save(model.state_dict(), f'./checkpoint/{settings.name}/ft_unfreezeL4_weights/best.pt')
    elif settings.freeze:
        save_dir = f'./checkpoint/{settings.name}/ft_weights/{dataset}'
        # torch.save(model.state_dict(), f'./checkpoint/{settings.name}/ft_weights/best.pt')
    else:
        save_dir = f'./checkpoint/{settings.name}/weights'
    # set up logger
    log_path = save_dir + '/train.log'
    log = create_logger(log_path)
    log.info(f"Training the R50_tf model on the {dataset} dataset - FT: {settings.freeze} - unfreezeL4: {settings.r50unfreezeL4}")
    log.info(f"Training settings: {json.dumps(vars(settings), indent=2, default=str)}")
    # breakpoint()
    

    early_stopping = None
    start_epoch = 1

    log.info(f"Early stopping enabled - patience: {settings.earlystop_epoch} epochs")

    if settings.resume:
        found = find_last_checkpoint(save_dir)
        if found is not None:
            last_epoch, ckpt_path = found
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full resume checkpoint written by this script.
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                es_best_score = checkpoint.get('early_stopping_best_score')
                es_count_down = checkpoint.get('early_stopping_count_down')
            else:
                # Fallback: file at this path is a plain model state_dict (e.g.
                # left behind by an interrupted run before the full checkpoint
                # got written - see the note about the mid-loop save below).
                # We can still recover the weights and the epoch number (from
                # the filename), but optimizer momentum and early-stopping
                # counters can't be reconstructed, so they restart clean.
                model.load_state_dict(checkpoint)
                es_best_score = None
                es_count_down = None
                print(f'Warning: {ckpt_path} is not a full resume checkpoint - loaded model weights only; '
                      f'optimizer and early-stopping state were reset \n')
                log.info(f"Checkpoint {ckpt_path} missing 'model_state_dict' key - loaded as a raw state_dict; "
                         f"optimizer and early-stopping state could not be restored")

            start_epoch = last_epoch + 1

            if es_best_score is not None:
                early_stopping = EarlyStopping(
                    init_score=es_best_score,
                    patience=settings.earlystop_epoch,
                    delta=0.001,
                    verbose=True,
                    logger=log,
                )
                early_stopping.count_down = es_count_down if es_count_down is not None else settings.earlystop_epoch
            print(f'Resuming training from epoch {start_epoch} (loaded {ckpt_path}) \n')
            log.info(f"Resumed from checkpoint {ckpt_path} - starting at epoch {start_epoch}, early_stopping best_score={getattr(early_stopping, 'best_score', None)}, count_down={getattr(early_stopping, 'count_down', None)}")

        # else:
        #     print(f'--resume set but no checkpoint found in {save_dir} - starting fresh \n')
        #     log.info(f"--resume set but no checkpoint found in {save_dir} - starting fresh")

    log.info(f"Training the model...")


    for epoch in range(start_epoch, settings.num_epoches+1):
        model.train()
        with tqdm(train_dataloader, unit='batch', mininterval=0.5) as tepoch:
            tepoch.set_description(f'Epoch {epoch}', refresh=False)
            if epoch > 0:
                for batch_idx, (data, label, _) in enumerate(tepoch):
                    data = data.to(device)
                    label = label.to(device).float()

                    scores = model(data).squeeze(1)

                    loss = criterion(scores, label).mean()
    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    tepoch.set_postfix(loss=loss.item())

        accuracy = check_accuracy(val_dataloader, model, settings)
        log.info(f"Epoch {epoch} - Validation accuracy: {accuracy:.4f}")

        if early_stopping is None:
            # First validation pass: initialize EarlyStopping with this score as
            # the baseline and always save it as the current best.
            early_stopping = EarlyStopping(
                init_score=accuracy,
                patience=settings.earlystop_epoch,
                delta=0.001, 
                verbose=True,
                logger=log,
            )
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
            print(f'Save best model at epoch {epoch} with val accuracy = {accuracy:.4f} \n')
            # log.info(f"Epoch {epoch} - Save best model - early stopping initialized with patience={settings.patience}, min_delta={settings.min_delta}")
            log.info(f"Save best model at epoch {epoch} with val acc = {accuracy} - early stopping initialized with patience = {settings.earlystop_epoch} and delta = 0.001")
        else:
            if early_stopping(accuracy):
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
                print(f'New best model saved with accuracy {accuracy:.4f} \n')
                log.info(f"Epoch {epoch} - New best model saved with accuracy {accuracy:.4f} - EarlyStopping count_down: {early_stopping.count_down} on {early_stopping.patience}")

        # Save a full per-epoch checkpoint (model + optimizer + early-stopping
        # state) so training can be resumed exactly where it left off. This is
        # separate from best.pt, which stays a plain state_dict for test.py.
        os.makedirs(save_dir, exist_ok=True)
        epoch_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'early_stopping_best_score': early_stopping.best_score,
            'early_stopping_count_down': early_stopping.count_down,
        }
        torch.save(epoch_checkpoint, os.path.join(save_dir, f'{epoch}.pt'))
        log.info(f"Epoch {epoch} - Saved resume checkpoint to {epoch}.pt")

        if early_stopping.early_stop:
            print(f'Early stopping triggered at epoch {epoch} - no improvement for {settings.earlystop_epoch} epochs \n')
            log.info(f"Epoch {epoch} - Early stopping - no improvement for {settings.earlystop_epoch} epochs")
            break

    log.info(f"Training completed. Best accuracy: {early_stopping.best_score:.4f} - number of Epochs: {epoch+1}")

    # Training finished (either ran all epochs or early-stopped for good) -
    # the per-epoch checkpoints were only needed to support --resume, so
    # clean them up now and keep just best.pt.
    removed = 0
    for fname in os.listdir(save_dir):
        name, ext = os.path.splitext(fname)
        if ext == '.pt' and name.isdigit():
            os.remove(os.path.join(save_dir, fname))
            removed += 1
    print(f'Removed {removed} intermediate epoch checkpoint(s), kept best.pt \n')
    log.info(f"Removed {removed} intermediate epoch checkpoint(s) from {save_dir} - kept best.pt")


if __name__ == "__main__":
    parser = get_parser()
    settings = parser.parse_args()
    print(f"settings: {settings}")

    
    
    print(f"settings.freeze: {settings.freeze}")
    print(f"r50unfreezeL4", settings.r50unfreezeL4)
    dataset = settings.dataset.replace(os.sep, '_')
    # dataset += '_unfreezeL4' if opt.r50unfreezeL4 else ''
    if settings.ft and settings.r50unfreezeL4:
        dataset += '_ft_unfreezeL4'
    # breakpoint()
    device = torch.device(settings.device if torch.cuda.is_available() else 'cpu')

    model = ImageClassifier(settings)

    # if ft load the best model and ft it
    if settings.ft:
        settings.freeze = True
        # Bootstrap fine-tuning from the previously-trained baseline model
        # for this --name (produced by a prior non-ft run). Loading after
        # model.to(device) but the freeze decisions made inside
        # ImageClassifier.__init__ (based on settings.freeze/r50unfreezeL4)
        # are unaffected - freezing only sets requires_grad, it doesn't
        # depend on parameter values, so loading weights afterward is safe.
        load_path = f'./checkpoint/{settings.name}/weights/best.pt'
        if os.path.isfile(load_path):
            print(f'[FT] Loading pretrained weights from {load_path}')
            state_dict = torch.load(load_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            print(f'[FT] WARNING: no checkpoint found at {load_path} - '
                  f'fine-tuning from ImageNet-pretrained weights only.')
    
    model.to(device)
    os.makedirs(f'./checkpoint/{settings.name}/weights/', exist_ok=True)

    # breakpoint()
    
    with open(f'./checkpoint/settings.txt', 'w') as f:
        f.write(str(settings))
    if settings.tf2k:
        train_dataloader = tf2k_create_dataloader(settings, split='train')
        val_dataloader = tf2k_create_dataloader(settings, split='val')
    else:
        train_dataloader = create_dataloader(settings, split='train')
        val_dataloader = create_dataloader(settings, split='val')

    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=settings.lr)

    criterion = nn.BCEWithLogitsLoss(reduction='none')

    train(train_dataloader, val_dataloader, model, optimizer, dataset, settings)