# ----------------------------------------------------------------------------
# IMPORTS
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
from dataset import create_dataloader
from sklearn.metrics import balanced_accuracy_score
from tf2k_dataset import tf2k_create_dataloader
from logger import create_logger
import json

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


def train(train_dataloader, val_dataloader, model, dataset, settings):
    best_accuracy = 0
    lr_decay_counter = 0

    # ----------------------------------
    # Logger
    # ----------------------------------
    print(f"logging results ")
    if settings.freeze and settings.r50unfreezeL4:
        save_dir = f'./checkpoint/{settings.name}/ft_unfreezeL4_weights/{dataset}'
        # torch.save(model.state_dict(), f'./checkpoint/{settings.name}/ft_unfreezeL4_weights/best.pt')
    elif settings.freeze:
        save_dir = f'./checkpoint/{settings.name}/ft_weights/{dataset}'
        # torch.save(model.state_dict(), f'./checkpoint/{settings.name}/ft_weights/best.pt')
    else:
        save_dir = f'./checkpoint/{settings.name}/weights'
    # set up logger
    log_path = save_dir + '/exp_log/output.log'
    log = create_logger(log_path)
    log.info(f"Training the R50_tf model on the {dataset} dataset - FT: {settings.freeze} - unfreezeL4: {settings.r50unfreezeL4}")
    log.info(f"Training settings: {json.dumps(vars(settings), indent=2, default=str)}")
    # breakpoint()
    # # print some info on the model architecture
    # log.info("Model informations:")
    # log.info(f"Model Name: R50_TF")
    # # log.info(f"Pretrained model weights: {pretrained_model_path if pretrained_model_path != '' else 'ImageNet pre-trained model weights'}")
    # log.info(f"Optimizer: {optimizer}")
    # log.info(f"Loss function: {criterion}")
    # log.info(f"Learning rate: {settings.lr}")
    # log.info(f"Learning rate decay epochs: {settings.lr_decay_epochs}")
    # log.info(f"Learning rate minimum: {settings.lr_min}")
    # log.info(f"Batch size: {settings.batch_size}")
    # log.info(f"Number of epochs: {settings.num_epochs}")

    # log.info(f"Resume training from epoch: ", checkpoint['epoch']+1) if args.resume else print("training model from scratch")
    # log.info(f"Early Stopping-Patience: {patience}")
    # log.info(f"Dataset: {dataset}")
    # log.info(f"Dataset path: {settings.data_root}")
    # log.info(f"Split file: {settings.split_file}")
    # log.info(f"{train_transform}")
    # log.info(f"Model path: {model_path}")
    log.info(f"Training the model...")


    for epoch in range(0, settings.num_epoches):
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
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            
            
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))

            print(f'New best model saved with accuracy {best_accuracy:.4f} \n')
            lr_decay_counter = 0
            log.info(f"Epoch {epoch} - New best model saved with accuracy {best_accuracy:.4f} - lr_decay_counter: {lr_decay_counter}")
            

        elif settings.lr_decay_epochs > 0:
            lr_decay_counter += 1
            if lr_decay_counter == settings.lr_decay_epochs:
                if optimizer.param_groups[0]['lr'] > settings.lr_min:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.1
                    print('Learning rate decayed \n')
                    log.info(f"Epoch {epoch} - Learning rate decayed - lr_decay_counter: {lr_decay_counter}")
                    lr_decay_counter = 0
                else:
                    print('Learning rate already at minimum \n')
                    log.info(f"Epoch {epoch} - Learning rate already at minimum - lr_decay_counter: {lr_decay_counter}")
                    break
    
    log.info(f"Training completed. Best accuracy: {best_accuracy:.4f} - number of Epochs: {epoch+1}")

if __name__ == "__main__":
    parser = get_parser()
    settings = parser.parse_args()
    print(f"settings: {settings}")

    if settings.ft:
        settings.freeze = True
    
    print(f"settings.freeze: {settings.freeze}")
    print(f"r50unfreezeL4", settings.r50unfreezeL4)
    dataset = settings.dataset.replace(os.sep, '_')
    # dataset += '_unfreezeL4' if opt.r50unfreezeL4 else ''
    if settings.r50unfreezeL4:
        dataset += '_r50unfreezeL4'
    # breakpoint()
    device = torch.device(settings.device if torch.cuda.is_available() else 'cpu')

    model = ImageClassifier(settings)
    
    model.to(device)
    os.makedirs(f'./checkpoint/{settings.name}/weights/', exist_ok=True)
    
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

    train(train_dataloader, val_dataloader, model, dataset, settings)