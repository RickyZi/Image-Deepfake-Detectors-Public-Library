import os
import tqdm
# from utils import TrainingModel, create_dataloader, EarlyStopping
from utils.training import TrainingModel
# from utils.finetuning import FTModel
from utils import EarlyStopping
from utils.dataset import create_dataloader
from utils.tf2k_dataset import tf2k_create_dataloader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from utils.processing import add_processing_arguments
from parser import get_parser
import torch

from utils.logger import create_logger

if __name__ == "__main__":
    parser = get_parser()
    parser = add_processing_arguments(parser)

    opt = parser.parse_args()

    # ------------------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------------------
    dataset = opt.dataset.replace(os.sep, '_')
    if opt.ft and opt.mlp:
        training_strategy = 'ft_MLP'
        ckpt_dir = os.path.join('checkpoint', opt.name, 'ft_MLP_weights', dataset)
    elif opt.ft:
        training_strategy = 'ft'
        ckpt_dir = os.path.join('checkpoint', opt.name, 'ft_weights', dataset)
    else:
        training_strategy = 'pretrained'
        ckpt_dir = os.path.join('checkpoint', opt.name, 'weights', dataset)
    os.makedirs(ckpt_dir, exist_ok=True)
 
    # ------------------------------------------------------------------
    # Dataloaders 
    # ------------------------------------------------------------------
    if opt.tf2k:
        print("Using tf2k dataloaders")
        train_data_loader = tf2k_create_dataloader(opt, split='train')
        valid_data_loader = tf2k_create_dataloader(opt, split='val')
    else:
        train_data_loader = create_dataloader(opt, split='train')
        valid_data_loader = create_dataloader(opt, split='val')

    print()
    print("# validation batches = %d" % len(valid_data_loader))
    print("#   training batches = %d" % len(train_data_loader))

    model_wrapper = TrainingModel(opt) # if opt.ft else TrainingModel(opt) # CLIP-D = frozen CLIP bb + binary fc layer (trained)
    
    # if opt.ft:
    #     load_path = f'./checkpoint/{opt.name}/weights/best.pt' # load best pretrained weights from TB
    #     model_wrapper.load_networks(load_path)


    # ----------------------------------------------------- #
    # CLIP-D architecture investigation #
    # inner = model_wrapper.model # OpenClipLinear instance

    # print("\n=== OpenClipLinear top-level attributes ===")
    # for name, module in inner.named_children():
    #     print(f"  {name}: {type(module).__name__}")
    # # breakpoint()
    # print("\n=== All named modules (2 levels deep) ===")
    # for name, module in inner.named_modules():
    #     depth = name.count('.')
    #     if depth <= 2:
    #         print(f"  {'  '*depth}{name}: {type(module).__name__}")
    # # breakpoint()
    # print("\n=== Raw __dict__ keys of model.model ===")
    # for k, v in model_wrapper.model.__dict__.items():
    #     print(f"  {k}: {type(v).__name__}")
    # # breakpoint()
    # # Check registered modules (fc)
    # print("=== Registered parameters (via named_parameters) ===")
    # for name, param in model_wrapper.model.named_parameters():
    #     print(f'  requires_grad={param.requires_grad}\t{name}')

    # # Check backbone manually since bb is a plain list
    # print("\n=== Backbone parameters (bb[0], not registered) ===")
    # for name, param in model_wrapper.model.bb[0].named_parameters():
    #     print(f'  requires_grad={param.requires_grad}\t{name}')

    # # breakpoint()
    
    # # Run one fake batch through

    # print("--------- Run one fake batch through the model --------- ")

    # model_wrapper.model.train()
    # dummy_img   = torch.randn(2, 3, 224, 224).to(model_wrapper.device)
    # dummy_label = torch.tensor([0.0, 1.0]).to(model_wrapper.device)

    # output = model_wrapper.model(dummy_img)
    # loss   = model_wrapper.loss_fn(output.squeeze(1), dummy_label)
    # loss.backward()

    # print("=== Gradient check after backward pass ===")
    # print("\n-- fc (should have gradients) --")
    # for name, param in model_wrapper.model.named_parameters():
    #     has_grad = param.grad is not None and param.grad.abs().sum().item() > 0
    #     print(f'  has_grad={has_grad}\t{name}')

    # print("\n-- bb[0] backbone (should be None everywhere) --")
    # for name, param in model_wrapper.model.bb[0].named_parameters():
    #     has_grad = param.grad is not None and param.grad.abs().sum().item() > 0
    #     print(f'  has_grad={has_grad}\t{name}')

    # # breakpoint()
    # ----------------------------------------------------- #

    early_stopping = None
    start_epoch = model_wrapper.total_steps // len(train_data_loader)
    # print()


    # log training info
    logger = create_logger(os.path.join(ckpt_dir, 'train.log'))
    logger.info(f"Model name: {opt.name}")
    logger.info(f"Training strategy: {training_strategy}")
    logger.info(f"Training dataset: {opt.data_keys}")
    logger.info(f"Training dataset keys: {opt.data_keys.split('&')}")
    logger.info(f"Number of training batches: {len(train_data_loader)}")
    logger.info(f"Number of validation batches: {len(valid_data_loader)}")
    logger.info(f"Starting training from epoch {start_epoch} (total_steps={model_wrapper.total_steps})")
    logger.info(f"Checkpoint directory: {ckpt_dir}")
    # learning rate, optimizer, early stopping
    logger.info(f"Initial learning rate: {model_wrapper.get_learning_rate()}")
    logger.info(f"Optimizer: {model_wrapper.optimizer.__class__.__name__}")
    logger.info(f"Early stopping patience: {opt.earlystop_epoch} epochs")
    logger.info(f"Learning rate {model_wrapper.get_learning_rate()} will be reduced by 10 if early stopping is triggered and training will continue until learning rate < 1e-6")

    for epoch in range(start_epoch, opt.num_epoches+1):
        if epoch > start_epoch:
            # Training
            pbar = tqdm.tqdm(train_data_loader)
            for data in pbar:
                loss = model_wrapper.train_on_batch(data).item()
                total_steps = model_wrapper.total_steps
                pbar.set_description(f"Train loss: {loss:.4f}")
            
            # Save model
            model_wrapper.save_networks(epoch)

        # Validation
        print("Validation ...", flush=True)
        y_true, y_pred, y_path = model_wrapper.predict(valid_data_loader)
        acc = balanced_accuracy_score(y_true, y_pred > 0.0)
        auc = roc_auc_score(y_true, y_pred)
        lr = model_wrapper.get_learning_rate()
        print("After {} epoches: val acc = {}; val auc = {}".format(epoch, acc, auc), flush=True)
        logger.info(f"After {epoch} epoches: val acc = {acc}; val auc = {auc}; learning rate = {lr}")

        # Early Stopping
        if early_stopping is None:
            early_stopping = EarlyStopping(
                init_score=acc, 
                patience=opt.earlystop_epoch,
                delta=0.001, 
                verbose=True,
                logger=logger
            )
            # print(f"early_stopping: {early_stopping}")
            # breakpoint()
            print('Save best model', flush=True)
            model_wrapper.save_networks('best')
            logger.info(f"Save best model at epoch {epoch} with val acc = {acc}; val auc = {auc}; learning rate = {lr} - early stopping initialized with patience = {opt.earlystop_epoch} and delta = 0.001")
        else:
            if early_stopping(acc):
                print('Save best model', flush=True)
                model_wrapper.save_networks('best')
                logger.info(f"Save best model at epoch {epoch} with val acc = {acc}; val auc = {auc}; learning rate = {lr}") # - early stop {early_stopping.count_down}/{opt.earlystop_epoch}")
                # breakpoint()
            if early_stopping.early_stop:
                cont_train = model_wrapper.adjust_learning_rate()
                if cont_train:
                    print("Learning rate dropped by 10, continue training ...", flush=True)
                    early_stopping.reset_counter()
                    logger.info(f"Learning rate dropped by 10, reset early stopping counter and continue training ...")
                else:
                    print("Early stopping.", flush=True)
                    logger.info(f"Early stopping at epoch {epoch} with val acc = {acc}; val auc = {auc}; learning rate = {lr}")
                    logger.info(f"Best model was saved at epoch {early_stopping.best_epoch} with val acc = {early_stopping.best_score}; val auc = {early_stopping.best_auc}; learning rate = {early_stopping.best_lr}")
                    break
