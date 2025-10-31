import os
from tqdm import tqdm
import torch
import pandas as pd
from networks import create_architecture, count_parameters
from utils.dataset import create_dataloader
from utils.processing import add_processing_arguments
from parser import get_parser

def test(loader, model, settings):
    model.eval()

    csv_filename = f'./train/{settings.name}/data/{settings.data_keys}/results.csv'
    # df = pd.DataFrame(columns=['name', 'pro','flag'])
    with open(csv_filename, 'w') as f:
        f.write(f"{','.join(['name', 'pro', 'flag'])}\n")
    with torch.no_grad():
        with tqdm(loader, unit='batch', mininterval=0.5) as tbatch:
            tbatch.set_description(f'Validation')
            for data_dict in tbatch:
                data = data_dict['img'].to(device)
                labels = data_dict['target'].to(device)
                paths = data_dict['path']

                scores = model(data).squeeze(1)
                
                with open(csv_filename, 'a') as f:
                    for score, label, path in zip(scores, labels, paths):
                        f.write(f"{path}, {score.item()}, {label.item()}\n")
                        # df = df._append({'name': path,'pro': score.item(),'flag':label.item()}, ignore_index=True)

    # df.to_csv(csv_filename, index=False)

if __name__ == '__main__':
    parser = get_parser()
    parser = add_processing_arguments(parser)
    settings = parser.parse_args()
    
    device = torch.device(settings.device if torch.cuda.is_available() else 'cpu')

    os.makedirs(f'./train/{settings.name}/data/{settings.data_keys}', exist_ok=True)
    test_dataloader = create_dataloader(settings, split='test')

    model = create_architecture(settings.arch, pretrained=True, num_classes=1).to(device)
    num_parameters = count_parameters(model)
    print(f"Arch: {settings.arch} with #parameters {num_parameters}")
    
    load_path = f'./train/{settings.name}/models/best.pt'
    
    print('loading the model from %s' % load_path)
    model.load_state_dict(torch.load(load_path, map_location=device)['model'])
    model.to(device)

    test(test_dataloader, model, settings)
