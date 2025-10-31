import pickle
import os
import glob

dataroot = "/media/mmlab/Volume2/TrueFake"

dataset_structure = [
    'Fake/FLUX.1/animals',
    'Fake/FLUX.1/faces',
    'Fake/FLUX.1/general',
    'Fake/FLUX.1/landscapes',
    'Fake/StableDiffusion1.5/animals',
    'Fake/StableDiffusion1.5/faces',
    'Fake/StableDiffusion1.5/general',
    'Fake/StableDiffusion1.5/landscapes',
    'Fake/StableDiffusion2/animals',
    'Fake/StableDiffusion2/faces',
    'Fake/StableDiffusion2/general',
    'Fake/StableDiffusion2/landscapes',
    'Fake/StableDiffusion3/animals',
    'Fake/StableDiffusion3/faces',
    'Fake/StableDiffusion3/general',
    'Fake/StableDiffusion3/landscapes',
    'Fake/StableDiffusionXL/animals',
    'Fake/StableDiffusionXL/faces',
    'Fake/StableDiffusionXL/general',
    'Fake/StableDiffusionXL/landscapes',
    'Fake/StyleGAN/images-psi-0.5',
    'Fake/StyleGAN/images-psi-0.7',
    'Fake/StyleGAN2/conf-f-psi-0.5',
    'Fake/StyleGAN2/conf-f-psi-1',
    'Fake/StyleGAN3/conf-t-psi-0.5',
    'Fake/StyleGAN3/conf-t-psi-0.7',
    'Real/FFHQ',
    'Real/FORLAB'
]

if __name__ == "__main__":
    with open("./classes_nosocial.pkl", "rb") as f:
        results = pickle.load(f)

    results_shared = {}
    for social in ['Facebook', 'Telegram', 'Twitter']:
        social_root = f'{dataroot}/{social}'

        for dataset in dataset_structure:
            print(f'Processing {social}/{dataset}')
            current_dir = f'{social_root}/{dataset}'
            images = os.listdir(current_dir)

            for image in images:
                try:
                    image_key = f'{dataset}/{os.path.splitext(image)[0]}.png'
                    results_shared[f'{social}/{dataset}/{image}'] = results[image_key]
                except KeyError:
                    image_key = f'{dataset}/{os.path.splitext(image)[0]}.jpg'
                    results_shared[f'{social}/{dataset}/{image}'] = results[image_key]


    results_presocial = {}
    for key, value in results.items():
        results_presocial[f'PreSocial/{key}'] = value

    print(len(results))
    print(len(results_presocial))
    print(len(results_shared))
    print(len(results_shared)+len(results_presocial))

    results_all = {**results_presocial, **results_shared}
    print(len(results_all))

    with open("./classes.pkl", "wb") as f:
        pickle.dump(results_all, f, protocol=pickle.HIGHEST_PROTOCOL)
