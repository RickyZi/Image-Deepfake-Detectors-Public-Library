from parser import get_parser
import subprocess

if __name__ == "__main__":
    parser = get_parser()
    settings = parser.parse_args()
    print(settings)

    with open ("configs/test_template.json", "r") as f:
        training_template = f.read()
        training_template = training_template.replace("${DATA_KEYS}", settings.data_keys)
        training_template = training_template.replace("${DATA_ROOT}", settings.data_root)
        training_template = training_template.replace("${SPLIT_FILE}", settings.split_file)
        training_template = training_template.replace("${NAME}", settings.name)
        training_template = training_template.replace("${DEVICE}", settings.device)
        training_template = training_template.replace("${TASK}", settings.task)

    with open("configs/test.json", "w") as f:
        f.write(training_template)
        print("Test config file created")

    subprocess.run(f'python -u src/eval.py --config configs/test.json', shell=True)