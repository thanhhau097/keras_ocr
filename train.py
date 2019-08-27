import comet_ml
from data_loader.ocr_data_loader import OCRDataLoader
from models.ocr_model import OCRModel
from models.attention_model import AttentionModel
from models.joint_model import JointModel
from trainers.ocr_trainer import OCRTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils.ocr_utils import build_vocab

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        # args = get_args()
        # config = process_config(args.config)
        config = process_config('configs/config.json')
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    # type 'ctc' and type 'attention' (+2)
    print('Building vocabulary')
    config.vocab_type = 'ctc'
    config.n_letters = build_vocab(config)

    print('Create the model.')
    model = OCRModel(config)

    config.downsample_factor = model.get_downsample_factor()
    print('Create the data generator.')
    data_loader = OCRDataLoader(config)
    val_data_loader = OCRDataLoader(config, phase='val')

    config.validation_steps = val_data_loader.get_steps()
    print('Create the trainer')
    trainer = OCRTrainer(model, data_loader, val_data_loader, config)

    print('Start training the model.')
    trainer.train()
    # test


if __name__ == '__main__':
    main()
