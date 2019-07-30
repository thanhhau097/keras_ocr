import comet_ml
from data_loader.ocr_data_loader import OCRDataLoader
from models.ocr_model import OCRModel
from trainers.ocr_trainer import OCRTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

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

    print('Create the model.')
    model = OCRModel(config)

    config.downsample_factor = model.get_downsample_factor()
    print('Create the data generator.')
    data_loader = OCRDataLoader(config)

    # data = next(data_loader.next_batch())[0]
    # images = data['the_input']
    # print(len(images))
    # from matplotlib import pyplot as plt
    # for i in range(16):
    #     plt.imshow(images[i])
    #     plt.show()

    print('Create the trainer')
    trainer = OCRTrainer(model.model, data_loader, config)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()
