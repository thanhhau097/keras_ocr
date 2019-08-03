from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from .training_callbacks import TrainingCallback


class OCRTrainer(BaseTrain):
    def __init__(self, model, data, val_data, config):
        super(OCRTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.val_data = val_data
        self.init_callbacks()

    def init_callbacks(self):
        # self.callbacks.append(
        #     ModelCheckpoint(
        #         filepath=os.path.join(self.config.callbacks.checkpoint_dir,
        #                               '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
        #         monitor=self.config.callbacks.checkpoint_monitor,
        #         mode=self.config.callbacks.checkpoint_mode,
        #         save_best_only=self.config.callbacks.checkpoint_save_best_only,
        #         save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
        #         verbose=self.config.callbacks.checkpoint_verbose,
        #     )
        # )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        self.callbacks.append(
            TrainingCallback(self.model.test_func, self.config.letters,
                             self.config.validation_steps, self.config.trainer.batch_size,
                             self.val_data.next_batch(), filepath='experiments/models/model.h5')
        )

        # if hasattr(self.config,"comet_api_key"):
        # if ("comet_api_key" in self.config):
        #     from comet_ml import Experiment
        #     experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp_name)
        #     experiment.disable_mp()
        #     experiment.log_multiple_params(self.config)
        #     self.callbacks.append(experiment.get_keras_callback())

    def train(self):
        self.model.model.fit_generator(generator=self.data.next_batch(),
                                 steps_per_epoch=5000,
                                 epochs=self.config.trainer.num_epochs,
                                 verbose=self.config.trainer.verbose_training,
                                 callbacks=self.callbacks)
