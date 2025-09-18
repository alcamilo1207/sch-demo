from stable_baselines3.common.callbacks import BaseCallback
import os

class SaveOnStepCallback(BaseCallback):
    """
    Callback for saving a model every `save_freq` steps.
    :param save_freq: (int) Save the model every `save_freq` steps
    :param save_path: (str) Path to the folder where the model will be saved
    :param verbose: (int) Verbosity level
    """
    def __init__(self, save_freq: int, save_path: str, verbose: int = 1):
        super(SaveOnStepCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f'model_{int(self.n_calls/1000)}k_steps')
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {model_path}")
        return True
