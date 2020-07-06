import wandb
import cv2
import pandas as pd

from torch.utils.tensorboard  import SummaryWriter
from pathlib                  import Path

class Logger:

  def __init__(self, log_dir:str):
        # define summery writer
      self.writer = SummaryWriter
      self.wandb = wandb.log()

      # define relevant paths
      self.trial_tb_dir = Path(log_dir) / 'tensorboard'   # tensorboard dir
      self.log_file = Path(log_dir) / 'metrics'           # file to store metrics for each iteration
      self.ckpt_dir = Path(log_dir) / 'checkpoints'       # directory to store saved models

      # define experiment/trial file structure
      self._init_trial_dir()

  def log_dict(self, dict_values, step, split):
      #write to wandb
      self.wandb(dict_values)
      # write to tensorboard
      
      for key, value in a_dict_values.items():
        self.writer.add_scalar(f"{key}/{split}", step)

  def log_image(self, img, step):

      img = img[0][0].unsqueeze(0)
      self.writer.add_image('images',img, step)

  def log_iteration(self, dict_values, step, split):
      """
       Log all relavent metrics to log file. Should be a csv file that looks
       like: 
            epoch | itration | train_loss | ...
       """
      metrics_file = self.log_file / Path('metrics.csv')

      dict_values.update({'step':step})
      dict_values.update({'split':split})
      df = pd.DataFrame(dict_values)

      if Path(metrics_file).is_file() == False:      
        df.to_csv(metrics_file)      
      else:        
        df_old = pd.read_csv(metrics_file)
        df = df.append(df_old)
        df.to_csv(metrics_file)

  def save_checkpoint(self, model):

      ckpt_dict = {'model_name': model.__class__.__name__, 
                   'model_args': model.args_dict(),
                   'model_state': model.state_dict()}.ckpt_path = os.path.join(args.save_dir, f"{model.__class__.__name__}_best.pth")
                     
      torch.save(ckpt_dict, celf.ckpt_dir)

  def _init_trial_dir(self):
      """
      # structure the log directory for this trial
      """
      if Path(self.trial_tb_dir).is_dir() == False:
        pathlib.Path(self.trial_tb_dir).mkdir(parents=True, exist_ok=False) #if the if command works, exception is never raised

      if Path(self.log_file).is_dir() == False:
        pathlib.Path(self.trial_tb_dir).mkdir(parents=True, exist_ok=False) 

      if Path(self.ckpt_dir).is_dir() == False:
        pathlib.Path(self.ckpt_dir).mkdir(parents=True, exist_ok=False)






