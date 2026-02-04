from src.models.blip2.trainer import BLIPTrainer
from src.utils.config import Config

config = Config('configs/config.yaml')
trainer = BLIPTrainer(config)
trainer.train()