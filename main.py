import argparse
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed
from trainer import GCFMCLTrainer
from model import GCFMCL
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='D2R', help='datasets')
parser.add_argument('--config', type=str, default='./properties/D2R.yaml', help='config.')
args, _ = parser.parse_known_args()
args.config_file_list = [
    'properties/D2R.yaml'
]
config = Config(
    model=GCFMCL,
    dataset=args.dataset,
    config_file_list=args.config_file_list
)
config['seed'] = 1
init_seed(config['seed'], config['reproducibility'])
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)
model = GCFMCL(config, train_data.dataset).to(config['device'])
trainer = GCFMCLTrainer(config, model)
trainer.fit(train_data, valid_data,dataset)