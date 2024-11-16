import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color, get_trainer
import os

from utils import get_model, create_dataset


def run_single(model_name, dataset, pretrained_file='', skip_training=False, **kwargs):
    # Get the directory containing run.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct absolute paths to config files
    config_dir = os.path.join(current_dir, 'config')
    props = [
        os.path.join(config_dir, 'overall.yaml'),
        os.path.join(config_dir, f'{model_name}.yaml')
    ]
    print(props)

    model_class = get_model(model_name)

    # configurations initialization
    config = Config(model=model_class, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = model_class(config, train_data.dataset).to(config['device'])

    # Load pre-trained model
    if pretrained_file != '':
        logger.info(f'Loading from {pretrained_file}')
        checkpoint = torch.load(pretrained_file, map_location=config['device'])
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
    logger.info(model)

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    if not skip_training:
        # model training
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=True, show_progress=config['show_progress']
        )
        # model evaluation with best model
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
    else:
        logger.info('Skipping training phase...')
        best_valid_score, best_valid_result = None, None
        # evaluate using current model state without loading
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='UniSRec', help='model name')
    parser.add_argument('-d', type=str, default='All_Beauty', help='dataset name')
    parser.add_argument('-p', type=str, default='', help='pre-trained model path')
    parser.add_argument('--eval-only', action='store_true', help='skip training and run evaluation only')
    args, unparsed = parser.parse_known_args()
    print(args)

    run_single(args.m, args.d, pretrained_file=args.p, skip_training=args.eval_only)
