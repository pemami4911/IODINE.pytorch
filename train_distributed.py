import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data.distributed import DistributedSampler

from sacred import Experiment, cli_option

from lib.datasets import ds
from lib.datasets import StaticHdF5Dataset
from lib.model import net
from lib.model import IODINE
from lib.visualization import visualize_slots

from tqdm import tqdm
from pathlib import Path
import shutil
import pprint

@cli_option('-r','--local_rank')
def local_rank_option(args, run):
    run.info['local_rank'] = args

ex = Experiment('TRAINING', ingredients=[ds, net], additional_cli_options=[local_rank_option])

@ex.config
def cfg():
    training = {
            'batch_size': 16,  # training mini-batch size
            'num_workers': 8,  # pytorch dataloader workers
            'iters': 500000,  # train steps if no curriculum
            'lr': 3e-4,  # Adam LR
            'mode': 'train',
            'tensorboard_freq': 100,  # how often to write to TB
            'tensorboard_delete_prev': False,  # delete TB dir if already exists
            'checkpoint_freq': 25000,  # save checkpoints every % steps
            'load_from_checkpoint': False,  # whether to load from a checkpoint or not
            'checkpoint': '',  # name of .pth file to load model state
            'run_suffix': 'debug',  # string to append to run name
            'out_dir': 'experiments'
        }

def save_checkpoint(step, model, model_opt, filepath):
    state = {
        'step': step,
        'model': model.state_dict(),
        'model_opt': model_opt.state_dict(),
    }
    torch.save(state, filepath)

@ex.automain
def run(training, seed, _run):
    # maybe create
    run_dir = Path(training['out_dir'], 'runs')
    checkpoint_dir = Path(training['out_dir'], 'weights')
    tb_dir = Path(training['out_dir'], 'tb')
    
    for dir_ in [run_dir, checkpoint_dir, tb_dir]:
        if not dir_.exists():
            #dir_.mkdir()
            print(f'Create {dir_} before running!')
            exit(1)

    tb_dbg = tb_dir / training['run_suffix']

    local_rank = 'cuda:{}'.format(_run.info['local_rank'])
    if local_rank == 'cuda:0':
        writer = SummaryWriter(tb_dbg)
    
    # Fix random seed
    print(f'setting random seed to {seed}')
    # Auto-set by sacred
    # torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    # Auto-set by sacred 
    #np.random.seed(seed)
        
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)

    model = IODINE(batch_size=training['batch_size'])

    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    model.train()

    # Optimization
    model_opt = torch.optim.Adam(model.parameters(), lr=training['lr'])

    if not training['load_from_checkpoint']:    
        step = 0 
        checkpoint_step = 0
    else:
        checkpoint = checkpoint_dir / training['checkpoint']
        map_location = {'cuda:0': local_rank}
        state = torch.load(checkpoint, map_location=map_location)
        model.load_state_dict(state['model'])
        model_opt.load_state_dict(state['model_opt'])
        step = state['step']
        checkpoint_step = step

    tr_dataset = StaticHdF5Dataset(d_set=training['mode'])
    batch_size = training['batch_size']
    tr_sampler = DistributedSampler(dataset=tr_dataset)
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
            batch_size=batch_size, sampler=tr_sampler, 
            num_workers=training['num_workers'], drop_last=True)
    
    max_iters = training['iters']

    while step <= max_iters:
        
        if local_rank == 'cuda:0':
            data_iter = tqdm(tr_dataloader)
        else:
            data_iter = tr_dataloader

        for batch in data_iter:
            img_batch = batch['imgs'].to(local_rank)
            model_opt.zero_grad()

            out_dict = model(img_batch)
    
            total_loss = out_dict['total_loss']
            kl = out_dict['kl']
            nll = out_dict['nll']
            
            total_loss.backward()
            # clip gradient norm to 5
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)

            model_opt.step()
                        
            # logging
            if step % training['tensorboard_freq'] == 0 and local_rank == 'cuda:0':
                writer.add_scalar('train/total_loss', total_loss, step)
                writer.add_scalar('train/KL', kl, step)
                writer.add_scalar('train/NLL', nll, step)
                visualize_slots(writer, img_batch, out_dict, step)
                
            if step > 0 and step % training['checkpoint_freq'] == 0 and local_rank == 'cuda:0':
                prefix = training['run_suffix']
                save_checkpoint(step, model, model_opt, 
                       checkpoint_dir / f'{prefix}-state-{step}.pth')
            
            if step >= max_iters:
                step += 1
                break
            step += 1

    if local_rank == 'cuda:0':
        writer.close()