import torch
import torchvision


def visualize_slots(writer, batch_data, model_outs, step):
    """
    Render images for each mask and slot reconstruction,
    as well as mask*slot 
    """
    with torch.no_grad():
        batch_size, C, H, W = batch_data.shape
        imgs = batch_data[0]
        
        mask_iter_grid = torchvision.utils.make_grid(model_outs[f'masks'][0].exp())
        mean_iter_grid = torchvision.utils.make_grid(model_outs[f'x_means'][0])
        recon_grid = torch.sum(model_outs[f'masks'][0].exp() * model_outs[f'x_means'][0], 0)

        writer.add_image(f'RGB', mean_iter_grid, step)
        writer.add_image(f'masks', mask_iter_grid, step)
        writer.add_image(f'reconstruction', recon_grid, step)

        writer.add_image('image', imgs, step)