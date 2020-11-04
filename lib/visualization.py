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
        
        # just show the last iteration, but this can be modified
        mask_iter_grid = torchvision.utils.make_grid(model_outs['masks'][-1][0].exp())
        mean_iter_grid = torchvision.utils.make_grid(model_outs['x_means'][-1][0])
        recon_grid = torch.sum(model_outs[f'masks'][-1][0].exp() * model_outs['x_means'][-1][0], 0)

        writer.add_image(f'RGB', mean_iter_grid, step)
        writer.add_image(f'masks', mask_iter_grid, step)
        writer.add_image(f'reconstruction', recon_grid, step)

        writer.add_image('image', imgs, step)
