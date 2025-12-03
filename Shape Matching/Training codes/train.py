from CoordMLP import CoordMLP, save_model, load_model
from data import  load_data
import torch
import torchvision
import torch.utils.tensorboard as tb
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from torch import nn
import os


class WeightedLogMSE(nn.Module):
    def __init__(self,
                 eps=2.5e-3,
                 y0_low=0.08, w_low=100,
                 y0_high=0.17, w_mid=25,
                 w_high=1.0,
                 reduction='mean'):
        super().__init__()
        self.eps = eps
        self.y0_low, self.w_low = y0_low, w_low
        self.y0_high, self.w_mid = y0_high, w_mid
        self.w_high = w_high
        self.reduction = reduction

    def forward(self, pred, target):
        # log‐transform
        lp = torch.log(pred + self.eps)
        lt = torch.log(target + self.eps)
        base_loss = (lp - lt).pow(2.0)

        # piecewise weight on original target
        w = torch.where(
            target < self.y0_low,
            torch.tensor(self.w_low, device=target.device),
            torch.where(
                target < self.y0_high,
                torch.tensor(self.w_mid, device=target.device),
                torch.tensor(self.w_high, device=target.device)
            )
        )

        weighted = w * base_loss
        return weighted.mean() if self.reduction=='mean' else weighted

class MAPELoss(torch.nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        # Avoid division by zero
        denominator = torch.abs(targets) + self.epsilon
        return torch.mean(torch.abs((targets - outputs) / denominator)) * 100

class MAPELossElementWise(torch.nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        # Calculate absolute percentage error for each element
        # Avoid division by zero
        denominator = torch.abs(targets) + self.epsilon
        individual_errors = torch.abs((targets - outputs) / denominator) * 100
        return individual_errors

"""class WeightedL1_piecewise(nn.Module):
    def __init__(self, y0=0.17, w1=4.2, w2=1.0, reduction='mean'):
        super().__init__()
        self.y0, self.w1, self.w2 = y0, w1, w2
        self.reduce = reduction

    def forward(self, pred, target):
        w = torch.where(target <= self.y0,
                        torch.tensor(self.w1, device=target.device),
                        torch.tensor(self.w2, device=target.device))
        if(self.reduce == 'mean'):
            return (w * (pred-target).abs()).mean()
        else:
            return (w * (pred-target).abs())"""

class WeightedL1_piecewise(nn.Module):
    def __init__(self, y0_low=0.08, w_low=1, y0_high=0.17, w_mid=1, w_high=1.0, reduction='mean'):
        super().__init__()
        self.y0_low, self.w_low = y0_low, w_low
        self.y0_high, self.w_mid = y0_high, w_mid
        self.w_high = w_high
        self.reduce = reduction

    def forward(self, pred, target):
        # Define weights based on three conditions
        # If target < y0_low (0.08), weight is w_low (8.4)
        # Else if target < y0_high (0.17), weight is w_mid (4.2)
        # Else (target >= y0_high), weight is w_high (1.0)
        w = torch.where(target < self.y0_low,
                        torch.tensor(self.w_low, device=target.device),
                        torch.where(target < self.y0_high,
                                    torch.tensor(self.w_mid, device=target.device),
                                    torch.tensor(self.w_high, device=target.device)))

        if(self.reduce == 'mean'):
            return (w * (pred-target).abs()).mean()
        else:
            return (w * (pred-target).abs())
import subprocess
def train(args):
    from os import path
    model = CoordMLP()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """ 
    Your code here, modify your HW1 / HW2 code
    
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = CoordMLP(neurons=args.no_neurons).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, fused=True)
    #step_size = 10  # Reduce LR every 4 epochs
    #gamma = 1/5  # Reduction factor
    #total_epochs = 40
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch, eta_min=1e-4)
    # Create the scheduler
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    startepoch = 0
    if args.continue_training:
        checkpoint_path =args.checkpointpath
        startepoch, model, optimizer,scheduler = load_model(model, optimizer, scheduler,checkpoint_path, device)
        startepoch+=1
        #model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'coord_mlp.th')))

    loss = torch.nn.L1Loss()
    mapeloss_scalar = MAPELoss(epsilon = 1e-3)
    mapeloss_element = MAPELossElementWise(epsilon = 1e-3)
    weighted_loss = WeightedL1_piecewise()
    weighted_loss_val = WeightedL1_piecewise(reduction='val')
    
    huberloss = torch.nn.HuberLoss(reduction='mean', delta=args.delta)
    lossval = torch.nn.L1Loss(reduction='none')
    best_val_loss = best_std = 1000
    save_model(model, optimizer, scheduler,-1, f"pcamodel_{best_val_loss}")
    import inspect
    #mcatrain = "MCA_standard_rotated_center_1000_train_normal.npy"
    #mcavalid = "MCA_standard_rotated_center_1000_valid_normal.npy"
    pca_path = "/storage/pcaUDFVertProcess_all_TNetalign.npy"
    #pca_path = "/storage/pcaUDFMeshVertProcess_all_notnorm.npy" #"/storage/pca_std_0_25fil_rot_center_scale_voxels_norm.npy"
    scalefactor_path = "ScaleFactorSmallMeshVertices.npy"
    meshvertices = "/storage/MeshVerticesAligned.npz"#"MeshVertices_RemeshedSMAL_NOUDF.npz"#
    #meshvertices ="/storage/VerticesSMALProcessedKaolinRotate.npz"
    #mcavalid = "Xpca_voxels_dense_rot_center_cleannoise_valid_0_025fil_norm.npy"
    #trainindex = "TrainindexCAFM.npy"
    #validindex = "ValindexCAFM.npy"
    #dest_coords = "DestsCoord_CAFM.npz"
    #source_coords = "SourcesCoord_CAFM.npz"
    #dist_data = "DiffEuclidean_CAFM.npz"
    npzdatatrain = ["geodesic_data_0.npz", "geodesic_data_1.npz"]
    npzdataval = ["geodesic_data_val.npz"]
    train_data = load_data(meshvertices, npzdatatrain, pca_path, scalefactor_path, num_workers=8, batch_size = args.batch_size)
    valid_data = load_data(meshvertices, npzdataval, pca_path, scalefactor_path, num_workers=8, batch_size = 30384)
    print("Data loaded")
    global_step = 0
    global_step_val = 0
    
    val_epoch = 1
    
    for epoch in range(startepoch, args.num_epoch):
        #if(epoch > 8):
        #    break
        model.train()
        lossavrg = None
        nobatch = 0
        print("Epoch :", epoch)
        #if(epoch >= 1):
        #    break
        for mesh1, mesh2, source, dest, dist1, dist2 in tqdm(train_data, miniters=2000, mininterval=35):
            
            nobatch+=1
            mesh1, mesh2, source, dest, dist1, dist2 = mesh1.to(device, non_blocking=True),mesh2.to(device, non_blocking=True), source.to(device, non_blocking=True), \
                  dest.to(device, non_blocking=True), dist1.to(device, non_blocking=True), dist2.to(device, non_blocking=True)

            pred = model(source,dest, mesh1, mesh2)
            #gt = torch.cat((dist1,dist2),dim=-1)
            loss_val = loss(pred,(dist1 +dist2)/2.0)#(huberloss(pred, (dist1 +dist2)/2.0))
            #if(loss_val.isnan().any()):
            #    print("Nan happens")
            if(lossavrg is None):
                lossavrg = loss_val.detach().cpu().numpy()
                
            else :
                lossavrg += loss_val.detach().cpu().numpy()

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            if(nobatch % 10_000 == 0):
                print("Loss : ",lossavrg/nobatch)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        print("Train loss : ", lossavrg/nobatch)
        model.eval()
        if(epoch%val_epoch==0):
            total_items = 0 # Count of all individual items processed
            sum_losses = 0.0 # Sum of all individual item losses
            sum_sq_losses = 0.0 # Sum of the squares of all individual item losses
            with torch.no_grad():
                for mesh1, mesh2, source, dest, dist1, dist2 in tqdm(valid_data):
            
                    nobatch+=1
                    mesh1, mesh2, source, dest, dist1, dist2 = mesh1.to(device, non_blocking=True),mesh2.to(device, non_blocking=True), source.to(device, non_blocking=True), \
                        dest.to(device, non_blocking=True), dist1.to(device, non_blocking=True), dist2.to(device, non_blocking=True)

                    pred = model(source,dest, mesh1, mesh2)
                    
                    target = (dist1 + dist2)/2.0 # Your target calculation
                    #gt = torch.cat((dist1,dist2),dim=-1)
                    # Create a boolean mask where either pred or target is < 0.17
                    mask = (pred < 0.17) | (target < 0.17)

                    # Apply the mask
                    pred_filtered = pred[mask]
                    target_filtered = target[mask]

                    # Compute the loss only on the filtered values
                    if pred_filtered.numel() > 0:
                        per_item_losses = lossval(pred_filtered, target_filtered)
                    else:
                        per_item_losses = torch.tensor(0.0, device=pred.device)  # Fallback if no values match


                    # Flatten the per-item losses into a 1D tensor
                    per_item_losses_flat = per_item_losses.view(-1)

                    # Ensure the flat tensor is on the correct device for summation if not already
                    per_item_losses_flat = per_item_losses_flat.to(device)

                    # Get the number of items in this batch (which is your batch size, potentially less for the last batch)
                    batch_items = per_item_losses_flat.numel()

                    if batch_items > 0:
                        # Accumulate total number of items processed across all batches
                        total_items += batch_items

                        # Accumulate sum of losses and sum of squared losses across all items
                        # Use .sum() on the tensor, which is efficient on GPU
                        sum_losses += per_item_losses_flat.sum()
                        sum_sq_losses += (per_item_losses_flat ** 2).sum()
                sum_losses_scalar = sum_losses.item() if isinstance(sum_losses, torch.Tensor) else sum_losses
                sum_sq_losses_scalar = sum_sq_losses.item() if isinstance(sum_sq_losses, torch.Tensor) else sum_sq_losses

                # Calculate the overall mean per-item loss across the entire validation dataset
                overall_mean_per_item_loss = sum_losses_scalar / total_items if total_items > 0 else 0.0
                print("Val loss (mean per-item across epoch): ", overall_mean_per_item_loss)

                # Calculate the standard deviation of ALL individual item losses across the entire validation dataset
                overall_std_per_item_loss = 0.0
                variance = (sum_sq_losses_scalar - (sum_losses_scalar**2) / total_items) / (total_items - 1)

                # Ensure variance is non-negative due to potential floating point inaccuracies with large numbers
                variance = max(0.0, variance)

                # Standard deviation is the square root of variance
                overall_std_per_item_loss = np.sqrt(variance) # Using np.sqrt for simplicity after moving to CPU

                print("Val loss (std dev per-item across epoch): ", overall_std_per_item_loss)
                #print(acc_tot/nobatch)
                if(overall_mean_per_item_loss < best_val_loss or overall_std_per_item_loss < best_std):
                    print("Best Val Loss or Val Std improved")
                    best_val_loss = overall_mean_per_item_loss
                    best_std = overall_std_per_item_loss
                    print(best_val_loss)
                    print(best_std)
                save_model(model, optimizer, scheduler,epoch, f"pcamodel_ep_{epoch}_l_{overall_mean_per_item_loss}_std_{overall_std_per_item_loss}")
                command = f"python testshapematchefficient.py -neu {args.no_neurons} -ck pcamodel_ep_{epoch}_l_{overall_mean_per_item_loss}_std_{overall_std_per_item_loss}.pth"
                subprocess.call(command, shell=True)
        scheduler.step()
        

        #if valid_logger is None or train_logger is None:
        #print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, confusion.global_accuracy,
        #                                                            val_confusion.global_accuracy))
        #save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-bs', '--batch_size', type=int, default=3072)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-neu', '--no_neurons', type=int,default=400)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-ck', '--checkpointpath')
    parser.add_argument('-d', '--delta', type=float,default = 0.035)
    
    
    args = parser.parse_args()
    train(args)