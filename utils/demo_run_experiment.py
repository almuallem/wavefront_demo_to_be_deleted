import sys

sys.path.append("..")

import torch
from zernike import zern_abb
from pupil import polymask
from psf import psf
from matplotlib import pyplot as plt
from methods.proj_methods import gerchberg_saxton, fienup_hio
from fourier import ft2
from metrics import strehl, corr
import torch.optim as optim
import torch.nn as nn
from methods.xi_encoded_inr import INR_xi_encoded
from dataset_prep import pupil_dataset
from dataset_prep import s2p_alphas_betas
from zernike import zernike_gram_schmidt
import time
import imageio.v3 as iio
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os
import glob
import numpy as np
from IPython.display import HTML, display
from base64 import b64encode


def run_experiment(pupil_shape, N, inr=False, data = None):
    print("Running a noiseless experiment with ", pupil_shape, " pupil of size ", str(N))

    if data is not None:
        pupil, scale, dataset, tset, vset, psfs, phases, psfs_train, phases_train, xi, alphas, betas, psf_basis, psf_flat_cen, betas, alphas = data
        print ("Data already prepared!. Excuting the methods...")
    else:
        pupil, scale, dataset, tset, vset, psfs, phases, psfs_train, phases_train, xi, alphas, betas, psf_basis, psf_flat_cen, betas, alphas = prepare_data(pupil_shape, N)
    # pupil, scale = pupil_dataset(pupil_shape, N)
    # dataset = torch.load(f"datasets/table1_{pupil_shape}_{N}_demo.pt", weights_only=False)
    # tset = torch.load(f"datasets/y_phi_{pupil_shape}_{N}_train.pt", weights_only=False)
    # vset = torch.load(f"datasets/y_phi_{pupil_shape}_{N}_val.pt", weights_only=False)
    # L = len(vset) + len(tset)

    # print("Preparing the dataset...")
    # psfs = []
    # phases = []
    # for psf_idx, phase in dataset:
    #     psfs.append(psf_idx)
    #     phases.append(phase)
    # psfs_train = []
    # phases_train = []
    # for psf_idx, phase in tset:
    #     psfs_train.append(psf_idx)
    #     phases_train.append(phase)
    # for psf_idx, phase in vset:
    #     psfs_train.append(psf_idx)
    #     phases_train.append(phase)

    # psfs = torch.stack(psfs)
    # phases = torch.stack(phases)
    # psfs_train = torch.stack(psfs_train)
    # phases_train = torch.stack(phases_train)

    # print("Calculating S2P basis")
    # xi, alphas, betas, psf_basis, psf_mu = s2p_alphas_betas(
    #     psfs_train, phases_train, pupil_shape, N, zoom=30, xi_count=21
    # )
    # psf_flat_cen = psfs[
    #     :, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30
    # ].reshape(psfs.shape[0], -1) - torch.mean(
    #     psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30].reshape(
    #         psfs.shape[0], -1
    #     ),
    #     dim=0,
    #     keepdim=True,
    # )
    # # for newly generated data
    # betas = psf_flat_cen @ torch.linalg.pinv(psf_basis[:100].reshape(100, -1))
    # alphas = phases.reshape(phases.shape[0], -1) @ torch.linalg.pinv(xi.reshape(21, -1))

    # Dictionary to store results: {'method': [pr_mean, pr_std, we_mean, we_std, time]}
    results = {}
    
    # Projection methods
    # Gerchberg-Saxton
    print("Running the Gerchberg Saxton")
    start_time = time.perf_counter()
    out, GS_preds = gerchberg_saxton(torch.sqrt(psfs), pupil.unsqueeze(0), 500, return_predictions= True)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    psf_hat = psf(pupil, out)
    
    # print(
    #     f"Gerchberg-Saxton, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(out, phases, pupil))} +- {torch.std(strehl(out, phases, pupil))}. Elapsed time: {elapsed_time} seconds"
    # )

    pr_mean_gs = torch.mean(corr(psf_hat, psfs)).item()
    pr_std_gs = torch.std(corr(psf_hat, psfs)).item()
    we_mean_gs = torch.mean(strehl(out, phases, pupil)).item()
    we_std_gs = torch.std(strehl(out, phases, pupil)).item()

    results['Gerchberg-Saxton'] = [pr_mean_gs, pr_std_gs, we_mean_gs, we_std_gs, elapsed_time]

    phigs, psfgs = out * 1.0, psf_hat * 1.0

    # Fienup's Hybrid Input-Output
    print("Running Fienup's HIO")
    start_time = time.perf_counter()
    out, HIO_preds= fienup_hio(torch.sqrt(psfs), pupil.unsqueeze(0), 500, beta=0.5, return_predictions= True)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    psf_hat = psf(pupil, out)
    # print(
    #     f"HIO, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(out, phases, pupil))} +- {torch.std(strehl(out, phases, pupil))}. Elapsed time: {elapsed_time} seconds"
    # )

    pr_mean_hio = torch.mean(corr(psf_hat, psfs)).item()
    pr_std_hio = torch.std(corr(psf_hat, psfs)).item()
    we_mean_hio = torch.mean(strehl(out, phases, pupil)).item()
    we_std_hio = torch.std(strehl(out, phases, pupil)).item()

    results['HIO'] = [pr_mean_hio, pr_std_hio, we_mean_hio, we_std_hio, elapsed_time]

    phihio, psfhio = out * 1.0, psf_hat * 1.0

    # xi-Encoded INR, very VERY slow :( off by default
    if inr:
        print("Running xi-Encoded INR")
        psfs_inr = []
        phases_inr = []
        k = 0
        for psf_idx, phase in dataset:
            k = k + 1
            inr = INR_xi_encoded(21, 75, 1, pupil, scale, N_proj_approx=N)
            criterion = nn.MSELoss()
            opt_inr = optim.Adam(inr.parameters(), lr=80e-3)

            for i in range(300):
                # if i % 199 == 0 and i > 0:
                #     print(i, k, loss.item())
                opt_inr.zero_grad()
                phase_est = inr()
                psf_est = psf(pupil, phase_est.reshape(N, N).unsqueeze(0))[0]
                loss = criterion(psf_idx, psf_est)
                loss.backward()
                opt_inr.step()
            with torch.no_grad():
                psfs_inr.append(psf_est)
                phases_inr.append(phase_est.reshape(N, N))
            
            psfs_inrt = torch.stack(psfs_inr)
            phases_inrt = torch.stack(phases_inr)
            if k > 1:
                print(
                    f"{k}, xi-Encoded, PR: {torch.mean(corr(psfs_inrt, psfs[:k]))} +- {torch.std(corr(psfs_inrt, psfs[:k]))}, WE:  {torch.mean(strehl(phases_inrt, phases[:k], pupil))} +- {torch.std(strehl(phases_inrt, phases[:k], pupil))}"
                )

    # phiinr, psfinr = phases_inrt * 1.0, psfs_inrt * 1.0

    # # S2P-like methods

    s2p = torch.load(f"datasets/s2p_{pupil_shape}_{N}_train.pt", weights_only=False)
    print("Running the S2P method")
    start_time = time.perf_counter()
    alphas = s2p(betas)
    out = torch.einsum("dB, Bxy->dxy", alphas, xi)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    psf_hat = psf(pupil, out)
    # print(
    #     f"S2P, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(out, phases, pupil))} +- {torch.std(strehl(out, phases, pupil))}.  Elapsed time: {elapsed_time} second."
    # )

    pr_mean_s2p = torch.mean(corr(psf_hat, psfs)).item()
    pr_std_s2p = torch.std(corr(psf_hat, psfs)).item()
    we_mean_s2p = torch.mean(strehl(out, phases, pupil)).item()
    we_std_s2p = torch.std(strehl(out, phases, pupil)).item()
    
    results['S2P'] = [pr_mean_s2p, pr_std_s2p, we_mean_s2p, we_std_s2p, elapsed_time]

    phis2p, psfs2p = out * 1.0, psf_hat * 1.0
    # with torch.no_grad():
    #     for i in range(4):
    #         plt.figure(figsize=(18, 10))
    #         plt.subplot(2, 4, 1)
    #         plt.title("Ground Truth PSF")
    #         plt.imshow(psfs[i]) # Ground Truth PSF
            
    #         plt.subplot(2, 4, 2)
    #         plt.title("PSF Gerchberg Saxton")
    #         plt.imshow(psfgs[i])
            
    #         plt.subplot(2, 4, 3)
    #         plt.title("PSF HIO")
    #         plt.imshow(psfhio[i])

    #         plt.subplot(2, 4, 4)
    #         plt.title("PSF S2P")
    #         plt.imshow(psfs2p[i])

    #         # Row 2: Phases
    #         plt.subplot(2, 4, 5)
    #         plt.title("Ground Truth Phase")
    #         plt.imshow(phases[i] * pupil) # Ground Truth Phase

    #         plt.subplot(2, 4, 6)
    #         plt.title("Phase Gerchberg Saxton")
    #         plt.imshow(phigs[i])

    #         plt.subplot(2, 4, 7)
    #         plt.title("Phase HIO")
    #         plt.imshow(phihio[i])

    #         plt.subplot(2, 4, 8)
    #         plt.title("Phase S2P")
    #         plt.imshow(phis2p[i])
            
    #         # Optional: Add tight_layout to prevent overlap
    #         plt.tight_layout()
    #         plt.show()
    return results, [psfs, psfgs, psfhio, psfs2p], [phases, phigs, phihio, phis2p], pupil, GS_preds, HIO_preds

def parse_metrics(output_string):
    parts = output_string.split(' +- ')
    mean = float(parts[0].split(': ')[-1])
    std = float(parts[1].split(',')[0])
    return mean, std

def plot_experiment_results(all_psfs, all_phases, all_pupil):
    """Iterates through all collected experiment data and plots the PSF and Phase comparisons."""
    
    psf_titles = ["Ground Truth PSF", "PSF Gerchberg Saxton", "PSF HIO", "PSF S2P"]
    phase_titles = ["Ground Truth Phase", "Phase Gerchberg Saxton", "Phase HIO", "Phase S2P"]
    
    # Iterate over each completed experiment
    for exp_key in all_psfs.keys():
        print(f"\n--- Displaying Plots for Experiment: {exp_key} ---")
        
        psf_data_list = all_psfs[exp_key] # [psfs, psfgs, psfhio, psfs2p]
        phase_data_list = all_phases[exp_key] # [phases, phigs, phihio, phis2p]
        pupil = all_pupil[exp_key]

        # The data lists contain tensors of shape (N_samples, N, N)
        # We will iterate over the first 4 samples for visualization
        for i in range(4):
            plt.figure(figsize=(18, 10))
            
            # Plot PSFs (Row 1)
            for j in range(4):
                plt.subplot(2, 4, j + 1)
                plt.title(psf_titles[j] + f" (Sample {i})")
                # Detach and convert to numpy for plotting
                plt.imshow(psf_data_list[j][i].detach().cpu().numpy())
                
            # Plot Phases (Row 2)
            for j in range(4):
                plt.subplot(2, 4, j + 5)
                plt.title(phase_titles[j] + f" (Sample {i})")
                
                phase_to_plot = phase_data_list[j][i].detach().cpu().numpy()
                pupil_np = pupil.detach().cpu().numpy()
                
                if j == 0: # Ground Truth Phase
                    # Original code applied pupil mask to ground truth
                    plt.imshow(phase_to_plot * pupil_np) 
                else:
                    # For estimated phases, just show the result
                    plt.imshow(phase_to_plot) 

            # Optional: Add tight_layout to prevent overlap
            plt.tight_layout()
            plt.show()
            print("\n" + "-"*80 + "\n")




def create_gif_with_colorbar(GS_preds_np, discard_every = 5, output_dir="output", duration_ms=1000/30, cmap_name='twilight', plot_title="Phase Evolution: GS Algorithm"):
    """
    Generates a color GIF with a counter, a colorbar, and a title.

    Args:
        GS_preds_np (np.ndarray): The input phase array of shape (N, 1, H, W).
        output_dir (str): The directory to save the output GIF.
        duration_ms (float): The duration of each frame in milliseconds (ms).
        cmap_name (str): The name of the Matplotlib colormap.
        plot_title (str): The main title for the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Using slice [::5] as found in your provided code
    images_raw = GS_preds_np[::discard_every, 0, :, :]
    num_frames = images_raw.shape[0]
    total_iterations = GS_preds_np.shape[0]

    # 1. Normalize the Phase Data to the [0, 1] range
    normalized_data = (images_raw + np.pi) / (2 * np.pi) 

    # 2. Setup Matplotlib Figure and Colorbar
    # fig, ax = plt.subplots(figsize=(5, 4)) 
    # plt.close(fig)

    # # Display the FIRST frame to set up the plot object for the colorbar
    # im = ax.imshow(normalized_data[0], cmap=cmap_name, origin='lower', vmin=0, vmax=1)
    
    fig, ax = plt.subplots(figsize=(5.5, 4.2)) # Adjusted for a slightly wider image and small margin
    plt.close(fig)

    # Crucial for removing whitespace around the plot area itself
    fig.subplots_adjust(left=0.001, right=0.92, top=0.9, bottom=0.05) 
    # Adjust 'right' to ensure colorbar fits. 'left', 'top', 'bottom' to remove blank space.

    # Display the FIRST frame to set up the plot object for the colorbar
    im = ax.imshow(normalized_data[0], cmap=cmap_name, origin='lower', vmin=0, vmax=1)

    # Hide axes ticks and labels on the image subplot for a cleaner look
    ax.set_xticks([]); ax.set_yticks([])

    # --- Add Colorbar ---
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Phase (radians)', rotation=270, labelpad=15)
    
    # Set Colorbar ticks and labels to show the original phase range (-pi to pi)
    cbar_ticks = np.linspace(0, 1, 5)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    
    color_frames = []
    #print(f"Generating {num_frames} frames with counter, colorbar, and title at {1000/duration_ms:.2f} FPS...")

    # 3. Process each frame
    for i in range(num_frames):
        # Update the image data and ensure axes are clear
        ax.clear() 
        im = ax.imshow(normalized_data[i], cmap=cmap_name, origin='lower', vmin=0, vmax=1) # Redraw image
        ax.set_xticks([]); ax.set_yticks([]) # Keep axes clean
        
        # Add the main TITLE
        ax.set_title(plot_title, color='black', fontsize=12)#, backgroundcolor='black')
        
        # Add the frame counter text
        current_iter = 5 * (i)
        counter_text = f"Iter: {current_iter} / {total_iterations}"
        ax.text(
            0.05, 0.95, 
            counter_text, 
            transform=ax.transAxes, 
            color='white', 
            fontsize=10, 
            bbox={'facecolor': 'black', 'alpha': 0.6, 'pad': 2}
        )

        # Convert the Matplotlib figure (image + colorbar + title + counter) to a numpy array
        fig.canvas.draw()
        rgb_image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3] 
        color_frames.append(rgb_image)

    # 4. Generate the GIF
    output_filename = output_dir + f"/output_{plot_title}.gif"
    #print(f"Generating GIF: {output_filename}...")
    
    iio.imwrite(
        output_filename, 
        color_frames,
        duration=duration_ms, 
        loop=0
    )



def display_gifs_side_by_side(gif_directory="output"):
    """
    Reads all GIF files from a directory, Base64 encodes them, and displays
    them side-by-side in a Jupyter/Colab notebook using HTML flexbox layout.
    """
    
    # 1. Find all GIF files
    search_pattern = os.path.join(gif_directory, "*.gif")
    gif_files = sorted(glob.glob(search_pattern))
    
    if not gif_files:
        print(f"No GIF files found in '{gif_directory}'. Please check the directory and file extensions.")
        return

    # 2. Prepare HTML content for each GIF
    html_elements = []
    labels = []
    width_percent = 100 / len(gif_files)
    
    for filename in gif_files:
        # Read the file data
        with open(filename, 'rb') as f:
            gif_data = f.read()
            
        # Base64 encode the data
        data_url = "data:image/gif;base64," + b64encode(gif_data).decode()
        
        # Get a label from the filename (e.g., "output_30FPS_colorbar_twilight.gif" -> "twilight")
        label = os.path.basename(filename).split('_')[-1].replace('.gif', '').capitalize()
        labels.append(label)
        
        # HTML for the image container (the image itself and a label beneath it)
        gif_html = f"""
        <div style="flex: 1 1 {width_percent}%; text-align: center; margin: 5px;">
            <img src="{data_url}" style="width: 100%; height: auto; border: 1px solid #ccc;">
            <p style="margin-top: 5px; font-weight: bold;">{label}</p>
        </div>
        """
        html_elements.append(gif_html)

    # 3. Assemble the final HTML string with a flex container
    html_output = f"""
    <div style="display: flex; flex-direction: row; justify-content: space-around; width: 100%;">
        {''.join(html_elements)}
    </div>
    """
    
    # 4. Display the HTML in the notebook
    display(HTML(html_output))
    #print(f"Successfully displayed {len(gif_files)} GIFs side-by-side.")
    
