import os  
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

# Ensure 'results' folder exists or create it
if not os.path.exists('results'):
    os.makedirs('results')

# Set model path and device
model_path = 'models/RRDB_ESRGAN_x4.pth'  # Update with your model file if different
device = torch.device('cpu')  # If you want to run on CPU, change 'cuda' -> 'cpu'
# device = torch.device('cpu')

# Set the folder containing low-resolution images
test_img_folder = 'LR/*'  # Ensure the LR folder contains images

# Initialize model
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=False)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)

    # Read image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255  # Normalize the image
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    # Inference with model
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    # Post-process the output
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    # Save result
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)
