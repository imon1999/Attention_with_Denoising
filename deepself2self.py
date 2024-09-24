import numpy as np
import os
import glob
import torch
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.autograd import Variable
from model import DeepSelf2Self

def image_loader(image, device, p1, p2):
    """Load image, returns CUDA tensor."""
    loader = T.Compose([T.RandomHorizontalFlip(torch.round(torch.tensor(p1))),
                        T.RandomVerticalFlip(torch.round(torch.tensor(p2))),
                        T.ToTensor()])
    image = Image.fromarray(image.astype(np.uint8))
    image = loader(image).float()
    image = image.unsqueeze(0)
    return image.to(device)

if __name__ == "__main__":
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)
    image_list = [filepath for filepath in glob.iglob("Input/*.png")]
    path = "Output/"
    folder_list = glob.glob(path + '*', recursive=False)
    for z, image_path in enumerate(image_list):
        img = Image.open(image_path)
    # Initialize the model
    model = DeepSelf2Self(3).to(device)

    # Load the image
    img = np.array(Image.open(image_path))
    if len(img.shape) == 2:  
        img = np.stack((img,) * 3, axis=-1)
    print("Start new image running")
    print("Image shape:", img.shape)

    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    rate = 0.5
    NPred = 100
    x = torch.Tensor(img).to(device)
    w, h, c = img.shape
    slice_avg = torch.tensor([1, 3, w, h]).to(device)
    i = []
    l = []

    # Main processing loop
    for itr in range(100000):
        input_reshape = x.view(x.size()[0], -1)
        stdev = torch.std(input_reshape, 0, True).cpu().numpy()
        probs = np.zeros(input_reshape.size()[1])
        for j in range(len(probs)):
            probs[j] = np.random.normal(loc=1 - rate, scale=stdev[j], size=1)
            probs[j] = np.clip(probs[j], 0.0, 1.0)  
        probs = np.tile(probs, (input_reshape.size(0), 1))
        mask = Variable(torch.bernoulli(torch.Tensor(probs)), requires_grad=True)
        mask = mask.view(x.size())
        convert = mask.clone().detach().requires_grad_(True)

        p1 = np.random.uniform(size=1)
        p2 = np.random.uniform(size=1)
        img_input_tensor = image_loader(img, device, p1, p2)
        y = image_loader(img, device, p1, p2)
        mask = np.expand_dims(np.transpose(convert.detach().cpu().numpy(), [2, 0, 1]), 0)
        mask = torch.tensor(mask).to(device, dtype=torch.float32)
        model.train()
        img_input_tensor = img_input_tensor * mask
        output = model(img_input_tensor, mask)

        loader = T.Compose([T.RandomHorizontalFlip(torch.round(torch.tensor(p1))),
                            T.RandomVerticalFlip(torch.round(torch.tensor(p2)))])
        if itr == 0:
            slice_avg = loader(output)
        else:
            slice_avg = slice_avg * 0.99 + loader(output) * 0.01

        loss = torch.sum(abs(output - y)*(1 - mask))/torch.sum(1 - mask)
        # loss = torch.sum((output-y)*(output-y)*(1-mask))/torch.sum(1-mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Iteration {itr + 1}, loss = {loss.item() * 100:.4f}")

        i.append(itr + 1)
        l.append(loss.item() * 100)

        if (itr + 1) % 1000 == 0:
            model.eval()
            img_array = []
            sum_preds = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
            for j in range(NPred):
                input_reshape = x.view(x.size()[0], -1)
                stdev = torch.std(input_reshape, 0, True).cpu().numpy()
                probs = np.zeros(input_reshape.size()[1])
                for j in range(len(probs)):
                    probs[j] = np.random.normal(loc=1 - rate, scale=stdev[j], size=1)
                    probs[j] = np.clip(probs[j], 0.0, 1.0)
                probs = np.tile(probs, (input_reshape.size(0), 1))
                mask = Variable(torch.bernoulli(torch.Tensor(probs)), requires_grad=True)
                mask = mask.view(x.size())
                convert = mask.clone().detach().requires_grad_(True)

                img_input = img * convert.detach().cpu().numpy()
                img_input_tensor = image_loader(img_input, device, 0.1, 0.1)
                mask = np.expand_dims(np.transpose(convert.detach().cpu().numpy(), [2, 0, 1]), 0)
                mask = torch.tensor(mask).to(device, dtype=torch.float32)

                output_test = model(img_input_tensor, mask)
                sum_preds[:, :, :] += np.transpose(output_test.detach().cpu().numpy(), [2, 3, 1, 0])[:, :, :, 0]
                img_array.append(np.transpose(output_test.detach().cpu().numpy(), [2, 3, 1, 0])[:, :, :, 0])

            k = 0
            # print("k =", k, "image saving done")
            average = np.squeeze(np.uint8(np.clip(np.average(img_array, axis=0), 0, 1) * 255))
            base_filename = os.path.basename(image_path)
            input_name = os.path.splitext(base_filename)[0]
            avg_path = os.path.join(output, f"{input_name}_avg_{itr + 1}.png")
            write_img = Image.fromarray(average)
            write_img.save(avg_path)

    print(f"Finished processing {image_path}")

