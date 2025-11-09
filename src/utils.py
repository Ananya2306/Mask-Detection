from torchvision import transforms

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3,0.3,0.2,0.02),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# IMPORTANT: folder names are alphabetically sorted by ImageFolder
# ensure "mask" and "no_mask" match this label order
LABELS = ["mask", "no_mask"]  # because "mask" < "no_mask" alphabetically
