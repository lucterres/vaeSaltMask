import pandas as pd
from sklearn.model_selection import train_test_split
from MVAE import *
from SeismicT import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CSV = 'train.csv'
DEPTH_CSV = 'depths.csv'
TRAIN_IMAGE_DIR = './train/images/'
TRAIN_MASK_DIR = './train/masks/'
TEST_IMAGE_DIR = './test/images/'

df_train = pd.read_csv(TRAIN_CSV)
df_depth = pd.read_csv(DEPTH_CSV)

df = pd.merge(df_depth, df_train)
# 0 = no_salt #1 = salt
df['salt'] = df['rle_mask'].notnull().replace([False, True], [0, 1])
df['salt'].head()

X_trainval, X_test, y_trainval, y_test = train_test_split(df['id'].values,
                                                          df['salt'].values,
                                                          stratify=df['salt'].values,
                                                          test_size=0.05, random_state=97)
X_train, X_val = train_test_split(X_trainval, stratify=y_trainval, test_size=0.1, random_state=97)
len(X_train), len(X_val), len(X_test)

mask_set = Masks(TRAIN_MASK_DIR, X_train)
train_loader = DataLoader(mask_set, batch_size=32, shuffle=True)

# ### MAB-VAE : The code of the MAB-VAE is available in the library `MVAE.py` present in the repository. Let us
# propose different type of decoders that we will use for the two first datasets. (We can't use exactly the same one
# for the last dataset because it is made of RGB images while the other images are binary). Remark : The encoder is
# considered as fixed by simplicity => The latent_dim should be the same for all decoders

Decoders = nn.ModuleList([Decoder_MLP(latent_dim=100, in_channel=1, im_size=32, hiddens=[128, 256, 512]),
                          Decoder_MLP(latent_dim=100, in_channel=1, im_size=32, hiddens=[256, 512, 1024]),
                          Decoder_Conv(latent_dim=100, in_channel=1, im_size=32, hiddens=[1024, 512, 256, 128], init=2),
                          Decoder_Conv(latent_dim=100, in_channel=1, im_size=32, hiddens=[1024, 512, 256], init=4),
                          Decoder_Linear_Conv(latent_dim=100, in_channel=1, im_size=32, hiddens=[512, 256, 128, 64],
                                              init=2)])

# Creation of a MabVAE instance
MVAE = MabVAE(train_loader, Decoders, eps=0.3, i=0)

# Use the GPU for making computations faster
trainer = Trainer(gpus=0, max_epochs=5)
trainer.fit(MVAE)
