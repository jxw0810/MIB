from keras.models import load_model
from utils.losses import dice_coef, dice_coef_loss

weight_path = './weights.h5'
model = load_model(weight_path,custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})