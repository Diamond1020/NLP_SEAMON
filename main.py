import wordninja

from config import Config
from trainer import train_and_fit
from infer import infer_from_trained

words = wordninja.split('mrmorristhemartianwasflyingaroundthesolarsystemonedaywhenhesawastrangelightinfrontofhimwhatisthathethoughttohimselfmorriswasscaredbutheflewalittlebitclosersothathecouldseeitbetterhellohecalledouttherewasnoreplyhelloisanyonetherehecalledbutagaintherewasnoreplysuddenlyacreatureappearedinfrontofthelightbooitshoutedpoormorriswasreallyscaredandheflewoffhomeandhidunderhisbedeguk')
pre_sentences = " ".join(words)
print(pre_sentences)
config = Config() # loads default argument parameters as above
config.data_path = "./data/train.tags.en-fr.en"
config.batch_size = 32
config.lr = 5e-5 # change learning rate
config.model_no = 1 # sets model to PuncLSTM
train_and_fit(config) # starts training with configured parameters
inferer = infer_from_trained(config) # initiate infer object, which loads the model for inference, after training model
res = inferer.infer_sentence(pre_sentences)
print(res)