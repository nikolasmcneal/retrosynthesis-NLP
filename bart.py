# !pip install ohmeow-blurr -q
# !pip install bert-score -q

import os

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import pandas as pd
from fastai.text.all import *
from transformers import *
from blurr.data.all import *
from blurr.modeling.all import *

df_train = pd.read_csv('train_data.csv', sep=',')
# df_valid = pd.read_csv('valid_data.csv', sep=',')
df_test = pd.read_csv('test_data.csv', sep=',')

# df = pd.concat([df_train, df_valid, df_test], axis=0)
df_train = df_train.reset_index()
df_train = df_train.head(200)


df_test = df_test.reset_index()
# df = df.dropna().reset_index()

print("length df_train: " + str(len(df_train)))


# In[7]:


# #Clean text
# df['content'] = df['content'].apply(lambda x: x.replace('/',''))
# df['content'] = df['content'].apply(lambda x: x.replace('\xa0',''))
# df.head()


# In[8]:


pretrained_model_name = "facebook/bart-large-cnn"
hf_arch, hf_config, hf_tokenizer, hf_model = BLURR.get_hf_objects(pretrained_model_name, 
                                                                  model_cls=BartForConditionalGeneration)


# In[9]:


hf_batch_tfm = HF_Seq2SeqBeforeBatchTransform(hf_arch, hf_config, hf_tokenizer, hf_model, task='summarization',
text_gen_kwargs={'max_length': 248,
 'min_length': 10,
 'do_sample': False,
 'early_stopping': True,
 'num_beams': 4,
 'temperature': 1.0,
 'top_k': 50,
 'top_p': 1.0,
 'repetition_penalty': 1.0,
 'bad_words_ids': None,
 'bos_token_id': 0,
 'pad_token_id': 1,
 'eos_token_id': 2,
 'length_penalty': 2.0,
 'no_repeat_ngram_size': 3,
 'encoder_no_repeat_ngram_size': 0,
 'num_return_sequences': 1,
 'decoder_start_token_id': 2,
 'use_cache': True,
 'num_beam_groups': 1,
 'diversity_penalty': 0.0,
 'output_attentions': False,
 'output_hidden_states': False,
 'output_scores': False,
 'return_dict_in_generate': False,
 'forced_bos_token_id': 0,
 'forced_eos_token_id': 2,
 'remove_invalid_values': False})

blocks = (HF_Seq2SeqBlock(before_batch_tfm=hf_batch_tfm), noop)

dblock = DataBlock(blocks=blocks, get_x=ColReader('Product'), get_y=ColReader('Reactants')) # splitter =


# In[10]:


dls = dblock.dataloaders(df_train, bs=2)


# In[11]:


# seq2seq_metrics = {
#         'rouge': {
#             'compute_kwargs': { 'rouge_types': ["rouge1", "rouge2", "rougeL"], 'use_stemmer': True },
#             'returns': ["rouge1", "rouge2", "rougeL"]
#         },
#         'bertscore': {
#             'compute_kwargs': { 'lang': 'fr' },
#             'returns': ["precision", "recall", "f1"]
#         }
#     }


# In[12]:


model = HF_BaseModelWrapper(hf_model)
learn_cbs = [HF_BaseModelCallback]
# fit_cbs = [HF_Seq2SeqMetricsCallback(custom_metrics=seq2seq_metrics)]

learn = Learner(dls, 
                model,
                opt_func=ranger,
                loss_func=CrossEntropyLossFlat(),
                metrics=[accuracy],
                cbs=learn_cbs,
                splitter=partial(seq2seq_splitter, arch=hf_arch)).to_fp16()

learn.create_opt() 
learn.freeze()


# In[13]:


learn.fit_one_cycle(5, lr_max=3e-5)


# In[14]:


# Learner.export(learn)


# In[ ]:





# In[15]:


# df_test['Product'][3]


# In[16]:


# len(df_test['Reactants'])


# In[18]:


from tqdm import tqdm
predictions = []
actual = []
#for idx, i in tqdm(enumerate(df_test['Reactants'])):
for idx, i in tqdm(range(200)):
    outputs = learn.blurr_generate(df_test['Reactants'][idx], early_stopping=False, num_return_sequences=3)
#     predictions.append(outputs)
    tmp = []
    for q in outputs:
        tmp.append(q.strip(' '))
    predictions.append(tmp)    
    
#     print(outputs[0])

# for idx, o in enumerate(outputs):
#     print(f'=== Prediction {idx+1} ===\n{o}\n')
    
#for idx, i in enumerate(df_test['Reactants']):
for idx in range(200):
    output = df_test['Reactants'][idx]
    actual.append(output)
    
# outputs = learn.blurr_generate(text_to_generate, early_stopping=False, num_return_sequences=1)

# for idx, o in enumerate(outputs):
#     print(f'=== Prediction {idx+1} ===\n{o}\n')


# In[22]:


#df_test["Product"][6]


# In[ ]:


# from sklearn.metrics import accuracy_score
# from sklearn.metrics import top_k_accuracy_score

# print(accuracy_score(actual, predictions))

# # top_k_accuracy_score(actual, predictions, k=3)


# In[19]:


import numpy as np

np_predictions = np.array(predictions)
np_actual = np.array(actual)

np.save('predictions.npy', predictions)
np.save('actual.npy', actual)

