#!/usr/bin/env python
# coding: utf-8

# # Final Project 607

# ## Introduction
# 
# Today I will attempt to use a neural network model to detect reading levels based on a provided text.

# In[1]:


import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import BertTokenizer
import tensorflow as tf


# ## Dataset
# 
# This dataset was generated from ChatGPT 3.5, I asked 20 questions and ask for their response in every grade level.

# In[2]:


csv_file =  "https://raw.githubusercontent.com/jamilton08/reading_level_language/main/grade_level_dataset%20-%20Sheet1.csv"

df = pd.read_csv(csv_file)


# In[3]:


df.head()


# In[4]:


df = pd.melt(df,id_vars = ["question"], var_name = "grade", value_name = "response")


# In[5]:


##clean data removing punctuations and etc
df["grade"] = df["grade"].str.replace("grade_", "")
df["response"] = df["response"].str.replace("\.|!|\"|-|,|'", "")
df["grade"] = df["grade"].astype(int)
df["id"] = df.reset_index().index
df.tail()


# In[6]:


df.info()


# In[7]:


df.head()


# ## Tokenizing

# In[8]:


# bert tokenizes words by putting words to their base models which serves as a consistent
#way for analysis being that every word is brung to their most basic form
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


# In[9]:


# sample of response
df["response"].iloc[0]


# In[10]:


# many deep learning architectures processes language at the token level and hugging face has 
# bert which does just that for us
token = tokenizer.encode_plus(
df['response'].iloc[0],
max_length = 512,
truncation = True,
padding = 'max_length',
add_special_tokens=True,
return_tensors="tf")


# In[11]:


token


# ## Initializing Training Data

# In[12]:


# create ids and masks to populate them with what we recieve from tokenizer
X_input_ids = np.zeros((len(df), 256))
X_attn_masks = np.zeros((len(df), 256))


# In[13]:


X_input_ids.shape


# In[14]:


# build a function that will loop trough each row of the my dataframe is text and encode
#them and tokenize them then populate our id and mask dataframe
def generate_training_data(df, ids, masks, tokenizer):
    for i, text in tqdm(enumerate(df["response"])):
        tokenized_text = tokenizer.encode_plus(
        text,
        max_length = 256,
        truncation = True,
        padding = 'max_length',
        add_special_tokens = True,
        return_tensors = 'tf')
        ids[i, :] = tokenized_text.input_ids
        masks[i, :] = tokenized_text.attention_mask
    return ids, masks


# In[15]:


X_input_ids, X_attn_masks = generate_training_data(df, X_input_ids, X_attn_masks, tokenizer)


# In[16]:


X_input_ids


# In[17]:


## labels based on how many outputs we have 
labels = np.zeros((len(df), 12))


# In[18]:


## one hot encoding where our classification index indicator will become 1 in an n by m 
# matrix
labels[np.arange(len(df)), df["grade"].values - 1] = 1


# In[19]:


labels


# In[20]:


## tensorflow does a good job in create a mapped dataset of our inputs to the expected outputs
dataset = tf.data.Dataset.from_tensor_slices((X_input_ids, X_attn_masks, labels))


# In[21]:


dataset.take(1)


# In[22]:


## will be used to rearrange the output of the result when we map
def  GradeDatasetMapFunction(input_ids, attn_masks, labels):
    return {
        'input_ids':input_ids,
        'attention_masks' : attn_masks
    }, labels


# In[23]:


dataset = dataset.map(GradeDatasetMapFunction)


# In[24]:


#shuffle 
dataset = dataset.shuffle(10000).batch(16, drop_remainder = True)


# In[25]:


## get train size based on what percent we want to be training which is 80% of our data
p = 0.8
train_size = int((len(df)//16)*p)


# In[26]:


train_size


# In[27]:


df.shape


# In[28]:


## common in all machine learning to seperate data,, one section to train and the other to test
train_data = dataset.take(train_size)
testing_data = dataset.skip(train_size)


# ## Structure  Network

# In[30]:


from transformers import BertConfig, TFBertModel

config = BertConfig.from_pretrained('bert-base-cased')
bert_model = TFBertModel.from_pretrained('bert-base-cased', config=config)


# In[31]:


#input values
input_ids = tf.keras.layers.Input(shape=(256,), name='input_ids', dtype = 'int32')
attention_masks = tf.keras.layers.Input(shape=(256,), name='attention_masks', dtype = 'int32')

# hidden layer, my understanding is that were not dealing with probability here cause it's
# not just a binary classification which means we are working with values greater then 1
# hence it serves us better for this multiclassification problem, sigmoid is continously,
# divides one by one it self or greater making it one or less which if it goes over a threshold
# based on probability then we classifh
bert_embds = bert_model.bert(input_ids, attention_mask = attention_masks)[1]
intermediate_layer = tf.keras.layers.Dense(512, activation='relu', name = 'intermediate_layer')(bert_embds)

#output layer, we use softmax since were dealing with a multiclass which is a vector of 
#probabilities and based on where we fall for each output is probability we can identify 
# our class with softmax
output_layer = tf.keras.layers.Dense(12, activation = 'softmax', name="output_layer")(intermediate_layer)

model = tf.keras.Model(inputs = [input_ids, attention_masks], outputs=output_layer)
model.summary()


# In[32]:


# optim will reduce our loss, which backproporgation calculate the gradient of our loss to 
# update the weights and when the weights are optimized we will get our best accuracy.
#  Categorical cross entropy is used in multiclassification and our accuracy is based on 
# how often the predictions of this model matches the one hot encode from earlier.
optim = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss_func = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')


# model.compile(optimizer = optim, loss = loss_func, metrics=[acc])

# ## Build Network

# In[33]:


model.compile(optimizer = optim, loss = loss_func, metrics=[acc])


# ## Train Network
# 
# I I began to notice during cross validation that accuracy would decrease in some instance and feared that my model was gearing towards overfitting leading me to stop at 50 epochs.

# In[39]:


# training model, I did 50 epochs and the accuracy was not changing much at the end.
# epochs use cross validation and uses a blocks of the dataset to train the model, reads
# the loss, calculate the gradients and repeats every epoch and then recalculates blocks of data
# and repeats trying to find the diffirences and adjust accordingly.
hist = model.fit(
    train_data,
    validation_data = testing_data,
    epochs = 5
)


# In[62]:


model.save("reading_level_model")


# ## Test Network
# 
# I used the same dataset I trained the netowrk with as generating another dataset is very time consuming. 

# In[35]:


load_model = tf.keras.models.load_model("reading_level_model")


# In[36]:


## Tokenize regular input now 
def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length = 256,
        truncation = True,
        padding = 'max_length',
        add_special_tokens=True,
        return_tensors="tf"
    )
    return {
        'input_ids' : tf.cast(token.input_ids, tf.float64),
        'attention_masks' : tf.cast(token.attention_mask, tf.float64)
    }


# In[37]:


text = "Skunks have a special way to protect themselves when they feel scared or threatened. They can spray a really stinky liquid from their bodies. This liquid smells really bad, like rotten eggs or a stinky garbage can! Skunks spray this smelly stuff to scare away animals that might want to hurt them. So, when you smell a skunk, it's because it's trying to stay safe by making itself smell really awful to other animals."


# In[38]:


t_text = prepare_data(text, tokenizer)


# In[39]:


t_text


# In[40]:


softmaxvect = load_model(t_text)


# In[41]:


softmaxvect


# In[42]:


m = np.argmax(softmaxvect[0])
m


# In[43]:


## load test 
csv_file =  "https://raw.githubusercontent.com/jamilton08/reading_level_language/main/grade_level_dataset%20-%20Sheet1.csv"

df = pd.read_csv(csv_file)
df.head()


# In[44]:


##clean data removing punctuations and etc
df = pd.melt(df,id_vars = ["question"], var_name = "grade", value_name = "response")
df["grade"] = df["grade"].str.replace("grade_", "")
df["response"] = df["response"].str.replace("\.|!|\"|-|,|'", "")
df["grade"] = df["grade"].astype(int)
df["id"] = df.reset_index().index
df.tail()


# In[45]:


empt_list = list()

for index, row in df.iterrows():
    print(row['response'])
    tokened_text = prepare_data(row['response'], tokenizer)
    pred = load_model.predict(tokened_text)
    grade = np.argmax(pred[0])
    empt_list.append(grade)
    print(grade)
    


# In[46]:


df.shape[0]


# In[47]:


print(empt_list)


# In[48]:


df["predicted"] = empt_list
df["predicted"] = df["predicted"] + 1
df["correct"] = np.where(df['grade'] == df["predicted"], 1, 0)


# In[49]:


df


# ## Accuracy

# In[50]:


corrects = df["correct"].value_counts()

accuracy = corrects[1] / df.shape[0]

print("the full dataset predicted's accuracy is : {}".format(accuracy))


# Realistically the dataset is too small to train the network for real life use, I would have to gather a much bigger
# dataset as 75 percent accuracy seems to be a good value if in real life use it predicted with such probability. I will know see the distribution of how far off were the incorrect grades 

# In[51]:


wrong_values = df[df["correct"] == 0]


# ## Incorrect Results Analysis
# 

# In[52]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.countplot(x='grade', data=wrong_values)
plt.xlabel('Grades')
plt.ylabel('n times predicted wrong')
plt.show()


# High school grades seem to get confused alot, but now we will check by how much were they wrong in general

# In[53]:


wrong_values["diff"] = wrong_values["predicted"]- wrong_values["grade"] 

wrong_values.head()


# ## Grade Level Distance Distribution

# In[54]:


plt.figure(figsize=(8, 6))
sns.histplot(wrong_values["diff"], bins=5, kde=False, color='skyblue')
plt.title('Histogram of Data')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()


# ## High School Levels Justification
# This means grades were mainly predicted to be above reading level by one but no more then three and not lower then by one grade level, I decided to further research why is it that high school levels tend to be  most confusing and found that high school reading levels usually have the same reading level hence explaning the reading level neural network's confusion . ![lexile_chart.jpg](attachment:lexile_chart.jpg)

# ## Comparison To Other Algorithms
# I gave other algorithms a seventh grade reading level to see how they perform since the reading level's neural network didn't seem to have much trouble classifying them and it resulted in not a so good estimate. Might be because of the dataset generation proccess or simply that my model is being tested on dataset it used to train.
# Below you can find the document to see the results of an seventh grade from diffirent algorithms.
# 
# https://github.com/jamilton08/reading_level_language/blob/main/Readability%20Scoring%20System.pdf
# 
# 

# # # Conclusion

# Reading level are very measurable for a neural network, and a neural network does it effectively as well. The grades that were predicted wrong were still within their range for the most part according to the Lexile Text Standards. With a larger training dataset, this will surely be more effectly at distinguighing reading levels. 70% to 90% accuracy is the sweet spot for a model to perform in the real word and with a small sample, this did well. 

# In[ ]:




