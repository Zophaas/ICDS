import tensorflow as tf
from transformers import BertTokenizer, TFBertForMaskedLM

# 初始化 BERT 分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForMaskedLM.from_pretrained('bert-base-uncased')

# 输入文本
text = "Hello, how are you doing today?"

# 分词并添加特殊标记
inputs = tokenizer(text, return_tensors='tf', max_length=512, truncation=True, padding='max_length')

# 创建掩码标记
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
labels = tf.identity(input_ids)

# 随机选择一些词进行掩码
rand = tf.random.uniform(shape=tf.shape(input_ids))
mask_arr = (rand < 0.15) & (input_ids != tokenizer.cls_token_id) & \
           (input_ids != tokenizer.sep_token_id) & (input_ids != tokenizer.pad_token_id)

selection = []

for i in range(tf.shape(input_ids)[0]):
    selection.append(
        tf.where(mask_arr[i]).numpy().flatten().tolist()
    )

for i in range(tf.shape(input_ids)[0]):
    input_ids = tf.tensor_scatter_nd_update(input_ids, tf.constant([[i, j] for j in selection[i]]), tf.constant([tokenizer.mask_token_id] * len(selection[i])))

# 将输入传递给模型
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

# 获取损失和预测结果
loss = outputs.loss
logits = outputs.logits

print(f"Loss: {loss.numpy()}")