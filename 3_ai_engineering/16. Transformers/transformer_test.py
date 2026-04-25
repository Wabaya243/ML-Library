from tf_keras.utils import plot_model
from tf_keras.models import Model
from tf_keras.layers import Dense, Input, LayerNormalization, Add, MultiHeadAttention


#On defini un encoder de block transformers simplifié

def transformer_encoder(input_dim, num_heads, ff_dim):
    inputs = Input(shape=(None, input_dim))
    # Attention Multi-tête
    attention_outputs = MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)(inputs, inputs)
    attention_outputs = Add()([inputs, attention_outputs])  # <-- ici, liste entre crochets
    attention_outputs = LayerNormalization()(attention_outputs)
    
    # Feed-Forward Neuronal 
    ff_outputs = Dense(ff_dim, activation='relu')(attention_outputs)
    ff_outputs = Dense(input_dim)(ff_outputs)
    outputs = Add()([attention_outputs, ff_outputs])  # <-- ici aussi
    outputs = LayerNormalization()(outputs)
    
    return Model(inputs, outputs)

# Créer et visualiser
encoder_block = transformer_encoder(input_dim=64, num_heads=8, ff_dim=128)
plot_model(encoder_block, show_shapes=True, to_file='Images/transformer_encoder.png')






# Transformer pre_entrainé


from transformers import BertTokenizer, TFBertModel


#Charger un Bert Pre_entrainé

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_tf = TFBertModel.from_pretrained('bert-base-uncased')


#Tokanisons un exmeple d'entree

text_tf = 'Les Transofrmers sont des models puissants pour les taches des NLP et autres'
inputs_tf = tokenizer(text_tf, return_tensors='tf')



# On passe l'entrée a travers les models

outputs_tf = model_tf(**inputs_tf)
print("La forme de l'etat caché: ", outputs_tf.last_hidden_state.shape)
plot_model(model_tf, show_shapes=True, to_file='Images/transformer_tf_encoder.png')






#Pytorch Version

from transformers import BertTokenizer, BertModel

#Charger un Bert Pre_entrainé

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


#Tokanisons un exmeple d'entree

text = 'Les Transofrmers sont des models puissants pour les taches des NLP'
inputs = tokenizer(text, return_tensors='pt')



# On passe l'entrée a travers les models

outputs = model(**inputs)
print("La forme de l'etat caché: ", outputs.last_hidden_state.shape)







