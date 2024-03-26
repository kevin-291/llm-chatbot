from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicNER")

model = AutoModelForTokenClassification.from_pretrained("ai4bharat/IndicNER")

def get_predictions( sentence, tokenizer, model ):
  # Let us first tokenize the sentence - split words into subwords
  tok_sentence = tokenizer(sentence, return_tensors='pt')

  with torch.no_grad():
    # we will send the tokenized sentence to the model to get predictions
    logits = model(**tok_sentence).logits.argmax(-1)
    
    # We will map the maximum predicted class id with the class label
    predicted_tokens_classes = [model.config.id2label[t.item()] for t in logits[0]]
    
    predicted_labels = []
    
    previous_token_id = 0
    # we need to assign the named entity label to the head word and not the following sub-words
    word_ids = tok_sentence.word_ids()
    for word_index in range(len(word_ids)):
        if word_ids[word_index] == None:
            previous_token_id = word_ids[word_index]
        elif word_ids[word_index] == previous_token_id:
            previous_token_id = word_ids[word_index]
        else:
            predicted_labels.append( predicted_tokens_classes[ word_index ] )
            previous_token_id = word_ids[word_index]
    
    return predicted_labels
  
def get_labels(input_sentence):
    predicted_labels = get_predictions(sentence=input_sentence, 
                                   tokenizer=tokenizer,
                                   model=model
                                   )

    for index in range(len(input_sentence.split(' '))):
        return (input_sentence.split(' ')[index] + '\t' + predicted_labels[index])