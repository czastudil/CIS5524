from numba import jit
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import losses
from sentence_transformers import SentenceTransformer, models
import random
from utilities import *

@jit
def run():
    # Shuffle the article pairs
    sims = read_adj_list()
    random.shuffle(sims)
    # Select 10,000 random examples for training
    n_examples = 10000
    training_sims = sims[:n_examples]
    # Save the rest of the unused article pairs for future use (if needed)
    remaining_sims = sims[n_examples:]
    f = open('test_examples_articles.txt', 'w')
    for x in remaining_sims:
        f.write(f"{x}\n")

    # Construct the base model
    word_embedding_model = models.Transformer('distilroberta-base')
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Convert all of the article pairs into input examples
    train_examples = []
    for i in training_sims:
        train_examples.append(InputExample(texts=[i[0],i[1]]))
    # Load the article pairs into a dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    # Set the loss function for the model training
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Fine-tune the model
    num_epochs = 1
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=100)
    model.save('saved_model_articles')

if __name__=="__main__":
    run()