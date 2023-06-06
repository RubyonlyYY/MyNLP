'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

#### Flair for few-shot learning ####
### Import modules
from flair.data import Corpus, Sentence
from flair.datasets import SentenceDataset, ClassificationCorpus
from flair.trainers import ModelTrainer
#from flair.models.text_classification_model import TARSClassifier
from flair.models import TARSClassifier

def train() -> None:
    ### Helper function
    def convert_to_sentence_data(sentences, labels, task_name):
        res = SentenceDataset([
            Sentence(s).add_label(task_name, labels[i]) for i, s in enumerate(sentences)
        ])
        return res


    ###### Body
    ## 0. convert example data

    # training sentences
    train_sent = ['I give my team authority over issues within the department.',
                  'I am concerned that my team reach their goal.', 'I recognize my team strong and weak sides.',
                  'I focus on developing human capital (the individual capabilities, knowledge, skills and experience of a firm\'s employees).',
                  'I develop or help develop standard operating procedures and standardized processes.','I clarify task performance strategies.',
                  'I act as a representative of the team with other parts of the organization (e.g., other teams, management).',
                  'I challenge my team to think about old problems in new ways.','I ensure that the team has a clear understanding of its purpose.',
                  'I help provide a clear vision of where the team is going.','I communicate expectations for high team performance.',
                  'I define and structures my own work and the work of the team.','I go beyond my own interests for the good of the team.',
                  'I ask questions that prompt my employees to think.','I have stimulated my employees to rethink the way they do things.',
                  'I treat my employees by considering their personal feelings.']

    # each training example has just one label with 5 unique labels in total
    train_labels = ['autonomy and empowerment', 'autonomy and empowerment', 'autonomy and empowerment', 'creativity',
                    'task management', 'task management', 'external representation and boundary management', 'creativity',
                    'task management', 'task management', 'task management', 'task management', 'personal care and support',
                    'creativity', 'creativity', 'personal care and support']

    # use helper to convert to SentenceDataSet
    train_data = convert_to_sentence_data(train_sent, train_labels, "behaviors")

    ## 1. make corpus
    crpus = Corpus(train=train_data)


    ### Run predictions
    ## 2. load TARS
    label_dict=crpus.make_label_dictionary(label_type='behaviors')

    tars = TARSClassifier(task_name="behaviors",
                          embeddings="bert-base-uncased",
                          #document_embeddings="paraphrase-mpnet-base-v2",
                          multi_label=True)

    # 5a: alternatively, comment out previous line and comment in next line to train a new TARS model from scratch instead
    # tars = TARSClassifier(embeddings="bert-base-uncased")
    # 6. switch to a new task (TARS can do multiple tasks so you must define one)
    tars.add_and_switch_to_new_task(task_name="behaviors",
                                    label_dictionary=label_dict,
                                    label_type='behaviors',
                                    )
    '''
    #1. make corpus
    crpus = Corpus(train=train_data)
    label_dict = crpus.make_label_dictionary()
    label_dict.multi_label = True
    
    #Run predictions
    #2. load TARS
    tars = TARSClassifier(task_name = "behaviors",
                        label_dictionary = label_dict,
                        document_embeddings = "paraphrase-mpnet-base-v2",
                        batch_size = 16)
    '''
    ## 3. initialize the text classifier trainer
    trainer = ModelTrainer(tars, crpus)

    ## 4. train model
    trainer.train(base_path="./example/dir/",
                  learning_rate=5.0e-5,
                  mini_batch_size=8,
                  max_epochs=10,
                  embeddings_storage_mode="cpu")

def predict():
    ## 5. load model
    ft_tars = TARSClassifier.load('./example/dir/best-model.pt')
    label_set = ['autonomy and empowerment', 'creativity','task management','external representation and boundary management',
                    'personal care and support']
    ## 6. Predict
    test_data = [
        'i had a conversation  with each receptionist regarding being more efficient in their recording of products in and out',
        'get requests to increase or decrease budgets on projects and to close out projects as they are finished',
        'marketing sales', 'delegated tasks for closing',
        'i smiled at her and let her know what i would be finished with the reports within 15 minutes',
        'i wanted to make the client comfortable  and i alerted everyone to be professional and be on time',
        'we made sure we had enough expertise and knowledge to show the client that we were the best for business']

    test_sentences : list = [Sentence(s) for s in test_data]
    for s in test_sentences:
        #ft_tars.predict(s)
        ft_tars.predict_zero_shot(s,label_set)
        print(s,s.labels)

if __name__ == '__main__':
    train()
    predict()