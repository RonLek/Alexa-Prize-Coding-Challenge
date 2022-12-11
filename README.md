# Alexa Prize Coding Challenge

The repository is a submission for the Alexa Prize Coding Challenge. Description of the challenge can be found [here](https://rentry.co/alexa-prize-coding-test).

```
Submitter: Rohan Lekhwani
Challenge Received: December 8, 2022
Challenge Completed: December 11, 2022
```

## Table of Contents

- [Alexa Prize Coding Challenge](#alexa-prize-coding-challenge)
  - [Table of Contents](#table-of-contents)
  - [How to Run](#how-to-run)
  - [Dataset](#dataset)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
    - [Hyperparameters](#hyperparameters)
    - [Metrics](#metrics)
  - [Deployment](#deployment)
    - [Creating an AMI](#creating-an-ami)
    - [Load Balancing, Target Groups and Auto Scaling Groups](#load-balancing-target-groups-and-auto-scaling-groups)
  - [Development](#development)
  - [Possible Improvements](#possible-improvements)
    - [Model Improvements](#model-improvements)
    - [Deployment Improvements](#deployment-improvements)
  - [References](#references)

## How to Run

The URL for the deployed model is: `http://apcc-asg-1-1665300744.us-east-1.elb.amazonaws.com/predict`. Make a `POST` request with the text to be classified within the body.

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "Will I be making my own pasta for this recipe or using the packaged store bought kind?"}' http://apcc-asg-1-1665300744.us-east-1.elb.amazonaws.com/predict
# {"intent":"ask_question_ingredients_tools","score":"0.62276465"}
```

## Dataset

The [Wizards of Tasks](https://registry.opendata.aws/wizard-of-tasks/) dataset has been used to train an intent classification model. The dataset comprises of two domains: `cooking` and `diy`. During training we only make use of the `text` and `intent` attributes of the samples. The `cooking` and `diy` domains have been combined to create a join dataset.

There are a total of 11 labels combined across the two domains:

```python
labels = ['answer_question_external_fact', 'answer_question_recipe_steps',
       'ask_question_ingredients_tools', 'ask_question_recipe_steps',
       'ask_student_question', 'chitchat', 'misc', 'request_next_step',
       'return_list_ingredients_tools', 'return_next_step', 'stop']
```

## Data Preprocessing

The splits into train, test and validation have been done in accordance with the split `data_split` label within the dataset.

A few samples have `null` values for `text` or `intent` these have been removed from the split sets. The JSON data is converted to a Pandas dataframe for easier processing.

## Model Training

A pre-trained [BERT](https://arxiv.org/abs/1810.04805) model has been used to train on the Wizards of Tasks dataset. [Tensorflow Hub](https://tfhub.dev/google/collections/bert/1) was used to obtain a pre-trained BERT model. More specifically, the deployment makes use of the Small BERT model `small_bert/bert_en_uncased_L-2_H-256_A-4`. This model has 2 hidden layers (transformer blocks), a hidden size of 256 and 4 attention heads. More information about this model can be found [here](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/2).

To transform the text inputs numeric token ids and to arrange them in Tensors before being input to BERT we make use of pre-processing models. TensorFlow Hub provides a matching preprocessing model for each of the BERT models, which implements this transformation using TF ops from the TF.text library.

A customized model has been built upon the pre-trained Small BERT model. It consists of the preprocessing layer, the BERT encoder, dropout layer and a dense layer to predict the classification output. The model summary can be seen below:

```python
Model: "model_2"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 text (InputLayer)              [(None,)]            0           []                               
                                                                                                  
 preprocessing (KerasLayer)     {'input_type_ids':   0           ['text[0][0]']                   
                                (None, 128),                                                      
                                 'input_word_ids':                                                
                                (None, 128),                                                      
                                 'input_mask': (Non                                               
                                e, 128)}                                                          
                                                                                                  
 BERT_encoder (KerasLayer)      {'encoder_outputs':  9591041     ['preprocessing[0][0]',          
                                 [(None, 128, 256),               'preprocessing[0][1]',          
                                 (None, 128, 256)],               'preprocessing[0][2]']          
                                 'sequence_output':                                               
                                 (None, 128, 256),                                                
                                 'pooled_output': (                                               
                                None, 256),                                                       
                                 'default': (None,                                                
                                256)}                                                             
                                                                                                  
 dropout_2 (Dropout)            (None, 256)          0           ['BERT_encoder[0][3]']           
                                                                                                  
 classifier (Dense)             (None, 11)           2827        ['dropout_2[0][0]']              
                                                                                                  
==================================================================================================
Total params: 9,593,868
Trainable params: 9,593,867
Non-trainable params: 1
```

### Hyperparameters

The CategoricalCrossentropy loss is used since the goal is a non-binary classification into intents. Adam optimizer with a learning rate of `1e-5` has been used while training. The model is trained in batches of 32 over 5 epochs.

### Metrics

After the final epoch the following are the metrics obtained with the current model

```python
training_loss: 0.7695
categorical_accuracy: 0.7319
val_loss: 0.6979
val_categorical_accuracy: 0.7587
test_loss: 0.7573978900909424
test_accuracy: 0.7222222089767456
```

## Deployment

### Creating an AMI

The model obtained is saved to the Drive in an `.hd5` format. `server.py` is responsible for loading this model and starting a flask server on `PORT=5000`.

A `t2.micro` EC2 instance is started with 1 vCPU and 1GiB memory. The repository is cloned and all dependencies are installed through the `requirements.txt` file within the instance.

```bash
pip install -r requirements.txt
```

A systemd service `apcc.service` is created to run the flask server within `/etc/systemd/system`.

```bash
[Unit]
Description=APCC Service
After=network.target

[Service]
User=root
WorkingDirectory=/home/ec2-user/apcc
ExecStart=/bin/python3 server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

The service is started to verify correct deployment.

```bash
sudo systemctl start apcc.service
```

An Amazon Machine Image (AMI) of the instance is created. Doing so enables spinning up ready-to-serve VMs within an autoscaler.

### Load Balancing, Target Groups and Auto Scaling Groups

A launch tempalte using the AMI created above is created to templatize new instance creations. An Auto Scaling Group is provision which makes use of this launch template. The following are capacity limits of the ASG in the current deployment.

```txt
Desired Capacity: 2
Minimum Capacity: 1
Maximum Capacity: 3
```

With the above limits, two instances are always guaranteed to be running the application in two different zones which ensures scalability.

A target group is created for port `5000` for the instances. The Internet-facing Application Load Balancer created next forwards its HTTP requests on `PORT=80` to this target group.

The following is the architecture diagram of the deployment.

![architecture](./architecture.png)

## Development

Clone the repo and `cd` into the `apcc` directory.

```bash
git clone git@github.com:RonLek/Alexa-Prize-Coding-Challenge.git
cd apcc
```

Download a model to deploy from the [Google Drive](https://drive.google.com/drive/folders/17oGnAMfBkuxeXM_vI4qb3a9DZr2yPuqN?usp=share_link) and place it in the current directory. Make sure to change the name of the model to load in `server.py` correspondingly.

Install the dependencies using `pip` and start the server. If you are low on memory make use of the `--no-cache-dir` flag.

```bash
sudo pip3 install -r requirements.txt --no-cache-dir
sudo python3 server.py
```

> Note: `sudo` is required when serving on `PORT=80`.

The server should be up and running on `PORT=80`. You can test the server by sending a POST request to the server with the following command.

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "Will I be making my own pasta for this recipe or using the packaged store bought kind?"}' http://localhost:80/predict
```

## Possible Improvements

The following are some of the possible improvements that can be made to the model and the deployment strategy.

### Model Improvements

- From analysis it is seen that the model accuracy is not very high. This is due to the fact that the model size had to be constrained (larger models take up more memory to load) owing to the 1vCPU and 1GiB memory limit of the free tier EC2 instance. Larger models (like the uncased BERT, and 8 layer Small BERT models) were trained and the accuracy was seen to improve. These models can be found within the [Google Drive](https://drive.google.com/drive/folders/17oGnAMfBkuxeXM_vI4qb3a9DZr2yPuqN?usp=share_link).

- The model accuracy is seen to improve linearly with the number of epochs. However owing to time constraints the model was trained for only 5 epochs. The model can be trained for more epochs to improve the accuracy.

- The current model is a pre-trained BERT model with additional Dense and Dropout layers. A better model is bound to give better results on the dataset.

### Deployment Improvements

- The entire deployment can be containerized on EKS. Helm charts can be used to start a Kubernetes cluster and deploy the application on it. $0.10 per hour is charged for the EKS cluster, however since the challenge required using only free resources this was not done.

- AWS SageMaker could be optionally use to train, test and validate the model along with deploying it for API consumption. SageMaker also handles scaling the resources dedicated to the model to handle increased traffic.

- Currently instances are started in two different zones. This can be extended to all 6 zones to ensure high availability.

- Although healthchecks are in place. Observability and monitoring alerts can be set up to ensure that the application is running smoothly.

## References

The following papers and resources were used to complete the project.

- Wizard of Tasks: A Novel Conversational Dataset for Solving Real-World Tasks in Conversational Settings by Jason Ingyu Choi, Saar Kuzi, Nikhita Vedula, Jie Zhao, Giuseppe Castellucci, Marcus Collins, Shervin Malmasi, Oleg Rokhlenko and Eugene Agichtein

- Devlin, J., Chang, M.W., Lee, K. and Toutanova, K., 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

- AWS Documentation
