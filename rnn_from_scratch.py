# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:46:55 2017

@author: anand
"""

import numpy as np
import sys
from datetime import datetime
from get_penn_data import function_get_data


vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "START"
sentence_end_token = "END"
batch_size = 128

class RNNNumpy:
    
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        
def forward_propagation(self, x):
    # The total number of time steps
    T = len(x)
    # During forward propagation we save all hidden states in s because need them later.
    # We add one additional element for the initial hidden, which we set to 0
    s = np.zeros((T + 1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)
    # The outputs at each time step. Again, we save them for later.
    o = np.zeros((T, self.word_dim))
    # For each time step...
    for t in np.arange(T):
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
        o[t] = softmax(self.V.dot(s[t]))
    return [o, s]

def predict(self, x):
    # Perform forward propagation and return index of the highest score
    o, s = self.forward_propagation(x)
    return np.argmax(o, axis=1)

def calculate_total_loss(self, x, y):
    L = 0
    # For each sentence...
    for i in np.arange(len(y)):
        o, s = self.forward_propagation(x[i])
        # We only care about our prediction of the "correct" words
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        # Add to the loss based on how off we were
        L += -1 * np.sum(np.log(correct_word_predictions))
    return L

def calculate_loss(self, x, y):
    # Divide the total loss by the number of training examples
    N = np.sum((len(y_i) for y_i in y))
    return self.calculate_total_loss(x,y)/N

def bptt(self, x, y):
    T = len(y)
    # Perform forward propagation
    o, s = self.forward_propagation(x)
    # We accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.
    # For each output backwards...
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)
        # Initial delta calculation
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        # Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            dLdW += np.outer(delta_t, s[bptt_step-1])              
            dLdU[:,x[bptt_step]] += delta_t
            # Update delta for next step
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
    return [dLdU, dLdV, dLdW]

# Performs one step of SGD.
def numpy_sdg_step(self, x, y, learning_rate):
    # Calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y)
    # Change parameters according to gradients and learning rate
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW
    
# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, i,batch_size,learning_rate=0.005, nepoch=100, evaluate_loss_after=1):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = batch_size*i
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)
    
def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    count = 0
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            #samples = np.random.multinomial(1, next_word_probs[-1])
            o = next_word_probs[0]
            s = next_word_probs[1]
            temp1 = s[0]
            temp2 = s[1]
            samples = np.random.multinomial(1,temp1)    
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
        count+=1
        if(count>100):
            break
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str
    
def main():
    index_to_word,word_to_index,tokenized_sentences = function_get_data()

    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    
    
    RNNNumpy.forward_propagation = forward_propagation
    RNNNumpy.predict = predict
    RNNNumpy.calculate_total_loss = calculate_total_loss
    RNNNumpy.calculate_loss = calculate_loss
    RNNNumpy.bptt = bptt
    RNNNumpy.sgd_step = numpy_sdg_step    
    
    np.random.seed(10)    
    model = RNNNumpy(vocabulary_size) 
    
    # Print an training data example
    x_example, y_example = X_train[10], y_train[10]
    print tokenized_sentences[10]
    print "x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example)
    print "\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example)
    o, s = model.forward_propagation(X_train[10])
    print o.shape
    print o
    
    predictions = model.predict(X_train[10])
    print predictions.shape
    print predictions
    
    print "Expected Loss for random predictions: %f" % np.log(vocabulary_size)
    print "Intial losss on Untrained model: %f" % model.calculate_loss(X_train[:1000], y_train[:1000])
    
    num_of_batches = int(len(X_train)/batch_size)
    for i in range(0,num_of_batches+1,1):
        print "Batch=",i
        next_x = X_train[batch_size*i:min((i+1)*batch_size,len(X_train))]
        next_y = y_train[batch_size*i:min((i+1)*batch_size,len(y_train))]
        train_with_sgd(model, next_x,next_y,i,batch_size,nepoch=10, evaluate_loss_after=1)
    
    print "Loss after training the RNN model: %f" % model.calculate_loss(X_train, y_train)    
    
    #Generating new sentences
    num_sentences = 10
    senten_min_length = 1
    
    for i in range(num_sentences):
        sent = []
        # We want long sentences, not sentences with one or two words
        while len(sent) < senten_min_length:
            sent = generate_sentence(model)
        print " ".join(sent)


if __name__ == '__main__':
    main()





