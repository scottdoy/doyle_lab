# Fix to turn off matplotlib errors on Mac OSX
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np

import os, time, argparse, shutil
from datetime import timedelta
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import scipy.misc

# Import the model class we're going to train
from models.alexnet import AlexNet

    
# Main function =================================================================

def main():
    '''
    Based on what you pass as arguments, appropriate model actions (training,
    testing, sampling, etc.) are performed.

    'main()' is just an argument parser which then calls the appropriate functions
    defined below to perform various tasks. There should be at least as many
    functions as there are parsers (unless some functions are overloaded or
    something).

    For example, passing 'train' to the script will call the 'train' function
    below, which interacts with the 'train' method of the model. Similarly, 'test'
    will call the test function, which in turn interacts with the ModelSession
    method 'restore' (which is inherited by the model), allowing you to load up
    previously-trained models as defined in the '--model-directory' argument. (We
    may move over to using tf.FLAGS later on if I can be convinced that they are
    better.)
    '''
    
    # Arguments
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="AlexNet Model")
    parser.set_defaults(func=lambda _: parser.print_usage())
    
    # Shared arguments ----------------------------------------------------------
    shared_arguments = argparse.ArgumentParser(add_help=False)
    shared_arguments.add_argument("--datadir", default="data", help="data files directory, default: ./data/")
    shared_arguments.add_argument("--model-directory", default="checkpoint", help="Checkpoint directory, default: ./checkpoint/")
    
    # Training -----------------------------------------------------------------
    train_parser = subparsers.add_parser("train", parents=[shared_arguments],
                                         description="Train an AlexNet Model", help="train model")
    train_parser.add_argument("--training-epochs", type=int, default=75, help="Number of training epochs, default 75")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Training batch size, default 128")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate, default 1e-3")
    train_parser.add_argument("--report-interval", type=int, default=50, help="How often to report training batch accuracy, default 50 iterations")
    train_parser.add_argument("--validation-interval", type=int, default=100, help="How often to run validation, default 100 iterations")
    train_parser.add_argument("--checkpoint-interval", type=int, default=500, help="How often to save checkpoint model, default 500 iterations")
    train_parser.set_defaults(func=train)
    
    # Testing ------------------------------------------------------------------
    test_parser = subparsers.add_parser("test", parents=[shared_arguments], 
                                        description="Test the AlexNet classifier", help="test model")
    test_parser.add_argument("--batch-size", type=int, default=128, help="Testing batch size, default 128 (same as training)")
    test_parser.set_defaults(func=test)
    
    # Compile the arguments and execute the indicated function
    args = parser.parse_args()
    args.func(args)

# Model Actions =================================================================
# 

def train(args):
    ''' Train an AlexNet model. '''
    # Load data -- separate vars for training and validation
    training_data = input_data.read_data_sets(args.datadir, one_hot=True).train
    validation_data = input_data.read_data_sets(args.datadir, one_hot=True).validation

    # Restore model (if the path exists) or instantiate a new one
    if os.path.exists(args.model_directory):
        # Since we're training, delete the old checkpoint dir
        #model = AlexNet.restore(args.model_directory)
        shutil.rmtree(args.model_directory)
        
    os.makedirs(args.model_directory)
    model = AlexNet.create(batch_size=args.batch_size)
        
    # Create the summary of the variables that we're going to print to the log for
    # TensorBoard
    summ = tf.summary.merge_all()
    
    # Create the logfile directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
       
    # Construct the writer for the logs and add the graph to it
    writer = tf.summary.FileWriter('logs')
    writer.add_graph(model.session.graph)
    
    # Training Cycle
    total_batch = training_data.num_examples // args.batch_size
    iter = 1
    for epoch in range(args.training_epochs):
        
        # Iterate through this batch
        for i in range(total_batch):
            x, y = training_data.next_batch(args.batch_size)
            x = x.reshape([-1, 28, 28, 1])
        
            # Run one iteration of training
            [train_step, iteration] = model.train(x, y, args.learning_rate)
            
            # Write the result of this iteration to the summary logfile
            # (Needs the same arguments as the training function)
            s = model.session.run(summ, feed_dict={model.x: x, 
                                                   model.y: y, 
                                                   model.learning_rate: args.learning_rate})
            writer.add_summary(s, iter)
            
            # Update the iterator
            iter += 1

            # Report!
            if iteration % args.report_interval == 0:
                training_loss = model.test(x,y)
                print('{}: training batch loss {}'.format(model,
                                                          training_loss))

            if iteration % args.checkpoint_interval == 0:
                model.save(args.model_directory)

    # Save the finished model after all epochs have run
    model.save(args.model_directory)
    print('Final model {}'.format(model))

def test(args):
    ''' Function to test the reconstruction loss of the VAE classifier '''
    
    # Grab some data to test the classifier (NOT data it's currently using for
    # training or validation)
    test_data = input_data.read_data_sets(args.mnist, one_hot=True).test
    test_batch_x, test_batch_labels = test_data.next_batch(args.batch_size)
    test_batch_x = text_batch_x.reshape([-1, 28, 28, 1])
    
    # Restore the previously-trained model (will throw an error if not already trained)
    model = AlexNet.restore(args.model_directory)
    
    # Calculate the cost and print the result
    cost = model.test(test_batch_x, test_batch_labels)[0]
    print('Test cost {:0.4f}'.format(cost))

if __name__ == '__main__':
    main()
