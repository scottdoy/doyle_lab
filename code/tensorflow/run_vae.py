# ===============================================================================
# Following the architecture from:
# https://danijar.github.io/structuring-your-tensorflow-models
#
# This code is meant to be an example for loading and running models located in
# the 'models' subdirectory. It will also have some explanatory text to help
# newbies learn a bit about Tensorflow, although other tutorials are better
# suited for a general introduction (see for example here:
# https://github.com/Hvass-Labs/TensorFlow-Tutorials).
#
# Copy this function and use it to train / test your own models, modifying the
# overall structure as needed. Feel free to delete the tutorial comments if you
# like.
# ===============================================================================

# Imports =======================================================================

# Fix to turn off matplotlib errors on Mac OSX
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np

import os, time, argparse
from datetime import timedelta
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import scipy.misc

# Import the model class we're going to train
from models.vae import VAEModelSession

    
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
    subparsers = parser.add_subparsers(title="MNIST VAE Model")
    parser.set_defaults(func=lambda _: parser.print_usage())
    
    # Shared arguments ----------------------------------------------------------
    shared_arguments = argparse.ArgumentParser(add_help=False)
    shared_arguments.add_argument("--mnist", default="mnist.data", help="MNIST files directory, default mnist.data")
    shared_arguments.add_argument("--model-directory", metavar="model-directory", default="checkpoint",
                                  help="Model directory, default checkpoint")
    
    # Training -----------------------------------------------------------------
    train_parser = subparsers.add_parser("train", parents=[shared_arguments],
                                         description="Train an MNIST VAE Model", help="train model")
    train_parser.add_argument("--encoder-nodes", type=int, default=500, help="Hidden nodes in the encoder network, default 500")
    train_parser.add_argument("--decoder-nodes", type=int, default=500, help="Hidden nodes in the decoder network, default 500")
    train_parser.add_argument("--latent-dims", type=int, default=20, help="Number of latent dimensions to encode, default 20")
    train_parser.add_argument("--training-examples", type=int, help="Number of training examples, default one epoch")
    train_parser.add_argument("--training-epochs", type=int, default=75, help="Number of training epochs, default 75")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Training batch size, default 64")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate of VAE, default 1e-3")
    train_parser.add_argument("--report-interval", type=int, default=50, help="How often to report training batch accuracy, default 50 iterations")
    train_parser.add_argument("--validation-interval", type=int, default=100, help="How often to run validation, default 100 iterations")
    train_parser.add_argument("--checkpoint-interval", type=int, default=500, help="How often to save checkpoint model, default 500 iterations")
    train_parser.set_defaults(func=train)
    
    # Testing ------------------------------------------------------------------
    test_parser = subparsers.add_parser("test", parents=[shared_arguments], 
                                        description="Test an MNIST VAE classifier", help="test model")
    test_parser.add_argument("--batch-size", type=int, default=64, help="Testing batch size, default 64 (same as training)")
    test_parser.set_defaults(func=test)
    
    # Additional Parsers --------------------------------------------------------
    #
    # Example: Here we have:
    # 1. Plotting the latent space of a variational autoencoder.
    # 2. Reconstructing a grid of results sampled from the latent space
    # 3. Reconstructing a single input from the latent space.
    
    plot_latent_parser = subparsers.add_parser("plot_latent", parents=[shared_arguments],
                                            description="Reconstruct the latent space", help="reconstruct samples")
    plot_latent_parser.add_argument("--batch-size", type=int, default=5000, help="Testing batch size, default 64 (same as training)")
    plot_latent_parser.add_argument("--save-dir", metavar="save-dir", default="checkpoint", help="Dir to save the reconstructions")
    plot_latent_parser.set_defaults(func=plot_latent_space)
    
    reconstr_grid_parser = subparsers.add_parser("reconstr_grid", parents=[shared_arguments],
                                            description="Reconstruct a grid of MNIST digits from the latent space", help="reconstruct grid of samples")
    reconstr_grid_parser.add_argument("--batch-size", type=int, default=64, help="Testing batch size, default 64 (same as training)")
    reconstr_grid_parser.add_argument("--save-dir", metavar="save-dir", default="checkpoint", help="Dir to save the reconstructions")
    reconstr_grid_parser.set_defaults(func=reconstruct_grid)
    
    reconstr_parser = subparsers.add_parser("reconstr", parents=[shared_arguments],
                                            description="Reconstruct an MNIST digit from the latent space", help="reconstruct samples")
    reconstr_parser.add_argument("--batch-size", type=int, default=64, help="Testing batch size, default 64 (same as training)")
    reconstr_parser.add_argument("--save-dir", metavar="save-dir", default="checkpoint", help="Dir to save the reconstructions")
    reconstr_parser.set_defaults(func=reconstruct)
    
    # Compile the arguments and execute the indicated function
    args = parser.parse_args()
    args.func(args)

# Model Actions =================================================================
# 

def train(args):
    ''' Train a VAE model. '''
    # Load data -- separate vars for training and validation
    training_data = input_data.read_data_sets(args.mnist, one_hot=True).train
    validation_data = input_data.read_data_sets(args.mnist, one_hot=True).validation

    # Restore model (if the path exists) or instantiate a new one
    if os.path.exists(args.model_directory):
        model = VAEModelSession.restore(args.model_directory)
    else:
        os.makedirs(args.model_directory)
        model = VAEModelSession.create(latent_dims=args.latent_dims)
        
    # Create the summary of the variables that we're going to print to the log for
    # TensorBoard
    summ = tf.summary.merge_all()
    
    # Create the logfile directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
       
    # Construct the writer for the logs and add the graph to it
    writer = tf.summary.FileWriter('logs')
    writer.add_graph(model.session.graph)
    
    # How many training samples to use (default: all)
    if args.training_examples is None:
        args.training_examples = training_data.num_examples
    
    # Training Cycle
    total_batch = args.training_examples // args.batch_size
    iter = 1
    for epoch in range(args.training_epochs):
        
        # Iterate through this batch
        for i in range(total_batch):
            x, y = training_data.next_batch(args.batch_size)
        
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
                training_batch_cost, training_reconstr_loss, training_latent_loss = model.test(x,y)
                print('{}: training batch cost {}'.format(model, 
                                                                training_batch_cost))

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
    
    # Restore the previously-trained model (will throw an error if not already trained)
    model = VAEModelSession.restore(args.model_directory)
    
    # Calculate the cost and print the result
    cost = model.test(test_batch_x, test_batch_labels)[0]
    print('Test cost {:0.4f}'.format(cost))
    
def plot_latent_space(args):
    ''' Create a scatterplot of the latent space '''
    
    # Load the previously-trained model 
    if os.path.exists(args.model_directory):
        model = VAEModelSession.restore(args.model_directory)
    else:
        print('Need a model saved at: {}'.format(args.model_directory))

    # Grab a 'batch_size'-sized batch of validation data
    x_sample, y_sample = input_data.read_data_sets(args.mnist, one_hot=True).validation.next_batch(args.batch_size)
    
    # Get a representation of the latent space by calculating the mean of the
    # latent vectors 'z' for the input batch
    z_mu = model.transform(x_sample)
    
    # Plot and save the latent space (scatter)
    # Note that this will plot the first and second latent dims ONLY
    # This will still work, but will look weird, if your trained model has > 2 dims
    plt.figure(figsize=(8,6))
    plt.scatter(z_mu[:,0], z_mu[:,1], c=np.argmax(y_sample,1))
    plt.colorbar()
    plt.grid()
    plt.savefig('VAE Latent Space pts.png')
        
def reconstruct(args):
    ''' Compute the reconstruction for a batch of samples selected from the testing set '''
    
    # Load the model
    if os.path.exists(args.model_directory):
        model = VAEModelSession.restore(args.model_directory)
    else:
        print('Need a model saved at: {}'.format(args.model_directory))
        
    # Grab a batch of validation data to reconstruct
    x_sample = input_data.read_data_sets(args.mnist, one_hot=True).validation.next_batch(64)[0]
    
    # Perform the reconstruction on the sample
    x_reconstruct = model.reconstruct(x_sample) 
    
    # Plot the reconstruction
    for i in range(5):
        plt.subplot(5, 2, 2*i+1)
        plt.imshow(x_sample[i].reshape(28,28), vmin=0, vmax=1, cmap='gray')
        plt.title('Test Input')
        plt.colorbar()
        plt.subplot(5, 2, 2*i+2)
        plt.imshow(x_reconstruct[i].reshape(28,28), vmin=0, vmax=1, cmap='gray')
        plt.title('Reconstruction')
        plt.colorbar()
    plt.tight_layout()
    plt.savefig('test.png')
    
def reconstruct_grid(args):
    ''' Create a reconstruction for a grid of points in the latent space '''
    
    # Load the model 
    if os.path.exists(args.model_directory):
        model = VAEModelSession.restore(args.model_directory)
    else:
        print('Need a 2D model saved at: {}'.format(args.model_directory))
     
    # VERY hacky way to get the size of the latent space -- should refactor
    # the model definition to set this as an attribute of the model
    tmp_data = np.random.rand(1,28*28)
    tmp_output = model.transform(tmp_data)
    n_latent = np.shape(tmp_output)[1]
    
    assert n_latent == 2, 'Need a 2D latent space! Re-train model with latent_dims = 2'
    
    # Construct the grid we're going to sample from Because of the nature of the
    # latent space, we expect the grid to be centered on 0 Adding points to nx,
    # ny will create a larger grid (more samples), whereas increasing the limits
    # will select from areas where there is no good latent space representation
    gridspacing = 30
    x_values = np.linspace(-3, 3, gridspacing)
    y_values = np.linspace(-3, 3, gridspacing)
    
    # Construct a canvas into which we will drop our reconstructions
    canvas = np.empty((28*gridspacing, 28*gridspacing))
    
    # Cycle through the grid
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            
            # For each pair of points on the grid, create an input z_mu at that
            # point, and calculate the reconstruction
            z_mu = np.array([[xi, yi]]*args.batch_size)
            x_mean = model.generate(z_mu)
      
            # The reconstruction is the output of the decoder step; resize this
            # to the size of the inputs and "paint" it into the canvas matrix
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28,28)

    # Plot the canvas as an image and save
    plt.figure(figsize=(8, 10))
    #Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    plt.savefig('VAE MNIST Grid.png')

if __name__ == '__main__':
    main()
