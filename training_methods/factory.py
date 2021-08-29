import training_methods.gan as gan
import training_methods.gan_with_posterior_test as gan_test_posterior
import training_methods.mse as mse

factory = {
    'gan': gan.GAN,
    'gan_test_posterior': gan_test_posterior.GANWithPosteriorTest,
    'mse': mse.MSE
}