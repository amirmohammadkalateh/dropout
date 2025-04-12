Dropout
Dropout is a regularization technique used in neural networks to prevent overfitting. During training, dropout randomly deactivates a fraction of neurons in a layer by setting their outputs to zero. This process has two key effects:

Prevents Co-adaptation: By randomly dropping neurons, dropout forces the network to learn more robust features that are not dependent on the presence of specific other neurons.

Ensembles Sub-networks: Each dropout application effectively trains a different sub-network architecture. At test time, all neurons are active, and their outputs are scaled to approximate the average prediction of these sub-networks.

Alpha Dropout
Alpha Dropout is a variant of dropout designed specifically for deep neural networks using self-normalizing activation functions like SELU.  It addresses a limitation of standard dropout, which can disrupt the mean and variance of activations, potentially hindering training in deep models.

Key differences:

Standard Dropout: Randomly sets neuron outputs to zero, which can alter the statistical properties of the activations.

Alpha Dropout: Instead of simply setting activations to zero, Alpha Dropout scales and shifts the remaining activations to preserve the original mean and variance of the input. This helps maintain the self-normalizing property of certain network architectures, preventing issues like vanishing gradients in very deep networks.

In essence, Alpha Dropout is a more advanced form of dropout that is better suited for deep neural networks that employ self-normalizing activation functions.
