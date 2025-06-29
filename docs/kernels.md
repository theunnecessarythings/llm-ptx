The kernels in this project are written in both PTX and CUDA. This document provides a high-level overview of the kernels and their purpose.

- **`attention_kernel.ptx`**: This kernel performs the attention mechanism, which is a key component of the transformer architecture. It calculates the attention scores between the query, key, and value vectors.
- **`gelu_kernel.ptx`**: This kernel applies the GELU (Gaussian Error Linear Unit) activation function, which is used in the feed-forward network.
- **`layernorm_kernel.ptx`**: This kernel applies layer normalization to the hidden states. This helps to stabilize the training process and improve the performance of the model.
- **`matmul_kernel.ptx`**: This kernel performs matrix multiplication, which is a fundamental operation in deep learning.
- **`residual_kernel.ptx`**: This kernel adds the input of a sub-layer to its output. This is known as a residual connection and it helps to prevent the vanishing gradient problem.
- **`softmax_kernel.ptx`**: This kernel applies the softmax function to the output of the attention mechanism. This converts the attention scores into a probability distribution.
