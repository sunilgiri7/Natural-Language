import numpy as np
import math

L, d_k, d_v = 4, 8, 8
q = np.random.randn(L, d_k)
k = np.random.randn(L, d_k)
v = np.random.randn(L, d_v)

# print("Q\n", q)

# Self Attention
# print(np.matmul(q, k.T))

# this is why we need to sqrt(d_k) in denominator
# print(q.var(), k.var(), np.matmul(q, k.T).var())

scaled = np.matmul(q, k.T) / math.sqrt(d_k)
# print(q.var(), k.var(), scaled.var())

# Notice reduction in variance of the product
# print(scaled)

# Masking
# This is to ensure words don't get context from words generated in the future.
# Not required in the encoders, but required int he decoders

mask = np.tril(np.ones((L,L)))
# print(mask)

mask[mask == 0] = -np.infty
mask[mask == 1] = 0
# print(mask)

# print(scaled + mask)

def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=1)).T

attention = softmax(scaled + mask)
# print(attention)

new_v = np.matmul(attention, v)
# print(new_v)

# Wrapping in function
def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=1)).T
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scaled = np.matmul(q, k.T) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled + mask
    attention = softmax(scaled)
    out = np.matmul(attention, v)
    return out, attention

value, attention = scaled_dot_product_attention(q, k, v, mask)
print("value \n", value)
print("attention \n", attention)