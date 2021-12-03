import poptorch
import torch
from torch import nn


class SerializedEmbedding(nn.Module):
    """
    Wrapper for `nn.Embedding` layer that performs the embedding look-up into
    smaller serialized steps in order to reduce memory in the embedding gradient
    calculation.

    Args:
        embedding: A `nn.Embedding` to wrap
        serialization_factor: The number of serialized embedding look-ups
    """

    def __init__(self, embedding: nn.Embedding, serialization_factor: int):
        super().__init__()
        self.serialization_factor = serialization_factor
        self.num_embeddings = embedding.num_embeddings

        # Num embeddings should be divisible by the serialization factor
        assert self.num_embeddings % self.serialization_factor == 0, \
            f'num_embeddings (={self.num_embeddings}) is not divisible by the serialization factor (={self.serialization_factor})'
        self.split_size = self.num_embeddings // self.serialization_factor
        self.split_embeddings = nn.ModuleList(
            [nn.Embedding.from_pretrained(embedding.weight[i * self.split_size:(i + 1) * self.split_size, :].detach(),
                                          freeze=False, padding_idx=embedding.padding_idx if i == 0 else None)
             for i in range(self.serialization_factor)])

    def deserialize(self):
        """
        Deserialize the internal wrapped embedding layer and return it as a
        `nn.Embedding` object.

        Returns:
            `nn.Embedding` layer
        """
        return nn.Embedding.from_pretrained(torch.cat([l.weight for l in self.split_embeddings]), padding_idx=0)

    def forward(self, indices):
        # iterate through the splits
        x_sum = None
        for i in range(self.serialization_factor):
            # mask out the indices not in this split
            split_indices = indices.to(torch.int) - i * self.split_size
            mask = (split_indices >= 0) * (split_indices < self.split_size)
            mask = mask.detach()
            split_indices *= mask.to(torch.int)

            # do the embedding lookup
            x = self.split_embeddings[i](split_indices)

            # multiply the output by mask
            x *= mask.unsqueeze(-1)

            # add to partial
            if x_sum is not None:
                x_sum += x
            else:
                x_sum = x
        return x_sum


class SerializedLinear(nn.Linear):
    """
        Exactly equivalent to `nn.Linear` layer, but with the matrix multiplication replaced with
        a serialized matrix multiplication: `poptorch.serializedMatMul`.
        The matrix multiplication is split into separate smaller multiplications, calculated one after the other,
        to reduce the memory requirements of the multiplication and its gradient calculation.

        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            factor: Number of serialized multiplications. Must be a factor of
                the dimension to serialize on.
            bias: If set to False, the layer will not learn an additive bias.
                Default: True
            mode: Which dimension of the matmul to serialize on:
                for matrix A (m by n) multiplied by matrix B (n by p).
                * InputChannels: Split across the input channels (dimension m).
                * ReducingDim: Split across the reducing dimension (n).
                * OutputChannels: Split across the output channels (dimension p).
                * Disabled: Same as an ordinary matrix multiplication.
    """

    def __init__(self, in_features, out_features, factor, bias=False,
                 mode=poptorch.MatMulSerializationMode.OutputChannels):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.factor = factor

    def deserialize(self):
        linear = nn.Linear(self.in_features, self.out_features, self.bias is not None)
        linear.weight = self.weight
        linear.bias = self.bias
        return linear

    def forward(self, x):
        output = poptorch.serializedMatMul(x, self.weight.t(), self.mode, self.factor)
        if self.bias is not None:
            output += self.bias
        return output.reshape(*x.shape[:-1], self.out_features)
