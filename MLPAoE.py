from megablocks.layers import common
from megablocks.layers import moe
from megablocks.layers import mpu
from megablocks.layers.arguments import Arguments
import megablocks.ops as ops
import numpy as np
import stk
import torch
from megablocks.layers.activation_fn import act_fn
from megablocks.layers.moe import save_load_balancing_loss
from megablocks.layers.mlp import create_dmoe_expert_weights, resolve_dtensor, ScaleGradient

def promote_scalar(x):
    return x.view(1) if not len(x.size()) else x

scale_gradient = ScaleGradient.apply

class SparseGLUAoE(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()

        self.args = args
        self._num_rows_per_rank = (
            (mpu.experts_per_rank(args) * mpu.features_per_rank(args)) //
            mpu.get_weight_parallel_world_size(args)
        )

        d_low = os.getenv('D_LOW')

        self.wup = torch.nn.Parameter(torch.empty(
            self._num_rows_per_rank,
            d_low,
            device=args.device,
            dtype=common.dtype(args)))
        self.w2 = torch.nn.Parameter(torch.empty(
            self._num_rows_per_rank,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))
        self.v1 = torch.nn.Parameter(torch.empty(
            self._num_rows_per_rank,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))
        with torch.no_grad():
            self.wup.copy_(create_dmoe_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                d_low, args.init_method))
            self.w2.copy_(create_dmoe_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                args.hidden_size, args.output_layer_init_method))
            self.v1.copy_(create_dmoe_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                args.hidden_size, args.init_method))

        self._should_set_parallelism_attribute = (
            args.moe_expert_model_parallelism or args.moe_weight_parallelism)
        mpu.set_expert_model_parallel_attributes(
            self.wup, self._should_set_parallelism_attribute)
        mpu.set_expert_model_parallel_attributes(
            self.w2, self._should_set_parallelism_attribute)
        mpu.set_expert_model_parallel_attributes(
            self.v1, self._should_set_parallelism_attribute)

        self.gradient_scale = None
        if self.args.moe_expert_model_parallelism:
            self.gradient_scale = 1 / mpu.get_expert_parallel_world_size(self.args)

        if self.args.moe_weight_parallelism:
            raise NotImplementedError("Weight parallelism not yet supported with GLU.")

    def forward(self, x, acts, topo):
        if self.args.memory_optimized_mlp:
            raise NotImplementedError("Memory optimized implementation not yet supported with GLU with sparse kernels.")

        wup, v1, w2 = self.scale_grad(self.wup), self.scale_grad(self.v1), self.scale_grad(self.w2)
        wup, v1, w2 = resolve_dtensor(wup), resolve_dtensor(v1), resolve_dtensor(w2)

        # Compute the GLU.
        x1 = stk.ops.sdd(acts, wup.t(), topo)
        x2 = stk.ops.sdd(x, v1.t(), topo)

        activation_fn_out = act_fn(x1, self.args.activation_fn)
        x1 = stk.ops.mul(activation_fn_out, x2)

        return stk.ops.dsd(x1, w2)

    def scale_grad(self, w):
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)
    
class ParallelDroplessMLPAoE(moe.ParallelMLP):

    def __init__(self, args : Arguments):
        super(ParallelDroplessMLPAoE, self).__init__(args)
        self.hidden_size = args.hidden_size
        self.ffn_hidden_size = mpu.features_per_rank(args)
        self.blocking = 128
        self.mlp = SparseGLUAoE(args)

        # Calculate the number of bits needed to represent the column indices
        # in the intermediate sparse matrix.
        max_column_index = (
            (self.ffn_hidden_size * self.num_experts) // self.blocking)
        self.transpose_sort_end_bit = max(
            int(np.ceil(np.log2(max_column_index))), 1)

    def sparse_transpose(self, size, row_indices, column_indices, offsets):
        block_columns = size[1] // self.blocking

        # Sort row indices by column indices to get the transposed matrix's
        # column indices.
        #
        # NOTE: Our sort operation uses the same width indices as the input values.
        # To avoid overflow when we have large activation matrices we cast to
        # 32-bit before sorting.
        _, gather_indices = ops.sort(
            column_indices.int(), self.transpose_sort_end_bit)

        # There are a constant number of blocks in every row of the sparse matrix.
        # A blocks offset is:
        #
        # row_index * blocks_per_row + column_index % blocks_per_row
        #
        # Once we have the block offsets ordered for transposition we can divide
        # by blocks_per_row to get the transposed column indices.
        column_indices_t = row_indices.gather(0, gather_indices.long())
        block_offsets_t = gather_indices.int()

        zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
        nnz_per_column = ops.histogram(column_indices, block_columns)
        nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
        if nnz_per_column.dim() == 0:
            # This addresses an edge case when ffn_hidden_size is equal to self.blocking.
            nnz_per_column = nnz_per_column.unsqueeze(0)
        offsets_t = torch.cat([zero, nnz_per_column])
        return column_indices_t, offsets_t, block_offsets_t

    def topology(self, x, padded_bins):
        padded_tokens, _ = x.size()
        assert padded_tokens % self.blocking == 0
        if self.ffn_hidden_size % self.blocking != 0:
            raise ValueError(f'The ffn_hidden_size {self.ffn_hidden_size} must be divisible by ' +
                             f'the block size {self.blocking}. Please update your configuration.')

        # Offsets for the sparse matrix. All rows have the
        # same number of nonzero blocks dictated by the
        # dimensionality of a single expert.
        block_rows = padded_tokens // self.blocking
        blocks_per_row = self.ffn_hidden_size // self.blocking
        offsets = torch.arange(
            0,
            block_rows * blocks_per_row + 1,
            blocks_per_row,
            dtype=torch.int32,
            device=x.device)

        # Indices for the sparse matrix. The indices for
        # the intermediate matrix are dynamic depending
        # on the mapping of tokens to experts.
        column_indices = ops.topology(padded_bins,
                                      self.blocking,
                                      block_rows,
                                      blocks_per_row)

        # TODO(tgale): This is unused. Remove the need for this in stk.
        # For now, use meta init to save the device memory.
        data = torch.empty(
            column_indices.numel(),
            self.blocking,
            self.blocking,
            dtype=common.dtype(self.args),
            device='meta')
        shape = (
            padded_tokens,
            self.ffn_hidden_size * mpu.experts_per_rank(self.args)
        )
        row_indices = stk.ops.row_indices(
            shape, data, offsets, column_indices)
        column_indices_t, offsets_t, block_offsets_t = self.sparse_transpose(
            shape, row_indices, column_indices, offsets)
        return stk.Matrix(shape, data, row_indices, column_indices, offsets,
                          column_indices_t, offsets_t, block_offsets_t)

    def indices_and_padded_bins(self, top_experts):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        top_experts = top_experts.int()
        bin_ids, indices = ops.sort(top_experts, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        tokens_per_expert = ops.histogram(top_experts, self.num_experts)

        # Round the token counts up to the block size used in
        # the matrix muliplications. Caculate the starting
        # position of each bin.
        padded_tokens_per_expert = ops.round_up(
            tokens_per_expert, self.blocking)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        padded_bins = promote_scalar(padded_bins)

        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = promote_scalar(bins)
        return indices, bin_ids, bins, padded_bins, tokens_per_expert

    def sparse_forward_once(self, x, acts, expert_weights, top_experts):
        # x: [sl, bs, hs]
        # acts: # sl * bs, top-k, d_low
        # expert_weights: [sl * bs, top-k]
        # top_experts: [sl * bs, top-k]
        expert_weights = expert_weights.flatten()
        top_experts = top_experts.flatten()
        with torch.no_grad():
            indices, bin_ids, bins, padded_bins, tokens_per_expert = (
                self.indices_and_padded_bins(top_experts))

        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])
        x = ops.padded_gather(
            x,
            indices,
            bin_ids,
            bins,
            padded_bins,
            self.top_k)
        
        # sort acts by indices
        acts = acts.view(-1, acts.shape[-1])
        acts = ops.padded_gather(
            acts,
            indices,
            bin_ids,
            bins,
            padded_bins,
            1) # just pad acts to align with x
        
        # Create the sparse matrix topology.
        with torch.no_grad():
            topo = self.topology(x, padded_bins)

        # # Perform the expert computation.
        x = self.mlp(x, acts, topo)

        # # Un-route the data for the MoE output.
        x = ops.padded_scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            padded_bins,
            self.top_k)
        return x, tokens_per_expert
   
    def forward(self, x, acts, scores, logits, expert_weights, top_experts):
        in_shape = x.size()

        # Compute the experts.
        x, tokens_per_expert = self.sparse_forward_once(x, acts, expert_weights, top_experts)
        
        if self.training and self.args.moe_loss_weight > 0:
            save_load_balancing_loss((tokens_per_expert, scores, logits))
        x = x.view(in_shape)
        if self.bias is not None:
            if self.args.return_bias:
                return x, self.bias
            return x + self.bias
        return x