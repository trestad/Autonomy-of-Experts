from megablocks.layers.dmoe import dMoE
from .MLPAoE import ParallelDroplessMLPAoE
class VanillaAoE(dMoE):

    def __init__(self, args):
        super().__init__(args)

        self.args = args

        if hasattr(self, 'router'):
            del self.router

        self.d_low = os.getenv('D_LOW')
        self.wg = nn.Parameter(torch.empty(args.hidden_size, self.d_low * args.num_experts))

    def _init_experts_mlp(self, args):
        return ParallelDroplessMLPAoE(args)
    
    def forward(self, x):
        # NOTE: If we're going to cast the activations to lower precision
        # do it before we permute the tokens to save bandwidth.
        x = common.cast_if_autocast_enabled(x)

        acts = (x.view(-1, x.shape[-1]) @ self.wg).view(-1, self.args.num_experts, self.d_low)
        logits = torch.norm(acts, dim=-1)
        scores = logits.softmax(dim=-1)

        expert_weights, expert_indices = torch.topk(scores, self.args.topk, dim=-1)

        topk_acts = acts[torch.arange(acts.shape[0]).unsqueeze(1), expert_indices] # total_tokens, k, d_low

        # Compute the experts.
        out = self.experts(x, topk_acts, scores, logits, expert_weights, expert_indices)
        if self.shared_expert is not None:
            shared_expert_out = self.shared_expert(x)
            out = self.shared_expert.add_experts_sharedexpert(shared_expert_out, out)
        return out