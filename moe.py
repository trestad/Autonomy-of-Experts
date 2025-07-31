import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mirage.init import init_normal_
from typing import Tuple, List, Union
import os
ExpertRecipe = Union[None, int, Tuple[int, ...], List[int]]

def _normalize_expert_recipe(expert_recipe: ExpertRecipe):
    if expert_recipe is None:
        return ()
    elif isinstance(expert_recipe, int):
        return tuple([1] * expert_recipe)
    else:
        return tuple(expert_recipe)

def load_balancing_loss_func(
    logits,
    num_experts,
    top_k=2,
):
    """
        I deleted code regarding attention mask
        this function only can be used in pretraining: no padding, all tokens are processed
    """

    # compute_device = gate_logits[0].device
    # concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class Expert(nn.Module):

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.gw = nn.Parameter(torch.empty((dim, hidden_dim))) # W_in
        self.pw = nn.Parameter(torch.empty((dim, hidden_dim))) # 
        self.ow = nn.Parameter(torch.empty((hidden_dim, dim)))
        self.reset_parameters()

    def extra_repr(self) -> str:
        return f"{self.dim} -> {self.hidden_dim} -> {self.dim}"

    def reset_parameters(self) -> None:
        init_normal_(self.pw, std=0.02)
        init_normal_(self.gw, std=0.02)
        init_normal_(self.ow, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        g: Tensor = x @ self.gw
        x = g * F.silu(x @ self.pw)
        x = x @ self.ow
        return x

class BigRouter(nn.Module):
    
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.gate1 = nn.Linear(self.dim, 1012, bias=False)
        self.gate2 = nn.Linear(1012, self.num_experts, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_normal_(self.gate1.weight, std=0.02)
        init_normal_(self.gate2.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return self.gate2(self.gate1(x))

class MoE(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        top_k: int,
        shared_expert_recipe: ExpertRecipe = None,
        wander_expert_recipe: ExpertRecipe = None,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.shared_expert_recipe = _normalize_expert_recipe(shared_expert_recipe)
        self.wander_expert_recipe = _normalize_expert_recipe(wander_expert_recipe)

        fuel_quantity = sum(self.shared_expert_recipe) + sum(self.wander_expert_recipe)
        if fuel_quantity == 0:
            raise ValueError(
                "Choose at least one of shared_expert_recipe and dynamic_expert_recipe to configure."
            )
        if hidden_dim % fuel_quantity != 0:
            raise ValueError(
                f"Provide hidden_dim({hidden_dim}) cannot be equally divided by fuel({fuel_quantity})."
            )
        
        self.fuel_dim = hidden_dim // fuel_quantity

        self.wander_experts = nn.ModuleList()

        for fuel in self.wander_expert_recipe:
            self.wander_experts.append(Expert(dim, fuel * self.fuel_dim))
        
        self.num_experts = len(self.wander_experts)
        
        # self.gate = BigRouter(self.dim, self.num_experts) 
        self.gate = nn.Linear(self.dim, self.num_experts, bias=False)
        
        self.shared_hidden_dim = sum(self.shared_expert_recipe) * self.fuel_dim

        self.wander_hidden_dim = sum(self.wander_expert_recipe) * self.fuel_dim
        
        self.jitter_noise = float(os.getenv('JITTER_NOISE', -1.0))
        print(f"JITTER_NOISE: {self.jitter_noise}")
        
        self.activated_knowledge_per_token = self.top_k * self.fuel_dim # (mask * expert_dims).sum(1) + shared_hidden_dim
        
        self.activated_knowledge_ratio = self.top_k * self.fuel_dim / hidden_dim

    def extra_repr(self) -> str:
        return (
            f"{self.dim} -> "
            f"{self.hidden_dim} [{self.shared_expert_recipe} + {self.wander_expert_recipe}] -> "
            f"{self.dim}"
        )

    def forward(self, hidden_states: torch.Tensor, tokens: torch.Tensor=None) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # print(self.training)
        # print(self.jitter_noise)
        if self.training and self.jitter_noise > 0:
            # print('add jitter')
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
            
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.wander_experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        # activated_wander_experts = expert_mask
        moe_states = {
            "activated_knowledge_per_token": self.activated_knowledge_per_token,
            "activated_knowledge_ratio": self.activated_knowledge_ratio,
            }

        bl_loss = load_balancing_loss_func(router_logits, self.num_experts, self.top_k)

        return final_hidden_states, moe_states, router_logits, bl_loss

class ExpertAoE(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, dim4route: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dim4route_to_hidden_dim = nn.Parameter(torch.empty((dim4route, hidden_dim)))
        self.dim_to_hidden_dim = nn.Parameter(torch.empty((dim, hidden_dim))) # W_in
        self.dim_to_dim4route = nn.Parameter(torch.empty((dim, dim4route))) # 
        self.hidden_dim_to_dim = nn.Parameter(torch.empty((hidden_dim, dim)))
        self.reset_parameters()

    def extra_repr(self) -> str:
        return f"{self.dim} -> {self.hidden_dim} -> {self.dim}"

    def reset_parameters(self) -> None:
        init_normal_(self.dim4route_to_hidden_dim, std=0.02)
        init_normal_(self.dim_to_hidden_dim, std=0.02)
        init_normal_(self.dim_to_dim4route, std=0.02)
        init_normal_(self.hidden_dim_to_dim, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        w1_state = F.silu(x @ self.dim_to_dim4route @ self.dim4route_to_hidden_dim)
        w3_state: Tensor = x @ self.dim_to_hidden_dim
        x = w3_state * w1_state
        x = x @ self.hidden_dim_to_dim
        return x

class AoE(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        top_k: int,
        shared_expert_recipe: ExpertRecipe = None,
        wander_expert_recipe: ExpertRecipe = None,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.shared_expert_recipe = _normalize_expert_recipe(shared_expert_recipe)
        self.wander_expert_recipe = _normalize_expert_recipe(wander_expert_recipe)

        fuel_quantity = sum(self.shared_expert_recipe) + sum(self.wander_expert_recipe)
        if fuel_quantity == 0:
            raise ValueError(
                "Choose at least one of shared_expert_recipe and dynamic_expert_recipe to configure."
            )
        if hidden_dim % fuel_quantity != 0:
            raise ValueError(
                f"Provide hidden_dim({hidden_dim}) cannot be equally divided by fuel({fuel_quantity})."
            )
        
        self.wander_experts = nn.ModuleList()

        self.dim4route = 128
        self.fuel_dim = 4096
       
        for fuel in self.wander_expert_recipe:
            self.wander_experts.append(ExpertNoRouter(dim, fuel * self.fuel_dim, self.dim4route))

        self.num_experts = len(self.wander_experts)
        
        self.activated_knowledge_per_token = self.top_k * (hidden_dim // fuel_quantity) # (mask * expert_dims).sum(1) + shared_hidden_dim
        
        self.activated_knowledge_ratio = self.top_k / self.num_experts
        
        self.jitter_noise = float(os.getenv('JITTER_NOISE', -1.0))
        print(f"JITTER_NOISE: {self.jitter_noise}")
        
    @property
    def shared_hidden_dim(self):
        return sum(self.shared_expert_recipe) * self.fuel_dim

    @property
    def wander_hidden_dim(self):
        return sum(self.wander_expert_recipe) * self.fuel_dim

    def extra_repr(self) -> str:
        return (
            f"{self.dim} -> "
            f"{self.hidden_dim} [{self.shared_expert_recipe} + {self.wander_expert_recipe}] -> "
            f"{self.dim}"
        )

    def forward(self, hidden_states: torch.Tensor, tokens: torch.Tensor=None) -> torch.Tensor:
        """ """
        batch_size, sequence_length, dim = hidden_states.shape
        # print(self.training)
        # print(self.jitter_noise)
        if self.training and self.jitter_noise > 0:
            # print('add jitter')
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
            
            
        hidden_states = hidden_states.view(-1, dim) # (batch * sequence_length, dim)
        
        expert_act_norms = []
        expert_acts = []

        # Naive loop implementation here
        # Optimization suggestion: Define a combined W_down in AoE class's init 
        # e.g., self.Wdown = nn.Parameter(torch.empty((dim, self.dim4router, self.num_experts)))
        # This allows replacing the loop with 3 concise lines:
        # expert_act = (hidden_states @ self.Wdown).reshape(-1, self.num_experts, self.dim4router)
        # expert_act_norms = expert_act.mean(dim=-1)
        # expert_act = expert_act.permute(1, 0, 2)
        for expert in self.wander_experts:
            expert_act = hidden_states @ expert.dim_to_dim4route # (batch * sequence_length, inter_dim)
            expert_acts.append(expert_act)
            expert_act_norms.append(torch.norm(expert_act, dim=-1))
            
        expert_act_norms = torch.stack(expert_act_norms, dim=1) # (batch * sequence_length, num_experts)

        routing_weights = F.softmax(expert_act_norms, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(hidden_states.dtype)
        
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):

            expert_layer = self.wander_experts[expert_idx]
            
            idx, top_x = torch.where(expert_mask[expert_idx])

            w3_state = hidden_states[None, top_x].reshape(-1, dim) @ expert_layer.dim_to_hidden_dim

            w1_state = expert_acts[expert_idx][None, top_x].reshape(-1, self.dim4route)
            
            current_hidden_states = w3_state * F.silu(w1_state @ expert_layer.dim4route_to_hidden_dim)
            current_hidden_states = (current_hidden_states @ expert_layer.hidden_dim_to_dim) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, dim)

        moe_states = {
            "activated_knowledge_per_token": self.activated_knowledge_per_token,
            "activated_knowledge_ratio": self.activated_knowledge_ratio,
            }

        bl_loss = load_balancing_loss_func(expert_act_norms, self.num_experts, self.top_k)
        
        return final_hidden_states, moe_states, expert_act_norms, bl_loss