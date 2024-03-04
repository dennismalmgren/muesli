import torch
#how to generate head cells and place cells
from torch.distributions import MultivariateNormal, Normal, VonMises

max_per_dim = torch.tensor([2.3198, 0.3236, 0.8711, 1.3061, 1.8617, 2.7261, 3.5292, 4.7789])
min_per_dim = torch.tensor([-0.1643, -1.7658, -1.7496, -0.4820, -2.6647, -1.8809, -5.4557, -3.9380])

N_place_cells = 256
N_head_cells = 12

place_cell_centers = torch.rand((N_place_cells, 8)) * (max_per_dim - min_per_dim) + min_per_dim
torch.save(place_cell_centers, "place_cell_centers.pt")
head_cell_centers = torch.rand((N_head_cells, 8)) * 2 - 1 # from -1 to 1
torch.save(head_cell_centers, "head_cell_centers.pt")
head_cell_centers = head_cell_centers / torch.linalg.norm(head_cell_centers, dim=-1, keepdim=True)
place_cell_scale = 3
head_cell_scale = 15  # 20 degrees in radians
place_cell_scales = place_cell_scale **2 * torch.eye(N_place_cells )

loc1 = torch.zeros(8)
loc2 = torch.ones(8)
place_cell_scores = torch.linalg.vector_norm(place_cell_centers - loc2, dim=-1)**2
place_cell_scores = -place_cell_scores / (2 * place_cell_scale ** 2)
all_exponents = torch.logsumexp(place_cell_scores, dim=-1)
normalized_scores = place_cell_scores - all_exponents
place_cell_activations = torch.exp(normalized_scores)

heading = (loc2 - loc1) / torch.norm(loc2 - loc1)
head_cell_scores = heading @ head_cell_centers.T
head_cell_scores = head_cell_scale * head_cell_scores
all_exponents = torch.logsumexp(head_cell_scores, dim=-1)
normalized_scores = head_cell_scores - all_exponents
head_cell_activations = torch.exp(normalized_scores)

# def calculate_place_cell_activations(pos: torch.Tensor):
#     pos = pos.unsqueeze(0)
#     return torch.exp(-((pos - place_cell_centers) / place_cell_scale).pow(2).sum(dim=-1))

# pos = torch.zeros(8)

#place_cell_activations = calculate_place_cell_activations(pos)
print(place_cell_activations)
print(head_cell_activations)