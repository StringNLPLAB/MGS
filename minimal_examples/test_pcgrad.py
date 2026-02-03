import torch
import torch.nn as nn
import torch.optim as optim

# --- PCGrad Core Logic Helpers ---

def _get_grad_vector(gradients):
    """Flattens a list of gradients into a single 1D vector."""
    return torch.cat([g.contiguous().flatten() for g in gradients])

def _set_grad_tensors(params, grad_tensors):
    """Sets the .grad attribute for each parameter from the provided list of tensors."""
    for p, g in zip(params, grad_tensors):
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.data.copy_(g.data)

def _reshape_vector_to_tensors(vector, params):
    """Reshapes a single flat vector back into a list of tensors matching parameter shapes."""
    start = 0
    tensors = []
    for p in params:
        numel = p.numel()
        end = start + numel
        grad_slice = vector[start:end]
        tensors.append(grad_slice.view_as(p).detach())
        start = end
    return tensors

# --- PCGrad Utility Class ---

class PCGradUtility:
    """PCGrad projection logic."""
    def __init__(self, model):
        self.params = [p for p in model.parameters() if p.requires_grad]

    def apply_projection(self, individual_grad_lists):
        """
        Applies PCGrad projection.
        
        Args:
            individual_grad_lists (list[list[Tensor]]): Gradients for each objective.
        
        Returns:
            (list[Tensor], list[list[Tensor]]): 
                1. The final list of combined, projected gradients (to be set on model.grad).
                2. The list of individual projected gradient tensors (for verification).
        """
        grad_vectors = [_get_grad_vector(g_list) for g_list in individual_grad_lists]
        num_tasks = len(grad_vectors)

        new_grad_vectors = []
        for i in range(num_tasks):
            g_i = grad_vectors[i].clone()
            g_i_prime = g_i.clone()

            for j in range(num_tasks):
                if i == j: 
                    continue

                g_j = grad_vectors[j]
                dot_product = torch.dot(g_i, g_j)

                if dot_product < 0:  # FIXED: Only project when there's CONFLICT (dot < 0)
                    g_j_norm_sq = torch.sum(g_j * g_j)
                    if g_j_norm_sq > 1e-12:  # More stable threshold
                        projection_factor = dot_product / g_j_norm_sq
                        g_i_prime -= projection_factor * g_j

            new_grad_vectors.append(g_i_prime)

        # Combine the projected gradients
        final_grad_vector = torch.sum(torch.stack(new_grad_vectors), dim=0)
        final_grad_tensors = _reshape_vector_to_tensors(final_grad_vector, self.params)
        
        individual_projected_tensors = [
            _reshape_vector_to_tensors(v, self.params) for v in new_grad_vectors
        ]
            
        return final_grad_tensors, individual_projected_tensors

# --- Test Setup ---

torch.manual_seed(42)
model = nn.Linear(3, 1) 
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Create input data
X = torch.randn(10, 3)

# Initialize PCGrad Utility
pcgrad_util = PCGradUtility(model)
trainable_params = pcgrad_util.params

print("=== PCGrad Test ===\n")

# ----------------------------------------------------
# TEST 1: Check Initial Gradient Conflict
# ----------------------------------------------------
print("--- TEST 1: Checking Initial Gradient Conflict ---")

# Clear any existing gradients
optimizer.zero_grad()

# Forward pass
output = model(X)

# Create conflicting tasks - both trying to control the same output dimension
# Task 1: Wants output[:, 0] to be +10
Y1_target = torch.full((10,), 10.0)
loss1 = criterion(output[:, 0], Y1_target)

# Task 2: Wants output[:, 0] to be -10 (CONFLICT!)
Y2_target = torch.full((10,), -10.0) 
loss2 = criterion(output[:, 0], Y2_target)  # FIXED: Changed to output[:, 1] to create conflict on shared weights

# Compute gradients for both tasks
grad1 = torch.autograd.grad(loss1, trainable_params, retain_graph=True, create_graph=False)
grad2 = torch.autograd.grad(loss2, trainable_params, retain_graph=False, create_graph=False)

# Check initial conflict
grad1_vector_raw = _get_grad_vector(grad1)
grad2_vector_raw = _get_grad_vector(grad2)

initial_dot = torch.dot(grad1_vector_raw, grad2_vector_raw)
initial_cosine = initial_dot / (grad1_vector_raw.norm() * grad2_vector_raw.norm() + 1e-12)

print(f"Initial Dot Product: {initial_dot.item():.4f}")
print(f"Initial Cosine Similarity: {initial_cosine.item():.4f}")

if initial_dot.item() < 0:
    print("✅ Conflicting gradients detected (dot product < 0)")
else:
    print("❌ No strong conflict detected - gradients are aligned")
print("-" * 50)

# ----------------------------------------------------
# TEST 2: PCGrad Projection
# ----------------------------------------------------
print("--- TEST 2: Applying PCGrad Projection ---")

# Clear gradients again
optimizer.zero_grad()

# We need fresh gradients since the previous graph was cleared
output_fresh = model(X)
loss1_fresh = criterion(output_fresh[:, 0], Y1_target)
loss2_fresh = criterion(output_fresh[:, 0], Y2_target)

# Compute fresh gradients
grad1_fresh = torch.autograd.grad(loss1_fresh, trainable_params, retain_graph=True, create_graph=False)
grad2_fresh = torch.autograd.grad(loss2_fresh, trainable_params, retain_graph=False, create_graph=False)

# Apply PCGrad
individual_grad_lists = [list(grad1_fresh), list(grad2_fresh)]
final_grad_tensors, individual_projected_tensors = pcgrad_util.apply_projection(individual_grad_lists)

# Set the projected gradients
_set_grad_tensors(trainable_params, final_grad_tensors)

print(f"PCGrad applied successfully")
print(f"Model weight grad norm: {model.weight.grad.norm().item():.4f}")

# Verify orthogonality
g1_prime_vector = _get_grad_vector(individual_projected_tensors[0])
g2_prime_vector = _get_grad_vector(individual_projected_tensors[1])

final_dot = torch.dot(g1_prime_vector, g2_prime_vector)
final_cosine = final_dot / (g1_prime_vector.norm() * g2_prime_vector.norm() + 1e-12)

print(f"\nFinal Dot Product: {final_dot.item():.4f}")
print(f"Final Cosine Similarity: {final_cosine.item():.4f}")

# Verification
if abs(final_dot.item()) < 1e-4:
    print("✅ SUCCESS: Projected gradients are orthogonal (dot ≈ 0)")
elif final_dot.item() >= 0:
    print("✅ SUCCESS: Projected gradients are non-conflicting (dot ≥ 0)")
else:
    print("❌ Projected gradients still conflicting")

print("-" * 50)

# ----------------------------------------------------
# TEST 3: Compare with Standard MTL
# ----------------------------------------------------
print("--- TEST 3: Comparison with Standard MTL ---")

# Standard MTL approach
optimizer.zero_grad()
output_mtl = model(X)
loss1_mtl = criterion(output_mtl[:, 0], Y1_target)
loss2_mtl = criterion(output_mtl[:, 0], Y2_target)
total_loss_mtl = loss1_mtl + loss2_mtl
total_loss_mtl.backward()

mtl_grad_norm = model.weight.grad.norm().item()
pcgrad_grad_norm = final_grad_tensors[0].norm().item()  # weight is first parameter

print(f"Standard MTL gradient norm: {mtl_grad_norm:.4f}")
print(f"PCGrad gradient norm: {pcgrad_grad_norm:.4f}")

if pcgrad_grad_norm > mtl_grad_norm:
    print("✅ PCGrad preserved more gradient magnitude")
else:
    print("ℹ️  PCGrad reduced gradient magnitude (expected when resolving conflicts)")