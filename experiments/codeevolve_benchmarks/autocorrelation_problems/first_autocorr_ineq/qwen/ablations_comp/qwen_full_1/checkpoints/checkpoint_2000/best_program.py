# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import convolve
import random

def compute_c1(sequence):
    """Compute C1 constant for a given sequence."""
    if len(sequence) == 0 or sum(sequence) < 0.01:
        return float('inf')
    
    n = len(sequence)
    # Always use FFT-based convolution for better performance
    conv_result = np.convolve(sequence, sequence, mode='full')
    
    max_conv = np.max(conv_result)
    sum_sq = sum(sequence) ** 2
    
    if sum_sq == 0:
        return float('inf')
    
    # Add numerical stability check
    if max_conv < 1e-12:
        return float('inf')
    
    return 2 * n * max_conv / sum_sq

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence using more sophisticated local search."""
    n = len(sequence)
    if n == 0:
        return None
        
    sum_sequence = sum(sequence)
    if sum_sequence < 0.01:
        return None
    
    # Try multiple local search strategies to find better directions
    best_sequence = sequence.copy()
    best_c1 = compute_c1(sequence)
    
    # Strategy 1: Enhanced random perturbations with adaptive step size and better sampling
    for _ in range(120):  # More iterations for better exploration
        test_sequence = sequence.copy()
        # Perturb fewer elements with larger perturbations for more focused search
        num_perturb = max(1, n // 6)  # Perturb ~16% of elements for more focused search
        indices = random.sample(range(n), num_perturb)
        for idx in indices:
            # Add larger random noise with adaptive magnitude
            base_magnitude = min(2.0, sequence[idx] * 0.9)  # Even larger perturbations
            perturbation = random.uniform(-base_magnitude, base_magnitude)
            test_sequence[idx] = max(0, test_sequence[idx] + perturbation)
        
        # Check if this improves our result
        test_c1 = compute_c1(test_sequence)
        if test_c1 < best_c1 and test_c1 != float('inf'):
            best_c1 = test_c1
            best_sequence = test_sequence
    
    # Strategy 1b: Enhanced perturbation focused on critical regions (ends and center)
    if n > 30:
        try:
            test_sequence = sequence.copy()
            # Focus on end regions and center where convolution behavior is most critical
            critical_indices = []
            if n > 10:
                critical_indices.extend(range(min(5, n//4)))  # First few elements
                critical_indices.extend(range(max(0, n-5), n))  # Last few elements
            if n > 20:
                critical_indices.extend(range(n//2-2, n//2+3))  # Center region
            
            # Remove duplicates and sample appropriately
            critical_indices = list(set(critical_indices))
            num_perturb = max(1, min(len(critical_indices), len(critical_indices) // 2))
            indices = random.sample(critical_indices, num_perturb)
            
            for idx in indices:
                # Add larger perturbations to critical elements
                base_magnitude = min(1.5, sequence[idx] * 0.8)
                perturbation = random.uniform(-base_magnitude, base_magnitude)
                test_sequence[idx] = max(0, test_sequence[idx] + perturbation)
            
            test_c1 = compute_c1(test_sequence)
            if test_c1 < best_c1 and test_c1 != float('inf'):
                best_c1 = test_c1
                best_sequence = test_sequence
        except Exception:
            pass
    
    # Strategy 2: Enhanced smoothing approach with better kernel selection
    try:
        # Apply a well-designed smoothing kernel that's optimized for convolution reduction
        if n > 5:
            smoothed = np.array(sequence)
            # Use a Gaussian kernel with adaptive width
            kernel_width = min(15, max(3, n // 5))
            kernel = np.exp(-np.arange(-kernel_width, kernel_width+1)**2 / (2 * (kernel_width/3)**2))
            kernel = kernel / np.sum(kernel)
            
            # Convolve with padding
            padded = np.pad(smoothed, kernel_width, mode='edge')
            smoothed = np.convolve(padded, kernel, mode='valid')
            
            # Ensure non-negativity and normalize
            smoothed = np.maximum(0, smoothed[:n])
            total_old = sum(sequence)
            total_new = sum(smoothed)
            if total_new > 0:
                smoothed = smoothed * total_old / total_new
            
            # Clip to reasonable bounds
            smoothed = np.clip(smoothed, 0, 1000)
            
            test_c1 = compute_c1(smoothed.tolist())
            if test_c1 < best_c1 and test_c1 != float('inf'):
                best_c1 = test_c1
                best_sequence = smoothed.tolist()
    except Exception:
        pass
    
    # Strategy 3: Enhanced exponential decay with more strategic bases
    if n > 10:
        # Try exponential decays with more strategic bases that have shown effectiveness
        bases = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995, 0.999]
        for base in bases:
            exp_pattern = [base**i for i in range(n)]
            total_exp = sum(exp_pattern)
            if total_exp > 0:
                exp_scaled = [x * sum_sequence / total_exp for x in exp_pattern]
                test_c1 = compute_c1(exp_scaled)
                if test_c1 < best_c1 and test_c1 != float('inf'):
                    best_c1 = test_c1
                    best_sequence = exp_scaled
    
    # Strategy 4: Refine power-law decay with more effective alpha values
    if n > 10:
        try:
            # Try power law decay patterns with alpha values that have shown effectiveness
            alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
            for alpha in alphas:
                power_pattern = [1.0 / ((i+1)**alpha) for i in range(n)]
                total_power = sum(power_pattern)
                if total_power > 0:
                    power_scaled = [x * sum_sequence / total_power for x in power_pattern]
                    test_c1 = compute_c1(power_scaled)
                    if test_c1 < best_c1 and test_c1 != float('inf'):
                        best_c1 = test_c1
                        best_sequence = power_scaled
        except Exception:
            pass
    
    # Strategy 5: Enhanced hybrid approach with better mathematical grounding
    if n > 20:
        # Try more sophisticated hybrid combinations with mathematical justification
        hybrid_patterns = []
        
        # Pattern 1: Geometric with linear blend (as before)
        hybrid1 = [0.9**i * (1.0 - i/(n-1)) if n > 1 else 1.0 for i in range(n)]
        total1 = sum(hybrid1)
        if total1 > 0:
            hybrid1 = [x * sum_sequence / total1 for x in hybrid1]
            hybrid_patterns.append(('geo_linear', hybrid1))
        
        # Pattern 2: Weighted combination of multiple decay types
        hybrid2 = []
        for i in range(n):
            geo_val = 0.9**i
            lin_val = 1.0 - i/(n-1) if n > 1 else 0
            log_val = 1.0 / (np.log(i + 2) + 0.5) if i > 0 else 1.0
            # Blend with optimized weights
            hybrid2.append(0.3 * geo_val + 0.5 * lin_val + 0.2 * log_val)
        total2 = sum(hybrid2)
        if total2 > 0:
            hybrid2 = [x * sum_sequence / total2 for x in hybrid2]
            hybrid_patterns.append(('multi_blend', hybrid2))
        
        # Pattern 3: Adaptive hybrid that changes based on position
        hybrid3 = []
        for i in range(n):
            if i < n//2:
                val = 0.9**i
            else:
                val = 0.95**(i - n//2)
            hybrid3.append(val)
        total3 = sum(hybrid3)
        if total3 > 0:
            hybrid3 = [x * sum_sequence / total3 for x in hybrid3]
            hybrid_patterns.append(('adaptive', hybrid3))
        
        # Evaluate hybrid patterns
        for name, pattern in hybrid_patterns:
            test_c1 = compute_c1(pattern)
            if test_c1 < best_c1 and test_c1 != float('inf'):
                best_c1 = test_c1
                best_sequence = pattern
    
    # Strategy 6: Enhanced sinusoidal modulation with multiple frequencies
    if n > 30:
        try:
            # Try sinusoidal modulation with multiple frequencies for better exploration
            frequencies = [n//6, n//8, n//10, n//12]
            best_modulated = None
            best_mod_c1 = best_c1
            
            for freq in frequencies:
                if freq > 0:
                    modulated = [sequence[i] * (1 + 0.25 * np.sin(i * np.pi / freq)) for i in range(n)]
                    modulated = [max(0, x) for x in modulated]
                    total_mod = sum(modulated)
                    if total_mod > 0:
                        modulated = [x * sum_sequence / total_mod for x in modulated]
                        test_c1 = compute_c1(modulated)
                        if test_c1 < best_mod_c1 and test_c1 != float('inf'):
                            best_mod_c1 = test_c1
                            best_modulated = modulated
            
            if best_modulated is not None:
                best_c1 = best_mod_c1
                best_sequence = best_modulated
        except Exception:
            pass
    
    # Strategy 7: Spike-and-decay pattern with more refined parameters
    if n > 20:
        try:
            # Create a spike at the beginning followed by exponential decay
            spike_pattern = [0.0] * n
            spike_pattern[0] = sum_sequence * 0.7  # Balanced spike
            for i in range(1, n):
                spike_pattern[i] = spike_pattern[i-1] * 0.85  # Standard decay
            
            # Normalize to match sum
            total_spike = sum(spike_pattern)
            if total_spike > 0:
                spike_pattern = [x * sum_sequence / total_spike for x in spike_pattern]
                
            test_c1 = compute_c1(spike_pattern)
            if test_c1 < best_c1 and test_c1 != float('inf'):
                best_c1 = test_c1
                best_sequence = spike_pattern
        except Exception:
            pass
    
    return best_sequence

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Try multiple random initializations to find good starting points
    best_sequence = None
    best_c1 = float('inf')
    
    # Try different initialization strategies - focused on most effective patterns plus new promising ones
    init_strategies = [
        # Uniform distribution
        lambda n: [1.0] * n,
        # Geometric decay (more aggressive)
        lambda n: [0.95**i for i in range(n)],
        # Linear decay
        lambda n: [max(0, 1.0 - i/(n-1)) for i in range(n)],
        # Spike at beginning
        lambda n: [1.0 if i == 0 else 0.0 for i in range(n)],
        # Random with variance control
        lambda n: [random.uniform(0.1, 15.0) for _ in range(n)],
        # Power law decay (more refined)
        lambda n: [1.0 / ((i+1)**0.6) for i in range(n)],
        # Gaussian-like pattern
        lambda n: [np.exp(-((i - n/2)**2) / (2 * (n/6)**2)) for i in range(n)],
        # Logarithmic decay
        lambda n: [1.0/np.log(i+2) for i in range(n)] if n > 1 else [1.0] * n,
        # Sine wave pattern
        lambda n: [0.5 * (1 + np.sin(i * 2 * np.pi / (n//4))) for i in range(n)],
        # Inverted exponential
        lambda n: [0.95**(n-1-i) for i in range(n)],
        # Step function pattern
        lambda n: [1.0 if i < n//2 else 0.3 for i in range(n)],
        # Parabolic decay
        lambda n: [1.0 - (i/(n-1))**2 for i in range(n)],
        # Exponential with base 0.98
        lambda n: [0.98**i for i in range(n)],
        # Heavy-tailed power law
        lambda n: [1.0 / ((i+1)**1.3) for i in range(n)],
        # Optimized power law decay
        lambda n: [1.0 / ((i+1)**0.7) for i in range(n)],
        # Fast decay exponential
        lambda n: [0.85**i for i in range(n)],
        # Modified exponential decay
        lambda n: [0.92**i for i in range(n)],
        # Inverted bell curve
        lambda n: [1 - np.exp(-((i - n/2)**2) / (2 * (n/5)**2)) for i in range(n)],
        # Concave pattern
        lambda n: [(n-i)/n for i in range(n)],
        # Anti-correlated pattern
        lambda n: [1.0 if i % 3 == 0 else 0.2 if i % 3 == 1 else 0.8 for i in range(n)],
        # "Cone" pattern - high at start, decreases rapidly
        lambda n: [max(0, 1.0 - i/(n*0.3)) for i in range(n)],
        # "Bump" pattern - small values, then sharp rise, then fall
        lambda n: [0.0 if i < n//3 else (1.0 if i < 2*n//3 else 0.0) for i in range(n)],
        # "Tapered exponential" - fast initial decay, then slower
        lambda n: [0.95**i * (1.0 + 0.1 * np.sin(i * 0.5)) for i in range(n)],
        # "Multi-scale" pattern - combines multiple scales
        lambda n: [0.9**i * (1.0 + 0.2 * np.sin(i * 0.3)) if i < n//2 else 0.95**(i-n//2) * (1.0 + 0.2 * np.sin(i * 0.3)) for i in range(n)]
    ]
    
    for attempt in range(45):  # Slightly more attempts for better exploration
        # Use more varied sequence lengths with emphasis on smaller ranges for efficiency
        # Increase probability of exploring larger sequences to find breakthroughs
        if random.random() < 0.65:
            n = random.randint(20, 600)  # Focus on smaller ranges for more efficient search
        else:
            n = random.randint(600, 1500)  # Include larger sequences for breakthroughs
        
        # Choose initialization strategy
        strategy_idx = attempt % len(init_strategies)
        sequence = init_strategies[strategy_idx](n)
        
        # Apply optimization multiple times for better convergence
        current_sequence = sequence.copy()
        # Adjust optimization steps based on sequence characteristics
        if n > 1000:
            # For very large sequences, use fewer but more intense optimization steps
            num_steps = max(5, min(12, n // 100))
        else:
            # For smaller sequences, use more steps
            num_steps = max(10, min(20, n // 15))
        for _ in range(num_steps):
            h_function = get_good_direction_to_move_into(current_sequence)
            if h_function is not None:
                current_sequence = h_function
            else:
                break
                
        # Evaluate
        c1 = compute_c1(current_sequence)
        if c1 < best_c1 and c1 != float('inf'):
            best_c1 = c1
            best_sequence = current_sequence.copy()
    
    # If no good sequence found, start with a better baseline
    if best_sequence is None:
        # Try a more carefully constructed sequence
        n = 350
        # Use a combination that often works well: geometric with slight modification
        best_sequence = [0.92**i * (1.0 + 0.02 * random.random()) for i in range(n)]
        # Normalize to have reasonable sum
        s = sum(best_sequence)
        if s > 0:
            best_sequence = [x/s * 80 for x in best_sequence]
    
    # Final refinement with fewer passes to preserve time
    for _ in range(15):  # Reduced refinement passes for efficiency
        refined = get_good_direction_to_move_into(best_sequence)
        if refined is not None:
            c1_refined = compute_c1(refined)
            if c1_refined < best_c1 and c1_refined != float('inf'):
                best_c1 = c1_refined
                best_sequence = refined
        else:
            break
    
    # Additional specialized refinement for better convergence
    # Try to detect when we're close to optimal and do more focused search
    if best_c1 < 1.6:  # If we're already quite good, do more intensive refinement
        for _ in range(10):
            refined = get_good_direction_to_move_into(best_sequence)
            if refined is not None:
                c1_refined = compute_c1(refined)
                if c1_refined < best_c1 and c1_refined != float('inf'):
                    best_c1 = c1_refined
                    best_sequence = refined
            else:
                break
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
