import math

def input_size_from_output(output_size, num_layers):
    size = int(math.sqrt(output_size))  # Convert linear output to 2D size (assuming square)
    
    for _ in range(num_layers):
        # Reverse max pooling: 2x2, stride 2
        size *= 2
        
        # Reverse Conv layer 2: 3x3 same (output size same as input)
        
        # Reverse ReLU (no size change)
        
        # Reverse Conv layer 1: 3x3 conv, no padding, stride 1
        size += 2  # Adding 2 to account for the kernel size
        
    return size * size  # Return as linear size