import torch
import numpy as np

def generate_switching_sine_data(n_samples=1000, sequence_length=20):
    """Generate data that switches between different sine wave patterns"""
    t = np.linspace(0, 8*np.pi, n_samples)
    
    pattern1 = np.sin(t)
    pattern2 = np.sin(2*t)
    pattern3 = np.sin(t) * (0.5 + 0.5*np.sin(0.5*t))
    
    data = np.zeros_like(t)
    for i in range(len(t)):
        if i % (n_samples//3) < (n_samples//9):
            data[i] = pattern1[i]
        elif i % (n_samples//3) < 2*(n_samples//9):
            data[i] = pattern2[i]
        else:
            data[i] = pattern3[i]
    
    X = np.array([data[i:i+sequence_length] for i in range(len(data) - sequence_length)])
    y = np.array([data[i+sequence_length] for i in range(len(data) - sequence_length)])
    
    return torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1)

def generate_lorenz_data(n_samples=1000, sequence_length=20):
    """Generate data from Lorenz attractor"""
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return x_dot, y_dot, z_dot

    dt = 0.01
    x, y, z = 1, 1, 1
    data = []
    for i in range(n_samples):
        dx, dy, dz = lorenz(x, y, z)
        x = x + dx * dt
        y = y + dy * dt
        z = z + dz * dt
        data.append(x)
    
    X = np.array([data[i:i+sequence_length] for i in range(len(data) - sequence_length)])
    y = np.array([data[i+sequence_length] for i in range(len(data) - sequence_length)])
    
    return torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1)

def generate_mixed_frequency_data(n_samples=1000, sequence_length=20):
    """Generate data with multiple frequency components"""
    t = np.linspace(0, 8*np.pi, n_samples)
    
    data = (np.sin(t) + 
            0.5 * np.sin(3*t) + 
            0.25 * np.sin(7*t)) * (1 + 0.5 * np.sin(0.5*t))
    
    X = np.array([data[i:i+sequence_length] for i in range(len(data) - sequence_length)])
    y = np.array([data[i+sequence_length] for i in range(len(data) - sequence_length)])
    
    return torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1)

def generate_memory_data(n_samples=1000, sequence_length=20, pattern_length=5):
    """Generate data where future values depend on patterns from earlier in sequence"""
    data = []
    patterns = []
    
    for i in range(n_samples):
        if i % pattern_length == 0:
            pattern = np.random.choice([-1, 1], size=pattern_length)
            patterns.append(pattern)
        
        if i >= pattern_length:
            data.append(patterns[-2][i % pattern_length])
        else:
            data.append(0)
    
    X = np.array([data[i:i+sequence_length] for i in range(len(data) - sequence_length)])
    y = np.array([data[i+sequence_length] for i in range(len(data) - sequence_length)])
    
    return torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1)

def generate_language_data(sequence_length=20, vocab_size=27, n_samples=1000):
    """Generate simple language-like sequences"""
    words = ['the', 'cat', 'dog', 'sat', 'ran', 'jumped', 'on', 'mat', 'bed']
    
    sentences = []
    for _ in range(n_samples):
        if np.random.random() < 0.5:
            sentence = 'the ' + np.random.choice(['cat', 'dog']) + ' ' + \
                      np.random.choice(['sat', 'ran', 'jumped']) + ' on the ' + \
                      np.random.choice(['mat', 'bed'])
        else:
            sentence = np.random.choice(['cat', 'dog']) + ' ' + \
                      np.random.choice(['sat', 'ran', 'jumped']) + ' on the ' + \
                      np.random.choice(['mat', 'bed'])
        sentences.append(sentence)
    
    char_to_idx = {' ': 0}
    for c in 'abcdefghijklmnopqrstuvwxyz':
        char_to_idx[c] = len(char_to_idx)
    
    X = []
    y = []
    
    for sentence in sentences:
        chars = [char_to_idx[c] for c in sentence]
        for i in range(len(chars) - sequence_length):
            X.append(chars[i:i+sequence_length])
            y.append(chars[i+sequence_length])
    
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).reshape(-1, 1)
    
    return X, y