import random
import numpy as np

# Parameters
num_sequences = 1000  # Total number of sequences to generate
initial_seq_random_ratio = 0.7  # Initial fraction of sequences vs. random tokens
context_window_length = 512  # Length of the context window
vocabulary_size = 50000  # Increase the size of the vocabulary
training_steps = 10000  # Total number of training steps

# Simulate a larger vocabulary of tokens
vocabulary = ["token_{}".format(i) for i in range(1, vocabulary_size + 1)]

# Generate predefined sequences with variable lengths
sequences = [" ".join(random.choices(vocabulary, k=random.randint(5, 50))) for _ in range(num_sequences)]

def adjust_ratio(step, total_steps, initial_ratio, final_ratio=0.9):
    """Dynamically adjust the seq_random_ratio from initial_ratio to final_ratio over training steps."""
    progress = step / total_steps
    return initial_ratio + (final_ratio - initial_ratio) * progress

def create_context_window(seq_random_ratio, context_window_length, sequences, vocabulary):
    num_seq_tokens = int(context_window_length * seq_random_ratio)
    num_random_tokens = context_window_length - num_seq_tokens
    
    # Randomly select sequences until the number of sequence tokens is reached or exceeded
    selected_sequences = ""
    while len(selected_sequences.split()) < num_seq_tokens:
        selected_sequences += random.choice(sequences) + " "
    
    # Trim and adjust as before
    selected_sequence_tokens = selected_sequences.split()
    if len(selected_sequence_tokens) > num_seq_tokens:
        selected_sequence_tokens = selected_sequence_tokens[:num_seq_tokens]
    
    random_tokens = " ".join(random.choices(vocabulary, k=num_random_tokens))
    
    context_window = selected_sequence_tokens + random_tokens.split()
    
    if len(context_window) > context_window_length:
        context_window = context_window[:context_window_length]
    elif len(context_window) < context_window_length:
        context_window += random.choices(vocabulary, k=context_window_length - len(context_window))
    
    return " ".join(context_window)

# Example integration with a hypothetical training loop
for step in range(1, training_steps + 1):
    # Dynamically adjust the ratio
    current_ratio = adjust_ratio(step, training_steps, initial_seq_random_ratio)
    
    # Generate context window with the current ratio
    context_window = create_context_window(current_ratio, context_window_length, sequences, vocabulary)
    
    # Here, you would feed the context_window into your model as part of the training batch
    # For demonstration, we'll just print the ratio and an excerpt of the context window at certain steps
    if step % 2000 == 0:
        print(f"Step {step}, seq_random_ratio: {current_ratio:.2f}, Example tokens: {' '.join(context_window.split()[:10])}")

