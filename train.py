import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from models.models import PresidentialModel
from models.tokenizer import TokenMapper
from datasets.datasets import PresidentialDataset
from tqdm import tqdm

# Load global configuration
CONFIG = {
    'pretrained_model_name': 'meta-llama/Llama-3.2-1B',
    'batch_size': 2,
    'seq_len': 10,
    'mask_prob': 0.15,
    'epochs': 10,
    'learning_rate': 0.001,
    'n_transformers': 1,
    'hidden_size': 1024,
    'n_heads': 8,
    'attn_dropout': 0.1,
    'n_mlp_layers': 2
}


# Training loop
def train():
    config = CONFIG
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize tokenizer and dataset
    token_mapper = TokenMapper.load('token_mapper.pth')
    # token_mapper = TokenMapper.init_from_dataset(PresidentialDataset.get_text(), CONFIG['pretrained_model_name'])
    # token_mapper.save('token_mapper.pth')

    dataset = PresidentialDataset(
        tokenizer=token_mapper,
        dataset=PresidentialDataset.get_text(),
        seq_len=CONFIG['seq_len'],
        mask_prob=CONFIG['mask_prob']
    )

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Initialize model
    model = PresidentialModel(config, tokenizer=token_mapper)
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    best_loss = float('inf')

    model.train()
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        total_loss = 0
        num_batches = len(dataloader)
        for i, (input_seq, labels) in tqdm(enumerate(dataloader), total=num_batches):
            input_seq, labels = input_seq.to(device), labels.to(device)

            batch_size, seq_len = input_seq.size(0), input_seq.size(1)  # Assuming input_seq is of shape [batch_size, seq_len]
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
            causal_mask = (1.0 - causal_mask) * -1e9  # Convert to 0 for unmasked, -1e9 for masked
            causal_mask = torch.tile(causal_mask.unsqueeze(0).unsqueeze(0), (batch_size, 1, 1, 1)) 


            # Forward pass
            outputs = model(input_seq, causal_mask)
            outputs = outputs.reshape(-1, token_mapper.vocab_size)
            loss = criterion(outputs, labels.reshape(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {total_loss/len(dataloader):.4f}")
        if total_loss < best_loss:
            best_loss = total_loss
            model.save('best_model.pth')

if __name__ == '__main__':
    train()
