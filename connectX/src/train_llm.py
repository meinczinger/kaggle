from llm_models import ModelFactory
from tqdm import tqdm
import pandas as pd
from llm_models.tools import ModelManagement, ModelParameters
import torch
import torch.nn as nn
from pathlib import Path
import wandb
from datetime import datetime

import torch._dynamo

torch._dynamo.config.suppress_errors = True

vocab_size = 8
dtype = torch.bfloat16

start_iteration = 0
train_iters = 21
eval_interval = 10
weight_decay = 0.01
lr = 3e-4
grad_clip = 1.0
eval_iters = 3
batch_size = 32

GAMES_FOLDER = Path("resources/games/")
params_path = "src/config/model_params.yaml"

checkpoint_dir = "resources/llm_models/"  # Where do we store checkpoints?

checkpoint_fn = "best_model.pt"
# Name of checkpoint file to be saved during training

checkpoint_load_fn = "best_model.pt"

encode = lambda s: s
decode = lambda l: l

wandb_log = True
wandb_project = "connectx"
wandb_run_name = "basic-gpt" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

if wandb_log:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name)


# Return a batch of either training or evaluation data
def get_batch(data, context, batch_size, device):
    # BS = Batch Size / SL = Sequence Length or context length
    inds = torch.randint(len(data) - context, (batch_size,))  # (BS)
    x = torch.stack([torch.tensor(data[i][:context]) for i in inds])  # (BS,SL)
    y = torch.stack([torch.tensor([data[i][1:]]) for i in inds])  # (BS,SL)

    # Examples of what it returns
    # # First 10 elements of first batch of inputs and labels
    # x[0][:10] -> tensor([ 664,  278, 4031, 4056, 4065, 4062, 4062, 4051, 13, 13])
    # y[0][:10] -> tensor([ 278, 4031, 4056, 4065, 4062, 4062, 4051,   13, 13, 4066])

    x, y = x.to(device), y.to(device)
    return x, y


# Calculate the Loss
@torch.no_grad()  # Prevent gradient calculation
def calculate_loss(train_data, val_data, context, batch_size, device):
    out = {}
    model.eval()
    # for split in ["train", "eval"]:
    l = torch.zeros(eval_iters)  # Create a tensor of zeros the size of eval_iters
    for i in range(eval_iters):
        x, y = get_batch(
            train_data, context, batch_size, device
        )  # Get a new batch of data
        _, loss = model(x, y)  # Calculate the loss
        l[i] = loss  # Store the loss in the next position of tensor
    out["train"] = l.mean().item()  # Calculate the mean and extract the final value

    l = torch.zeros(eval_iters)  # Create a tensor of zeros the size of eval_iters
    for i in range(eval_iters):
        x, y = get_batch(
            val_data, context, batch_size, device
        )  # Get a new batch of data
        _, loss = model(x, y)  # Calculate the loss
        l[i] = loss  # Store the loss in the next position of tensor
    out["eval"] = l.mean().item()  # Calculate the mean and extract the final value

    model.train()
    return out


# Generate a new sample
@torch.no_grad()
def generate_sample(input, model, params):
    t1 = torch.tensor(
        encode(input), dtype=torch.long, device=params.params["device"]
    )  # Tokenize string -> (tensor of ids)
    t1 = t1[None, :]  # (1 , [size of ids])
    newgen = model.generate(t1, max=1)[
        0
    ].tolist()  # call the generate method, limit output size
    result = decode(
        newgen
    )  # decode the result with the tokenizer to get back characters
    print(f"{result}")


def get_train_val_data(games_path: Path, params):
    games_file = "train_actions.csv"
    with open(games_path / games_file) as f:
        lines = f.readlines()
    lines = [[int(a) for a in l.strip().split(",")] for l in lines]
    data = []
    context = params["context"]
    for l in lines:
        reward = l[-1]
        player = l[-2]
        for i in range(
            1 if reward == 0 or player == 1 else 0, len(l) - 1, 1 if reward == 0 else 2
        ):
            actions = [0 for _ in range(context - i)] + l[0 : (i + 1)]
            data.append(actions)
            # print(actions)
        # print(f"Actions: {actions}, player: {player}, reward: {reward}")
        # train_data.append({"actions": actions, "player": player, "reward": reward})

    data_size = len(data)  # Get the size of the dataset

    spl = int(0.9 * data_size)  # set the split at 90%-10%
    return data[:spl], data[spl:]


def create_opt_sched():
    # Set Weight Decay differently for different kinds of parameters
    # parameter dictionary where keys are parameter names, and values are the parameter themselves
    p_dict = {
        p_name: p for p_name, p in model.named_parameters() if p.requires_grad
    }  # len: 370

    # isolate weight matrices as they benefit specially from weight decay
    weight_decay_p = [p for n, p in p_dict.items() if p.dim() >= 2]  # len: 171

    # isolate other parameters like bias parameters, that don't benefit from weight decay
    no_weight_decay_p = [p for n, p in p_dict.items() if p.dim() < 2]  # len: 199

    # store the parameter types in a list of dictionaries
    optimizer_groups = [
        {"params": weight_decay_p, "weight_decay": weight_decay},
        {"params": no_weight_decay_p, "weight_decay": 0.0},
    ]

    # Declare optimizer, it helps us compute gradients, update parameters, manage learning rate, apply weight decay
    optimizer = torch.optim.AdamW(optimizer_groups, lr=lr, betas=(0.9, 0.99))
    # betas: control the exponential moving averages of the gradient and its square,
    # which are essential components of the Adam and AdamW optimization algorithms.

    # Declare scheduler to change learning rate through the training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, train_iters, eta_min=lr / 10
    )
    # learning rate will descend till a minimum of a tenth of the lr

    return optimizer, scheduler


def train(optimizer, scheduler, train_data, val_data, context, batch_size, device):
    try:
        start_iteration = 0
        best_val_loss = float("inf")  # Track best loss value

        for i in tqdm(range(start_iteration, train_iters)):
            xb, yb = get_batch(
                train_data, context, batch_size, device
            )  # Get a new batch of data
            logits, loss = model(xb, yb)  # Run the LLM and get the logits and the loss

            if i % eval_interval == 0 or i == train_iters - 1:  # Calculate the loss
                l = calculate_loss(train_data, val_data, context, batch_size, device)
                print(f"\n{i}: train loss: {l['train']} / val loss: {l['eval']}")

                # We do a quick test so that we observe the evolution through the training
                # Remember that we use a very small dataset which doesn't include all topics
                # generate_sample("The mountain in my city is")  # Generate a sample

                if (
                    l["eval"] < best_val_loss
                ):  # If we improved the best loss, save a checkpoint
                    best_val_loss = l["eval"]
                    print("[CHECKPOINT]: Saving with loss: ", best_val_loss)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": best_val_loss,
                            "iteration": i,
                        },
                        checkpoint_dir + checkpoint_fn,
                    )

            if wandb_log:
                wandb.log(
                    {
                        "loss/train": l["train"],
                        "loss/val": l["eval"],
                        "lr": scheduler.get_last_lr()[0],
                    },
                    step=i,
                )

            optimizer.zero_grad(set_to_none=True)  # Reset gradients
            loss.backward()  # Calculate new gradients

            # This line clips the gradients to prevent the exploding gradient problem during training.
            # Exploding gradients can occur when gradients become too large, causing unstable updates to model weights.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()  # Update the model parameters
            scheduler.step()  # Update the learning rate value

        if wandb_log:
            wandb.finish()

    except KeyboardInterrupt:
        print("Training interrupted. Cleaning up...")

    finally:
        # Release GPU memory
        torch.mps.empty_cache()
        print("GPU memory released.")

    if wandb_log:
        wandb.finish()
    torch.mps.empty_cache()


def load_checkpoint(path, model, optimizer):
    print("LLM - Loading model")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])  # Load parameters
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )  # Load optimizer state
    iteration = checkpoint["iteration"]  # In what iteration did we save the model?
    loss = checkpoint["loss"]  # What was the last loss value?
    print(f"Loaded iter {iteration} with loss {loss}")
    return model, optimizer


params = ModelParameters(params_path)
model = ModelFactory("basic_gpt", params, vocab_size)

# model = GPT() # Instantiate LLM
model = model.to(dtype)  # Set the precision type
model = model.to(params.params["device"])  # Move it to the right device

# Torch.compile compiles a PyTorch model to an optimized version, aiming to improve runtime performance and efficiency.
# Disable if your system doesn't support it
if compile:
    print("Torch :: Compiling model")
    model = torch.compile(model)


# Print the number of parameters of our model (19 million in our case)
print(sum(p.numel() for p in model.parameters()) / 1e6, " Million parameters")

# generate_sample([0, 0, 0, 0, 0], model, params)

train_data, val_data = get_train_val_data(GAMES_FOLDER, params.params)

# print(train_data[0:1])

# print(get_batch(train_data, params.params["context"], 1, params.params["device"]))

opt, sched = create_opt_sched()

# train(
#     opt,
#     sched,
#     train_data,
#     val_data,
#     params.params["context"],
#     batch_size,
#     params.params["device"],
# )

model, opt = load_checkpoint(checkpoint_dir + checkpoint_load_fn, model, opt)

model.eval()

# print(generate_sample([0 for _ in range(41)] + [4], model, params))

t1 = torch.tensor(
    [0 for _ in range(41)] + [4], dtype=torch.long, device=params.params["device"]
)  # Tokenize string -> (tensor of ids)
t1 = t1[None, :]  # (1 , [size of ids])

newgen = model.generate(t1, max=1)[0].tolist()

probs = model.get_next_probs(t1)
print(probs[0].cpu().detach()[0].item())
