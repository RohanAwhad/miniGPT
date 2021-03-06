This is a CopyWork exercise of MinGPT from Andrej Karpathy.
Original Repo: https://github.com/karpathy/minGPT

## 18th April, 2022

Started with model.py

-   The start of the file contained a model summary and explanation. Very Brief
-   register_buffers:
    If you have parameters in your model, which should be saved and restored in the state_dict, but not trained by the optimizer, you should register them as buffers.Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
-   torch.tril: Returns the lower triangle of the matrix

## 19th April, 2022

Continuing the model.py

-   torch.nn.parameter.Parameter:
    Are tensor subclasses, whose weights we can define.
-   torch.nn.Module.apply: Applies function to every submodule as well as self
-   torch.numel: Returns number of elements in the input
-   Configuring optimizers to take into account decaying of certain parameters
    only and not all the parameters

## 21st April, 2022

Moving to the trainer.py

-   torch.nn.DataParallel: Parallelization across multiple devices. Stores the
    model in `module`. Parallelizes across batch dimension. Recommended to use
    torch.nn.parallel.DistributedDataParallel. I will be updating mine to the
    recommended one.
-   DataLoader(pin_memory): pin_memory first copies the Tensors into CUDA pinned
    memory. This speeds up the host-to-device data transfers
-   CosineLearningRateDecay: A cyclical LR scheduler based on cosine function

## 25th April, 2022

Now on to the utils.py and play_char.ipynb

-   torch.topk: returns top k items and their indices
-   temperature: Changes the output distribution. The logits computed from the
    NN are divided by temperate `T`. This changes the output distribution,
    making it either hard distribution (model is very confident) or soft
    distribution (model is less confident). If `T == 1` probabilities remain the
    same. If your model is extremely confident, it may produce very repetitive
    and uninteresting text.
