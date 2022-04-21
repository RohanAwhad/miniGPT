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
