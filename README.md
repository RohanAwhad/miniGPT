This is a CopyWork exercise of MinGPT from Andrej Karpathy.

## 18th April, 2022

Started with model.py

-   The start of the file contained a model summary and explanation. Very Brief
-   register_buffers:
    If you have parameters in your model, which should be saved and restored in the state_dict, but not trained by the optimizer, you should register them as buffers.Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
-   torch.tril: Returns the lower triangle of the matrix
