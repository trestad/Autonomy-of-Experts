Since the paper utilized Tencent's internal framework for training, I spent some time extracting the key algorithmic components and reimplemented an AoE class.

This implementation focuses on preserving the core mechanisms of the original algorithm while removing parts tightly coupled with the internal framework, ensuring it is allowed to be opened and easy to import. 
