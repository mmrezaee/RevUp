<!-- ![alt-text-1]("title-1") ![alt-text-2](image2.png "title-2") -->
The implementation of EACL 2023 paper 
["RevUp: Revise and Update Information Bottleneck for Event Representation".](https://arxiv.org/pdf/2010.04361.pdf)

<img src="figs/RevUpModel.png" width="600" height="200"/>

This Pytorch code implements the model and reproduces the results from the paper.
# Conda Environment:

```
conda create --name revup
conda install cudatoolkit=10.2 -c pytorch
pip install torchtext==0.2.3
pip install jsonlines
pip install wandb
conda install -c conda-forge pytablewriter
conda install pandas scikit-learn
```


# Training:
```
./train.sh 
```

Some parts of the code were inspired by [HAQAE](https://github.com/StonyBrookNLP/HAQAE) implementations.

