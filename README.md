# machine_learning_a4
L0, L1, and L2-Regularizations. Softmax classification. Cost of Multi-Nomial logistic regression.

Link to [solution](doc/a4ans.pdf)

2. Code included for l2/l1/l0-regularization: [code](code/linear_model.py)
3. Code included for one-vs-all/softmax: [code](code/linear_model.py)

```
data = utils.load_dataset("logisticData")
XBin, yBin = data['X'], data['y']
XBinValid, yBinValid = data['Xvalid'], data['yvalid']

model = linear_model.logRegL2(lammy=1.0, maxEvals=400)
model.fit(XBin,yBin)
```
