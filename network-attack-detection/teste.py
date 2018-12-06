# coding=utf-8
import numpy as np
from scipy.stats import skew,kurtosis,moment


vals = np.ndarray(shape=(2,3), dtype=float, order='F')
vals[0,0] = 1
vals[0,1] = 10
vals[0,2] = 1
vals[1,0] = 1
vals[1,1] = 200
vals[1,2] = 1
print("all:", vals)
print("\n:,0:", vals[:,0])
print(":,1:", vals[:,1])
print("mean:", vals.mean())
print("mean(0):", vals.mean(0))
print("skew(axis=0,True):", skew(vals,axis=0,bias=True))
print("skew(axis=0,False):", skew(vals,axis=0,bias=False))
print("kurtosis(axis=0):", kurtosis(vals,axis=0,fisher=True,bias=True))
print("kurtosis(axis=0):", kurtosis(vals,axis=0,fisher=False,bias=True))
print("moment1(axis=0):", moment(vals,moment=1,axis=0))
print("moment2(axis=0):", moment(vals,moment=2,axis=0))
print("moment3(axis=0):", moment(vals,moment=3,axis=0))
print("moment4(axis=0):", moment(vals,moment=4,axis=0))
print("moment5(axis=0):", moment(vals,moment=5,axis=0))
print("moment6(axis=0):", moment(vals,moment=6,axis=0))


# ver clube smiles de mamae
# ver pontos do livelo