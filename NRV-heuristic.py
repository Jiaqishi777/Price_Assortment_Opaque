#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import itertools
import sys
import pandas as pd
from scipy.optimize import minimize_scalar
import time


# In[2]:


def prob_trad( v, p):
  assert len(v) == len(p)
  p0 = np.r_[np.array(p), 0]
  v0 = np.r_[np.array(v), 0]
  w = np.exp(v0 - p0)
  return w/np.sum(w)     

def revenue_trad(v, p):
  probs = prob_trad(v, p)
  return np.sum(p * probs[:-1])


def prob_opq(v, p, rho):
    p = np.array(p)
    v = np.array(v)
    rho = np.nan_to_num(rho)
    assert rho <= np.min(p)
    asst_size = int(len(v))
    index_sets = itertools.product([0, 1], repeat=asst_size)
    probs_opa = np.zeros(int(len(v)) + 2)
    for index_set in index_sets:
        if not np.any(index_set):
            continue
        index_arr = np.array(index_set)
        set_length = np.sum(index_arr)
        p0 = p * (1 - index_arr) + rho * index_arr
        probs = prob_trad(v, p0)
        index_arr = np.r_[index_arr, 0]
        probs_opa[1:] += (-1) ** (set_length % 2 + 1) * probs * (1 - index_arr)
        probs_opa[0] += (-1) ** (set_length % 2 + 1) * np.sum(probs * index_arr)
    return probs_opa

def revenue_opq(v, p, rho):
    probs = prob_opq(v, p, rho)
    return np.sum(np.r_[rho, p] * probs[:-1])



def find_optimal_opaque_price(v, p):
    trad_rev = revenue_trad(v, p)
    ub = np.min(p)
    lb = 0
    objective_function = lambda x: -revenue_opq(v, p, x)
    result = minimize_scalar(objective_function, bounds=(lb, ub), method='bounded')
    opt = -result.fun
    if opt<=trad_rev:
        return ub
    else:
        return result.x
#         if abs(result.x - ub)<1e-4:
#           return ub
#         else:
#           return np.round(result.x,5)  


# In[3]:


def NRV(v, p): 
    best_asst = None
    best_opq_price = 0
    best_revenue = 0
  
    for p_thresh in p:
        for v_thresh in v:
            asst = np.array( (p >=p_thresh) & (v >= v_thresh), dtype=float)
            asst1 = asst.astype(bool)
            if np.sum(asst) == 0:
                continue
            v1 = v[asst1]
            p1 = p[asst1]    
            rho = find_optimal_opaque_price(v1, p1)
            rev = revenue_opq(v1, p1, rho)
            if rev > best_revenue:
                best_revenue = rev
                best_asst = np.copy(asst)
                best_opq_price = rho
    return {
                    "rev": best_revenue,
                    "asst": best_asst,
                    "rho": best_opq_price
                }

def brute_force(v, p):
    assert len(v) == len(p)
    best_asst = None
    best_opq_price = 0
    best_revenue = 0
    for asst in itertools.product([0, 1], repeat=len(v)):
        if not any(asst):
            continue
        asst1 = np.array(asst, dtype=bool)
        v1 = v[asst1]
        p1 = p[asst1]
        rho = find_optimal_opaque_price(v1, p1)
        rev = revenue_opq(v1, p1, rho)
        if rev > best_revenue:
            best_revenue = rev
            best_asst = np.copy(asst)
            best_opq_price = rho
    return {
        "rev": best_revenue,
        "asst": best_asst,
        "rho": best_opq_price
    }     

def brute_force_iterative(v, p, v0, p0):
    assert len(v) == len(p)
    best_asst = None
    best_opq_price = 0
    best_revenue = 0
    for asst in itertools.product([0, 1], repeat=len(v)):
        asst1 = np.array(asst, dtype=bool)
        v1 = np.r_[v[asst1], v0]
        p1 = np.r_[p[asst1], p0]
        rho = find_optimal_opaque_price(v1, p1)
        rev = revenue_opq(v1, p1, rho)
        if rev > best_revenue:
            best_revenue = rev
            best_asst = np.copy(np.r_[asst,1])
            best_opq_price = rho
    return {
        "rev": best_revenue,
        "asst": best_asst,
        "rho": best_opq_price
    }  


# In[4]:


def get_instances(size, n_inst):
    vs = np.random.lognormal(0, 0.3, size=(n_inst, size))
    rs = np.random.lognormal(0.5, 1.5, size=(n_inst, size))
    while np.any(rs > 300):
        rs = np.random.lognormal(0.5, 1.5, size=(n_inst, size))
    instances = [(vs[k, :], rs[k, :]) for k in range(n_inst)]
    return instances


# In[ ]:


alg_names = ["NRV","brute force"]
num_instances = 2000
result_df = pd.DataFrame(
        columns=["instance", "n", "v" ,"r", "alg", "rev", "asst", "size", "r_min", "opq_price"])
N = 9
np.random.seed(123 + N * 9)
instances_total = get_instances(N, num_instances)
# initial case
instances = [(vs[:2], rs[:2]) for vs, rs in instances_total]
instance_df = pd.DataFrame(columns=["instance", "n", "v", "r"])
start = time.time()
for j, (v, p) in enumerate(instances):
    inst_num = 2 * 1e4 + j 
    instance_df.loc[j] = [inst_num, 2, v, p]
    res = []
    res.append(NRV(v, p))
    res.append(brute_force(v, p))
    for i in range(len(alg_names)):
      alg = alg_names[i]
      result_df.loc[2*j + i] = [inst_num, 2, v, p, alg, res[i]["rev"], res[i]["asst"].astype(int), np.sum(res[i]["asst"]),np.min(p[np.array(res[i]["asst"], dtype=bool)]), res[i]["rho"] ]
end = time.time()
print(2,end -start)

for n in range(3,N+1):
  start = time.time()
  instances = [(vs[:n], rs[:n]) for vs, rs in instances_total]
  instance_df = pd.DataFrame(columns=["instance", "n", "v", "r"])
  for j, (v, p) in enumerate(instances):
    inst_num = n * 1e4 + j 
    instance_df.loc[j] = [inst_num, n, v, p]
    res = []
    res.append(NRV(v, p))
    res1 = brute_force_iterative(v[:-1], p[:-1], v[-1], p[-1])
    index = 2*j + 1 + num_instances*(n-3)*len(alg_names)
    assert result_df["v"][index][-1] == v[-2]
    if res1["rev"] > result_df["rev"][index]:
      res.append(res1)
    else:
      prev = {"rev": result_df["rev"][index], "asst": np.r_[result_df["asst"][index],0], "rho": result_df["opq_price"][index] }
      res.append(prev)  
    for i in range(len(alg_names)):
      alg = alg_names[i]
      result_df.loc[2*j + i + num_instances*(n-2)*len(alg_names) ] = [inst_num, n, v, p, alg, res[i]["rev"], res[i]["asst"].astype(int), np.sum(res[i]["asst"]), np.min(p[res[i]["asst"].astype(bool)]), res[i]["rho"] ]
  
  print(n,time.time() - start)


# In[ ]:


table = pd.DataFrame(
        columns=["n", "#w/ opaque", "# subopt", "max opt gap", "avg opt gap", "avg size"])
for n in range(2,10):
    num_opa = 0
    subopt = 0
    gap = np.zeros(num_instances)
    size = np.zeros(num_instances)
    
    for i in range(2000*(n-2),2000*(n-1)):
        if (result_df['r_min'][2*i]- result_df['opq_price'][2*i]) > 1e-6 :
            num_opa += 1
        if abs(result_df['rev'][2*i]- result_df['rev'][2*i+1])>1e-8:
            if (result_df['asst'][2*i] - result_df['asst'][2*i+1]).any() !=0:
                subopt +=1
        gap[i-2000*(n-2)] = result_df['rev'][2*i]/result_df['rev'][2*i+1]
        size[i-2000*(n-2)] = result_df['size'][2*i+1]
    print(n,np.max(gap),np.min(gap),np.mean(gap))
    row_df = pd.DataFrame({ "n": n, "#w/ opaque": num_opa, "# subopt": subopt, "max opt gap": 1 - np.min(gap),
                           "avg opt gap": 1- np.mean(gap), "avg size": np.mean(size)}, index=[0])
    table = pd.concat([table, row_df], ignore_index=True)
table    


# In[ ]:




