import torch.nn.functional as F
import torch
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np

class RBM (torch.nn.Module):
    
    def __init__ (_,vsize, hsize, CD_K =1): #CD_K es diferencia constractiva, me dice las vueltas que da para entrenar
        
        super().__init__()
        
        _.w = torch.nn.Parameter(torch.randn(hsize,vsize)*1e-2)
        _.bv = torch.nn.Parameter(torch.randn(vsize)*1e-2) #bias visible
        _.bh = torch.nn.Parameter(torch.randn(hsize)*1e-2)
        _.k = CD_K
        
    def sample_h (_,v): #Para estimar las ocultas le pasamos las visibles _.v
        
        prob_h = torch.sigmoid(F.linear(v,_.w,_.bh)) #Estimulo v por W y restamos bias y sigmoid
        samp_h = torch.bernoulli(prob_h) # Devuelve un vector con la misma dimension pero con valores 0 y 1
        return prob_h, samp_h
        
    def sample_v(_,h):

        prob_v = torch.sigmoid(F.linear(h,_.w.t(),_.bv)) #Estimulo v por W y restamos bias y sigmoid
        samp_v = torch.bernoulli(prob_v) # Devuelve un vector con la misma dimension pero con valores 0 y 1
        return prob_v, samp_v
        
    def free_energy (_,v): ## Defino mi loss function
        v_bv = v.mv(_.bv) ## Multiplica matriz por vector
        hlin = torch.clamp(F.linear(v,_.w, _.bh),-80,80) ## Clamp evita que me de valores muy grandes para que no diverja
        slog = hlin.exp().add(1).log().sum(1)
        return (-slog - v_bv).mean()
    
    def forward(_,v):
        vs = v
        for i in range(_.k): #Iteramos k veces (k Vueltas)
            hp, hs = _.sample_h(vs)
            vp, vs = _.sample_v(hs)
        return v,vs