// 우상이형



def FedAvg_FoolsGold(w, w_glob, losses, m, lr):
    device = torch.device('cuda:{}'.format(3) if torch.cuda.is_available() else 'cpu')
    #initialize variables
    w_tmp = copy.deepcopy(w)
    w_avg = copy.deepcopy(w_glob)
    w_init = copy.deepcopy(w[0])
    n_clients = m
    maxcs = {} 
    wv = {}
    
    #caculate Cosign similarity
    cs = CosSim_with_key(w_tmp)
    #print(cs)
    for k in w_init.keys():
        #cs[k] = cs[k] - np.eye(n_clients) 
        maxcs[k] = np.max(cs[k], axis = 1)

    
    
    for k in w_init.keys():
        for i in range(n_clients):
            for j in range(n_clients):
                if i==j:
                    continue
                if maxcs[k][i] < maxcs[k][j]:
                        
                    cs[k][i][j] = cs[k][i][j]*maxcs[k][i] / maxcs[k][j]
    
    
    for k in w_init.keys():
        wv[k] = 1. - (np.max(cs[k], axis = 1))
        
    for k in w_init.keys():
        wv[k][wv[k]>1] = 1.
        wv[k][wv[k]<0] = 0.
        alpha = np.max(cs[k], axis =1)
        tmp = np.max(wv[k])
        if tmp == 0.:
            tmp = .01
        else: 
            continue
        wv[k] = wv[k]/ tmp  
        wv[k][(wv[k]==1.)] = .99
        wv[k] = (np.log(wv[k]/(1.-wv[k])) + 0.5)
        wv[k][((np.isinf(wv[k]) + wv[k]) > 1)] = 1.
        wv[k][(wv[k]<0)] = 0.
    #print(w_avg)
        
    for k in w_init.keys():
        WV = torch.from_numpy(wv[k]).to(device)
        for i in range(m):    
            w_avg[k] = w_avg[k].double() + ((w_tmp[i][k]-w_glob[k]).double()) * WV[i].double()
    #print(w_avg)    
    return w_avg, losses