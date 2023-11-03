def FedAvg_FLAME(w, w_glob, losses, num_users, m, num_comps, comps_list):

   
    #Cosine distance for local weights
    score = CosSim(w)
    score = 1. - score
    weight = copy.deepcopy(w)
    weight_glob = copy.deepcopy(w_glob)
    loss = copy.deepcopy(losses)

    # To 2-norm 
    pca = PCA(2)
    transform = pca.fit_transform(score)
 
    # HDBSCAN model initailizing and clustering and Filfering? (-> cluster_selection_epsilon= 0.5)
    clustering = hdbscan.HDBSCAN(min_cluster_size=int((num_users/2) + 1),allow_single_cluster=True,  min_samples=1, gen_min_span_tree=True )
    clustering.fit(transform)
    labels = clustering.labels_

    # get cluster label and Outlier label // filtering by HDBSCAN
    cluster_labels = clustering.labels_
    outlier_scores_ = clustering.outlier_scores_


    print("Results are :")
    print(cluster_labels)

    # Get the indices of outliers and inliers based on cluster_labels
    outlier_indices = np.where(cluster_labels == -1)
    inlier_indices = np.where(cluster_labels != -1)[0]
    w_inline = [weight[idx] for idx in inlier_indices]
    loss_inline = [loss[idx] for idx in inlier_indices]

    #weight euclidian distance and S_t // clipping
    tmp = copy.deepcopy(weight[0])
    total_norm = [0. for i in range(len(weight))]

    for k in tmp.keys(): ## for each key  

        for i in range(len(weight)):
          w_tmp = weight[i][k].cpu().numpy()
          w_glob_tmp = weight_glob[k].cpu().numpy()
          norm_w = np.linalg.norm(w_tmp - w_glob_tmp)
          total_norm[i] += norm_w
          
    norm_tmp = [total_norm[idx] for idx in inlier_indices]
    S_t = np.median(total_norm)
    gamma = [0. for i in range(len(w_inline))]
    W_C = [weight_glob for i in range(len(w_inline))]
    
    for idx in range(len(w_inline)):
        #print(norm_tmp[idx])
        gamma[idx] = S_t/norm_tmp[idx]
    
    for k in tmp.keys():
        for idx in range(len(w_inline)):
            W_C[idx][k] = weight_glob[k] + (w_inline[idx][k]-weight_glob[k])*min(1,gamma[idx])        
   
    
    #BA
    #predict_backdoors = 0 
    #BA_count =  sum(1 for element in cluster_labels if element in comps_lables)
    #for i in len(comps_list):
     # if cluster_labels[comps_list[i]  
    
      #  predict_backdoors += 1  
    
   # print("Backdoor Accuracy (BA): ",  predict_backdoors/len(comps_list) )
    #noising // Adaptive Noising
    noise_eps = 0.1 # privacy tolerance
    noise_delta = 0.05 # tnoise distribution control paramete
    noise_lambda = (1/noise_eps)* math.sqrt(2 * math.log(1.25/noise_delta) )
    noise_level = S_t*noise_lambda


    # Avg
    w_avg = copy.deepcopy(w[0])
    l = len(W_C)
    for k in w_avg.keys():
        for i in range(l):
            if i==0:
                w_avg = copy.deepcopy(W_C[i])
            #print("Correct :", sorted_idx_score[i])
            else:
                w_avg[k] += W_C[i][k]
        #print()
        w_avg[k] = torch.div(w_avg[k], l) + noise_level

 
    return w_avg, losses