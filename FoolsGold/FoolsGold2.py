/// 김교수님


def FedAvg_FoolsGold(w, losses, num_users, m, num_comps): 
    # score is the result for this function
    score = np.array([[0. for i in range(len(w))]
                     for j in range(len(w))])  # n by n matrix
    w_avg = copy.deepcopy(w[0])  # To use keys in dict in w
    norm_total = [0. for i in range(len(w))]  # Denominator for CosSim

    tmp = copy.deepcopy(w)  # To compute CosSim
    for k in w_avg.keys():  # for each key
        for i in range(len(w)):  # for each local models
            ## Compute Denominator ##
            A = tmp[i][k].cpu().numpy()
            tmp_A = np.linalg.norm(A, ord=None)  # l2-norm
            tmp_A = tmp_A ** 2
            norm_total[i] += tmp_A
            flat_A = A.flatten()

            ## Compute numerator ##
            for j in range(len(w)):
                if j < i or j == i:  # because score is symmetric matrix
                    continue
                else:
                    B = tmp[j][k].cpu().numpy()
                    flat_B = B.flatten()
                    # Insert each numerator value into score matrix first
                    score[i][j] += np.dot(flat_A, flat_B)

    # Make symmetric matrix
    # Because we cannot compute for j < i for better computational complexity(De-duplicating for computing)
    score += score.T - np.diag(score.diagonal())
    norm_total = np.sqrt(norm_total)

    for i in range(len(w)):
        for j in range(len(w)):
            if i == j:
                score[i][j] = 0.
            else:
                score[i][j] = score[i][j] / (norm_total[i] * norm_total[j])

    print(score)

    epsilon = 1e-5
    n = len(w)
    d= 784

    #cs = smp.cosine_similarity(w) - np.eye(10)
    cs = score
    # Pardoning: reweight by the max value seen
    maxcs = np.max(cs, axis=1) + epsilon
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)

    wv[(wv == 1)] = 0.99

    # Logit function
    wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)

    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0
    print(w)

    print(w_avg)
    w_to_update= w_avg['layer_input.weight'].detach().cpu().numpy()

    # Apply the weight vector on this delta
    delta = np.reshape(w_to_update, (n, d))

    return np.dot(delta.T, wv), loss