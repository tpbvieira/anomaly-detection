function Dict = rlsdla(L, noIt, N, K, X, metPar, Dict, s)

    jD0 = mpv2.SimpleMatrix(Dict);
    jDicLea  = mpv2.DictionaryLearning(jD0, 1);
    jDicLea.setORMP(int32(s), 1e-6, 1e-6);
    jDicLea.setLambda( metPar{1}.lamM, metPar{1}.lam0, 1.0, (noIt*L)*metPar{1}.a );
    jDicLea.rlsdla( X(:), noIt );
    jD =  jDicLea.getDictionary();
    Dict = reshape(jD.getAll(), N, K);
    
    return;