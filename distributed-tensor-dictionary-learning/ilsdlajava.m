function Dict = ilsdlajava(noIt, N, K, X, Dict, s)

    jD0 = mpv2.SimpleMatrix(Dict);
    jDicLea  = mpv2.DictionaryLearning(jD0, 1);
    jDicLea.setORMP(int32(s), 1e-6, 1e-6);
    jDicLea.ilsdla( X(:), noIt );
    jD =  jDicLea.getDictionary();
    Dict = reshape(jD.getAll(), N, K);
    
    return;