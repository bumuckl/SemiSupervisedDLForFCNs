function Eval = EvalStruct()
%EVALSTRUCT Initialize an empty Eval struct
%
% @Author: Christoph Baur

    Eval = struct;
    Eval.TP = 0;
    Eval.FP = 0;
    Eval.FN = 0;
    Eval.TN = 0;
    Eval.P = 0;
    Eval.N = 0;
    Eval.Score = [];
    Eval.Target = [];

end

