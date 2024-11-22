using Plots 

gr(size = (600, 600))

# plot logistic (sigmoid) curve 

logistic(x) = 1 /  (1 + exp(-x))

p_log = plot(-6:0.1:6, logistic, 
             xlabel = "Inputs (x)", 
             ylabel = "Output (y)", 
             title = "Logistic (Sigmoid) Curve",
             legend = false, 
             linewidth = 2, 
             color = :blue) 