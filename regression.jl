using CSV, GLM, Plots, TypedTables

data  = CSV.File("housingdata.csv")

X = data.size 

Y = round.(Int, data.price / 1000) 

t = Table(X = X, Y = Y )

gr(size = (600, 600))

p_scatter = scatter(X, Y, 
                    xlims = (0, 5000), 
                    ylims = (0,  800), 
                    xlabel = "Size (sqft)", 
                    ylabel = "Price in 1000 dollars", 
                    title = "Housing Prices in Portland", 
                    legend = false, 
                    color = :red)

ols = lm(@formula(Y ~ X), t)

plot!(X, predict(ols), color = :green, linewidth  = 3 )

newX = Table(X = [1250])

predict(ols, newX)
println(t)


# Machine learning for scratch 

epochs = 0 

gr(size = (600, 600))

p_scatter = scatter(X, Y, 
                    xlims = (0, 5000), 
                    ylims = (0,  800), 
                    xlabel = "Size (sqft)", 
                    ylabel = "Price in 1000 dollars", 
                    title = "Housing Prices in Portland (epochs = $epochs)", 
                    legend = false, 
                    color = :red)

## initialization of the parameters of teh most basci liear regression 
theta0 = 0.0   # y-intercept 
theta1 = 0.0   #  slope

h(x) = theta0 .+ theta1 * x
plot!(X, h(X), color = :blue, linewidth = 3)

# using the cost function as teh mean-squared-error loss function 
m = length(X) 

y_hat = h(X)

function cost(X, Y)
    (1 / (2 * m))  * sum((y_hat - Y) .^ 2)
end 

J = cost(X, Y)

# push coast function value for history of the model 

J_history = []

function pd_thetha0(X, Y)
    (1/m) * sum(y_hat - Y)
end

function pd_thetha1(X, Y)
    (1/m) * sum((y_hat - Y) .* X)
end

# set learning rate (alpha)

alpha0 = 0.09 
alpha1 = 0.00000008

##########################################################
########## Begining the iterative training process #######
##########################################################
theta0_temp = pd_thetha0(X, Y)
theta1_temp = pd_thetha1(X, Y)

theta0 -= alpha0 * theta0_temp
theta1 -= alpha1 * theta1_temp

y_hat = h(X)
J = cost(X, Y)
push!(J_history, J )


epochs += 1 

plot!(X, y_hat, color = :blue, alpha = 0.5,
     title = "Housing Prices in Portland (epochs: $epochs)")

plot!(X, predict(ols), color = :green, linewidth = 3)

gr(size = (600, 600))

pline = plot(1:epochs, J_history, 
             xlabel = "Epochs",
             ylabel = "Loss", 
             title  = "learning Curve", 
             legend = false, 
             color = :blue,
             linewidth = 2)