# Import necessary libraries
using CSV               # For reading CSV files
using DataFrames        # For handling tabular data
using GLM               # For linear modeling (Ordinary Least Squares)
using Plots             # For data visualization (scatter plots, line plots)
using TypedTables       # For creating typed tables compatible with GLM

##########################################################
# Step 1: Load and preprocess the dataset
##########################################################

# Load data from a CSV file into a DataFrame
# The dataset should contain columns for house size and price
data = CSV.File("housingdata.csv") |> DataFrame

# Extract feature (size of the house) and target variable (price in 1000 dollars)
X = data.size                      # Independent variable: house size in square feet
Y = round.(Int, data.price / 1000) # Dependent variable: house price scaled to thousands

# Create a typed table for GLM to handle the data
# GLM requires a Table structure for input
t = Table(X = X, Y = Y)

##########################################################
# Step 2: Visualize the dataset
##########################################################

# Set up the GR backend for consistent plot sizes
gr(size = (600, 600))

# Scatter plot of the dataset
# - X-axis: House size
# - Y-axis: Price
# - Data points are red
p_scatter = scatter(X, Y, 
                    xlims = (0, 5000),              # Limit x-axis to a reasonable range
                    ylims = (0, 800),               # Limit y-axis to match price scale
                    xlabel = "Size (sqft)",         # Label for X-axis
                    ylabel = "Price in 1000 dollars",  # Label for Y-axis
                    title = "Housing Prices in Portland", # Title of the plot
                    legend = false,                # Disable legend for simplicity
                    color = :red)                  # Data points in red

##########################################################
# Step 3: Ordinary Least Squares Regression (OLS)
##########################################################

# Perform OLS regression using the GLM library
# @formula(Y ~ X): Formula specifies that Y (price) is modeled as a function of X (size)
ols = lm(@formula(Y ~ X), t)

# Add the OLS regression line to the scatter plot
plot!(X, predict(ols), color = :green, linewidth = 3)

# Predict the price for a new house size (e.g., 1250 sqft)
# - Create a new table with the input value
# - Use the `predict` function to calculate the corresponding price
newX = Table(X = [1250])
prediction = predict(ols, newX)

println("Predicted price for a house of 1250 sqft: $prediction")  # Display prediction
println(t)  # Display the dataset in tabular form for reference

##########################################################
# Step 4: Implementing Linear Regression from Scratch
##########################################################

# Initialize the number of training epochs
epochs = 0

# Re-plot scatter for manual linear regression visualization
# - This scatter plot is updated throughout training to visualize progress
gr(size = (600, 600))
p_scatter = scatter(X, Y, 
                    xlims = (0, 5000), 
                    ylims = (0, 800), 
                    xlabel = "Size (sqft)", 
                    ylabel = "Price in 1000 dollars", 
                    title = "Housing Prices in Portland (epochs = $epochs)", 
                    legend = false, 
                    color = :red)

# Initialize parameters for linear regression
theta0 = 0.0   # y-intercept (initial guess, no prior knowledge)
theta1 = 0.0   # slope (initial guess, no prior knowledge)

# Define the hypothesis function: h(x) = θ₀ + θ₁x
# This function models the relationship between size (X) and price (Y)
h(x) = theta0 .+ theta1 .* x

# Add the initial hypothesis line to the plot
plot!(X, h(X), color = :blue, linewidth = 3)

##########################################################
# Step 5: Define the cost function
##########################################################

# Cost function: Mean Squared Error (MSE)
# MSE measures the average squared difference between predictions (y_hat) and true values (Y)
function cost(X, Y)
    m = length(X)  # Number of training examples
    y_hat = h(X)   # Predictions using the current parameters
    (1 / (2 * m)) * sum((y_hat - Y) .^ 2)  # Mean Squared Error
end

# Compute the initial cost with the untrained parameters
J = cost(X, Y)
println("Initial Cost: $J")  # Display the cost before training

# Store cost values for analysis during training
J_history = Float64[]  # Initialize an array to track the cost history

##########################################################
# Step 6: Gradient Descent
##########################################################

# Partial derivatives for the cost function (used for updating parameters)
# - pd_theta0: Partial derivative of cost with respect to θ₀
# - pd_theta1: Partial derivative of cost with respect to θ₁

function pd_theta0(X, Y)
    m = length(X)
    y_hat = h(X)  # Predictions
    (1 / m) * sum(y_hat - Y)  # Gradient for θ₀
end

function pd_theta1(X, Y)
    m = length(X)
    y_hat = h(X)  # Predictions
    (1 / m) * sum((y_hat - Y) .* X)  # Gradient for θ₁
end

# Set learning rates (α) for gradient descent
# Learning rate controls the step size for parameter updates
alpha0 = 0.09         # Learning rate for θ₀
alpha1 = 0.00000008   # Learning rate for θ₁

##########################################################
# Step 7: Iterative Training Process
##########################################################

# Compute initial gradients
theta0_temp = pd_theta0(X, Y)
theta1_temp = pd_theta1(X, Y)

# Update parameters using the gradients and learning rates
theta0 -= alpha0 * theta0_temp
theta1 -= alpha1 * theta1_temp

# Recompute predictions and cost after parameter update
y_hat = h(X)
J = cost(X, Y)  # Updated cost
push!(J_history, J)  # Track the cost history
epochs += 1          # Increment epoch count

# Add updated hypothesis line to the plot
plot!(X, y_hat, color = :blue, alpha = 0.5,
      title = "Housing Prices in Portland (epochs: $epochs)")

# Overlay the OLS regression line for comparison
plot!(X, predict(ols), color = :green, linewidth = 3)

##########################################################
# Step 8: Visualize the Learning Curve
##########################################################

# Plot the learning curve (cost vs. epochs)
# This shows how the cost decreases as the model trains
gr(size = (600, 600))
pline = plot(1:epochs, J_history, 
             xlabel = "Epochs",
             ylabel = "Loss", 
             title = "Learning Curve", 
             legend = false, 
             color = :blue,
             linewidth = 2)

println("Final Cost: $(last(J_history))")  # Display the final cost after training
println("Number of Epochs: $epochs")      # Display total number of training epochs
