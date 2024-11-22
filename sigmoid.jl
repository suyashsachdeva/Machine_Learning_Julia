# Import the required library for visualization
using Plots  # Plots is a powerful visualization library for creating high-quality plots

# Set the GR backend and configure the plot size
gr(size = (600, 600))  # Configure the size of the plot to be 600x600 pixels

##########################################################
# Define the logistic (sigmoid) function
##########################################################

# The logistic function is commonly used in machine learning and statistics.
# It maps real-valued inputs (x) into the range (0, 1).
# Formula: logistic(x) = 1 / (1 + exp(-x))
function logistic(x)
    return 1 / (1 + exp(-x))  # Compute the logistic transformation
end

##########################################################
# Generate and plot the logistic curve
##########################################################

# Plot the logistic function over a range of inputs
# -6:0.1:6 defines the range of x values from -6 to 6 with a step of 0.1
p_log = plot(-6:0.1:6, logistic,  # Range of inputs and the function to plot
             xlabel = "Inputs (x)",        # Label for the x-axis
             ylabel = "Output (y)",        # Label for the y-axis
             title = "Logistic (Sigmoid) Curve",  # Title of the plot
             legend = false,              # Disable the legend for simplicity
             linewidth = 2,               # Set the line thickness to 2
             color = :blue)               # Use blue color for the curve

##########################################################
# Additional Notes
##########################################################

# The logistic function is widely used in:
# - Logistic regression: As an activation function to model probabilities.
# - Neural networks: For binary classification tasks.
# - Biological systems: To model growth patterns.
#
# This plot demonstrates the smooth, S-shaped curve of the logistic function,
# which asymptotically approaches 0 for large negative inputs and 1 for large positive inputs.
