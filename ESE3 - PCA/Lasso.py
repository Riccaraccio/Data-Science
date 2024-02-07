import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import model_selection


A = np.random.randn(100,10) # Matrix of possible predictors
x = np.array([0, 0, 1, 0, 0, 0, -1, 0, 0, 0]) #Two nonzero predictors out of 10

b = A @ x + 2*np.random.randn(100) #A*x + random noise(dimension = 100)

xL2 = np.linalg.pinv(A) @ b #least square regression

print(xL2) # should return x but doesnt 

# Cross-validation involves splitting the dataset into 'k' folds (in this case, 10), 
# training the model on 'k-1' folds, and validating it on the remaining fold
reg = linear_model.LassoCV(cv=10).fit(A, b)

xLasso = reg.coef_ #get the lasso coefficent x
print(xLasso)

plt.grid()

# Set the width of the bars
bar_width = 0.2

# Create positions for the bars
bar_positions_x = np.arange(len(x))
bar_positions_xL2 = bar_positions_x + bar_width
bar_positions_xLasso = bar_positions_xL2 + bar_width

# Create bar plots for each vector
plt.bar(bar_positions_x, x, width=bar_width, label='Vector 1')
plt.bar(bar_positions_xL2, xL2, width=bar_width, label='Vector 2')
plt.bar(bar_positions_xLasso, xLasso, width=bar_width, label='Vector 3')

# Add labels and title
plt.set_cmap("jet")
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Bar Plot of Vectors')

plt.legend()

# Show the plot
plt.show()

lasso = linear_model.Lasso(random_state=0, max_iter=10000)

alphas = np.logspace(-4, -0.5, 30) #alpha parameter for the lassoregression
tuned_parameters = [{'alpha': alphas}] #values of alpha are stored in a dictionary

#This sets up a grid search using cross-validation (cv=10) to find the optimal value for the hyperparameter alpha for the Lasso regression. 
# The refit=False parameter indicates that the model should not be refit with the best parameters found during the grid search.
clf = model_selection.GridSearchCV(lasso, tuned_parameters, cv=10, refit=False)
clf.fit(A, b) #perform the grid search usgin 

scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']
plt.semilogx(alphas, scores,'r-')

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(10)

plt.semilogx(alphas, scores + std_error, 'k--')
plt.semilogx(alphas, scores - std_error, 'k--')
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.1,color='k')

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[-1], alphas[0]])

plt.show()
