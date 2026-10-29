
# Assignment No. 5: Advanced E-commerce Recommendation System using SVD

import numpy as np  # Imports NumPy for mathematical calculations and matrix operations.
import pandas as pd  # Imports Pandas to create and manage tables.
from sklearn.decomposition import TruncatedSVD  # Imports SVD to find hidden patterns in ratings.
from sklearn.metrics import mean_squared_error  # Imports a function to calculate prediction error.


# STEP 1: CREATE A SAMPLE E-COMMERCE RATINGS DATASET

ratings_data = {  # Creates a dictionary containing user IDs, product names, and ratings.
    'user_id': [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],  # Identifies which user rated each product.
    'product_name': [  # Stores the names of the products rated by users.
        'Laptop', 'Headphones', 'Mouse', 'Keyboard',
        'Laptop', 'Smartphone', 'Smartwatch',
        'Headphones', 'Keyboard', 'Smartphone',
        'Mouse', 'Keyboard', 'Smartwatch',
        'Laptop', 'Headphones', 'Smartphone'
    ],
    'rating': [5, 4, 3, 4, 5, 4, 3, 4, 5, 2, 2, 3, 4, 4, 5, 4]  # Stores ratings given by users, out of 5.
}

df_ratings = pd.DataFrame(ratings_data)  # Converts the dictionary into a table called a DataFrame.


# STEP 2: BUILD THE USER-ITEM MATRIX

user_item_matrix = df_ratings.pivot(  # Converts the ratings table into a user-item matrix.
    index='user_id',  # Places each user ID in a separate row.
    columns='product_name',  # Places each product name in a separate column.
    values='rating'  # Fills the matrix with the corresponding ratings.
).fillna(0)  # Replaces missing ratings with 0, meaning the user has not rated that product.

print("Original User-Item Rating Matrix (0 = Unrated/No Interaction):")  # Prints a heading for the original matrix.
print(user_item_matrix)  # Displays the user-item matrix.
print("\n" + "=" * 70 + "\n")  # Prints a separator line to make the output easier to read.


# STEP 3: MATRIX FACTORIZATION USING SVD

n_components = min(3, user_item_matrix.shape[1] - 1)  # Chooses 3 components, or fewer if required by the number of products.

svd = TruncatedSVD(  # Creates the Truncated SVD model.
    n_components=n_components,  # Sets the number of components used to represent hidden patterns.
    random_state=42  # Makes the randomized calculations reproducible.
)

user_features = svd.fit_transform(user_item_matrix)  # Learns patterns from the matrix and transforms users into component features.
item_features = svd.components_  # Stores the component matrix that represents patterns across products.

predicted_ratings = np.dot(user_features, item_features)  # Multiplies the matrices to reconstruct approximate ratings.

predicted_df = pd.DataFrame(  # Converts the predicted ratings into a readable Pandas DataFrame.
    predicted_ratings,  # Supplies the calculated predicted rating values.
    index=user_item_matrix.index,  # Keeps the original user IDs as row labels.
    columns=user_item_matrix.columns  # Keeps the original product names as column labels.
)

print("Reconstructed (Predicted) Rating Matrix via SVD:")  # Prints a heading for the predicted matrix.
print(predicted_df.round(2))  # Displays predicted ratings rounded to two decimal places.
print("\n" + "=" * 70 + "\n")  # Prints a separator line.


# EVALUATE THE PREDICTION ERROR ON KNOWN RATINGS

known_mask = user_item_matrix.values > 0  # Identifies actual ratings and ignores entries representing unrated products.

mse = mean_squared_error(  # Calculates the Mean Squared Error between actual and predicted ratings.
    user_item_matrix.values[known_mask],  # Selects the actual ratings provided by users.
    predicted_ratings[known_mask]  # Selects the corresponding predicted ratings.
)

print(f"Mean Squared Error (MSE) on Known Ratings: {mse:.4f}")  # Displays the error rounded to four decimal places.
print("\n" + "=" * 70 + "\n")  # Prints a separator line.


# STEP 4: CREATE THE TOP-N RECOMMENDATION FUNCTION

def recommend_products(user_id, top_n=2):  # Defines a function to recommend products; the default is two products.
    """Recommends unrated products with the highest predicted scores."""  # Describes what the function does.

    if user_id not in user_item_matrix.index:  # Checks whether the requested user exists in the dataset.
        return f"User ID {user_id} not found."  # Returns a message if the user does not exist.

    user_orig = user_item_matrix.loc[user_id]  # Retrieves the user's original product ratings.
    user_pred = predicted_df.loc[user_id]  # Retrieves the predicted ratings for that user.

    unrated_products = user_orig[user_orig == 0].index  # Finds products that the user has not rated.

    recommendations = (  # Starts selecting and ranking products for recommendation.
        user_pred[unrated_products]  # Selects predicted scores only for unrated products.
        .sort_values(ascending=False)  # Sorts products from the highest predicted score to the lowest.
        .head(top_n)  # Selects up to the requested number of top products.
    )

    return recommendations  # Returns the selected product recommendations and their predicted scores.


# STEP 5: TEST THE RECOMMENDATION SYSTEM

if __name__ == '__main__':  # Runs this testing section only when the file is executed directly.

    target_user = 2  # Selects User 2 as the person receiving recommendations.
    top_n_count = 2  # Sets the number of products to recommend to two.

    recs = recommend_products(  # Calls the recommendation function and stores its result.
        user_id=target_user,  # Passes User 2's ID to the function.
        top_n=top_n_count  # Requests two recommendations.
    )

    print(f"Top {top_n_count} Product Recommendations for User {target_user}:")  # Displays the recommendation heading.

    if isinstance(recs, str):  # Checks whether the function returned an error message.
        print(recs)  # Prints the error message if the user does not exist.
    elif recs.empty:  # Checks whether there are any products left to recommend.
        print("No unrated products available to recommend.")  # Displays a message if no recommendations are available.
    else:  # Executes this block when valid recommendations are available.
        for product, score in recs.items():  # Goes through each recommended product and its predicted score.
            print(f" -> Product: {product:<15} (Predicted Score: {score:.2f})")  # Prints the product name and score to two decimal places.


The program imports Pandas and NumPy to manage product ratings and perform numerical calculations, along with the required Scikit-learn tools for matrix factorisation. User ratings for different products are stored in a dataset and organised into a user-item matrix, where rows represent users and columns represent products. The matrix is prepared for processing, and Singular Value Decomposition (SVD) is applied to identify hidden patterns in the ratings. These patterns help estimate how a user might rate products they have not rated before. The system uses the estimated ratings to rank products for a selected user. Products with higher predicted ratings are considered stronger recommendations, while products the user has already rated or purchased may be excluded. Finally, the program displays the recommended products. The quality of the recommendations depends on the amount and quality of the available user-rating data.
my project is an E-commerce Recommendation System that recommends products based on user ratings. First, I collect user-product ratings and arrange them into a user-item matrix. Each row represents a user, and each column represents a product. Then I apply Singular Value Decomposition, or SVD, to identify hidden patterns in the rating data. These patterns help the system estimate which products a user may prefer. The system then ranks the products according to their predicted ratings and recommends suitable products to the user. For example, if a user has shown interest in certain products, the system can identify related preference patterns and suggest other products. This approach is called collaborative filtering because it uses user-item interactions to generate recommendations
