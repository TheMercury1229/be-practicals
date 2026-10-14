# Title: Unsupervised K-Means Clustering and Elbow Method Analysis

## 1. Problem Statement

To implement the K-Means clustering algorithm on the four-dimensional Iris dataset using the four numerical features: Sepal Length, Sepal Width, Petal Length, and Petal Width. The Elbow Method is used to determine a suitable number of clusters by analyzing the Within-Cluster Sum of Squares (WCSS). The resulting clusters are visualized and compared with the actual Iris species to understand the effectiveness of unsupervised learning.

## 2. Learning Objectives

- To understand the fundamental concepts of unsupervised machine learning and clustering.
- To study the working principle of the K-Means clustering algorithm.
- To understand centroid initialization, data-point assignment, and centroid updating.
- To learn the importance of feature scaling before applying distance-based clustering algorithms.
- To determine a suitable number of clusters using the Elbow Method.
- To understand the concepts of cluster cohesion, cluster separation, and the Silhouette Coefficient.
- To visualize the clusters using scatter plots and interpret the clustering results.
- To compare the generated clusters with the actual species labels of the Iris dataset.

## 3. Introduction to Unsupervised Learning

Machine learning is a branch of artificial intelligence that enables computers to learn patterns from data and make decisions without being explicitly programmed for every task. Machine learning techniques are broadly classified into supervised learning, unsupervised learning, and reinforcement learning.

Unsupervised learning is a technique in which a model learns the underlying structure and patterns of a dataset without using predefined target labels during training. Unlike supervised learning, where the model learns from labeled examples, unsupervised learning identifies similarities, differences, and hidden relationships among observations.

Clustering is one of the most widely used unsupervised learning techniques. It divides a dataset into groups called clusters, such that data points within the same cluster are relatively similar to one another, while data points belonging to different clusters are relatively dissimilar.

Clustering is useful in customer segmentation, image segmentation, document grouping, pattern recognition, biological classification, and exploratory data analysis. In this assignment, K-Means clustering is applied to the Iris dataset to identify groups of flowers based on their physical measurements.

## 4. Iris Dataset

The Iris dataset is a well-known dataset used in machine learning and statistical analysis. It contains 150 observations of Iris flowers belonging to three species: Iris-setosa, Iris-versicolor, and Iris-virginica.

Each observation contains four numerical features describing the physical dimensions of the flower.

| Feature      | Description                                     |
| ------------ | ----------------------------------------------- |
| Sepal Length | Length of the sepal, measured in centimeters    |
| Sepal Width  | Width of the sepal, measured in centimeters     |
| Petal Length | Length of the petal, measured in centimeters    |
| Petal Width  | Width of the petal, measured in centimeters     |
| Species      | Actual species label associated with the flower |

The four numerical features are used as input variables for the clustering algorithm. The Species column is not used to train the K-Means model because the purpose is to perform unsupervised learning. However, it can be used after clustering to compare the generated groups with the known species labels.

This comparison helps us understand how closely the clusters correspond to the natural groups present in the dataset.

## 5. K-Means Clustering

K-Means is an unsupervised machine learning algorithm that partitions a dataset into a predefined number of clusters, represented by the parameter k. Each cluster is represented by a centroid, which is the mean position of all data points assigned to that cluster.

The primary objective of K-Means is to minimize the sum of squared distances between each observation and the centroid of its assigned cluster. This quantity is commonly called inertia or Within-Cluster Sum of Squares (WCSS).

Let the dataset contain n observations divided into k clusters:

S = {S₁, S₂, ..., Sₖ}

The objective function minimized by K-Means is:

J = Σⱼ₌₁ᵏ Σₓᵢ∈Sⱼ ||xᵢ − μⱼ||²

Where:

- k is the number of clusters.
- Sⱼ represents the set of observations belonging to cluster j.
- xᵢ represents an individual data point.
- μⱼ represents the centroid of cluster j.
- ||xᵢ − μⱼ||² represents the squared Euclidean distance between a data point and its assigned centroid.
- J represents the total within-cluster sum of squared distances.

The algorithm attempts to find cluster assignments and centroids that minimize this objective function.

### 5.1 Working Principle of K-Means

The K-Means algorithm works through the following iterative process:

**Step 1: Initialize the centroids**

The algorithm selects k initial centroids. The quality of the initial centroids can influence the final clustering result. The K-Means++ initialization technique improves the selection of initial centroids by spreading them apart more effectively.

**Step 2: Assign data points to clusters**

Each observation is assigned to the cluster whose centroid is closest to it according to Euclidean distance.

The Euclidean distance between two points x and μ is calculated as:

d(x, μ) = √Σᵢ₌₁ᵖ (xᵢ − μᵢ)²

Here, p represents the number of features in each observation. In this assignment, p = 4.

**Step 3: Update the centroids**

After assigning the data points, the algorithm calculates the mean of all observations belonging to each cluster. This mean becomes the new centroid.

μⱼ = (1 / |Sⱼ|) Σₓᵢ∈Sⱼ xᵢ

Here, |Sⱼ| represents the number of data points in cluster j.

**Step 4: Repeat the process**

The assignment and centroid-update steps are repeated until the centroids stop changing significantly or the algorithm reaches its stopping condition.

**Step 5: Obtain the final clusters**

Once the algorithm converges, each observation receives a cluster label. These labels represent the groups identified by the algorithm.

### 5.2 Voronoi Partitioning

K-Means divides the feature space into regions according to the nearest centroid. These regions form a Voronoi partition, in which each region contains points closer to its associated centroid than to any other centroid.

The boundaries between these regions are determined by the relative positions of the centroids. This geometric interpretation explains how K-Means assigns observations to clusters based on distance.

However, because standard K-Means uses Euclidean distance and mean-based centroids, it generally works best when clusters are reasonably compact and approximately spherical. It may not perform well when clusters have irregular shapes, substantially different densities, or many outliers.

## 6. Importance of Feature Scaling

Feature scaling is an important preprocessing step for distance-based algorithms such as K-Means. If the input features have significantly different scales or ranges, features with larger numerical values may contribute more strongly to the distance calculation.

Standardization transforms numerical features so that each feature has approximately zero mean and unit standard deviation.

The standardization formula is:

z = (x − μ) / σ

Where:

- x is the original feature value.
- μ is the mean of that feature.
- σ is the standard deviation of that feature.
- z is the standardized feature value.

In this assignment, the `StandardScaler` class from Scikit-learn is used to standardize the four numerical features of the Iris dataset.

Standardization allows the algorithm to consider the features on a more comparable scale, reducing the influence of differences in their original numerical ranges. The scaled data is then supplied to K-Means for clustering.

## 7. Within-Cluster Sum of Squares (WCSS)

Within-Cluster Sum of Squares (WCSS), also known as inertia in Scikit-learn, measures the total squared distance between observations and their assigned cluster centroids.

It is calculated as:

WCSS = Σⱼ₌₁ᵏ Σₓᵢ∈Sⱼ ||xᵢ − μⱼ||²

A lower WCSS value indicates that the observations are closer to their respective centroids. When the number of clusters increases, WCSS generally decreases because each cluster contains fewer observations and the centroids can represent the data more closely.

If every observation is assigned to its own cluster, WCSS can reach zero. However, such a result is not necessarily useful because it does not provide meaningful grouping.

Therefore, the goal is not simply to minimize WCSS but to select a suitable number of clusters that balances compactness and simplicity.

## 8. Elbow Method

The Elbow Method is a commonly used technique for estimating an appropriate number of clusters in K-Means clustering. It evaluates the WCSS values obtained for different values of k.

In this assignment, K-Means is executed for k values ranging from 1 to 10. For every value of k, the model is fitted to the standardized dataset and its inertia is stored.

The values are plotted on a graph with the number of clusters on the X-axis and WCSS on the Y-axis.

### 8.1 Procedure of the Elbow Method

1. Select a range of possible cluster counts, such as k = 1 to k = 10.
2. Fit a separate K-Means model for each value of k.
3. Calculate the WCSS or inertia for each model.
4. Store the inertia values in a list.
5. Plot the number of clusters against the corresponding WCSS values.
6. Identify the point where the rate of decrease in WCSS begins to slow down considerably.
7. Select a suitable cluster count based on the elbow, along with the characteristics of the dataset and other validation measures.

### 8.2 Interpretation of the Elbow Graph

Initially, increasing the number of clusters usually produces a substantial reduction in WCSS. This happens because the centroids can represent the observations more accurately when additional clusters are available.

After a certain point, adding more clusters produces only a relatively small improvement. The curve begins to flatten, creating a shape resembling an elbow.

The value of k near this bend is considered a possible choice for the number of clusters.

The elbow is not always sharply defined, and different datasets may produce different curve shapes. Therefore, the Elbow Method provides a useful estimate rather than a guaranteed optimal answer.

In the Iris dataset, three clusters are often considered a meaningful choice because the dataset contains three known species. Nevertheless, the final choice should be supported by the observed elbow curve and, where possible, additional evaluation measures.

## 9. Silhouette Analysis

Silhouette Analysis is a cluster evaluation technique used to measure how well an observation fits into its assigned cluster compared with other clusters.

The Silhouette Coefficient considers two quantities:

**a(i): Mean intra-cluster distance**

This is the average distance between observation i and the other observations in the same cluster. A smaller value indicates that the observation is close to the other points in its cluster.

**b(i): Mean nearest-cluster distance**

This is the smallest average distance between observation i and the observations belonging to another cluster. It represents the average distance to the nearest alternative cluster.

The Silhouette Coefficient for observation i is calculated as:

s(i) = [b(i) − a(i)] / max(a(i), b(i))

The resulting score ranges from −1 to +1.

### 9.1 Interpretation of Silhouette Scores

- **Close to +1:** The observation is well matched to its assigned cluster and well separated from neighboring clusters.
- **Close to 0:** The observation lies near the boundary between two clusters.
- **Below 0:** The observation may be assigned to a cluster that is less appropriate than a neighboring cluster.

The overall Silhouette Score is generally calculated as the average of the individual observations' Silhouette Coefficients.

A higher average score usually indicates better cluster cohesion and separation. However, the score should be interpreted in context because the best-scoring k does not always correspond to the most meaningful groups for a particular application.

For the Iris dataset, Silhouette Analysis can be used alongside the Elbow Method to compare different values of k. For example, k = 2 may produce a higher Silhouette Score than k = 3, even though three clusters may be more useful when the objective is to explore the three known Iris species.

**Implementation note:** The provided Python code currently calculates WCSS and visualizes clusters, but it does not calculate the Silhouette Score. To include numerical Silhouette Analysis in the practical results, the `silhouette_score` function from `sklearn.metrics` must be imported and used separately for k values greater than or equal to 2.

## 10. Cluster Visualization

Visualization helps us understand the distribution of observations and the separation between clusters. In this assignment, a scatter plot is created using Petal Length on the X-axis and Petal Width on the Y-axis.

Each observation is represented by a point, and different colors indicate the cluster assigned by K-Means. The scatter plot provides a two-dimensional representation of the clustering result.

Although the model uses all four standardized features, the visualization displays only two original features. Therefore, the plot provides a partial view of the clustering structure rather than a complete representation of the four-dimensional feature space.

The plot can help identify groups that are well separated and reveal areas where different clusters overlap. Petal measurements are especially useful for visualizing the Iris dataset because the species often show clearer differences in petal dimensions than in sepal dimensions.

## 11. Comparison of Clusters with Actual Species

K-Means is an unsupervised algorithm, so it does not use the actual species labels during model training. It generates cluster labels based only on similarities in the numerical features.

After clustering, the generated labels can be compared with the actual Species column using a cross-tabulation table.

The `pd.crosstab()` function creates a frequency table showing how many observations of each actual species belong to each predicted cluster.

This comparison helps determine whether the generated clusters correspond closely to the known species groups.

However, cluster labels such as 0, 1, and 2 are arbitrary identifiers. Cluster 0 does not automatically represent Iris-setosa, and cluster numbers may change between model runs or parameter settings. The labels should therefore be interpreted by examining the distribution of species within each cluster rather than assuming that their numerical values correspond directly to species names.

The comparison is useful for evaluating the practical meaning of the clusters, but it does not change the fact that the model itself was trained without species labels.

## 12. Algorithm

1. Start the program.
2. Import the required libraries, including Pandas, Matplotlib, Seaborn, StandardScaler, and KMeans.
3. Load the Iris dataset from the CSV file.
4. Examine the dataset dimensions, feature names, data types, missing values, duplicate records, and species distribution.
5. Select the four numerical features: Sepal Length, Sepal Width, Petal Length, and Petal Width.
6. Standardize the selected features using StandardScaler.
7. Initialize an empty list to store WCSS values.
8. For each value of k from 1 to 10:
   - Create a K-Means model with the specified number of clusters.
   - Fit the model to the standardized feature matrix.
   - Calculate and store the inertia value.
9. Plot the WCSS values against the number of clusters to generate the Elbow graph.
10. Select a suitable number of clusters based on the Elbow Method. In this implementation, k = 3 is used.
11. Fit the final K-Means model using three clusters.
12. Assign the predicted cluster labels to a new column named Cluster.
13. Display the number of observations assigned to each cluster.
14. Visualize the clusters using Petal Length and Petal Width.
15. Compare the generated clusters with the actual species labels using a cross-tabulation table.
16. Display the final dataset with the assigned cluster labels.
17. Report the total number of distinct clusters.
18. Stop the program.

## 13. Advantages of K-Means Clustering

- It is simple to understand and straightforward to implement.
- It is computationally efficient for many small and medium-sized datasets.
- It works effectively when the clusters are compact and reasonably well separated.
- It can process numerical datasets containing multiple features.
- It provides easily interpretable cluster centroids.
- It can be combined with visualization and evaluation techniques to explore hidden patterns in data.

## 14. Limitations of K-Means Clustering

- The number of clusters k must be specified before fitting the model.
- Results may depend on the initial positions of the centroids.
- It is sensitive to outliers because centroids are calculated using means.
- It may perform poorly on clusters with irregular shapes or substantially different densities.
- Distance calculations can be affected by differences in feature scales, making preprocessing important.
- The Elbow Method does not always identify a clear optimal number of clusters.
- Cluster labels have no predefined semantic meaning and require interpretation.
- K-Means is primarily intended for numerical data and does not directly handle categorical features in the same way as numerical features.

## 15. Applications of K-Means Clustering

K-Means clustering has several applications in practical data analysis:

- **Customer segmentation:** Grouping customers according to purchasing behavior and preferences.
- **Image segmentation:** Grouping pixels with similar color or intensity values.
- **Document clustering:** Organizing documents according to numerical text representations.
- **Biological data analysis:** Identifying groups of organisms or samples with similar measurements.
- **Market research:** Discovering groups of customers with similar characteristics.
- **Anomaly exploration:** Identifying observations that lie far from typical cluster patterns, although K-Means is not a dedicated anomaly-detection algorithm.

In this assignment, the algorithm is applied to biological measurements to discover groups of Iris flowers with similar characteristics.

## 16. Conclusion

K-Means clustering is an important unsupervised machine learning algorithm that groups observations according to their similarity in feature space. In this practical, the four numerical features of the Iris dataset are standardized before applying K-Means to ensure that the features contribute on comparable scales.

The Elbow Method is used to study the variation in Within-Cluster Sum of Squares for values of k ranging from 1 to 10. Based on the chosen elbow interpretation, a final model with three clusters is fitted, and each observation is assigned a cluster label.

The generated clusters are visualized using Petal Length and Petal Width, while a cross-tabulation table is used to compare the predicted clusters with the actual Iris species. Silhouette Analysis can provide an additional measure of cluster cohesion and separation when implemented.

Thus, the practical demonstrates the use of K-Means clustering, feature standardization, the Elbow Method, and cluster visualization for exploratory analysis of a multidimensional dataset. It also illustrates how unsupervised learning can reveal meaningful patterns without using the target labels during training.
