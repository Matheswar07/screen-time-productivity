import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

data = pd.read_csv(
    "dataset/productivity_data_unsupervised.csv"
)

print("===== DATASET SHAPE =====")
print(data.shape)


# ==========================================================
# 2. SELECT FEATURES FOR K-MEANS
# ==========================================================
#
# These 5 behavioural features are used for clustering:
#
# Screen Time
# Social Media
# Entertainment
# Productive Phone Use
# Study Time
#
# Phone Pickups is NOT used by K-Means.
# ==========================================================

features = data[
    [
        "Screen_Time_hr",
        "Social_Media_hr",
        "Entertainment_hr",
        "Productive_Phone_Use_hr",
        "Study_Time_hr"
    ]
]

print("\n===== SELECTED K-MEANS FEATURES =====")
print(features.head())

print("\n===== FEATURE SHAPE =====")
print(features.shape)


# ==========================================================
# 3. STANDARDIZE FEATURES
# ==========================================================

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

print("\n===== SCALED DATA SHAPE =====")
print(scaled_features.shape)


# ==========================================================
# 4. K-MEANS CLUSTERING
# ==========================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train K-Means
kmeans.fit(scaled_features)

# Assign each record to a cluster
data["Cluster"] = kmeans.labels_

print("\n===== CLUSTER COUNTS =====")
print(
    data["Cluster"]
    .value_counts()
    .sort_index()
)


# ==========================================================
# 5. SILHOUETTE SCORE
# ==========================================================

sil_score = silhouette_score(
    scaled_features,
    data["Cluster"]
)

print("\n===== SILHOUETTE SCORE =====")
print(f"Silhouette Score: {sil_score:.3f}")


# ==========================================================
# 6. PRODUCTIVITY INDEX
# ==========================================================
#
# The Productivity Index is used to interpret the clusters.
#
# Positive:
#   Higher Study Time
#
# Negative:
#   Higher Social Media
#   Higher Entertainment
#   Higher Phone Pickups
#
# Productive Phone Use is NOT separately added here because
# it is already part of Study Time.
#
# Screen Time is used by K-Means but is not directly
# penalized in the Productivity Index because some screen
# time can be used for academic activities.
# ==========================================================

score_features = data[
    [
        "Study_Time_hr",
        "Social_Media_hr",
        "Entertainment_hr",
        "Phone_Pickups"
    ]
]

# Normalize score features to the 0-1 range
score_scaler = MinMaxScaler()

normalized = pd.DataFrame(
    score_scaler.fit_transform(score_features),
    columns=score_features.columns
)

# Calculate Productivity Index
data["Productivity_Index"] = (
    normalized["Study_Time_hr"] * 0.50
    + (1 - normalized["Social_Media_hr"]) * 0.20
    + (1 - normalized["Entertainment_hr"]) * 0.20
    + (1 - normalized["Phone_Pickups"]) * 0.10
) * 100


# ==========================================================
# 7. AVERAGE PRODUCTIVITY INDEX OF EACH CLUSTER
# ==========================================================

cluster_index = (
    data.groupby("Cluster")["Productivity_Index"]
    .mean()
    .sort_values(ascending=False)
)

print("\n===== PRODUCTIVITY INDEX BY CLUSTER =====")

for cluster, score in cluster_index.items():
    print(
        f"Cluster {cluster}: {score:.2f}"
    )


# ==========================================================
# 8. ASSIGN PRODUCTIVITY PROFILES
# ==========================================================
#
# Highest average index  -> High
# Middle average index   -> Medium
# Lowest average index   -> Low
# ==========================================================

ordered_clusters = cluster_index.index.tolist()

cluster_profile = {
    ordered_clusters[0]: "High",
    ordered_clusters[1]: "Medium",
    ordered_clusters[2]: "Low"
}

data["Productivity_Profile"] = (
    data["Cluster"].map(cluster_profile)
)

print("\n===== CLUSTER PROFILES =====")

for cluster in sorted(cluster_profile.keys()):
    print(
        f"Cluster {cluster} → "
        f"{cluster_profile[cluster]}"
    )


# ==========================================================
# 9. CLUSTER BEHAVIOUR SUMMARY
# ==========================================================

cluster_summary = data.groupby("Cluster")[
    [
        "Screen_Time_hr",
        "Social_Media_hr",
        "Entertainment_hr",
        "Productive_Phone_Use_hr",
        "Study_Time_hr",
        "Phone_Pickups",
        "Productivity_Index"
    ]
].mean()

print("\n===== CLUSTER BEHAVIOUR SUMMARY =====")
print(cluster_summary.round(2))


# ==========================================================
# 10. PCA FOR VISUALIZATION
# ==========================================================
#
# PCA reduces the 5 clustering features to 2 dimensions
# only for visualization.
#
# K-Means still uses all 5 original scaled features.
# ==========================================================

pca = PCA(n_components=2)

pca_features = pca.fit_transform(
    scaled_features
)

data["PCA1"] = pca_features[:, 0]
data["PCA2"] = pca_features[:, 1]


# ==========================================================
# 11. PCA EXPLAINED VARIANCE
# ==========================================================

pc1 = pca.explained_variance_ratio_[0] * 100
pc2 = pca.explained_variance_ratio_[1] * 100

print("\n===== PCA EXPLAINED VARIANCE =====")
print(f"PC1: {pc1:.2f}%")
print(f"PC2: {pc2:.2f}%")
print(f"Total: {pc1 + pc2:.2f}%")


# ==========================================================
# 12. PCA CLUSTER VISUALIZATION
# ==========================================================

plt.figure(figsize=(8, 6))

for cluster in sorted(data["Cluster"].unique()):

    cluster_data = data[
        data["Cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["PCA1"],
        cluster_data["PCA2"],
        label=f"Cluster {cluster}",
        s=50
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Behavioural Clusters")

plt.legend()
plt.grid(True)

# Save graph
plt.savefig(
    "cluster_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 13. SAVE PROCESSED DATASET
# ==========================================================

output_file = (
    "dataset/productivity_clustered.csv"
)

data.to_csv(
    output_file,
    index=False
)

print(
    f"\nProcessed dataset saved as: "
    f"{output_file}"
)


# ==========================================================
# 14. SAVE TRAINED K-MEANS MODEL
# ==========================================================

joblib.dump(
    kmeans,
    "model/kmeans_model.pkl"
)

print(
    "K-Means model saved as: "
    "model/kmeans_model.pkl"
)


# ==========================================================
# 15. SAVE STANDARD SCALER
# ==========================================================

joblib.dump(
    scaler,
    "model/scaler.pkl"
)

print(
    "StandardScaler saved as: "
    "model/scaler.pkl"
)


# ==========================================================
# 16. SAVE PRODUCTIVITY SCORE SCALER
# ==========================================================

joblib.dump(
    score_scaler,
    "model/score_scaler.pkl"
)

print(
    "Score scaler saved as: "
    "model/score_scaler.pkl"
)


# ==========================================================
# 17. SAVE CLUSTER PROFILE MAPPING
# ==========================================================

joblib.dump(
    cluster_profile,
    "model/cluster_profiles.pkl"
)

print(
    "Cluster profile mapping saved as: "
    "model/cluster_profiles.pkl"
)


# ==========================================================
# 18. DISPLAY FIRST 10 PROCESSED RECORDS
# ==========================================================

print("\n===== FIRST 10 PROCESSED RECORDS =====")

print(
    data[
        [
            "Student_ID",
            "Day",
            "Day_Type",
            "Screen_Time_hr",
            "Social_Media_hr",
            "Entertainment_hr",
            "Productive_Phone_Use_hr",
            "Study_Time_hr",
            "Phone_Pickups",
            "Cluster",
            "Productivity_Index",
            "Productivity_Profile"
        ]
    ].head(10)
)


# ==========================================================
# 19. CHECK SAVED FILES
# ==========================================================

print("\n===== SAVED FILES =====")
print("1. dataset/productivity_clustered.csv")
print("2. cluster_visualization.png")
print("3. model/kmeans_model.pkl")
print("4. model/scaler.pkl")
print("5. model/score_scaler.pkl")
print("6. model/cluster_profiles.pkl")


# ==========================================================
# 20. PROGRAM COMPLETED
# ==========================================================

print("\n===== PROGRAM COMPLETED SUCCESSFULLY =====")