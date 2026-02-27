import pandas as pd
import numpy as np

# डेटा लोड करें
df = pd.read_csv("tech_employment_2000_2025.csv")

# NumPy एरेज़ तैयार करें
revenue = df['revenue_billions_usd'].values
layoffs = df['layoffs'].values
years = df['year'].values
companies = df['company'].values
new_hires = df["new_hires"].values

print("--- Task 1: Financial Health Check ---")
mean_rev = np.mean(revenue)
std_rev = np.std(revenue)
z_scores = (revenue - mean_rev) / std_rev  # Corrected Formula
print(f"Mean Revenue: ${mean_rev:.2f}B")
print(f"Revenue 25th Percentile: {np.percentile(revenue, 25)}")
print(f"First 5 Z-Scores: {z_scores[:5]}\n")

print("--- Task 2: The Crisis Filter (Post-2020 & High Layoffs) ---")
# उन कंपनियों को निकालें जहाँ 2020 के बाद 10k से ज्यादा छंटनी हुई
crisis_mask = (layoffs > 10000) & (years > 2020)
print(f"Companies in Crisis: {np.unique(companies[crisis_mask])}\n")

print("--- Task 3: Hiring-to-Firing Ratio ---")
# np.divide इस्तेमाल करें ताकि 0 से डिवाइड होने पर कोड न फटे
ratio = np.divide(new_hires, layoffs, out=np.zeros_like(new_hires, dtype=float), where=layoffs!=0)
print(f"Average Industry Hiring Ratio: {np.mean(ratio):.2f}")
print(f"Total NaN values in Layoffs: {np.sum(np.isnan(layoffs))}\n")

print("--- Task 4: Matrix Operations & Sorting ---")
# 1. टॉप 10 रेवेन्यू वैल्यूज निकालें
top_10_rev = np.sort(revenue)[-10:]
# 2. 2x5 मैट्रिक्स में बदलें
matrix_2x5 = top_10_rev.reshape(2, 5)
print("Top 10 Revenue (2x5 Matrix):\n", matrix_2x5)
# 3. Transpose (Matrix को घुमाना)
print("Transpose of Matrix:\n", matrix_2x5.T)
# 4. पूरे डेटा को Layoffs के हिसाब से सॉर्ट करें
sort_idx = np.argsort(layoffs)
print(f"Top 3 Companies with Highest Layoffs: {companies[sort_idx][-3:]}")
