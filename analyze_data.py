# %%
import pandas as pd

data = pd.read_csv("survey_data.csv")

data = data.iloc[1:, 1:]

data.iloc[:, 0] = ["a" if "е" in x else "b" for x in data.iloc[:, 0]]

def convert_to_binary(row):
    return [row[0]] + [1 if "of" in x else 0 for x in row[1:]]

data = data.apply(convert_to_binary, axis="columns", result_type="expand")
print(data)

# %%
trials = pd.read_json("trials.json")[["type", "proper_noun", "complex"]]
trials.fillna(0, inplace=True)
# %%

import matplotlib.pyplot as plt

# Convert type to categorical for better plotting
trials['type'] = pd.Categorical(trials['type'])

# Create figure with subplots for each trial characteristic
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

mean_df = pd.DataFrame({
    'Group A (s-genitive)': data[data.iloc[:,0] == 'a'].iloc[:,1:].mean(),
    'Group B (of-genitive)': data[data.iloc[:,0] == 'b'].iloc[:,1:].mean(),
    'Mean': data.iloc[:,1:].mean()
}).reset_index(drop=True)

factor = 2

# Plot by type
type_means = mean_df.join(trials['type'])

means = type_means.groupby('type').mean().reindex(['inanimate', 'animal', 'person'])
stds = type_means.groupby('type').std().reindex(['inanimate', 'animal', 'person']) / factor
means.plot(kind='bar', ax=ax1, yerr=stds, capsize=5)
ax1.set_title('Responses by Animacy')
ax1.set_xticklabels(['Inanimate', 'Animal', 'Person'])
ax1.set_xlabel('Possessor Animacy')
ax1.set_ylabel('Proportion "of" responses')

# Plot by proper noun
proper_means = mean_df.join(trials['proper_noun'])

means = proper_means.groupby('proper_noun').mean()
stds = proper_means.groupby('proper_noun').std() / factor
means.plot(kind='bar', ax=ax2, yerr=stds, capsize=5)
ax2.set_title('Responses by Proper Noun')
ax2.set_xticklabels(['Common', 'Proper'])
ax2.set_xlabel('Possessor Noun Type')
ax2.set_ylabel('Proportion "of" responses')

# Plot by complexity
complex_means = mean_df.join(trials['complex'])

means = complex_means.groupby('complex').mean()
stds = complex_means.groupby('complex').std() / factor
means.plot(kind='bar', ax=ax3, yerr=stds, capsize=5)
ax3.set_title('Responses by Complexity')
ax3.set_xticklabels(['Simple', 'Complex'])
ax3.set_xlabel('Possessor Complexity')
ax3.set_ylabel('Proportion "of" responses')

plt.tight_layout()
plt.show()

# %%
# Create detailed summary table showing proportions for all combinations
print("\nDetailed proportions for of-genitive responses:")

mean_df["total"] = mean_df.mean(axis=1)

# Join all categorical variables
summary_df = mean_df.join(trials[['type', 'proper_noun', 'complex']])

# Group by all combinations
detailed_stats = summary_df.groupby(['type', 'proper_noun', 'complex']).mean().round(2)
print("\nBreakdown by type, proper noun status, and complexity:")
print(detailed_stats)

# Get counts for each combination
counts = summary_df.groupby(['type', 'proper_noun', 'complex']).size()
print("\nNumber of items in each category:")
print(counts)

# %%

lm_results = pd.read_json("results.json")["ratio"]

mean_df["total"]

plt.figure(figsize=(8, 6))
plt.scatter(mean_df["total"], lm_results)
plt.xlabel("Human 'of' preference")
plt.ylabel("LM preference for 'of' construction\n(ratio)")

# Add correlation coefficient
correlation = mean_df["total"].corr(lm_results)
plt.title(f'Correlation between Human and LM Preferences\nr = {correlation:.3f}')

# Add trend line
z = np.polyfit(mean_df["total"], lm_results, 1)
p = np.poly1d(z)
plt.plot(mean_df["total"], p(mean_df["total"]), "r--", alpha=0.8)

plt.grid(True, alpha=0.3)
plt.show()


# %%
