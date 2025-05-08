import pandas as pd
import numpy as np
import itertools
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# Load data and map categorical codes to descriptive labels
def load_data(path='heart.csv'):
    # Read the CSV file 
    df = pd.read_csv(path)

    # mapping of each code based on the categories
    cat_maps = {
        'sex': {
            0: 'male',
            1: 'female'
        },
        'cp': {
            0: 'Typical angina',
            1: 'Atypical angina',
            2: 'Non‑anginal pain',
            3: 'Asymptomatic'
        },
        'fbs': {
            0: '≤120 mg/dl',
            1: '>120 mg/dl'
        },
        'restecg': {
            0: 'Normal',
            1: 'ST‑T wave abnormality',
            2: 'Left ventricular hypertrophy'
        },
        'exang': {
            0: 'no',
            1: 'yes'
        },
        'slope': {
            0: 'Upsloping',
            1: 'Flat',
            2: 'Downsloping'
        },
        'ca': {
            0: '0 vessels',
            1: '1 vessel',
            2: '2 vessels',
            3: '3 vessels',
            4: '4 vessels'
        },
        'thal': {
            0: 'Normal',
            1: 'Fixed defect',
            2: 'Reversible defect',
            3: 'Not described'
        }
        # leave 'target' as 0/1 for modeling
    }
    # label each code and return the frame with all the data
    for col, mapping in cat_maps.items():
        df[col] = df[col].map(mapping)

    return df

# Get each feature using the bins since we will be creating visualization
# based on the continuos data
def discretize(df, bins_dict=None):
    df_disc = df.copy()
    if bins_dict is None:
        bins_dict = {
            'age':     [29, 40, 50, 60, 70, 80],
            'trestbps':[90, 120, 140, 160, 200],
            'chol':    [100, 200, 300, 400],
            'thalach': [70, 100, 130, 160, 200],
            'oldpeak': [0.0, 1.0, 2.0, 3.0, 6.0]
        }
    for col, bins in bins_dict.items():
        # Create labels for each category
        labels = [f"{col}_{i}" for i in range(len(bins) + 1)]
        edges = [-np.inf] + bins + [np.inf]
        df_disc[col] = pd.cut(df_disc[col], bins=edges, labels=labels)
    return df_disc

# joint distribution P(X=x, Y=y) based on the basic implementation 
# and expanding upon it since we are dealing with a different dataset
def learn_joint(df, features, target='target'):
    counts = Counter()
    for row in df[features + [target]].itertuples(index=False):
        counts[tuple(row)] += 1
    total = sum(counts.values())
    # Finding the probabilities
    return {k: v/total for k, v in counts.items()}

# Inference by enumeration: P(Y=1 | evidence)
def enumeration_posterior(joint, features, evidence, target_index=-1):
    # initialize the values, find the values, extract them and then 
    # check if there is an evidence of a match
    p_num = p_den = 0.0
    for key, p in joint.items():
        feat_vals = key[:len(features)]
        y = key[target_index]
        if all(feat_vals[i] == evidence.get(features[i], feat_vals[i])
               for i in range(len(features))):
            p_den += p
            if y == 1:
                p_num += p
    # return the posterior
    return (p_num / p_den) if p_den > 0 else 0

# Compute single-variable posteriors
def variable_posteriors(joint, features):
    post = {feat: {} for feat in features}
    for feat in features:
        # Find possible values
        vals = {k[i] for k in joint.keys()
                for i, f in enumerate(features) if f == feat}
        for val in vals:
            post[feat][val] = enumeration_posterior(
                joint, features, {feat: val})
    return post

# Naive Bayes expanding on the basic implementation from the first 
# data set
def train_naive(df, features, target='target', laplace=0):
    # count each record, get the total and then we compute the
    # probabilities using laplace smoothing
    y_counts = df[target].value_counts().to_dict()
    total = len(df)
    py = {
        y: (count + laplace) / (total + laplace * len(y_counts))
        for y, count in y_counts.items()
    }
    # Get the conditional tables and unpack them
    cond = {feat: defaultdict(lambda: defaultdict(int))
            for feat in features}
    for row in df[features + [target]].itertuples(index=False):
        *x, y = row
        for i, feat in enumerate(features):
            cond[feat][y][x[i]] += 1
    # Compute feature and class based on value 
    p_cond = {feat: {} for feat in features}
    for feat in features:
        for y, counter in cond[feat].items():
            total_y = y_counts[y]
            distinct = set(counter.keys())
            p_cond[feat][y] = {
                val: (counter[val] + laplace) /
                     (total_y + laplace * len(distinct))
                for val in distinct
            }
    return py, p_cond

# Predict probabilities with Naive Bayes
def predict_proba(df, features, py, p_cond):
    probs = []
    for row in df[features].itertuples(index=False):
        scores = {}
        for y, prior in py.items():
            score = prior
            for i, feat in enumerate(features):
                score *= p_cond.get(feat, {}).get(y, {}).get(row[i], 0)
            scores[y] = score
        total_score = sum(scores.values())
        probs.append(scores.get(1, 0) / total_score
                     if total_score > 0 else 0)
    return np.array(probs)

# Plot ROC curves
def plot_rocs(y_true, probs_list, labels):
    plt.figure()
    for probs, label in zip(probs_list, labels):
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # --- Load & label data ---
    df = load_data()
    # --- Discretize continuous vars ---
    df_disc = discretize(df)

    # Prepare feature list
    features = [c for c in df_disc.columns if c != 'target']

    # --- Enumeration-based posterior per feature value ---
    joint = learn_joint(df_disc, features)
    var_posts = variable_posteriors(joint, features)

    # Print enumeration table
    rows = []
    for feat, vals in var_posts.items():
        for val, post in vals.items():
            rows.append({'feature': feat, 'value': val, 'posterior': post})
    df_enum = pd.DataFrame(rows)
    print("\nEnumeration-based posterior for each variable:")
    print(df_enum.to_string(index=False))

    # Plot grid of single‑feature posteriors
    import math
    n = len(features)
    cols = 4
    rows_plot = math.ceil(n/cols)
    fig, axes = plt.subplots(rows_plot, cols,
                             figsize=(4*cols, 3*rows_plot))
    axes = axes.flatten()
    for ax, feat in zip(axes, features):
        vals = var_posts[feat]
        ax.bar(vals.keys(), vals.values())
        ax.set_title(feat)
        ax.tick_params(axis='x', rotation=45)
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()

    # --- ROC comparison (enum vs. NB no smoothing vs. NB Laplace) ---
    train, test = train_test_split(df_disc, test_size=0.3,
                                   random_state=42)
    y_test = test['target'].values

    enum_probs = np.array([
        enumeration_posterior(joint, features, row._asdict())
        for row in test.itertuples(index=False)
    ])

    py0, pcond0 = train_naive(train, features, laplace=0)
    probs0 = predict_proba(test, features, py0, pcond0)

    py1, pcond1 = train_naive(train, features, laplace=1)
    probs1 = predict_proba(test, features, py1, pcond1)

    plot_rocs(
        y_test,
        [enum_probs, probs0, probs1],
        ['Enumeration (full joint)',
         'Naive Bayes (no smoothing)',
         'Naive Bayes (Laplace)']
    )
