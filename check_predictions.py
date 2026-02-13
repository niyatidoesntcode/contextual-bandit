import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Load data
train_users = pd.read_csv('data/train_users.csv')
test_users = pd.read_csv('data/test_users.csv')

# Preprocess
train_users['label'] = train_users['label'].str.replace('user_', 'User', regex=False).str.capitalize()
feature_cols = [col for col in train_users.columns if col not in ['user_id', 'label']]

X_train = train_users[feature_cols].fillna(0)
y_train = train_users['label']
X_test = test_users[feature_cols].fillna(0)

# Encode categoricals
for col in X_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
    le.fit(combined)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
le_y = LabelEncoder()
y_train_enc = le_y.fit_transform(y_train)
clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
clf.fit(X_train_scaled, y_train_enc)

# Predict on test set
preds = clf.predict(X_test_scaled)
pred_labels = le_y.inverse_transform(preds)

# Check distribution
print("="*70)
print("PREDICTION DISTRIBUTION ON 2000 TEST USERS")
print("="*70)
unique, counts = np.unique(pred_labels, return_counts=True)
for label, count in zip(unique, counts):
    pct = count/len(pred_labels)*100
    print(f"  {label}: {count:4d} users ({pct:5.1f}%)")
print(f"\nTotal: {len(pred_labels)} users")
print("="*70)

# Check learned Q-values from bandits
from rlcmab_sampler import sampler
env_sampler = sampler(80)

# Quick bandit training
class UCB:
    def __init__(self, c=2.0, n_arms=4):
        self.n = np.zeros(n_arms)
        self.q = np.zeros(n_arms)
        self.c = c
        self.t = 0
    def select_action(self):
        self.t += 1
        if 0 in self.n:
            return np.argmin(self.n)
        confidence = self.c * np.sqrt(np.log(self.t) / self.n)
        return np.argmax(self.q + confidence)
    def update(self, action, reward):
        self.n[action] += 1
        self.q[action] += (reward - self.q[action]) / self.n[action]

def get_global_arm(context_idx, category_idx):
    return context_idx * 4 + category_idx

# Train bandits
bandits = [UCB(c=2.0) for _ in range(3)]
for t in range(5000):
    ctx_idx = np.random.randint(0, 3)
    action = bandits[ctx_idx].select_action()
    global_arm = get_global_arm(ctx_idx, action)
    reward = env_sampler.sample(global_arm)
    bandits[ctx_idx].update(action, reward)

# Show learned Q-values
print("\nLEARNED Q-VALUES (Best action per context):")
print("="*70)
categories = ['ENTERTAINMENT', 'EDUCATION', 'TECH', 'CRIME']
contexts = ['User1', 'User2', 'User3']
for i, context in enumerate(contexts):
    best_cat_idx = np.argmax(bandits[i].q)
    best_cat = categories[best_cat_idx]
    best_q = bandits[i].q[best_cat_idx]
    print(f"  {context}: {best_cat} (Q={best_q:.2f})")
    print(f"    All Q-values: {dict(zip(categories, bandits[i].q.round(2)))}")
print("="*70)

# Now show recommendations for different contexts
print("\nRECOMMENDATIONS BY USER CONTEXT:")
print("="*70)
for i in range(3):
    context = contexts[i]
    best_action = np.argmax(bandits[i].q)
    best_category = categories[best_action]
    num_predicted = counts[i] if i < len(counts) else 0
    print(f"  {context} → {best_category} ({num_predicted} users predicted as {context})")
print("="*70)
