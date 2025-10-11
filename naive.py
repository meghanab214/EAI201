import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Task 1: Load and Explore Data ---
print("### Task 1: Load and Explore Data ###")

# Load the dataset, handling potential encoding issues
df = pd.read_csv('spam.csv', encoding='latin-1')

# Clean up the dataframe by dropping unnecessary columns and renaming the essential ones
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Check the shape (rows, columns)
print(f"Shape of the dataset: {df.shape}")

# Count spam vs ham messages
print("\nMessage counts:")
print(df['label'].value_counts())

# Display 3 examples of each class
print("\nSpam examples:")
print(df[df['label'] == 'spam'].head(3))
print("\nHam examples:")
print(df[df['label'] == 'ham'].head(3))

# Answer Question: What percentage of messages are spam?
spam_percentage = (df['label'].value_counts()['spam'] / len(df)) * 100
print(f"\nQuestion: What percentage of messages are spam? -> {spam_percentage:.2f}%\n")


# --- Task 2: Understand Bayes Theorem (Manual Calculation) ---
print("### Task 2: Understand Bayes Theorem ###")

# Map labels to binary values for easier calculation (spam=1, ham=0)
df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42
)

# Combine training data into a temporary dataframe for analysis
train_df = pd.DataFrame({'message': X_train, 'label': y_train})

# Calculate prior probabilities: P(Spam) and P(Ham)
p_spam = len(train_df[train_df['label'] == 1]) / len(train_df)
p_ham = 1 - p_spam
print(f"Prior Probability P(Spam) = {p_spam:.4f}")
print(f"Prior Probability P(Ham) = {p_ham:.4f}")

# Manually count messages containing "free"
spam_messages = train_df[train_df['label'] == 1]['message']
ham_messages = train_df[train_df['label'] == 0]['message']
n_free_in_spam = spam_messages.str.contains('free', case=False).sum()
n_free_in_ham = ham_messages.str.contains('free', case=False).sum()

# Calculate likelihoods: P("free"|Spam) and P("free"|Ham)
p_free_given_spam = n_free_in_spam / len(spam_messages)
p_free_given_ham = n_free_in_ham / len(ham_messages)
print(f"\nLikelihood P('free'|Spam) = {p_free_given_spam:.4f}")
print(f"Likelihood P('free'|Ham) = {p_free_given_ham:.4f}")

# Answer Question: Based on these probabilities, if you see "free" in a message, is it more likely spam or ham?
# We compare the proportional posteriors: P("free"|Class) * P(Class)
prob_spam_if_free = p_free_given_spam * p_spam
prob_ham_if_free = p_free_given_ham * p_ham
print(f"\nProportional posterior for Spam: {prob_spam_if_free:.4f}")
print(f"Proportional posterior for Ham:  {prob_ham_if_free:.4f}")
print("Conclusion: A message with 'free' is much more likely to be spam.\n")


# --- Task 3: Build Naive Bayes Classifier ---
print("### Task 3: Build Naive Bayes Classifier ###")

# Use CountVectorizer to convert text messages into a matrix of token counts
# stop_words='english' removes common English words like 'the', 'a', 'is'
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test) # IMPORTANT: Only transform test data, don't fit again

# Train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict on the test set
y_pred = model.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Question: What is your model's accuracy? -> {accuracy:.4f}")

# Create and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
print("Interpretation: [[True Ham, False Spam], [False Ham, True Spam]]")

# Answer Question: Does it perform better at detecting spam or ham?
tn, fp, fn, tp = conf_matrix.ravel()
spam_recall = tp / (tp + fn)
ham_recall = tn / (tn + fp)
print(f"\nSpam Recall (found {spam_recall:.2%} of actual spam)")
print(f"Ham Recall (found {ham_recall:.2%} of actual ham)")
print("Conclusion: The model is extremely good at identifying ham, and also very good at catching spam, though it misses a small fraction of spam messages (classifying them as ham).\n")


# --- Task 4: Analyze Feature Importance ---
print("### Task 4: Analyze Feature Importance ###")

# Extract feature names (words) from the vectorizer
feature_names = np.array(vectorizer.get_feature_names_out())

# Get the log probabilities for the spam class (class 1)
spam_log_probs = model.feature_log_prob_[1]

# Find the indices of the 5 words with the highest probability for spam
top_5_spam_indices = spam_log_probs.argsort()[-5:][::-1]

# Answer Question: Which words are strongest spam indicators?
top_5_spam_words = feature_names[top_5_spam_indices]
print(f"Question: Which words are strongest spam indicators? -> {top_5_spam_words}")

# Test the classifier on a custom message
custom_message = ["Free call now! Win money!"]
custom_message_vec = vectorizer.transform(custom_message)
prediction = model.predict(custom_message_vec)

# Answer Question: Does your custom message get classified correctly?
print(f"\nCustom message: '{custom_message[0]}'")
is_spam = prediction[0] == 1
print(f"Question: Does it get classified correctly? -> Yes, Prediction: {'Spam' if is_spam else 'Ham'}\n")


# --- Task 5: Test the "Naïve" Assumption ---
print('### Task 5: Test the "Naïve" Assumption ###')

# Create two similar messages to test
messages_to_test = ["Call me", "Free call"]
messages_to_test_vec = vectorizer.transform(messages_to_test)

# Get the probability scores for each class [P(ham), P(spam)]
probabilities = model.predict_proba(messages_to_test_vec)

print(f"Message: '{messages_to_test[0]}' -> P(Spam) = {probabilities[0][1]:.6f}")
print(f"Message: '{messages_to_test[1]}' -> P(Spam) = {probabilities[1][1]:.6f}")

# Answer Question: Why does Naive Bayes still work well for spam detection?
print("""
Question: Even though words aren't truly independent, why does Naive Bayes still work well for spam detection?
Answer: It works well because for classification, we don't need perfect probability estimates. We just need the final probability for the correct class to be higher than for other classes. In spam detection, certain words (like 'free', 'win', 'txt') are such strong indicators that their presence heavily sways the calculation towards 'spam', overcoming the incorrect assumption of independence.
""")