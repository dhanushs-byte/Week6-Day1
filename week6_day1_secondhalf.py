
import numpy as np
import pandas as pd
import re

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

print("WEEK 6 - DAY 1 SECOND HALF")

# LOAD MODEL
model = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================
# TASK 1: COSINE SIMILARITY FROM SCRATCH
# =====================================

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\nTASK 1: COSINE SIMILARITY")

tests = [
    ([1,2,3], [2,4,6]),
    ([1,0], [0,1]),
    ([1,0], [-1,0])
]

for a, b in tests:
    result = cosine_similarity(a,b)
    sklearn_result = sklearn_cosine(
        np.array(a).reshape(1,-1),
        np.array(b).reshape(1,-1)
    )[0][0]

    print(a, "vs", b, "->", round(result,4),
          "| sklearn:", round(sklearn_result,4))

a = np.array([1,2,3])
b = np.array([2,1,0])

a_norm = a / np.linalg.norm(a)
b_norm = b / np.linalg.norm(b)

print("Normalized dot product:", np.dot(a_norm,b_norm))

# =====================================
# TASK 2.1: SEMANTIC SEARCH
# =====================================

print("\nTASK 2.1: SEMANTIC SEARCH")

faqs = [
"How can I reset my account password?",
"How do I change my registered email address?",
"Where can I view my monthly invoice?",
"How can I cancel my subscription?",
"What payment methods are accepted?",
"How do I contact customer support?",
"Can I download my account data?",
"How do I enable two factor authentication?",
"Why was my payment declined?",
"How can I update my billing address?",
"Can I change my subscription plan?",
"How do I restore access to a locked account?",
"Where can I find the mobile application?",
"How do I request a refund?",
"How can I change notification preferences?"
]

faq_vectors = model.encode(faqs, convert_to_tensor=True)

queries = [
"I forgot my password and cannot login",
"I need my monthly bill",
"I want to cancel my plan"
]

for query in queries:
    q_vector = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(q_vector, faq_vectors)[0]
    top = scores.argsort(descending=True)[:3]

    print("\nQuery:", query)

    for i in top:
        print(faqs[int(i)], round(float(scores[i]),4))

# =====================================
# TASK 2.2: DUPLICATE DETECTION
# =====================================

print("\nTASK 2.2: DUPLICATE DETECTION")

complaints = [
"I was charged twice for the same order.",
"My card was billed two times for one purchase.",
"The delivery arrived three days late.",
"My package was delivered much later than expected.",
"I cannot login after resetting my password.",
"After changing my password I cannot access my account.",
"The product arrived broken.",
"The item received is damaged.",
"I want to update my phone number.",
"Please change my mobile number.",
"The app crashes when opening reports.",
"The reports section crashes every time."
]

threshold = 0.75

vectors = model.encode(complaints, convert_to_tensor=True)
matrix = util.cos_sim(vectors,vectors)

print("Threshold:",threshold)

duplicate_pairs = []

for i in range(len(complaints)):
    for j in range(i+1,len(complaints)):
        score = float(matrix[i][j])

        if score >= threshold:
            duplicate_pairs.append((i+1,j+1,score))
            print(
                f"Complaint {i+1} and {j+1}: {score:.4f}"
            )

# =====================================
# TASK 2.3: K-MEANS CLUSTERING
# =====================================

print("\nTASK 2.3: K-MEANS CLUSTERING")

headlines = [
"Central bank changes interest rates",
"Stock markets rise after earnings",
"Inflation slows down",
"Technology shares gain value",
"Government releases economic forecast",
"Major bank reports higher profits",
"Investors react to currency changes",
"New tax policy announced",
"Bond yields fall",
"Retail sales increase",
"New AI assistant launched",
"Technology company launches AI model",
"Researchers develop computer chip",
"Cybersecurity data breach reported",
"Cloud computing adoption grows",
"Smartphone maker launches new device",
"Robotics startup automates warehouse",
"Software update adds privacy controls",
"Scientists develop battery technology",
"Social media expands creator tools",
"National team wins football match",
"Athlete sets world record",
"Tournament final schedule announced",
"Cricket club signs international player",
"Tennis champion reaches semifinals",
"Coach discusses team strategy",
"Marathon runners compete",
"Basketball league announces awards",
"Olympic qualification rules announced",
"Fans celebrate dramatic victory"
]

headline_vectors = model.encode(headlines)

kmeans = KMeans(n_clusters=3,random_state=42,n_init=10)
clusters = kmeans.fit_predict(headline_vectors)

for i in range(3):
    print("\nCluster",i+1)

    for j in range(len(headlines)):
        if clusters[j] == i:
            print("-",headlines[j])

# =====================================
# TASK 2.4: PRODUCT RECOMMENDATION
# =====================================

print("\nTASK 2.4: PRODUCT RECOMMENDATION")

products = [
"Ultralight laptop with long battery life",
"Gaming laptop with powerful graphics",
"Tablet with keyboard",
"Noise cancelling headphones",
"Wireless mouse",
"USB C hub",
"Portable SSD storage",
"Laptop travel backpack",
"Smartphone",
"E-reader",
"27 inch monitor",
"HD webcam",
"Portable power bank",
"Mechanical keyboard",
"Adjustable laptop stand"
]

product_vectors = model.encode(products,convert_to_tensor=True)

query = "Lightweight laptop for travel and studying"

query_vector = model.encode(query,convert_to_tensor=True)

scores = util.cos_sim(query_vector,product_vectors)[0]

top = scores.argsort(descending=True)[:3]

for i in top:
    print(products[int(i)],round(float(scores[i]),4))

# =====================================
# BONUS: PRECISION AND RECALL
# =====================================

print("\nBONUS: PRECISION AND RECALL")

labelled_pairs = [
(0,1,1),(2,3,1),(4,5,1),
(6,7,1),(8,9,1),(10,11,1),
(0,2,0),(0,4,0),(2,6,0),
(4,8,0),(6,10,0),(8,11,0)
]

y_true = []
y_pred = []

for i,j,label in labelled_pairs:
    score = float(matrix[i][j])
    prediction = int(score >= threshold)

    y_true.append(label)
    y_pred.append(prediction)

print("Precision:",precision_score(y_true,y_pred,zero_division=0))
print("Recall:",recall_score(y_true,y_pred,zero_division=0))
print("F1 Score:",f1_score(y_true,y_pred,zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_true,y_pred))

print("\nALL TASKS COMPLETED")
