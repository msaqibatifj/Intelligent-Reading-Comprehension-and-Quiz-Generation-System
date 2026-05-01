"""
Check what's inside distractor_hint_generator_w2v_model file
"""
import os
import sys

model_dir = r"C:\Users\mosaq\Desktop\AI Proj\race_rc_project\models\model_b\traditional"
w2v_file = os.path.join(model_dir, 'distractor_hint_generator_w2v_model')

print(f"File: {w2v_file}")
print(f"File exists: {os.path.exists(w2v_file)}")
print(f"File size: {os.path.getsize(w2v_file)} bytes")

# Try to load as joblib pickle
try:
    import joblib
    obj = joblib.load(w2v_file)
    print(f"\n✓ Loaded as joblib pickle")
    print(f"Type: {type(obj)}")
    print(f"Object: {obj}")
    if hasattr(obj, '__dict__'):
        print(f"Attributes: {obj.__dict__.keys()}")
except Exception as e:
    print(f"✗ Not joblib pickle: {e}")

# Try to load as gensim Word2Vec model
try:
    from gensim.models import Word2Vec
    model = Word2Vec.load(w2v_file)
    print(f"\n✓ Loaded as gensim Word2Vec model")
    print(f"Type: {type(model)}")
    print(f"Vocabulary size: {len(model.wv)}")
    print(f"Vector size: {model.vector_size}")
except Exception as e:
    print(f"✗ Not gensim Word2Vec: {e}")

# Try to load as gensim KeyedVectors
try:
    from gensim.models import KeyedVectors
    kv = KeyedVectors.load(w2v_file)
    print(f"\n✓ Loaded as gensim KeyedVectors")
    print(f"Type: {type(kv)}")
    print(f"Vocabulary size: {len(kv)}")
except Exception as e:
    print(f"✗ Not gensim KeyedVectors: {e}")

# Try to load as pickle
try:
    import pickle
    with open(w2v_file, 'rb') as f:
        obj = pickle.load(f)
    print(f"\n✓ Loaded as pickle")
    print(f"Type: {type(obj)}")
except Exception as e:
    print(f"✗ Not pickle: {e}")
