"""
Simple pickle loader - no dependencies
"""
import pickle
import sys

file_path = r"C:\Users\mosaq\Desktop\AI Proj\race_rc_project\models\model_b\traditional\distractor_hint_generator_w2v_model"

try:
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    
    print(f"✓ Successfully loaded pickle file")
    print(f"Type: {type(obj).__name__}")
    print(f"Module: {type(obj).__module__}")
    
    # Check attributes
    if hasattr(obj, '__dict__'):
        attrs = obj.__dict__
        print(f"\nAttributes ({len(attrs)}):")
        for key in list(attrs.keys())[:10]:  # First 10
            val = attrs[key]
            print(f"  - {key}: {type(val).__name__}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
