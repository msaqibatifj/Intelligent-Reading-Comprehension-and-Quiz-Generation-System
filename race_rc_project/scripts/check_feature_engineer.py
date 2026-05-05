import joblib
from pathlib import Path

model_path = Path('race_rc_project/models/model_a/traditional/feature_engineer.pkl')
loaded = joblib.load(model_path)
print(f'Type: {type(loaded)}')
if isinstance(loaded, dict):
    print(f'Is dict with keys: {list(loaded.keys())}')
    for k, v in list(loaded.items())[:3]:
        print(f'  {k}: {type(v).__name__}')
else:
    print(f'Object type: {type(loaded).__name__}')
    print(f'Has extract_lexical_features: {hasattr(loaded, "extract_lexical_features")}')
