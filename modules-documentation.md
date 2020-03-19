The documentation shows some small and useful modules that helps the progress of the project

# 1. Layout detection

# 2. OCR
Download prepared pickled dataset at [here](https://drive.google.com/file/d/1-0bRc91c-50S38oC3JYE9BcWwogheiRg/view?usp=sharing)
Also the pickled labels for the dataset at here [here](https://drive.google.com/file/d/1-5jkZ7YT23tCd1-P_5AvKmR3cTyQIJ4n/view?usp=sharing)

Use module from `/helpers/FilePickling` to load pickled file, for example:

```python
from helpers.FilePickling import pkl_load
img_patches = pkl_load("patches.pkl")
img_patches_labels = pkl_load("labels.pkl") 
```

# 3. KV

# 4. Explainer
