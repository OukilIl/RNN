import kagglehub

# Download latest version
path = kagglehub.dataset_download("gulerosman/hg14-handgesture14-dataset")

print("Path to dataset files:", path)