Assignment 2: Stolen Model Detection
Team: shreyakolhapure9130-byte

How to Reproduce the Best Result

1. Clone the repository

git clone [https://github.com/shreyakolhapure9130-byte/stolen-model-detection.git](https://github.com/shreyakolhapure9130-byte/stolen-model-detection.git)

2. Install dependencies

pip install torch torchvision pandas tqdm safetensors

3. Prepare the data and models

Ensure the following folder structure exists:

tml26_task2/
target_model/
weights.safetensors

suspect_models/
suspect_000.safetensors
suspect_001.safetensors
...
suspect_359.safetensors

4. Run the detection

python task_template.py

This will generate submission.csv containing stealing confidence scores for all 360 suspect models and it will be used for leaderboard evaluation.

Expected Process:
The script loads the CIFAR-100 ResNet-18 target model, computes predictions on a fixed subset of CIFAR-100 test images, and evaluates all suspect models one by one. For each suspect model, it computes weight similarity using cosine similarity between fully connected layer weights and functional similarity based on prediction agreement with the target model. These two scores are combined to produce the final stealing confidence score:

score = 0.5 * weight_similarity + 0.5 * functional_similarity

The final results are saved into submission.csv.

Note on Approach:
This approach combines white-box parameter similarity and behavioral similarity to detect stolen or derived models, including direct copies, fine-tuned models, and models obtained via knowledge distillation.

---
