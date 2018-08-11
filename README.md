
# AutoEncoder for anime recommendation
This model was used for my entry for the Mangaki Data Challenge (http://research.mangaki.fr/2017/07/18/mangaki-data-challenge-en/)

Based on DeepRecommender from NVIDIA (https://github.com/NVIDIA/DeepRecommender)

## What I learned:

- A model that works well for the Netflix dataset *might* not work for another one (not enough data)
- Tensorflow doesn't like sparse data. It's still very difficult to load sparse matrices even with the new Datasets API
- Conventional Machine learning models probably work better at this scale
