Creating conda environment:
conda env update -f environment.yaml --prune

Activating conda environment:
conda activate country_guesser

Updating conda enviroment (after changing environment.yaml):
conda env update -f environment.yaml --prune

File functions:
- data
    - load_data: retrieves broad datapoints from mapillary (obtains image id and assigns country)
    - create_training_data: retrieves actual image URLs and splits data smartly.
    - inspect_data: contains functions for insights into the dataset.

Omitted countries:
    - Didn't fit nicely in the bounding boxes
        - Turkey, Georgia, Armenia, Azerbaijan
    - Microstates
        - Vatican, Andorra, Liechtenstein, Monaco, San Marino
    - Images not available
        - Ukraine