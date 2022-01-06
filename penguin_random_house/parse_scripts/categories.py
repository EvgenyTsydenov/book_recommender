import json
import os

from tqdm import tqdm

if __name__ == '__main__':

    # Paths
    path_raw_cats = os.path.join('..', 'data_raw', 'categories.txt')
    path_parsed_cats = os.path.join('..', 'data_interm', 'categories.txt')

    # Parse the file line by line
    with open(path_raw_cats, 'r') as categories_raw:
        with open(path_parsed_cats, 'w') as categories_parsed:
            for category in tqdm(categories_raw):
                # Load data about category
                category_data = json.loads(category)

                # Extract info
                info = {
                    'cat_id': category_data.get('catId'),
                    'description': category_data.get('description'),
                    'menu_text': category_data.get('menuText'),
                    'parent': category_data.get('parent')
                }

                # Save
                data_string = json.dumps(info)
                categories_parsed.write(data_string)
                categories_parsed.write('\n')
