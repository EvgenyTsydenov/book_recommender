import csv
import gzip
import json
import os

import gdown


def unpack_gz(path_load: str, path_save: str) -> None:
    """Unpack GZ archives to csv files.

    :param: path_load: path to archive.
    :param: path_save: path to save csv.
    """
    with gzip.open(path_load) as fin:
        with open(path_save, 'w', encoding='utf-8') as fout:
            header = True
            w = None
            for line in fin:
                d = json.loads(line)
                if header:
                    w = csv.DictWriter(fout, fieldnames=d.keys())
                    w.writeheader()
                    header = False
                w.writerow(d)


if __name__ == '__main__':
    """
    The files are located on Google Drive. 
    You can see how to download them here:
    https://github.com/MengtingWan/goodreads/blob/master/download.ipynb
    """

    # File ids on Google Drive
    file_ids = {
        '1CHTAaNwyzvbi1TR08MJrJ03BxA266Yxr': 'book_id_map.csv',
        '15ax-h0Oi_Oyee8gY_aAQN6odoijmiz6Q': 'user_id_map.csv',
        '1LXpK1UfqtP89H1tYy0pBGHjYk8IhigUK': 'goodreads_books.json.gz',
        '19cdwyXwfXx_HDIgxXaHzH0mrx8nMyLvC': 'goodreads_book_authors.json.gz',
        '1op8D4e5BaxU2JcPUgxM3ZqrodajryFBb': 'goodreads_book_series.json.gz',
        '1TLmSvzHvTLLLMjMoQdkx6pBWon-4bli7': 'goodreads_book_works.json.gz',
        '1zmylV7XW2dfQVCLeg1LbllfQtHD2KUon': 'goodreads_interactions.csv'
    }

    # Where to save files
    folder = 'data_original'
    if not os.path.exists(folder):
        os.mkdir(folder)

    # Download each file
    for file_id, file_name in file_ids.items():
        url = f'https://drive.google.com/uc?id={file_id}'
        path_save = os.path.join(folder, file_name)
        gdown.download(url, output=path_save, quiet=True)
        print(f'File "{file_name}" was downloaded to "{path_save}".')

        # Unzip
        name, file_ext = os.path.splitext(file_name)
        if file_ext == '.gz':
            new_path = os.path.join(folder, f'{name}.csv')
            unpack_gz(path_save, new_path)
            os.remove(path_save)
            print(f'File "{file_name}" was unzipped to "{new_path}".')
