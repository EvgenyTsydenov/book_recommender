import json
import os

from tqdm import tqdm


def parse_book(book_data: dict) -> dict:
    """Parse book core data."""
    info = {}
    for param in ['isbn', 'title', 'onsale', 'price', 'format',
                  'language', 'pages', 'publisher']:
        info[param] = book_data.get(param)
    info['cover'] = f'https://images.randomhouse.com/cover/{info["isbn"]}'
    info['format_family'] = book_data.get('formatFamily')
    info['projected_minutes'] = book_data.get('projectedMinutes')
    info['series_number'] = book_data.get('seriesNumber')
    return info


def parse_authors(authors_data: list) -> list:
    """Extract information about contributors."""
    authors = []
    for author in authors_data:
        authors.append({
            'author_id': author.get('authorId'),
            'first_name': author.get('first'),
            'last_name': author.get('last'),
            'company': author.get('company'),
            'client_source_id': author.get('clientSourceId'),
            'role': author.get('contribRoleCode')
        })
    return authors


def parse_categories(category_data: list) -> list:
    """Extract information about categories.

    Since we downloaded data about categories separately,
    keep here only category_id and the sequence.
    """
    categories = []
    for cat in category_data:

        # Read PRH docs about sequencing
        if cat.get('seq', 0) > 0:
            categories.append({
                'category_id': cat.get('catId'),
                'seq': cat.get('seq')
            })
    return categories


def parse_series(series_data: list) -> list:
    """Extract information about series."""
    series = []
    for item in series_data:
        series.append({
            'series_id': item.get('seriesCode'),
            'name': item.get('seriesName'),
            'description': item.get('description'),
            'series_count': item.get('seriesCount'),
            'is_numbered': item.get('isNumbered'),
            'is_kids': item.get('isKids')
        })
    return series


def parse_works(works_data: list) -> list:
    """Extract information about works."""
    works = []
    for work in works_data:
        works.append({
            'work_id': work.get('workId'),
            'title': work.get('title'),
            'author': work.get('author'),
            'onsale': work.get('onsale'),
            'language': work.get('language'),
            'series_number': work.get('seriesNumber')
        })
    return works


def parse_content(content_data: dict) -> dict:
    """Extract long text data."""
    content = {}
    for param in ['positioning', 'jacketquotes',
                  'flapcopy', 'keynote', 'excerpt']:
        content[param] = content_data.get(param)
    return content


if __name__ == '__main__':

    # Paths
    path_raw_books = os.path.join('..', 'data_raw', 'books.txt')
    path_parsed_books = os.path.join('..', 'data_interm', 'books.txt')

    # Parse the file line by line
    with open(path_raw_books, 'r') as books_raw:
        with open(path_parsed_books, 'w') as books_parsed:
            for book in tqdm(books_raw):
                book_data = json.loads(book)

                # Get core book data
                info = parse_book(book_data)

                # Parse relative info
                embeds = {}
                for embed in book_data['_embeds']:
                    embeds.update(embed)
                info['authors'] = parse_authors(embeds['authors'])
                info['categories'] = parse_categories(embeds['categories'])
                info['series'] = parse_series(embeds['series'])
                info['works'] = parse_works(embeds['works'])
                info.update(parse_content(embeds['content']))

                # Save
                data_string = json.dumps(info)
                books_parsed.write(data_string)
                books_parsed.write('\n')
