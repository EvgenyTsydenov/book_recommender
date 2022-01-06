import os

import pandas as pd

if __name__ == '__main__':
    # Set these roles manually
    roles = pd.DataFrame(
        columns=['code', 'description', 'seq'],
        data=[['A', 'Author', 10], ['I', 'Illustrator', 20],
              ['4', 'Read by', 30], ['U', 'Foreword by', 40],
              ['V', 'Introduction by', 50], ['1', 'Preface by', 60],
              ['E', 'Editor', 70], ['D', 'Afterword by', 80],
              ['Q', 'Epilogue by', 90], ['P', 'Photographer', 100],
              ['T', 'Translator', 110], ['L', 'Compiled by', 120],
              ['O', 'Designed by', 130],
              ['0', 'Text by (art/photo books)', 999],
              ['2', 'Prologue by', 999], ['3', 'Produced by', 999],
              ['5', 'Retold by', 999], ['6', 'Revised by', 999],
              ['7', 'Selected by', 999], ['8', 'Series Editor', 999],
              ['9', 'Supplement by', 999], ['A1', 'Appendix by', 999],
              ['A2', 'Arranged by (music)', 999], ['A3', 'Conductor', 999],
              ['A4', 'Dramatized by', 999], ['A5', 'Lyrics by', 999],
              ['A6', 'Maps by', 999], ['B', 'Abridged by', 999],
              ['C', 'Adapted by', 999], ['F', 'Annotations by', 999],
              ['G', 'As told to', 999], ['H', 'As told by', 999],
              ['J', 'Contribution by', 999], ['K', 'Commentaries by', 999],
              ['M', 'Created by', 999], ['N', 'Concept by', 999],
              ['R', 'Experiments by', 999], ['S', 'Footnotes by', 999],
              ['W', 'Memoir by', 999], ['Y', 'Narrated by', 999],
              ['Z', 'Notes by', 999]])
    roles.set_index('code', inplace=True)

    # Save
    path_roles_raw = os.path.join('..', 'data_raw', 'contributor_roles.csv')
    roles.to_csv(path_roles_raw)
