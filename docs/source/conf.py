import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

project = 'ImWIP'
copyright = '2022, Jens Renders'
author = 'Jens Renders'
version = "1.1"
release = '1.1.0'

extensions = [
   'sphinx.ext.duration',
   'sphinx.ext.doctest',
   'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
   'sphinx.ext.intersphinx',
   'sphinxcontrib.bibtex'
]

intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    "pylops": ("https://pylops.readthedocs.io/en/stable/", None),
}

bibtex_bibfiles = ['refs.bib']

templates_path = ['_templates']
exclude_patterns = []
autodoc_member_order = 'bysource'

html_theme = 'sphinx_rtd_theme'
html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "RendersJens", # Username
    "github_repo": "ImWIP", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/source/", # Path in the checkout to the docs root
}