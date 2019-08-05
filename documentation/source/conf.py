import os
import sys
sys.path.insert(0, os.path.abspath('../../gpys'))
sys.path.insert(0, os.path.abspath('../../gpys/archive'))
sys.path.insert(0, os.path.abspath('../..'))
# -- Project information -----------------------------------------------------
project = u'gpys'
copyright = u'2019, Peter Matheny'
author = u'Peter Matheny'
version = u''
release = u''
manpages_url = 'https://manpages.debian.org/{path}'
extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx'
]
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
language = None
exclude_patterns = []
pygments_style = None
html_theme = 'classic'
html_static_path = ['_static']
html_sidebars = {
   '**': ['globaltoc.html'],
   'using/windows': ['windowssidebar.html'],
}
html_use_index = False
htmlhelp_basename = 'ParallelGAMITdoc'
