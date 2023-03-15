# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
#sys.path.insert(0, os.path.abspath('..'))

#sys.path.insert(0, os.path.abspath('../../RAI'))


#sys.path.insert(0, os.path.abspath('../..'))


#so we can import RAI
#sys.path.append(os.path.join(os.path.dirname(__name__), ".."))
#sys.path.insert(0, os.path.abspath('../..'))
#import RAI

# source code directory, relative to this file, for sphinx-autobuild
#sys.path.insert(0, os.path.abspath('..'))

sys.path.insert(0, os.path.abspath('../'))

project = 'RAI Documentation'
copyright = '2023, sharfa@Cisco'
author = 'RAI Contributers'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration



source_suffix = '.rst'

master_doc = 'index'



extensions = ['sphinx.ext.autodoc','sphinx.ext.viewcode','sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_tabs.tabs','sphinx_togglebutton','sphinx_copybutton']

sphinx_tabs_valid_builders = ['linkcheck']
    

# Fix for read the docs
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    html_theme = 'default'
else:
    html_theme = 'sphinx_rtd_theme'


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []