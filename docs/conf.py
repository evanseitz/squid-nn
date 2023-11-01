# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'squid'
copyright = '2023, Evan Seitz, David McCandlish, Justin Kinney, Peter Koo'
author = 'Evan Seitz, David McCandlish, Justin Kinney, Peter Koo'
release = '0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme' #'alabaster'
html_static_path = ['_static']

html_logo = 'logo_light.png'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}
