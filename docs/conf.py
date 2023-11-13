# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os

project = 'squid'
copyright = '2023, Evan Seitz, David McCandlish, Justin Kinney, Peter Koo'
author = 'Evan Seitz, David McCandlish, Justin Kinney, Peter Koo'
release = '0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",  # for links
    "sphinx.ext.napoleon",  # for google style docstrings
    "sphinx.ext.viewcode",  # add links to code
    "autoapi.extension",  # to document the squid api
    "sphinx_click",  # to document click command line
    "sphinx_copybutton",  # add copy button to top-right of code blocks
    "numpydoc", # support for the Numpy docstring format
    #"nbsphinx", # required for reading jupyter notebooks
]

# Do NOT automatically execute notebooks when building.
#nbsphinx_execute = 'never'

# Internationalization.
language = "en"

# AutoAPI options.
autoapi_type = "python"
autoapi_dirs = ["../squid"]
autoapi_options = [
    "members",
    "undoc-members",
    # "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_ignore = ["*cli*", "*__main__.py"]

templates_path = ["_templates"]
autoapi_template_dir = "_templates/autoapi"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "openslide": ("https://openslide.org/api/python/", None),
}

# For editing the pages.
html_context = {
    "github_user": "evanseitz",
    "github_repo": "squid-nn",
    "github_version": "main",
    "doc_path": "docs",
}


html_theme = 'sphinx_rtd_theme' #"pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = 'logo_light_crop.png'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    'canonical_url': 'https://squid-nn.readthedocs.io',
    #'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'none', #'bottom',
    'style_external_links': False,
    #'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 3,
    'includehidden': True,
    'titles_only': False
}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'squiddoc'


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']


# -- Options for EPUB output
epub_show_urls = 'footnote'

#html_favicon = "_static/logo_light.png"
