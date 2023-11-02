API Reference
=============

The following section contains auto-generated API reference documentation [#f1]_.

.. toctree::
   :titlesonly:

   {% for page in pages %}
   {% if page.top_level_object and page.display %}
   {{ page.include_path }}
   {% endif %}
   {% endfor %}

.. [#f1] Created with `sphinx-autoapi <https://github.com/readthedocs/sphinx-autoapi>`_


An API flowchart is provided below, showing the flow of information between
separate modules in the SQUID Python pipeline.

.. image:: /api_flowchart.png
