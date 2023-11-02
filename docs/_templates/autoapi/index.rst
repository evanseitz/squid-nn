API Reference
=============

This page contains auto-generated API reference documentation.

.. toctree::
   :titlesonly:

   {% for page in pages %}
   {% if page.top_level_object and page.display %}
   {{ page.include_path }}
   {% endif %}
   {% endfor %}


An API flowchart is provided below, showing the flow of information between
separate modules in the SQUID Python pipeline.

.. image:: /api_flowchart.png
