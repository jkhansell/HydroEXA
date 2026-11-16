C++ API --- ``include/`` + ``src/``
==========================================

Generated from the Doxygen XML of the solver sources by Breathe.  The Doxyfile
names *directories*, so split or merged files never require a documentation
change; new public types appear here without an edit.

.. ifconfig:: has_doxygen_xml

   .. doxygenindex::
      :project: HydroEXA
      :outline:

.. ifconfig:: not has_doxygen_xml

   .. warning::

      Doxygen XML was not found, so the C++ reference is omitted from this
      build.  Generate it first:

      .. code-block:: console

         $ cd docs/doxygen && doxygen Doxyfile

      The XML lands in ``docs/doxygen/xml/`` (gitignored) and Breathe is
      enabled automatically on the next Sphinx run.

.. note::

   While the documentation pass is in progress the Doxyfile sets
   ``EXTRACT_ALL = YES`` and ``WARN_IF_UNDOCUMENTED = NO``: the structure is
   real, the prose is not yet.  The endpoint is ``WARN_IF_UNDOCUMENTED = YES``
   plus ``-W`` in CI, at which point every public entity must carry a
   doc-comment.
