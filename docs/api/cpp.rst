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

   Every entity in ``include/`` carries a doc comment, so the Doxyfile runs
   with ``WARN_IF_UNDOCUMENTED = YES`` and ``WARN_AS_ERROR = YES``: a new type,
   function or parameter without a comment fails the build rather than printing
   a warning into the log.  ``EXTRACT_ALL = YES`` stays on — it keeps the index
   structurally complete, so a gap shows up as an empty description in the
   output *and* as an error, instead of as a page that silently does not exist.
