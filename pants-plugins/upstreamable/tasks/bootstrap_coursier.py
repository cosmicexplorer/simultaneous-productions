from __future__ import (absolute_import, division, generators, nested_scopes,
                        print_function, unicode_literals, with_statement)

from pants.util.memo import memoized_property

from upstreamable.subsystems.coursier_source import CoursierSource
from upstreamable.tasks.bootstrap_jvm_source_tool import BootstrapJvmSourceTool


class BootstrapCoursier(BootstrapJvmSourceTool):

  workunit_component_name = 'coursier-from-source'

  @classmethod
  def subsystem_dependencies(cls):
    return super(BootstrapCoursier, cls).subsystem_dependencies() + (CoursierSource.scoped(cls),)

  @memoized_property
  def _coursier_source(self):
    return CoursierSource.scoped_instance(self)

  @memoized_property
  def binary_tool_target(self):
    return self._coursier_source.coursier_binary
