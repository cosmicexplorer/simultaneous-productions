from __future__ import (absolute_import, division, generators, nested_scopes,
                        print_function, unicode_literals, with_statement)

from pants.util.memo import memoized_property

from upstreamable.subsystems.ensime_gen_source import EnsimeGenSource
from upstreamable.tasks.bootstrap_jvm_source_tool import BootstrapJvmSourceTool


class BootstrapEnsimeGen(BootstrapJvmSourceTool):

  workunit_component_name = 'ensime-gen'

  @classmethod
  def subsystem_dependencies(cls):
    return super(BootstrapEnsimeGen, cls).subsystem_dependencies() + (EnsimeGenSource.scoped(cls),)

  @memoized_property
  def _ensime_gen_source(self):
    return EnsimeGenSource.scoped_instance(self)

  @memoized_property
  def binary_tool_target(self):
    return self._ensime_gen_source.ensime_gen_binary
