from __future__ import (absolute_import, division, generators, nested_scopes,
                        print_function, unicode_literals, with_statement)

from pants.base.exceptions import TargetDefinitionException
from pants.base.payload import Payload
from pants.base.payload_field import PrimitiveField
from pants.build_graph.target import Target


class PantsJvmBinarySubproject(Target):

  @classmethod
  def alias(cls):
    return 'pants_jvm_binary_subproject'

  def __init__(self,
               address=None,
               payload=None,
               sources=None,
               relative_target=None,
               **kwargs):

    payload = payload or Payload()
    payload.add_fields({
      'sources': self.create_sources_field(sources, address.spec_path, key_arg='sources'),
    })
    super(PantsSubproject, self).__init__(address=address, payload=payload, **kwargs)

    if not relative_target:
      raise TargetDefinitionException(self, "relative_target must be provided.")
    self._relative_target = relative_target
