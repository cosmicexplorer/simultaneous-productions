from __future__ import (absolute_import, division, generators, nested_scopes,
                        print_function, unicode_literals, with_statement)

import os

from pants.base.build_environment import get_pants_cachedir
from pants.option.custom_types import target_option
from pants.subsystem.subsystem import Subsystem
from pants.util.memo import memoized_property


class CoursierSource(Subsystem):

  options_scope = 'coursier-source'

  @classmethod
  def register_options(cls, register):
    super(CoursierSource, cls).register_options(register)

    register('--coursier-binary', type=target_option, default='//:coursier-source', advanced=True,
             help='The target to use for coursier sources.')

  @memoized_property
  def coursier_binary(self):
    return self.get_options().coursier_binary
