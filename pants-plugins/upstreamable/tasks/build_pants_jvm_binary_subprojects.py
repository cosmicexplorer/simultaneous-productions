from __future__ import (absolute_import, division, generators, nested_scopes,
                        print_function, unicode_literals, with_statement)

from pants.task.task import Task
from pants.util.objects import datatype
from upstreamable.tasks.bootstrap_jvm_source_tool import BootstrapJar


class BuildPantsJvmBinarySubprojects(Task):
  