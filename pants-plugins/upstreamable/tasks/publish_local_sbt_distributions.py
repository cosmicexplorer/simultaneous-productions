from __future__ import (absolute_import, division, generators, nested_scopes,
                        print_function, unicode_literals, with_statement)

import os

from pants.base.build_environment import get_buildroot
from pants.base.exceptions import TaskError
from pants.base.workunit import WorkUnit, WorkUnitLabel
from pants.java.distribution.distribution import DistributionLocator
from pants.task.task import Task
from pants.util.memo import memoized_property
from pants.util.objects import Exactly
from pants.util.process_handler import subprocess
from upstreamable.subsystems.sbt import Sbt
from upstreamable.targets.sbt_dist import SbtDist


class PublishLocalSbtDistributions(Task):

  @classmethod
  def product_types(cls):
    return ['sbt_local_publish']

  @classmethod
  def register_options(cls, register):
    super(PublishLocalSbtDistributions, cls).register_options(register)

    register('--skip', type=bool, default=False)

    register('--reported-scala-version', type=str, default=None,
             help='Scala version to use with sbt. Defaults to the sbt project scala version.')

  @memoized_property
  def _reported_scala_version(self):
    return self.get_options().reported_scala_version

  @classmethod
  def subsystem_dependencies(cls):
    return super(PublishLocalSbtDistributions, cls).subsystem_dependencies() + (
      DistributionLocator,
      Sbt.scoped(cls),
    )

  @memoized_property
  def _sbt(self):
    return Sbt.scoped_instance(self)

  class PublishLocalSbtDistsError(TaskError): pass

  source_target_constraint = Exactly(SbtDist)

  def execute(self):
    if self.get_options().skip:
      return

    sbt_dist_targets = self.context.targets(self.source_target_constraint.satisfied_by)

    jvm_dist_locator = DistributionLocator.cached()

    with self.invalidated(sbt_dist_targets, invalidate_dependents=True) as invalidation_check:
      # Check that we have at most one sbt dist per directory.
      seen_basedirs = {}
      for vt in invalidation_check.all_vts:
        base_dir = vt.target.address.spec_path
        if base_dir in seen_basedirs:
          prev_target = seen_basedirs[base_dir]
          raise self.PublishLocalSbtDistsError(
            "multiple sbt dists defined in the same directory: current = {}, previous = {}"
            .format(vt.target, prev_target))
        else:
          seen_basedirs[base_dir] = vt.target

      for vt in invalidation_check.invalid_vts:
        base_dir = vt.target.address.spec_path

        project_name_maybe = vt.target._project_name

        with self.context.new_workunit(
            name='publish-local-sbt-dists',
            labels=[WorkUnitLabel.COMPILER],
        ) as workunit:

          sbt_version_args = ['-sbt-version', self._sbt.version] if self._sbt.version else []

          scala_version_maybe = self._reported_scala_version
          scala_version_args = ['++{}'.format(scala_version_maybe)] if scala_version_maybe else []

          subproject_args = ['project {}'.format(project_name_maybe)] if project_name_maybe else []

          argv = ['sbt'] + sbt_version_args + scala_version_args + subproject_args + [
            '-java-home', jvm_dist_locator.home,
            '-ivy', self._sbt.local_publish_repo,
            '-batch',
            'publishLocal',
          ]

          try:
            subprocess.check_call(
              argv,
              cwd=os.path.join(get_buildroot(), base_dir),
              stdout=workunit.output('stdout'),
              stderr=workunit.output('stderr'))
          except OSError as e:
            workunit.set_outcome(WorkUnit.FAILURE)
            raise self.PublishLocalSbtDistsError(
              "Error invoking sbt with command {} for target {}: {}"
              .format(argv, vt.target, e),
              e)
          except subprocess.CalledProcessError as e:
            workunit.set_outcome(WorkUnit.FAILURE)
            raise self.PublishLocalSbtDistsError(
              "Error publishing local sbt dist with command {} for target {}. Exit code was: {}"
              .format(argv, vt.target, e.returncode),
              e)
