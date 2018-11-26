from __future__ import (absolute_import, division, generators, nested_scopes,
                        print_function, unicode_literals, with_statement)

import glob
import os
import shutil

from abc import abstractproperty
from future.utils import binary_type

from pants.base.build_environment import get_buildroot
from pants.base.exceptions import TaskError
from pants.base.workunit import WorkUnit, WorkUnitLabel
from pants.build_graph.address import Address
from pants.option.custom_types import target_option
from pants.task.task import Task
from pants.util.collections import assert_single_element
from pants.util.contextutil import temporary_dir
from pants.util.memo import memoized_property
from pants.util.objects import datatype
from pants.util.process_handler import subprocess
from pants.util.strutil import safe_shlex_join


def is_readable_file(path):
  return os.path.isfile(path) and os.access(path, os.R_OK)


class BootstrapJar(datatype([('tool_jar_path', binary_type)])): pass


class BootstrapJvmSourceTool(Task):

  # TODO: ???
  workunit_component_name = None

  @classmethod
  def product_types(cls):
    return [BootstrapJar]

  @classmethod
  def prepare(cls, options, round_manager):
    super(BootstrapJvmSourceTool, cls).prepare(options, round_manager)
    round_manager.optional_product('sbt_local_publish')

  @classmethod
  def register_options(cls, register):
    super(BootstrapJvmSourceTool, cls).register_options(register)

    register('--skip', type=bool, advanced=True,
             help='Whether to skip this task (e.g. if we are currently executing the subprocess).')

  @abstractproperty
  def binary_tool_target(self): pass

  @memoized_property
  def _bootstrap_config_files(self):
    all_config_files = self.get_options().pants_config_files

    bootstrap_ini_path = os.path.join(get_buildroot(), 'pants.ini.bootstrap')
    if is_readable_file(bootstrap_ini_path):
      all_config_files.append(bootstrap_ini_path)

    return all_config_files

  class BootstrapJvmSourceToolError(TaskError): pass

  def _collect_dist_jar(self, dist_dir):
    # We expect and assert that there is only a single file in the dist dir.
    dist_jar_glob = os.path.join(dist_dir, '*.jar')
    globbed_jars = glob.glob(dist_jar_glob)

    if globbed_jars:
      return assert_single_element(globbed_jars)
    else:
      return None

  _CLEAN_ENV = [
    'PANTS_ENABLE_PANTSD',
    'PANTS_ENTRYPOINT',
  ]

  def _get_subproc_env(self):
    env = os.environ.copy()

    for env_var in self._CLEAN_ENV:
      env.pop(env_var, None)

    return env

  def _build_binary(self, jvm_binary_target_spec):

    pants_config_files_args = ['"{}"'.format(f) for f in self._bootstrap_config_files]

    with temporary_dir() as tmpdir:
      cmd = [
        './pants',
        '--pants-config-files=[{}]'.format(','.join(pants_config_files_args)),
        '--pants-distdir={}'.format(tmpdir),
        'binary',
        jvm_binary_target_spec,
      ]

      env = self._get_subproc_env()

      with self.context.new_workunit(
          name='bootstrap-jvm-tool-subproc:{}'.format(self.workunit_component_name),
          labels=[WorkUnitLabel.COMPILER],
          cmd=safe_shlex_join(cmd),
      ) as workunit:

        try:
          subprocess.check_call(
            cmd,
            cwd=get_buildroot(),
            stdout=workunit.output('stdout'),
            stderr=workunit.output('stderr'),
            env=env)
        except OSError as e:
          workunit.set_outcome(WorkUnit.FAILURE)
          raise self.BootstrapJvmSourceToolError(
            "Error invoking pants for the {} binary with command {} from target {}: {}"
            .format(self.workunit_component_name, cmd, jvm_binary_target_spec, e),
            e)
        except subprocess.CalledProcessError as e:
          workunit.set_outcome(WorkUnit.FAILURE)
          raise self.BootstrapJvmSourceToolError(
            "Error generating the {} binary with command {} from target {}. "
            "Exit code was: {}."
            .format(self.workunit_component_name, cmd, jvm_binary_target_spec, e.returncode),
            e)

      dist_jar = self._collect_dist_jar(tmpdir)
      jar_fname = os.path.basename(dist_jar)
      cached_jar_path = os.path.join(self.workdir, jar_fname)
      shutil.move(dist_jar, cached_jar_path)

  @staticmethod
  def _add_product_at_target_base(product_mapping, target, value):
    product_mapping.add(target, target.target_base).append(value)

  def execute(self):
    if self.get_options().skip:
      return

    jvm_binary_target_spec = self.binary_tool_target
    binary_target_address = Address.parse(jvm_binary_target_spec)

    # Scan everything under the target dir, then check whether the binary target has been
    # invalidated. The default target dir is '', meaning scan all BUILD files -- but that's ok since
    # this project is small (for now).
    scala_root = os.path.join(get_buildroot(), binary_target_address.spec_path)
    new_build_graph = self.context.scan(scala_root)
    binary_target = new_build_graph.get_target_from_spec(jvm_binary_target_spec)

    with self.invalidated([binary_target], invalidate_dependents=True) as invalidation_check:
      if invalidation_check.invalid_vts:
        self._build_binary(jvm_binary_target_spec)

    built_jar = binary_type(self._collect_dist_jar(self.workdir))

    bootstrap_jar_product = self.context.products.get(BootstrapJar)
    self._add_product_at_target_base(bootstrap_jar_product, binary_target, BootstrapJar(built_jar))
