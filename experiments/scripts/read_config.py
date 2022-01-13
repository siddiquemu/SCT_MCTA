from typing import Generator, Tuple, Union
import inspect
import os.path
from sacred.utils import PathType
from typing import Sequence, Optional

from collections import OrderedDict

from sacred.config import (
    ConfigDict,
    ConfigScope,
    create_captured_function,
    load_config_file,
)
from sacred.dependencies import (
    PEP440_VERSION_PATTERN,
    PackageDependency,
    Source,
    gather_sources_and_dependencies,
)
from sacred.utils import CircularDependencyError, optional_kwargs_decorator, join_paths

class Configuration(object):
    def __init__(self, path):
        self.cfg_path = path
        self.config_hooks = []
        self.configurations = []
        self.named_configs = dict()
        self.logger = None
        self.captured_functions = []
        self.post_run_hooks = []
        self.pre_run_hooks = []
        self._is_traversing = False
        self.commands = OrderedDict()


    def add_config(self, **kw_conf):
        """
        Add a configuration entry to this ingredient/experiment.

        Can be called with a filename, a dictionary xor with keyword arguments.
        Supported formats for the config-file so far are: ``json``, ``pickle``
        and ``yaml``.

        The resulting dictionary will be converted into a
         :class:`~sacred.config_scope.ConfigDict`.

        :param cfg_or_file: Configuration dictionary of filename of config file
                            to add to this ingredient/experiment.
        :type cfg_or_file: dict or str
        :param kw_conf: Configuration entries to be added to this
                        ingredient/experiment.
        """
        self.configurations.append(self._create_config_dict(self.cfg_path, kw_conf))
        return self.configurations

    def _add_named_config(self, name, conf):
        if name in self.named_configs:
            raise KeyError('Configuration name "{}" already in use!'.format(name))
        self.named_configs[name] = conf

    @staticmethod
    def _create_config_dict(cfg_or_file, kw_conf):
        if cfg_or_file is not None and kw_conf:
            raise ValueError(
                "cannot combine keyword config with " "positional argument"
            )
        if cfg_or_file is None:
            if not kw_conf:
                raise ValueError("attempted to add empty config")
            return ConfigDict(kw_conf)
        elif isinstance(cfg_or_file, dict):
            return ConfigDict(cfg_or_file)
        elif isinstance(cfg_or_file, str):
            if not os.path.exists(cfg_or_file):
                raise OSError("File not found {}".format(cfg_or_file))
            abspath = os.path.abspath(cfg_or_file)
            return ConfigDict(load_config_file(abspath))
        else:
            raise TypeError("Invalid argument type {}".format(type(cfg_or_file)))

    def add_named_config(self, name, cfg_or_file=None, **kw_conf):
        """
        Add a **named** configuration entry to this ingredient/experiment.

        Can be called with a filename, a dictionary xor with keyword arguments.
        Supported formats for the config-file so far are: ``json``, ``pickle``
        and ``yaml``.

        The resulting dictionary will be converted into a
         :class:`~sacred.config_scope.ConfigDict`.

        See :ref:`named_configurations`

        :param name: name of the configuration
        :type name: str
        :param cfg_or_file: Configuration dictionary of filename of config file
                            to add to this ingredient/experiment.
        :type cfg_or_file: dict or str
        :param kw_conf: Configuration entries to be added to this
                        ingredient/experiment.
        """
        self._add_named_config(name, self._create_config_dict(cfg_or_file, kw_conf))