# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import json
import logging
import multiprocessing
import sys

import os
import poptorch
import random
import yaml

from .utils import set_random_seed
from .logger import Logger


class YAMLNamespace(argparse.Namespace):
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def get_available_configs(config_file):
    with open(config_file) as file:
        configs = yaml.full_load(file)
    return configs


def choices_from(enum):
    return [*filter(lambda x: not x.startswith('_'), enum.__dict__.keys())]


def get_docs_url(obj):
    qualname = obj.__qualname__
    if qualname.startswith('_'):
        qualname = f'options.{qualname}'
    name = f'poptorch.{qualname}'
    return f'https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/reference.html#{name}'


class UnboundedWidthHelpFormatter(argparse.HelpFormatter):
    def __init__(self,
                 prog,
                 indent_increment=2,
                 max_help_position=24):
        super(UnboundedWidthHelpFormatter, self).__init__(
            prog, indent_increment, max_help_position, sys.maxsize)

    def format_help(self):
        help = self._root_section.format_help()
        if help:
            help = self._long_break_matcher.sub('\n\n', help)
            help = help + '\n'
        return help


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self,
                 prog=None,
                 usage=None,
                 description=None,
                 epilog=None,
                 parents=[],
                 formatter_class=UnboundedWidthHelpFormatter,
                 prefix_chars='-',
                 fromfile_prefix_chars=None,
                 argument_default=None,
                 conflict_handler='error',
                 add_help=True,
                 allow_abbrev=True):
        super(ArgumentParser, self).__init__(
            prog, usage, description, epilog, parents, formatter_class, prefix_chars,
            fromfile_prefix_chars, argument_default, conflict_handler, add_help, allow_abbrev)

        self.add_argument('--log-level', choices=['FATAL', 'ERROR', 'WARN', 'INFO', 'DEBUG'], default='INFO',
                          help='Log level for standard python logging.')

        basic_options = self.add_argument_group('PopTorch Options (poptorch.Options)')
        basic_options.add_argument('--anchor-mode', choices=choices_from(poptorch.AnchorMode),
                                   help='Specify which data to return from a model.'
                                        f' [See {get_docs_url(poptorch.Options.anchorMode)}]')
        basic_options.add_argument('--auto-round-num-ipus', action='store_true',
                                   help='Round up the number of IPUs used automatically: the number of IPUs requested must be a power of 2. '
                                        'If you want to request the number of IPUs other than a power of 2, you will need this option.'
                                        f' [See {get_docs_url(poptorch.Options.autoRoundNumIPUs)}]')
        basic_options.add_argument('--connection-type', choices=choices_from(poptorch.ConnectionType),
                                   help='When to connect to the IPU.'
                                        f' [See {get_docs_url(poptorch.Options.connectionType)}]')
        basic_options.add_argument('--device-iterations', type=int, default=1,
                                   help='Number of iterations the IPU device should run over the data before returning to the user.'
                                        f' [See {get_docs_url(poptorch.Options.deviceIterations)}]')
        basic_options.add_argument('--cache-dir',
                                   help='Load/save Poplar executables to the specified path, using it as a cache, '
                                        'to avoid recompiling identical graphs.'
                                        f' [See {get_docs_url(poptorch.Options.enableExecutableCaching)}]')
        basic_options.add_argument('--profile-dir',
                                   help='Enable profiling report generation.'
                                        f' [See {get_docs_url(poptorch.Options.enableProfiling)}]')
        basic_options.add_argument('--enable-stable-norm', action='store_true',
                                   help='Use stable versions of norm operators. This stable version is slower, '
                                        'but more accurate than its unstable counterpart.'
                                        f' [See {get_docs_url(poptorch.Options.enableStableNorm)}]')
        basic_options.add_argument('--enable-synthetic-data', action='store_true',
                                   help='Disable host I/O and generate synthetic data on the IPU instead.'
                                        f' [See {get_docs_url(poptorch.Options.enableSyntheticData)}]')
        basic_options.add_argument('--log-cycle-count', action='store_true',
                                   help='Log the number of IPU cycles used in executing the main graph, which is printed by setting the environment '
                                        'variable POPTORCH_LOG_LEVEL=INFO. (Note: This will have a small detrimental impact on performance.)'
                                        f' [See {get_docs_url(poptorch.Options.logCycleCount)}]')
        basic_options.add_argument('--log-dir',
                                   help='Set the log directory.'
                                        f' [See {get_docs_url(poptorch.Options.logDir)}]')
        basic_options.add_argument('--model-name',
                                   help='Set the model name for profiling.'
                                        f' [See {get_docs_url(poptorch.Options.modelName)}]')
        basic_options.add_argument('--random-seed', type=int,
                                   help='Set the seed for the random number generator on the IPU.'
                                        f' [See {get_docs_url(poptorch.Options.randomSeed)}]')
        basic_options.add_argument('--relax-optimizer-attributes-checks', action='store_true',
                                   help='Controls whether unexpected attributes in setOptimizer() lead to warnings or debug messages.'
                                        f' [See {get_docs_url(poptorch.Options.relaxOptimizerAttributesChecks)}]')
        basic_options.add_argument('--replication-factor', type=int, default=1,
                                   help='Number of times to replicate the model.'
                                        f' [See {get_docs_url(poptorch.Options.replicationFactor)}]')
        basic_options.add_argument('--available-memory-proportion', type=float, nargs='+', default=[],
                                   help='Sets the amount of temporary memory made available on a per-IPU basis. '
                                        'List of float values or a single float value to use the same value for all IPUs.'
                                        f' [See {get_docs_url(poptorch.Options.setAvailableMemoryProportion)}]')
        basic_options.add_argument('--hide-compile-progress', action='store_true',
                                   help='Hide compile progress bar'
                                        f' [See {get_docs_url(poptorch.Options.showCompilationProgressBar)}]')
        basic_options.add_argument('--sync-pattern', choices=choices_from(poptorch.SyncPattern),
                                   help='Controls synchronisation in multi-IPU systems.'
                                        f' [See {get_docs_url(poptorch.Options.syncPattern)}]')
        basic_options.add_argument('--ipu-id', type=int,
                                   help='Use the IPU device specified by the ID (as provided by gc-info). You can use the the command-line tool '
                                        'gc-info: running gc-info -a, shows each device ID and a list of IPUs associated with the ID. '
                                        f' [See {get_docs_url(poptorch.Options.useIpuId)}]')
        basic_options.add_argument('--use-ipu-model', action='store_true',
                                   help='Use the IPU Model instead of physical hardware'
                                        f'(See {get_docs_url(poptorch.Options.useIpuModel)})')
        basic_options.add_argument('--use-offline-ipu-target', type=int, metavar='IPU_VERSION',
                                   help='Create an offline IPU target that can only be used for offline compilation. '
                                        '(Note: the offline IPU target cannot be used if the IPU model is enabled.) '
                                        f' [See {get_docs_url(poptorch.Options.useOfflineIpuTarget)}]')

        precision_options = self.add_argument_group('PopTorch Precision Options (poptorch.Options.Precision)')
        precision_options.add_argument('--disable-autocast', action='store_true',
                                       help='Disable automatic casting functionality.'
                                            f' [See {get_docs_url(poptorch.options._PrecisionOptions.autocastEnabled)}]')
        precision_options.add_argument('--enable-fp-exceptions', action='store_true',
                                       help='Enable floating point exceptions'
                                            f' [See {get_docs_url(poptorch.options._PrecisionOptions.enableFloatingPointExceptions)}]')
        precision_options.add_argument('--enable-stochastic-rounding', action='store_true',
                                       help='Enable stochastic rounding'
                                            f' [See {get_docs_url(poptorch.options._PrecisionOptions.enableStochasticRounding)}]')
        precision_options.add_argument('--half-float-casting', choices=choices_from(poptorch.HalfFloatCastingBehavior),
                                       help='Changes the casting behaviour for ops involving a float16 (half) and a float32'
                                            f' [See {get_docs_url(poptorch.options._PrecisionOptions.halfFloatCasting)}]')
        precision_options.add_argument('--force-batchnorm-fp32', action='store_true',
                                       help='Force the running mean and variance tensors of batch normalisation layers to be '
                                            'float32 regardless of input type.'
                                            f' [See {get_docs_url(poptorch.options._PrecisionOptions.runningStatisticsAlwaysFloat)}]')
        precision_options.add_argument('--partials-type', choices=['float16', 'float32'],
                                       help='The data type of partial results for matrix multiplication and convolution operators.'
                                            f' [See {get_docs_url(poptorch.options._PrecisionOptions.setPartialsType)}]')

        tensor_locations_options = self.add_argument_group('Tensor Location Options (poptorch.Options.TensorLocations)')
        tensor_locations_options.add_argument('--num-io-tiles', type=int,
                                              help='Assigns the number of tiles on the IPU to be IO rather than compute.'
                                                   f' [See {get_docs_url(poptorch.options._TensorLocationOptions.numIOTiles)}]')
        location_settings_help = 'A JSON string containing key-value pairs of one or more of the followings - ' \
                                 '"min_elements_off_chip": int, ' \
                                 '"min_elements_for_rts": int, ' \
                                 '"use_io_tiles_to_load": bool, ' \
                                 '"use_io_tiles_to_store": bool, ' \
                                 '"on_chip": bool, ' \
                                 '"use_rts": bool' \
                                 f' [See {get_docs_url(poptorch.TensorLocationSettings)}]'
        tensor_locations_options.add_argument('--accumulator-location-settings', type=json.loads,
                                              help='Tensor location settings for accumulator. '
                                                   + location_settings_help)
        tensor_locations_options.add_argument('--activation-location-settings', type=json.loads,
                                              help='Tensor location settings for activations. '
                                                   + location_settings_help)
        tensor_locations_options.add_argument('--optimizer-location-settings', type=json.loads,
                                              help='Tensor location settings for optimizer states. '
                                                   + location_settings_help)
        tensor_locations_options.add_argument('--weight-location-settings', type=json.loads,
                                              help='Tensor location settings for weights. '
                                                   + location_settings_help)

        training_options = self.add_argument_group('PopTorch Training Options (poptorch.Options.Training)')
        training_options.add_argument('--reduction-type', choices=choices_from(poptorch.ReductionType),
                                      help='Set the type of reduction applied to reductions in the graph.'
                                           f' [See {get_docs_url(poptorch.options._TrainingOptions.accumulationAndReplicationReductionType)}]')
        training_options.add_argument('--gradient-accumulation', type=int, default=1,
                                      help='Number of micro-batches to accumulate for the gradient calculation.'
                                           f' [See {get_docs_url(poptorch.options._TrainingOptions.gradientAccumulation)}]')
        training_options.add_argument('--enable-automatic-loss-scaling', action='store_true',
                                      help='Enable automatic loss scaling feature (experimental).'
                                           f' [See {get_docs_url(poptorch.options._TrainingOptions.setAutomaticLossScaling)}]')
        training_options.add_argument('--enable-convolution-dithering', action='store_true',
                                      help='Enable convolution dithering.'
                                           f' [See {get_docs_url(poptorch.options._TrainingOptions.setConvolutionDithering)}]')
        training_options.add_argument('--mean-reduction-strategy', choices=choices_from(poptorch.MeanReductionStrategy),
                                      help='Specify when to divide by a mean reduction factor when '
                                           'accumulationAndReplicationReductionType is set to ReductionType.Mean.'
                                           f' [See {get_docs_url(poptorch.options._TrainingOptions.setMeanAccumulationAndReplicationReductionStrategy)}]')

        popart_options = self.add_argument_group('PopART Options (poptorch.Options._Popart)')
        popart_options.add_argument('--recompute-mode', default="none", choices=['none', 'auto', 'manual'],
                                    help="Select single IPU recompute mode. If the model is multi  stage (pipelined) the recomputation is always "
                                         "enabled. Auto mode selects the recompute checkpoints automatically. Rest of the network will be recomputed. "
                                         "It is possible to extend the recompute checkpoints with the --recompute-checkpoints option. In manual mode "
                                         "no recompute checkpoint is added, they need to be determined by the user.")
        popart_options.add_argument('--enable-fast-groupnorm', action='store_true',
                                    help="There are two implementations of the group norm layer. If the fast implementation enabled, "
                                         "it couldn't load checkpoints, which didn't train with this flag. The default implementation can use any checkpoint.")

        data_loader_options = self.add_argument_group('PopTorch DataLoader Options (poptorch.DataLoader)')
        data_loader_options.add_argument('--batch-size', type=int, default=1,
                                         help='Micro batch size for training and inference')
        data_loader_options.add_argument('--data-loader-mode', choices=choices_from(poptorch.DataLoaderMode),
                                         help='poptorch.DataLoaderMode'
                                              f' [See {get_docs_url(poptorch.DataLoaderMode)}]')
        data_loader_options.add_argument('--num-workers', type=int,
                                         help='Number of worker processes to use to read the data.')
        data_loader_options.add_argument('--rebatched-worker-size', type=int,
                                         help='When using AsyncRebatched: batch size of the tensors loaded by the workers. '
                                              'Default to the combined batch size. If specified the rebatched_worker_size must be less than '
                                              'or equal to the combined batch size.')

        optimizer_options = self.add_argument_group('Optimizer Options (poptorch.optim)')
        optimizer_options.add_argument('--optimizer', choices=['SGD', 'Adam', 'AdamW', 'RMSprop', 'LAMB'], default='SGD',
                                       help='Optimizer for training (Default: SGD)')
        optimizer_options.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
        optimizer_options.add_argument('--weight-decay', type=float, default=0.0001, help="L2 parameter penalty")
        optimizer_options.add_argument('--optimizer-kwargs', type=json.loads,
                                       help='Optimizer-specific initializer keyword arguments other than "params" and "lr" in JSON format.'
                                            ' (Examples: \'{"momentum": 0.9, "accum_type": float16}\' for "SGD")'
                                            ' [See https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/reference.html#optimizers]')

        model_manipulation_options = self.add_argument_group('Model Manipulation Options')
        model_manipulation_options.add_argument('--pipeline-splits', type=str, nargs='+', default=[],
                                                help='List of the splitting layers for model parallel execution')
        model_manipulation_options.add_argument('--recompute-checkpoints', type=str, nargs='+', default=[],
                                                help='List of recomputation checkpoints. List of regex rules for the layer names must be provided. '
                                                     '(Example: Select all layers whose name containing "conv": .*conv.*)')
        model_manipulation_options.add_argument('--serialization-config', type=json.loads, nargs='+', default='{}',
                                                help='A JSON string containing key-value pairs, where each key is the address of a layer to serialize, '
                                                     'paired with the desired serialization factor as its value (int)')
        model_manipulation_options.add_argument('--sparsify-config', type=json.loads, nargs='+', default='{}',
                                                help='A JSON string containing key-value pairs, where each key is the address of a layer to sparsify, '
                                                     'paired with the desired sparsity as its value (float bewteen 0 and 1)')

    def parse_args(self, args=None, namespace=None):
        config_file = os.path.join(os.getcwd(), 'configs.yml')
        if os.path.isfile(config_file):
            configurations = get_available_configs(config_file)
            if configurations is not None:
                self.add_argument('--config', choices=configurations.keys(), help='Select from avalible configurations')
            args = super().parse_args(args, namespace)
            if hasattr(args, 'config') and args.config is not None:
                # Load the configurations from the YAML file and update command line arguments
                loaded_config = YAMLNamespace(configurations[args.config])
                # Check the config file keys
                for k in vars(loaded_config).keys():
                    assert k in vars(args).keys(), f"Couldn't recognise argument {k}."

                args = super().parse_args(namespace=loaded_config)
        else:
            logging.warning(f'Argument is parsed without configuration file. Please run "touch {config_file}" and add some configurations in it.')
            args = super().parse_args(args, namespace)

        return ArgumentParser.postprocess_arguments(args)

    @classmethod
    def postprocess_arguments(cls, args):
        num_workers_set_by_user = args.num_workers is not None
        if not num_workers_set_by_user:
            args.num_workers = min(32, multiprocessing.cpu_count())

        random_seed_set_by_user = args.random_seed is not None
        if not random_seed_set_by_user:
            args.random_seed = random.randint(0, 2 ** 32 - 1)
        set_random_seed(args.random_seed)

        # setup logging
        Logger.setup_logging_folder(args)

        if not num_workers_set_by_user:
            logging.info(f'Automatically set data loader worker to {args.num_workers}')

        if not random_seed_set_by_user:
            logging.info(f'Automatically generated random seed: {args.random_seed}')

        num_stages = len(args.pipeline_splits) + 1
        num_amps = len(args.available_memory_proportion)
        if num_stages > 1 and num_amps > 0 and num_amps != num_stages and num_amps != 1:
            logging.error(f'--available-memory-proportion number of elements should be '
                          f'either 1 or equal to the number of pipeline stages: {num_stages}')
            sys.exit()

        if num_stages > 1:
            logging.info('Recomputation is always enabled when using pipelining.')

        if args.recompute_mode == 'none' and len(args.recompute_checkpoints) > 0 and num_stages == 1:
            logging.warning('Recomputation is not enabled, while recomputation checkpoints are provided.')

        return args
