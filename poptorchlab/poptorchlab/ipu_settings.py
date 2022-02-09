# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse

import popart
import poptorch
import torch


def retrieve_tensor_location_settings(map: dict):
    settings = poptorch.TensorLocationSettings()
    if 'min_elements_off_chip' in map:
        settings.minElementsForOffChip(map.pop('min_elements_off_chip'))
    if 'min_elements_for_rts' in map:
        settings.minElementsForReplicatedTensorSharding(map.pop('min_elements_for_rts'))
    if 'use_io_tiles_to_load' in map:
        settings.useIOTilesToLoad(map.pop('use_io_tiles_to_load'))
    if 'use_io_tiles_to_store' in map:
        settings.useIOTilesToStore(map.pop('use_io_tiles_to_store'))
    if 'on_chip' in map:
        settings.useOnChipStorage(map.pop('on_chip'))
    if 'use_rts' in map:
        settings.useReplicatedTensorSharding(map.pop('use_rts'))
    if map.keys():
        raise KeyError(
            f'''Unrecognized key(s) provided in tensor location settings: {map.keys()}.
            Available keys are:
            * min_elements_off_chip: int > 0
            * min_elements_for_rts: int > 0
            * use_io_tiles_to_load: bool
            * use_io_tiles_to_store: bool
            * on_chip: bool
            * use_rts: bool'''
        )
    return settings


def inference_settings(opts: poptorch.Options, args: argparse.Namespace):
    if args.anchor_mode is not None:
        opts.anchorMode(getattr(poptorch.AnchorMode, args.anchor_mode))
    opts.autoRoundNumIPUs(args.auto_round_num_ipus)
    if args.connection_type is not None:
        opts.connectionType(getattr(poptorch.ConnectionType, args.connection_type))
    opts.deviceIterations(args.device_iterations)
    if args.cache_dir is not None:
        opts.enableExecutableCaching(args.cache_dir)
    if args.profile_dir is not None:
        opts.enableProfiling(args.profile_dir)
    opts.enableStableNorm(args.enable_stable_norm)
    opts.enableSyntheticData(args.enable_synthetic_data)
    opts.logCycleCount(args.log_cycle_count)
    if args.log_dir is not None:
        opts.logDir(args.log_dir)
    if args.model_name is not None:
        opts.modelName(args.model_name)
    opts.randomSeed(args.random_seed)
    opts.relaxOptimizerAttributesChecks(args.relax_optimizer_attributes_checks)
    opts.replicationFactor(args.replication_factor)
    num_stages = len(args.pipeline_splits) + 1
    if len(args.available_memory_proportion) == 1:
        opts.setAvailableMemoryProportion({f'IPU{i}': args.available_memory_proportion[0] for i in range(num_stages)})
    elif len(args.available_memory_proportion) > 1:
        opts.setAvailableMemoryProportion({f'IPU{i}': amp for i, amp in enumerate(args.available_memory_proportion)})
    opts.showCompilationProgressBar(not args.hide_compile_progress)
    if args.sync_pattern is not None:
        opts.syncPattern(getattr(poptorch.SyncPattern, args.sync_pattern))
    if args.ipu_id is not None:
        opts.useIpuId(args.ipu_id)
    opts.useIpuModel(args.use_ipu_model)
    if args.use_offline_ipu_target is not None:
        opts.useOfflineIpuTarget(args.use_offline_ipu_target)

    opts.Precision.autocastEnabled(not args.disable_autocast)
    opts.Precision.enableFloatingPointExceptions(args.enable_fp_exceptions)
    opts.Precision.enableStochasticRounding(args.enable_stochastic_rounding)
    if args.half_float_casting is not None:
        opts.Precision.halfFloatCasting(getattr(poptorch.HalfFloatCastingBehavior, args.half_float_casting))
    opts.Precision.runningStatisticsAlwaysFloat(args.force_batchnorm_fp32)
    if args.partials_type is not None:
        opts.Precision.setPartialsType(getattr(torch, args.partials_type))

    opts._Popart.set("groupNormStridedChannelGrouping", args.enable_fast_groupnorm)

    return opts


def training_settings(opts: poptorch.Options, args: argparse.Namespace):
    opts = inference_settings(opts, args)

    if args.num_io_tiles is not None:
        opts.TensorLocations.numIOTiles(args.num_io_tiles)
    if args.accumulator_location_settings is not None:
        opts.TensorLocations.setAccumulatorLocation(retrieve_tensor_location_settings(args.accumulator_location_settings))
    if args.activation_location_settings is not None:
        opts.TensorLocations.setActivationLocation(retrieve_tensor_location_settings(args.activation_location_settings))
    if args.optimizer_location_settings is not None:
        opts.TensorLocations.setOptimizerLocation(retrieve_tensor_location_settings(args.optimizer_location_settings))
    if args.weight_location_settings is not None:
        opts.TensorLocations.setWeightLocation(retrieve_tensor_location_settings(args.weight_location_settings))

    if args.reduction_type is not None:
        opts.Training.accumulationAndReplicationReductionType(getattr(poptorch.ReductionType, args.reduction_type))
    opts.Training.gradientAccumulation(args.gradient_accumulation)
    opts.Training.setAutomaticLossScaling(args.enable_automatic_loss_scaling)
    opts.Training.setConvolutionDithering(args.enable_convolution_dithering)
    if args.mean_reduction_strategy is not None:
        opts.Training.setMeanAccumulationAndReplicationReductionStrategy(
            getattr(poptorch.MeanReductionStrategy, args.mean_reduction_strategy))

    opts._Popart.set("scheduleNonWeightUpdateGradientConsumersEarly", True)
    opts._Popart.set("disableGradAccumulationTensorStreams", True)
    if not (args.recompute_mode == "none") and len(args.pipeline_splits) == 0:
        opts._Popart.set("explicitRecomputation", True)
        if opts.recompute_mode == "auto":
            opts._Popart.set("autoRecomputation", int(popart.RecomputationType.Standard))
        elif opts.recompute_mode == "manual":
            opts._Popart.set("autoRecomputation", int(popart.RecomputationType.RecomputeAll))

    return opts
