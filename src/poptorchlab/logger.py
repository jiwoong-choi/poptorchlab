# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import json
import logging
import string
import sys

import os
import random


def get_random_str(strlen=3):
    # We want this to be random even if random seeds have been set so that we don't overwrite
    # when re-running with the same seed
    random_state = random.getstate()
    random.seed()
    rnd_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(strlen))
    random.setstate(random_state)
    return rnd_str


class Logger:
    log_dir_configured: bool = False

    @classmethod
    def setup_logging_folder(cls, args):
        # If it's already configured, skip the reconfiguration
        if cls.log_dir_configured:
            return

        # Set up logging
        log = logging.getLogger()
        log.setLevel(getattr(logging, args.log_level))

        # get POPLAR_ENGINE_OPTIONS if it exists, as a Python dictionary
        eng_opts = json.loads(os.environ.get("POPLAR_ENGINE_OPTIONS", "{}"))
        profile_dir = eng_opts.get("autoReport.directory", None)
        options = {key: value for key, value in vars(args).items() if value not in [False, None, [], "", "none"]}
        if eng_opts:
            options["POPLAR_ENGINE_OPTIONS"] = eng_opts

        # Determine saving folder
        args.profiling = False
        if profile_dir is not None:
            args.log_dir = profile_dir
            args.profiling = True
            log.info(f'Overwriting logging directory from POPLAR_ENGINE_OPTIONS - {args.log_dir}')
        elif args.profile_dir is not None:
            args.log_dir = args.profile_dir
            args.profiling = True
            log.info(f'Overwriting logging directory from --profile-dir option - {args.log_dir}')

        if args.log_dir is not None and not args.profiling:
            if hasattr(args, 'config'):
                basename = args.config
            else:
                basename = f'{args.model_name or "anonymous"}_bs{args.batch_size}_r{args.replication_factor}_di{args.device_iterations}'
            while True:
                log_dir = os.path.join(args.log_dir, basename + "_" + get_random_str())
                if not os.path.exists(log_dir):
                    break
            args.log_dir = log_dir

        # remove stderr output logging
        if len(log.handlers) > 0:
            log.handlers.pop()
        stdout = logging.StreamHandler(sys.stdout)
        stdout_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        stdout.setFormatter(stdout_formatter)
        log.addHandler(stdout)
        if args.log_dir is not None:
            log.info(f'Setting up logging directory at {args.log_dir}')
            os.makedirs(args.log_dir, exist_ok=True)
            if args.profiling:
                profile_subdir = os.path.join(
                    args.log_dir, args.model_name if args.model_name is not None else 'training')
                os.makedirs(profile_subdir, exist_ok=True)
                app_info_path = os.path.join(profile_subdir, 'app.json')
            else:
                app_info_path = os.path.join(args.log_dir, 'app.json')
            with open(app_info_path, "w") as f:
                json.dump(options, f, sort_keys=True, indent=2)
                f.write('\n')
            fileh = logging.FileHandler(os.path.join(args.log_dir, 'log.txt'), 'a')
            file_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(module)s - %(funcName)s: %(message)s')
            fileh.setFormatter(file_formatter)
            log.addHandler(fileh)

        cls.log_dir_configured = True
