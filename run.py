import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
# CUDA caching-allocator tuning, set before anything imports torch. On a 16 GB card
# with layer offloading, the training OOMs were fragmentation: a 288 MiB request
# failed while 2.31 GiB sat reserved but unusable. PyTorch's usual fix,
# expandable_segments, is not supported on Windows; max_split_size_mb is. It stops
# small allocations from carving up large cached blocks, so big free blocks stay whole
# and can be reused -- or released and re-allocated -- when a large request would
# otherwise fail. setdefault: an explicitly set PYTORCH_CUDA_ALLOC_CONF always wins.
if os.name == "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
import sys
from typing import Union
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, os.getcwd())
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)
import argparse
from toolkit.job import get_job
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file

accelerator = get_accelerator()


def print_end_message(jobs_completed, jobs_failed):
    if not accelerator.is_main_process:
        return
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def main():
    parser = argparse.ArgumentParser()

    # require at lease one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    
    parser.add_argument(
        '-l', '--log',
        type=str,
        default=None,
        help='Log file to write output to'
    )
    args = parser.parse_args()
    
    if args.log is not None:
        setup_log_to_file(args.log)

    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0

    if accelerator.is_main_process:
        print_acc(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")
        print_acc(f"CUDA allocator: PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF') or '(default)'}")
    if os.name == "nt":
        # Before any DataLoader worker exists: every child process joins this job and
        # is killed by Windows when the trainer ends, however it ends (see the module).
        from toolkit.windows_job import kill_children_with_this_process
        status = kill_children_with_this_process()
        if accelerator.is_main_process:
            print_acc(f"Child processes (data-loader workers) end with the trainer: {status}")
    if accelerator.is_main_process:
        # workers orphaned by earlier jobs (e.g. from before the job object existed)
        from toolkit.memory_management.cleanup import kill_orphaned_workers
        kill_orphaned_workers(printer=print_acc)

    for config_file in config_file_list:
        job = None
        try:
            job = get_job(config_file, args.name)
            job.run()
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print_acc(f"Error running job: {e}")
            jobs_failed += 1
            try:
                if job is not None:
                    job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise
        except KeyboardInterrupt as e:
            try:
                if job is not None:
                    job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise
        finally:
            # Whether the job finished or failed, the next config in this process starts
            # a fresh pinned arena; old slabs are freed once the finished job's model (the
            # last users of their views) is gone.
            job = None
            try:
                from toolkit.memory_management import reset_pinned_arena
                reset_pinned_arena()
            except Exception:
                pass


if __name__ == '__main__':
    main()
