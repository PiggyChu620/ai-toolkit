from collections import OrderedDict
import os
import sqlite3
import asyncio
import concurrent.futures
from datetime import datetime, timedelta
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from jobs.process.BaseSDTrainProcess import SamplingInterrupted
from toolkit.current_step_feed import CurrentStepFeed
from typing import Literal, Optional
import threading
import time
import signal
import torch

try:
    import psutil
except ImportError:
    psutil = None

AITK_Status = Literal["running", "stopped", "error", "completed"]

# Job info while a stop waits for the step to be saved; the UI's stop route recognises it and
# holds off its force-kill (ui/src/utils/stopping.ts has the same text)
SAVING_BEFORE_STOP = "Saving before stopping"


class DiffusionTrainer(SDTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(DiffusionTrainer, self).__init__(process_id, job, config, **kwargs)
        self.sqlite_db_path = self.config.get("sqlite_db_path", "./aitk_db.db")
        self.job_id = os.environ.get("AITK_JOB_ID", None)
        self.job_id = self.job_id.strip() if self.job_id is not None else None
        self.is_ui_trainer = True
        if not os.path.exists(self.sqlite_db_path):
            self.is_ui_trainer = False
        else:
            print(f"Using SQLite database at {self.sqlite_db_path}")
        if self.job_id is None:
            self.is_ui_trainer = False
        else:
            print(f"Job ID: \"{self.job_id}\"")
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        # stop requested during training: the step finishes, is saved, then the job stops
        self._graceful_stop_target = None  # "stopped" or "queued" once requested
        self._graceful_stop_deadline = 0.0
        self._graceful_stop_lock = threading.Lock()
        self._save_on_stop_timeout = max(60.0, float(self.config.get("ui_save_on_stop_timeout", 1200)))
        self._graceful_info = None  # the "Saving before stopping: ..." text, re-asserted by the watcher
        self._last_info_assert = 0.0
        self._last_save_requested_step = None
        self._last_save_ok_step = None
        
        if self.is_ui_trainer:
            self.is_stopping = False
            # Serialize UI/DB writes off the training thread.
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._async_tasks = []
            self._async_loop_ready = threading.Event()
            self._async_loop = asyncio.new_event_loop()
            self._async_loop_thread = threading.Thread(
                target=self._run_async_loop,
                name="aitk-ui-async",
                daemon=True,
            )
            self._async_loop_thread.start()
            self._async_loop_ready.wait(timeout=1.0)

            self._step_update_interval = max(0.1, float(self.config.get("ui_step_update_interval", 1.0)))
            self._stop_poll_interval = max(0.1, float(self.config.get("ui_stop_poll_interval", 1.0)))
            self._cache_update_interval = max(0.1, float(self.config.get("ui_cache_update_interval", 1.0)))
            self._last_step_update_at = 0.0
            self._last_stop_poll_at = 0.0
            self._last_cache_update_at = 0.0
            self._cached_should_stop = False
            self._cached_return_to_queue = False
            self._last_status_payload: Optional[tuple[AITK_Status, Optional[str]]] = None
            # Initialize the status
            self._run_async_operation(self._update_status("running", "Starting"))
            self._stop_watcher_started = False
            self.start_stop_watcher(interval_sec=self._stop_poll_interval)

    def _run_async_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop_ready.set()
        self._async_loop.run_forever()

    def _shutdown_async_loop(self):
        if not getattr(self, "_async_loop", None):
            return
        if self._async_loop.is_running():
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        if getattr(self, "_async_loop_thread", None) is not None:
            self._async_loop_thread.join(timeout=2.0)
        self._async_loop.close()

    def _format_bytes(self, num_bytes: float) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(num_bytes)
        for unit in units:
            if size < 1024.0 or unit == units[-1]:
                if unit == "B":
                    return f"{int(size)}{unit}"
                return f"{size:.1f}{unit}"
            size /= 1024.0

    def _format_duration(self, total_seconds: float) -> str:
        remaining_minutes = max(0, int(round(total_seconds / 60)))
        weeks, remainder = divmod(remaining_minutes, 7 * 24 * 60)
        days, remainder = divmod(remainder, 24 * 60)
        hours, minutes = divmod(remainder, 60)

        parts = []
        if weeks:
            parts.append(f"{weeks}w")
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes or not parts:
            parts.append(f"{minutes}m")
        return " ".join(parts)

    def _format_expected_finish_time(self, total_seconds: float) -> str:
        finish_time = datetime.now().astimezone() + timedelta(seconds=max(0.0, total_seconds))
        return finish_time.strftime("%Y-%m-%d %H:%M")

    def _build_runtime_stats(self, seconds_per_iter: float) -> str:
        if seconds_per_iter <= 0:
            speed = "AVG n/a"
        elif seconds_per_iter < 1:
            speed = f"AVG {1 / seconds_per_iter:.2f} iter/sec"
        else:
            speed = f"AVG {seconds_per_iter:.2f} sec/iter"

        parts = [speed]
        remaining_steps = max(0, self.train_config.steps - self.step_num)
        if seconds_per_iter > 0 and remaining_steps > 0:
            remaining_seconds = remaining_steps * seconds_per_iter
            parts.append(f"ETA {self._format_duration(remaining_seconds)}")
            parts.append(f"Expected Finish {self._format_expected_finish_time(remaining_seconds)}")

        return " | ".join(parts)

    def _build_cache_stats(
            self,
            cache_kind: str,
            processed: int,
            total: int,
            elapsed_seconds: float,
            dataset_path: str,
    ) -> str:
        cache_labels = {
            "latents": "Latents",
            "text_embeddings": "Text",
            "clip_vision": "CLIP",
        }
        label = cache_labels.get(cache_kind, cache_kind.replace("_", " ").title())
        dataset_name = os.path.basename(os.path.normpath(dataset_path)) if dataset_path else ""

        if processed <= 0 or elapsed_seconds <= 0:
            speed = "AVG n/a"
            eta = "ETA n/a"
        else:
            items_per_second = processed / elapsed_seconds
            if items_per_second >= 1:
                speed = f"AVG {items_per_second:.2f} it/s"
            else:
                speed = f"AVG {elapsed_seconds / processed:.2f} sec/item"
            remaining_items = max(0, total - processed)
            eta = f"ETA {self._format_duration(remaining_items / items_per_second)}" if remaining_items > 0 else "ETA 0s"

        prefix = f"{label}"
        if dataset_name:
            prefix += f" [{dataset_name}]"
        return f"{prefix}: {speed} | {eta} ({processed}/{total})"
    
    def start_stop_watcher(self, interval_sec: float = 5.0):
        """
        Start a daemon thread that periodically checks should_stop()
        and terminates the process immediately when triggered.
        """
        if not self.is_ui_trainer:
            return
        if getattr(self, "_stop_watcher_started", False):
            return
        self._stop_watcher_started = True
        t = threading.Thread(
            target=self._stop_watcher_thread, args=(interval_sec,), daemon=True
        )
        t.start()

    def _stop_watcher_thread(self, interval_sec: float):
        while True:
            try:
                should_stop = self.should_stop(force=True)
                should_return_to_queue = self.should_return_to_queue(force=True)
                if should_stop or should_return_to_queue:
                    if self._enter_save_before_stop(to_queue=should_return_to_queue and not should_stop):
                        # the training thread finishes the step, saves it and stops by itself
                        if time.monotonic() < self._graceful_stop_deadline:
                            self._reassert_saving_info()
                            time.sleep(interval_sec)
                            continue
                        print(f"Saving before stopping took over {self._save_on_stop_timeout:.0f}s; "
                              f"stopping now (the previous checkpoint is intact)")
                    else:
                        # stopping at once (not training yet, or save on stop is off), but a
                        # checkpoint may be half written: let it land first
                        self._let_pending_write_finish()
                        should_stop = should_stop or self.should_stop(force=True)
                    # Mark and update status (non-blocking; uses existing infra)
                    self.is_stopping = True
                    target_status = "stopped" if should_stop else "queued"
                    reason = ""
                    if getattr(self, "_in_train_loop", False):
                        reason = ("; saving before stopping took too long and was cut off, the last "
                                  "finished checkpoint is kept"
                                  if self._graceful_stop_target is not None
                                  else "; the step in progress was not saved")
                    target_info = ("Job stopped (remote" if should_stop else "Job queued (remote") + reason + ")"
                    self._run_async_operation(
                        self._update_status(target_status, target_info)
                    )
                    # Best-effort flush pending async ops
                    try:
                        asyncio.run(self.wait_for_all_async())
                    except RuntimeError:
                        pass
                    # Try to stop DB thread pool quickly
                    try:
                        self.thread_pool.shutdown(wait=False, cancel_futures=True)
                    except TypeError:
                        self.thread_pool.shutdown(wait=False)
                    print("")
                    print("****************************************************")
                    print("    Stop signal received; terminating process.      ")
                    print("****************************************************")
                    os.kill(os.getpid(), signal.SIGINT)
                time.sleep(interval_sec)
            except Exception:
                time.sleep(interval_sec)

    def _prune_async_tasks(self):
        self._async_tasks = [task for task in self._async_tasks if not task.done()]

    def _run_async_operation(self, coro):
        """Queue UI work on a background event loop instead of blocking training."""
        if not self.is_ui_trainer:
            return
        if not self._async_loop_ready.is_set():
            self._async_loop_ready.wait(timeout=1.0)
        self._prune_async_tasks()
        task = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
        self._async_tasks.append(task)

    async def _execute_db_operation(self, operation_func):
        """Execute a database operation in a separate thread to avoid blocking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, operation_func)

    def _db_connect(self):
        """Create a new connection for each operation to avoid locking."""
        conn = sqlite3.connect(self.sqlite_db_path, timeout=10.0)
        conn.isolation_level = None  # Enable autocommit mode
        return conn

    def _refresh_remote_control_flags(self, force: bool = False):
        if not self.is_ui_trainer:
            return
        now = time.monotonic()
        if not force and (now - self._last_stop_poll_at) < self._stop_poll_interval:
            return

        def _check_flags():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT stop, return_to_queue FROM Job WHERE id = ?", (self.job_id,)
                )
                row = cursor.fetchone()
                if row is None:
                    return False, False
                return row[0] == 1, row[1] == 1

        self._cached_should_stop, self._cached_return_to_queue = _check_flags()
        self._last_stop_poll_at = now

    def should_stop(self, force: bool = False):
        self._refresh_remote_control_flags(force=force)
        return self._cached_should_stop

    def should_return_to_queue(self, force: bool = False):
        self._refresh_remote_control_flags(force=force)
        return self._cached_return_to_queue

    def _save_on_stop_enabled(self) -> bool:
        return (bool(getattr(self.save_config, "save_on_stop", True))
                and os.environ.get("AI_TOOLKIT_SAVE_ON_STOP", "1") != "0")

    def _enter_save_before_stop(self, to_queue: bool) -> bool:
        """Switch a stop request into "save this step, then stop"; True if that is (now) the plan.

        Only during training: before it (loading, caching) nothing new would be saved, and those
        stops stay immediate. Called from the stop watcher and from the training thread.
        """
        if self._graceful_stop_target is not None:
            if not to_queue:
                self._graceful_stop_target = "stopped"  # an explicit Stop wins over a queue pause
            return True
        if not self.is_ui_trainer or not getattr(self, "_in_train_loop", False) or not self.accelerator.is_main_process:
            return False
        if not self._save_on_stop_enabled() or not self.background_save_possible():
            # (a synchronous save writes over the real files, so it is never interrupted on purpose)
            if not getattr(self, "_stop_without_save_noted", False):
                self._stop_without_save_noted = True
                reason = ("save_on_stop is off" if not self._save_on_stop_enabled()
                          else "this setup saves checkpoints synchronously")
                print(f"\nStopping without saving the current step first ({reason}); "
                      f"training resumes from the last saved checkpoint")
            return False
        with self._graceful_stop_lock:
            if self._graceful_stop_target is None:
                self._graceful_stop_deadline = time.monotonic() + self._save_on_stop_timeout
                self._graceful_stop_target = "queued" if to_queue else "stopped"
                print("\nStop requested: the current step finishes and is saved, then the job stops")
                self._set_saving_info(f"{SAVING_BEFORE_STOP}: finishing the current step")
        return True

    def _set_saving_info(self, text):
        self._graceful_info = text
        self._last_info_assert = time.monotonic()
        self._run_async_operation(self._update_info_if_running(text))

    def _reassert_saving_info(self):
        """Every few seconds while saving before a stop: put the text back if something (a sample
        progress update, the queue, a second stop request) replaced it; the UI's stop route and
        its "Stop now" button go by it."""
        text = self._graceful_info
        if text and time.monotonic() - self._last_info_assert >= 5.0:
            self._last_info_assert = time.monotonic()
            self._run_async_operation(self._update_info_if_running(text))

    async def _update_info_if_running(self, info):
        if not self.accelerator.is_main_process:
            return

        def _do_update():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    # never over a final status: that one is written after the checkpoint is on disk
                    cursor.execute("UPDATE Job SET info = ? WHERE id = ? AND status = 'running'",
                                   (info, self.job_id))
                finally:
                    cursor.execute("COMMIT")

        await self._execute_db_operation(_do_update)

    def _let_pending_write_finish(self):
        writer = getattr(self, "_checkpoint_writer", None)
        if writer is None or not writer.busy:
            return
        step = writer.pending_step
        print(f"Stop: letting the checkpoint for step {step} finish writing first")
        self._set_saving_info(f"{SAVING_BEFORE_STOP}: writing the checkpoint for step {step}")
        writer.join(self._save_on_stop_timeout)

    def save_before_stop_requested(self) -> bool:
        if self._graceful_stop_target is None and self.is_ui_trainer and getattr(self, "_in_train_loop", False):
            # catch a stop the watcher has not seen yet (it polls about once a second)
            stop = self.should_stop()
            to_queue = self.should_return_to_queue()
            if stop or to_queue:
                self._enter_save_before_stop(to_queue=to_queue and not stop)
        return self._graceful_stop_target is not None

    def maybe_stop(self, force: bool = False, at_step_end: bool = False):
        if not self.is_ui_trainer:
            return
        stop = self.should_stop(force=force)
        to_queue = self.should_return_to_queue(force=force)
        if not (stop or to_queue):
            return
        if self._enter_save_before_stop(to_queue=to_queue and not stop):
            # stop only once the step that just ended has been saved; a stop noticed after the
            # step's save check gets one more step, which is then saved
            if not at_step_end or not getattr(self, "step_was_saved", False):
                return
            if getattr(self, "_stop_save_step", None) is None:
                # first seen here, after the step's own (regular) save: that is the checkpoint it leaves
                self._stop_save_step = self.step_num - 1
                self._stop_save_reused = False
        # the final status is set in on_error, once any checkpoint being written is on disk
        self.is_stopping = True
        raise Exception("Job stopped" if stop else "Job returning to queue")

    async def _update_key(self, key, value):
        if not self.accelerator.is_main_process:
            return

        def _do_update():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    # Convert the value to string if it's not already
                    if isinstance(value, str):
                        value_to_insert = value
                    else:
                        value_to_insert = str(value)

                    # Use parameterized query for both the column name and value
                    update_query = f"UPDATE Job SET {key} = ? WHERE id = ?"
                    cursor.execute(
                        update_query, (value_to_insert, self.job_id))
                finally:
                    cursor.execute("COMMIT")

        await self._execute_db_operation(_do_update)

    def update_step(self, force: bool = False):
        """Update the step count, but not so often that UI writes dominate fast steps."""
        if self.accelerator.is_main_process and self.is_ui_trainer:
            now = time.monotonic()
            if not force and (now - self._last_step_update_at) < self._step_update_interval:
                return
            self._last_step_update_at = now
            self._run_async_operation(self._update_key("step", self.step_num))

    def update_db_key(self, key, value):
        """Non-blocking update a key in the database."""
        if self.accelerator.is_main_process and self.is_ui_trainer:
            self._run_async_operation(self._update_key(key, value))

    def cache_progress_hook(
            self,
            cache_kind: str,
            processed: int,
            total: int,
            elapsed_seconds: float,
            dataset_path: str,
    ):
        if not self.accelerator.is_main_process or not self.is_ui_trainer or total <= 0:
            return

        now = time.monotonic()
        if processed < total and (now - self._last_cache_update_at) < self._cache_update_interval:
            return

        self._last_cache_update_at = now
        self.update_db_key(
            "cache_speed_string",
            self._build_cache_stats(cache_kind, processed, total, elapsed_seconds, dataset_path),
        )

    async def _update_status(self, status: AITK_Status, info: Optional[str] = None):
        if not self.accelerator.is_main_process or not self.is_ui_trainer:
            return

        def _do_update():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    if info is not None:
                        cursor.execute(
                            "UPDATE Job SET status = ?, info = ? WHERE id = ?",
                            (status, info, self.job_id)
                        )
                    else:
                        cursor.execute(
                            "UPDATE Job SET status = ? WHERE id = ?",
                            (status, self.job_id)
                        )
                finally:
                    cursor.execute("COMMIT")

        await self._execute_db_operation(_do_update)

    def update_status(self, status: AITK_Status, info: Optional[str] = None, force: bool = False):
        """Non-blocking update of status."""
        if self.accelerator.is_main_process and self.is_ui_trainer:
            if status == "running" and info is not None:
                if self._graceful_stop_target is not None and not info.startswith(SAVING_BEFORE_STOP):
                    info = f"{SAVING_BEFORE_STOP}: {info}"  # the UI's stop handling goes by this text
                if info.startswith(SAVING_BEFORE_STOP):
                    self._graceful_info = info
                    self._last_info_assert = time.monotonic()
            payload = (status, info)
            if not force and payload == self._last_status_payload:
                return
            self._last_status_payload = payload
            self._run_async_operation(self._update_status(status, info))

    async def wait_for_all_async(self):
        """Wait for all tracked async operations to complete."""
        if not self._async_tasks:
            return

        try:
            tasks = [asyncio.wrap_future(task) for task in self._async_tasks]
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            pass
        finally:
            # Clear the task list after completion
            self._async_tasks.clear()

    def on_error(self, e: Exception):
        self._close_step_feed()
        super(DiffusionTrainer, self).on_error(e)  # waits for a checkpoint still being written
        if self.is_ui_trainer:
            stop_requested = self.should_stop(force=True)
            return_to_queue_requested = self.should_return_to_queue(force=True)
            if stop_requested or return_to_queue_requested:
                self.is_stopping = True
            if self.accelerator.is_main_process:
                self._graceful_info = None  # no more re-asserting the "saving" text
                suffix = ""
                if self._graceful_stop_target is not None:
                    stop_step = getattr(self, "_stop_save_step", None)
                    if stop_step is not None and (getattr(self, "_stop_save_reused", False)
                                                  or stop_step == self._last_save_ok_step):
                        suffix = f" (saved at step {stop_step})"
                        if not getattr(self, "_stop_save_reused", False) and getattr(self, "_last_save_incomplete", False):
                            suffix = f" (saved at step {stop_step}, but not all of its optimizer/data-loader state)"
                    elif stop_step is not None:
                        suffix = f" (saving step {stop_step} failed; it resumes from the previous checkpoint)"
                    if str(e) not in ("Job stopped", "Job returning to queue"):
                        suffix += f" after an error: {e}"
                if stop_requested or return_to_queue_requested:
                    self.update_db_key("speed_string", "")
                    self.update_db_key("cache_speed_string", "")
                if stop_requested:  # an explicit Stop wins over a queue pause
                    self.update_status("stopped", "Job stopped" + suffix, force=True)
                elif return_to_queue_requested:
                    self.update_status("queued", "Job queued" + suffix, force=True)
                elif self.is_stopping:
                    self.update_status("stopped", "Job stopped" + suffix, force=True)
                else:
                    self.update_status("error", str(e), force=True)
            self.update_step(force=True)
            asyncio.run(self.wait_for_all_async())
            self.thread_pool.shutdown(wait=True)
            self._shutdown_async_loop()

    def handle_timing_print_hook(self, timing_dict):
        if "train_loop" not in timing_dict:
            print("train_loop not found in timing_dict", timing_dict)
            return
        seconds_per_iter = self.timer.get_lifetime_average("train_loop")
        if seconds_per_iter <= 0:
            seconds_per_iter = timing_dict["train_loop"]
        self.update_db_key("speed_string", self._build_runtime_stats(seconds_per_iter))

    def done_hook(self):
        self._close_step_feed()
        super(DiffusionTrainer, self).done_hook()
        if self.is_ui_trainer:
            self.update_step(force=True)
            self.update_status("completed", "Training completed")
            # Wait for all async operations to finish before shutting down
            asyncio.run(self.wait_for_all_async())
            self.thread_pool.shutdown(wait=True)
            self._shutdown_async_loop()

    def end_step_hook(self):
        super(DiffusionTrainer, self).end_step_hook()
        if self.is_ui_trainer:
            self.update_step()
            self.maybe_stop(at_step_end=True)

    def hook_background_save_finished(self, result):
        super().hook_background_save_finished(result)
        if result.step is not None:
            self._last_save_ok_step = result.step
            self._last_save_incomplete = bool(result.notes)
        if self.is_ui_trainer and self._graceful_stop_target is None:
            self.update_status("running", self._training_status_info())

    def hook_waiting_for_background_save(self, step):
        super().hook_waiting_for_background_save(step)
        if self.is_ui_trainer and self.accelerator.is_main_process:
            # the stop route holds its force-kill while it sees this text
            self.update_status("running", f"{SAVING_BEFORE_STOP}: writing the checkpoint for step {step}", force=True)

    def hook_before_model_load(self):
        super().hook_before_model_load()
        if self.is_ui_trainer:
            self.maybe_stop(force=True)
            self.update_status("running", "Loading model")

    def before_dataset_load(self):
        super().before_dataset_load()
        if self.is_ui_trainer:
            self.maybe_stop(force=True)
            self.update_status("running", "Loading dataset")

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        if self.is_ui_trainer:
            self.maybe_stop(force=True)
            # Report the ACTUAL training target (after Train-by-Epoch resolution and
            # train-on-top-of-loaded-LoRA, both resolved once the dataset is loaded)
            # so the UI shows "Step X of <real total>" instead of the static config.
            self.update_db_key("total_steps", int(self.train_config.steps))
            self.update_step(force=True)
            self.update_status("running", self._training_status_info())
            self.timer.add_after_print_hook(self.handle_timing_print_hook)
            if self.accelerator.is_main_process and getattr(self, "_current_step_feed", None) is None:
                try:
                    self._current_step_feed = CurrentStepFeed(self.save_root)
                except Exception as e:
                    self._current_step_feed = None
                    print(f"Note: the UI's current-step images are unavailable for this run ({e})")

    def _training_status_info(self) -> str:
        # "Training" gets re-set after every save and sample, so the OOM tally has
        # to live in it or the UI would forget it at the next checkpoint.
        skipped = getattr(self, "_oom_skip_total", None) or 0
        rescued = getattr(self, "_oom_rescued_total", None) or 0
        if skipped <= 0 and rescued <= 0:
            return "Training"
        parts = []
        if skipped > 0:
            last = getattr(self, "_oom_last_step", None)
            last_txt = f" (last at step {last})" if last is not None else ""
            parts.append(f"{skipped} step{'s' if skipped != 1 else ''} skipped for out-of-memory{last_txt}")
        if rescued > 0:
            parts.append(f"{rescued} OOM{'s' if rescued != 1 else ''} rescued by retry")
        return "Training - " + "; ".join(parts) + "; see skipped_oom.txt"

    # The job page shows the images of the step being trained, under the log. The feed only
    # stores references here; its own thread writes them out (see toolkit/current_step_feed.py).
    def _step_feed_call(self, method, *args):
        feed = getattr(self, "_current_step_feed", None)
        if feed is None:
            return
        try:
            getattr(feed, method)(*args)
        except Exception as e:  # a display aid must never stop training
            print(f"Note: the UI's current-step images are disabled for this run ({e})")
            self._close_step_feed(timeout=0)  # empty the panel; never wait on the training thread

    def _close_step_feed(self, timeout=2.0):
        feed = getattr(self, "_current_step_feed", None)
        self._current_step_feed = None
        if feed is not None:
            try:
                feed.close(timeout=timeout)
            except Exception:
                pass

    def hook_before_train_step(self, batch_list):
        super().hook_before_train_step(batch_list)
        # step_num counts finished steps (what the UI's progress shows); this one is the next
        self._step_feed_call("begin_step", int(self.step_num) + 1, int(self.train_config.steps), batch_list)

    def hook_before_forward_pass(self, microbatch_index, batch):
        super().hook_before_forward_pass(microbatch_index, batch)
        self._step_feed_call("forward_pass", microbatch_index)

    def hook_after_train_step(self):
        super().hook_after_train_step()
        self._step_feed_call("end_step")

    def hook_oom_tally_changed(self):
        super().hook_oom_tally_changed()
        if self.is_ui_trainer:
            self.update_status("running", self._training_status_info())

    def status_update_hook_func(self, string):
        self.update_status("running", string)

    def hook_after_sd_init_before_load(self):
        super().hook_after_sd_init_before_load()
        if self.is_ui_trainer:
            self.maybe_stop(force=True)
            self.sd.add_status_update_hook(self.status_update_hook_func)

    def sample_step_hook(self, img_num, total_imgs):
        super().sample_step_hook(img_num, total_imgs)
        if self.is_ui_trainer:
            self.maybe_stop()
            if self.save_before_stop_requested():
                raise SamplingInterrupted()  # the step is saved next, then the job stops
            self.update_status(
                "running", f"Generating images - {img_num + 1}/{total_imgs}")

    def sample(self, step=None, is_first=False):
        self.maybe_stop(force=True)
        if self.save_before_stop_requested():
            print("Skipping samples: the job is stopping")
            return
        total_imgs = len(self.sample_config.prompts)
        self.update_status("running", f"Generating images - 0/{total_imgs}")
        super().sample(step, is_first)
        self.maybe_stop(force=True)
        self.update_status("running", self._training_status_info())

    def save(self, step=None):
        self.maybe_stop(force=True)  # during training a stop request now means "save, then stop"
        if step is not None:
            self._last_save_requested_step = step
        if self._graceful_stop_target is not None:
            what = f"step {step}" if step is not None else "the end of training"
            self.update_status("running", f"{SAVING_BEFORE_STOP}: writing the checkpoint for {what}")
        else:
            self.update_status("running", "Saving model")
        super().save(step)
        if step is not None and not getattr(self, "last_save_in_background", False):
            self._last_save_ok_step = step  # written synchronously, and save() did not raise
            self._last_save_incomplete = False
        self.maybe_stop(force=True)
        writer = getattr(self, "_checkpoint_writer", None)
        pending = writer.pending_step if writer is not None else None
        if self._graceful_stop_target is not None:
            if pending is not None:
                self.update_status("running", f"{SAVING_BEFORE_STOP}: writing the checkpoint for step {pending}")
        elif pending is not None:
            self.update_status("running", f"{self._training_status_info()} (writing the step {pending} checkpoint in the background)")
        else:
            self.update_status("running", self._training_status_info())
