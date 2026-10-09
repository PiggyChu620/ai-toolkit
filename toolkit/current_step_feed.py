"""Tells the UI which images the training step is working on.

The job page shows the images of the step in progress, under the log. Each image appears just
before its forward pass, so with gradient accumulation one step fills in pass by pass. The panel
empties when the step ends.

The training thread only hands over references it already holds; that takes microseconds and
does no I/O and no formatting. A background thread turns the latest state into JSON and replaces
.current_step.json in the job folder. The UI server watches that file, pushes each change to the
browser, and the browser loads the images straight from the dataset by filename. Nothing is
copied. A slow or failing disk can delay or drop an update but can never stall or fail training.
Updates are latest-wins: if a write is slow, older states are skipped, because every state is
complete on its own.
"""
import json
import os
import threading
import time

# Dot-prefixed, like .job_config.json: checkpoint lookups glob "<job name>*" in the same folder
# (get_latest_save_path), and a job named "c" or "current" must never resume from this file.
FILE_NAME = ".current_step.json"
FORMAT_VERSION = 1
_CAPTION_LIMIT = 2000
_REPLACE_RETRIES = 10  # Windows: a reader or a virus scanner can hold the file for a moment


class CurrentStepFeed:
    def __init__(self, folder):
        self.path = os.path.join(folder, FILE_NAME)
        self._cond = threading.Condition()
        self._pending = None  # newest state not yet written; written states are dropped
        self._writing = False
        self._closed = False
        self._seq = 0
        self._warned = False
        # training-thread state
        self._step = None
        self._total_steps = None
        self._passes = ()
        self._thread = threading.Thread(target=self._run, name="aitk-current-step", daemon=True)
        self._thread.start()
        self._publish(None)  # a new run starts empty, never showing the previous run's last step

    # -- training thread: each call only stores references and wakes the writer ------------------

    def begin_step(self, step, total_steps, batch_list):
        """The step's batches, before any of them is trained on. Nothing is shown yet."""
        self._step = step
        self._total_steps = total_steps
        self._passes = tuple(
            tuple(getattr(batch, "file_items", None) or ()) if batch is not None else ()
            for batch in (batch_list or ())
        )

    def forward_pass(self, index):
        """Micro-batch `index` (0-based) is about to run its forward pass: show it and the ones before."""
        if self._step is not None:
            self._publish(index)

    def end_step(self):
        """The step is done (trained, or skipped after an out-of-memory error): empty the panel."""
        if self._step is not None:
            self._publish(None)
            self._passes = ()

    def close(self, timeout=2.0):
        """Hide the panel and wait briefly for the write; the process may be about to exit."""
        self._step = None  # no step at all, not "step N finished": nothing more will come
        self._total_steps = None
        self._passes = ()
        self._publish(None)
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._thread.join(timeout)

    def _publish(self, active_index):
        self._seq += 1
        passes = self._passes if active_index is not None else ()
        state = (self._seq, self._step, self._total_steps, active_index, passes)
        with self._cond:
            self._pending = state  # replaces any state the writer has not reached yet
            self._cond.notify_all()

    # -- background thread ------------------------------------------------------------------------

    def flush(self, timeout=5.0):
        """Wait until everything published so far is written. For tests and shutdown."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._pending is not None or self._writing:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def _run(self):
        while True:
            with self._cond:
                while self._pending is None and not self._closed:
                    self._cond.wait()
                state, self._pending = self._pending, None
                if state is None:
                    return  # closed, and nothing left to write
                self._writing = True
            try:
                self._write(self._payload(state))
            except Exception as e:
                if not self._warned:
                    self._warned = True
                    print(f"Note: the UI's current-step images could not be updated ({e}); training is unaffected")
            finally:
                with self._cond:
                    self._writing = False
                    self._cond.notify_all()

    @staticmethod
    def _payload(state):
        seq, step, total_steps, active_index, passes = state
        slots, items = [], []
        if active_index is not None:
            for pass_index, file_items in enumerate(passes):
                for fi in file_items:
                    width = _as_int(getattr(fi, "width", None))
                    height = _as_int(getattr(fi, "height", None))
                    slots.append({"pass": pass_index + 1, "width": width, "height": height})
                    if pass_index > active_index:
                        continue  # its forward pass has not started yet: reserve the tile only
                    caption = getattr(fi, "raw_caption", None) or getattr(fi, "caption", None) or ""
                    items.append({
                        "path": str(getattr(fi, "path", "") or ""),
                        "caption": str(caption)[:_CAPTION_LIMIT],
                        "pass": pass_index + 1,
                        "width": width,
                        "height": height,
                        "bucket": [_as_int(getattr(fi, "crop_width", None)),
                                   _as_int(getattr(fi, "crop_height", None))],
                        "flip_x": bool(getattr(fi, "flip_x", False)),
                        "flip_y": bool(getattr(fi, "flip_y", False)),
                        "is_reg": bool(getattr(fi, "is_reg", False)),
                    })
        return {
            "version": FORMAT_VERSION,
            "pid": os.getpid(),
            "seq": seq,
            "step": step,
            "total_steps": total_steps,
            "pass": (active_index + 1) if active_index is not None else 0,
            "passes": len(passes) if active_index is not None else 0,
            "slots": slots,
            "items": items,
        }

    def _write(self, payload):
        # ASCII-only JSON: any caption or file name encodes, even one with a lone surrogate
        data = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        tmp_path = self.path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(data)
            # Normally the replace is atomic. But on Windows it fails while anyone has the target
            # open, whatever their share mode (classic rename semantics; exFAT, like the USB drive
            # the job folders are on, has no POSIX rename). The UI server's reads take about a
            # millisecond, so retry briefly.
            for attempt in range(_REPLACE_RETRIES):
                try:
                    os.replace(tmp_path, self.path)
                    return
                except PermissionError:
                    if attempt == _REPLACE_RETRIES - 1:
                        break
                    time.sleep(0.02)
            # Still held: write in place. A reader can then catch a torn file; the UI server skips
            # it and reads again on this write's own change notification.
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(data)
        finally:
            try:
                os.remove(tmp_path)  # gone after a successful replace; otherwise do not leave it behind
            except OSError:
                pass


def _as_int(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None
