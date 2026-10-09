from tqdm import tqdm
import time


class ToolkitProgressBar(tqdm):
    DEFAULT_BAR_FORMAT = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {step_speed}{postfix}]"
    )

    def __init__(self, *args, **kwargs):
        now = time.time()
        self.paused = False
        self.last_time = now
        self.last_step_duration = None
        self.last_step_update_t = now
        kwargs.setdefault("bar_format", self.DEFAULT_BAR_FORMAT)
        super().__init__(*args, **kwargs)
        self.last_time = self._time()
        self.last_step_update_t = getattr(self, "start_t", self._time())

    def pause(self):
        if not self.paused:
            self.paused = True
            self.last_time = self._time()

    def unpause(self):
        if self.paused:
            self.paused = False
            cur_t = self._time()
            paused_duration = cur_t - self.last_time
            self.start_t += paused_duration
            self.last_print_t += paused_duration
            self.last_step_update_t += paused_duration

    def update(self, *args, **kwargs):
        if not self.paused:
            delta_n = args[0] if len(args) > 0 else kwargs.get("n", 1)
            if delta_n > 0:
                cur_t = self._time()
                elapsed = max(0.0, cur_t - self.last_step_update_t)
                if elapsed > 0:
                    self.last_step_duration = elapsed / delta_n
                self.last_step_update_t = cur_t
            super().update(*args, **kwargs)

    @property
    def format_dict(self):
        format_dict = super().format_dict
        last_step_duration = getattr(self, "last_step_duration", None)
        if last_step_duration is None or last_step_duration <= 0:
            format_dict["step_speed"] = "last n/a"
        else:
            format_dict["step_speed"] = f"last {last_step_duration:.2f}s/step"
        return format_dict
