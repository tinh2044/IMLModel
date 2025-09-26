import os
import sys
import time
import datetime
from collections import defaultdict, deque

import torch
import torch.distributed as dist
from loguru import logger
import re

from utils import is_dist_avail_and_initialized


class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter=", ", log_file=""):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = Logger(log_file=log_file)

        self.color_schemes = {
            "loss": {"key": "\033[91m", "value": "\033[0m"},
            "lr": {"key": "\033[94m", "value": "\033[0m"},
            "time": {"key": "\033[93m", "value": "\033[0m"},
            "data": {"key": "\033[96m", "value": "\033[0m"},
            "memory": {"key": "\033[95m", "value": "\033[0m"},
            "eta": {"key": "\033[92m", "value": "\033[0m"},
            "step": {"key": "\033[1m", "value": "\033[0m"},
            "accuracy": {"key": "\033[92m", "value": "\033[0m"},
            "precision": {"key": "\033[94m", "value": "\033[0m"},
            "recall": {"key": "\033[95m", "value": "\033[0m"},
            "f1": {"key": "\033[93m", "value": "\033[0m"},
            "default": {"key": "\033[0m", "value": "\033[0m"},
        }

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            color_scheme = self._get_color_scheme(name)
            colored_name = f"{color_scheme['key']}{name}{color_scheme['value']}"
            colored_value = f"{color_scheme['value']}{str(meter)}"
            loss_str.append(f"{colored_name}: {colored_value}")

        return self.delimiter.join(loss_str)

    def _get_color_scheme(self, metric_name):
        for key, scheme in self.color_schemes.items():
            if key in metric_name.lower():
                return scheme
        return self.color_schemes["default"]

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    if os.environ.get("LOCAL_RANK", "0") == "0":
                        log_str = self._format_log(
                            i, len(iterable), eta_string, iter_time, data_time, MB
                        )
                        self.logger.info(log_str)
                else:
                    log_str = self._format_log(
                        i, len(iterable), eta_string, iter_time, data_time, MB
                    )
                    self.logger.info(log_str)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            if os.environ.get("LOCAL_RANK", "0") == "0":
                final_str = "{} Total time: {} ({:.4f} s / it)".format(
                    header, total_time_str, total_time / len(iterable)
                )
                self.logger.info(final_str)
        else:
            final_str = "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
            self.logger.info(final_str)

    def _format_log(self, i, total_len, eta_string, iter_time, data_time, MB):
        step_color = self.color_schemes["step"]["key"]
        eta_color = self.color_schemes["eta"]["key"]
        time_color = self.color_schemes["time"]["key"]
        data_color = self.color_schemes["data"]["key"]
        memory_color = self.color_schemes["memory"]["key"]
        reset_color = self.color_schemes["default"]["value"]

        if torch.cuda.is_available():
            return f"{step_color}Step [{i}/{total_len}]{reset_color}{self.delimiter}{eta_color}ETA: {eta_string}{reset_color}{self.delimiter}{str(self)}{self.delimiter}{time_color}Time: {str(iter_time)}{reset_color}{self.delimiter}{data_color}Data: {str(data_time)}{reset_color}{self.delimiter}{memory_color}Max Memory: {torch.cuda.max_memory_allocated() / MB:.0f}MB{reset_color}"
        else:
            return f"{step_color}Step [{i}/{total_len}]{reset_color}{self.delimiter}{eta_color}ETA: {eta_string}{reset_color}{self.delimiter}{str(self)}{self.delimiter}{time_color}Time: {str(iter_time)}{reset_color}{self.delimiter}{data_color}Data: {str(data_time)}{reset_color}"


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        is_ddp = is_dist_avail_and_initialized()
        current_rank = dist.get_rank() if is_ddp else 0
        is_main_process = current_rank == 0

        if not is_main_process:
            self.log_file = None
            return
        logger.remove()

        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {message}",
            level="INFO",
            colorize=True,
        )
        logger.add(
            log_file,
            format=self._file_formatter,
            level="INFO",
            rotation="10MB",
        )

        logger.info(f"Logging to {log_file}")

    def write(self, message):
        logger.info(message.strip())

    def flush(self):
        pass

    @staticmethod
    def info(msg):
        logger.info(msg)

    @staticmethod
    def warning(msg):
        logger.warning(msg)

    @staticmethod
    def error(msg):
        logger.error(msg)

    @staticmethod
    def _strip_ansi(text):
        ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
        return ansi_re.sub("", text)

    def _file_formatter(self, record):
        time_str = record["time"].strftime("%Y-%m-%d %H:%M:%S")
        message = self._strip_ansi(record["message"])
        return f"{time_str} | {message}\n"
