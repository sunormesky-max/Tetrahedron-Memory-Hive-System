import threading
import time


class ReadWriteLock:
    def __init__(self):
        self._lock = threading.Lock()
        self._can_read = threading.Condition(self._lock)
        self._can_write = threading.Condition(self._lock)
        self._readers = 0
        self._writers = 0
        self._writer_waiting = 0

    def acquire_read(self):
        with self._can_read:
            while self._writers > 0 or self._writer_waiting > 0:
                self._can_read.wait(timeout=1.0)
            self._readers += 1

    def release_read(self):
        with self._can_read:
            self._readers -= 1
            if self._readers == 0:
                self._can_write.notify()

    def acquire_write(self):
        with self._can_write:
            self._writer_waiting += 1
            while self._readers > 0 or self._writers > 0:
                self._can_write.wait(timeout=1.0)
            self._writer_waiting -= 1
            self._writers += 1

    def release_write(self):
        with self._can_write:
            self._writers -= 1
            self._can_write.notify()
            self._can_read.notify_all()

    class _ReadContext:
        def __init__(self, rwl):
            self._rwl = rwl
        def __enter__(self):
            self._rwl.acquire_read()
            return self
        def __exit__(self, *a):
            self._rwl.release_read()

    class _WriteContext:
        def __init__(self, rwl):
            self._rwl = rwl
        def __enter__(self):
            self._rwl.acquire_write()
            return self
        def __exit__(self, *a):
            self._rwl.release_write()

    def read_locked(self):
        return self._ReadContext(self)

    def write_locked(self):
        return self._WriteContext(self)
