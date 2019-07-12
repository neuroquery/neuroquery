import time
from functools import wraps


class Sleep(object):

    def __init__(self, delay=2):
        self.delay_ = delay

    def __call__(self, *args, **kwargs):
        time.sleep(self.delay_)


def re_raise(exception):
    raise (exception)


class try_n_times(object):
    def __init__(self, action=Sleep(), on_fail=re_raise, n_tries=3):
        self.action_ = action
        self.n_tries_ = n_tries
        self.on_fail_ = on_fail

    def __call__(self, fun):
        @wraps(fun)
        def decorate(*args, **kwargs):
            error = None
            tries = 0
            while (True):
                try:
                    return fun(*args, **kwargs)
                except Exception as e:
                    tries += 1
                    error = e
                    if (tries == self.n_tries_):
                        break
                self.action_(*args, **dict(kwargs, error=error))
            self.on_fail_(error)

        return decorate
