from unittest import mock

import pytest

from neuroquery import _utils


def test_try_n_times():
    f = mock.Mock(side_effect=Exception)
    action = mock.Mock()
    on_fail = mock.Mock()
    nf = _utils.try_n_times(action, on_fail, 7)(f)
    nf()
    assert f.call_count == 7
    assert action.call_count == 6
    assert on_fail.call_count == 1
