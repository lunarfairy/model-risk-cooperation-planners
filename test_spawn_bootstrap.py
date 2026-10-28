"""Check acceptance-bootstrap equivalence under spawn using synthetic records.

Run: python test_spawn_bootstrap.py
No human records or full calibration are needed.
"""
from multiprocessing import get_context

import numpy as np
import pandas as pd

import calib2 as C


def synthetic_records():
    rng = np.random.default_rng(13)
    n = 512
    records = pd.DataFrame({
        'group': np.repeat(np.arange(8), n // 8),
        'pid': np.repeat(np.arange(64), n // 64),
        'val': rng.choice([-1, 1], n),
        'ref_coop': rng.integers(0, 2, n),
        'deg_dec': rng.integers(1, 16, n),
        'deg_ref': rng.integers(1, 16, n),
        'acc': rng.integers(0, 2, n),
    })
    return records


if __name__ == '__main__':
    records = synthetic_records()
    C._init(None, records)
    expected = C.acc_boot_task(5000)
    with get_context('spawn').Pool(1, initializer=C._init, initargs=(None, records)) as pool:
        actual = pool.map(C.acc_boot_task, [5000])[0]
    for serial, parallel in zip(expected, actual):
        np.testing.assert_allclose(serial, parallel, rtol=1e-12, atol=1e-12)
    print('PASS: acceptance bootstrap agrees between serial and spawned workers.')
