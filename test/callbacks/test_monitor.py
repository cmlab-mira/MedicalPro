import pytest
from pathlib import Path

from src.callbacks.monitor import Monitor


def test_monitor():
    monitor = Monitor(checkpoints_dir=Path('./models/test/checkpoints'),
                      mode='max',
                      target='loss',
                      saved_freq=2,
                      early_stop=2)
    assert monitor.is_saved(1) == None
    assert monitor.is_saved(2) == Path('./models/test/checkpoints/model_2.pth')
    assert monitor.is_early_stopped() == False
