import tensorflow as tf


class TestModel:
    def test_device(self):
        devices = tf.config.list_physical_devices()
        # assert "CPU" in devices
