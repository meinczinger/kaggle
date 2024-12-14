import tensorflow as tf


class TestModel:
    def test_device(self):
        devices = tf.config.list_physical_devices()
        assert any(["GPU" in i for i in devices])
