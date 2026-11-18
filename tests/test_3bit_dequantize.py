import unittest

import torch

from hqq.core.quantize import Quantizer


class TestThreeBitPacking(unittest.TestCase):
    def test_dequantization_matches_unpacked_weights(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            weights = torch.linspace(-1, 1, 48, device=device).reshape(6, 8)
            for axis in (0, 1):
                for channel_wise in (False, True):
                    for group_size in (None, 4):
                        with self.subTest(
                            device=device,
                            axis=axis,
                            channel_wise=channel_wise,
                            group_size=group_size,
                        ):
                            config = dict(
                                nbits=3,
                                axis=axis,
                                channel_wise=channel_wise,
                                group_size=group_size,
                                optimize=False,
                                device=device,
                            )
                            unpacked, meta = Quantizer.quantize(
                                weights, bitpack=False, **config
                            )
                            expected = Quantizer.dequantize(unpacked, meta)
                            packed, meta = Quantizer.quantize(weights, **config)
                            torch.testing.assert_close(
                                Quantizer.dequantize(packed, meta), expected
                            )
