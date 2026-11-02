"""INT4 Converter regression test; verifies nibble packing and fixture conversion."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import load_file, save_file
from prepare_olmoe_int4 import convert, pack_int4_nibbles


class Int4ConverterTest(unittest.TestCase):
    def test_pack_int4_nibbles_values(self):
        # Test values covering entire signed 4-bit range [-8, 7]
        values = torch.tensor([-8, -7, -2, -1, 0, 1, 6, 7], dtype=torch.int8)
        packed = pack_int4_nibbles(values)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(len(packed), 4)

        # Unpack manually to verify bit alignment
        unpacked = []
        for b in packed.tolist():
            # low nibble
            lo = b & 0x0F
            lo_signed = lo if lo < 8 else lo - 16
            # high nibble
            hi = (b >> 4) & 0x0F
            hi_signed = hi if hi < 8 else hi - 16
            unpacked.extend([lo_signed, hi_signed])

        self.assertEqual(unpacked, values.tolist())

    def test_round_trip_fixture(self):
        fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/olmoe/unnormalized-mha"
        config = json.loads((fixture / "config.json").read_text())
        tensors = load_file(str(fixture / "model.safetensors"))
        source_weights = {k: v for k, v in tensors.items() if ".mlp.experts." not in k}
        d, i = config["hidden_size"], config["intermediate_size"]
        for layer in range(config["num_hidden_layers"]):
            for expert in range(config["num_experts"]):
                p = f"model.layers.{layer}.mlp.experts.{expert}"
                q, s = tensors[p + ".merged_weight"], tensors[p + ".qs"]
                for name, weights, scales, rows, cols in [
                    ("gate_proj", q[: i * d], s[:i], i, d),
                    ("up_proj", q[i * d : 2 * i * d], s[i : 2 * i], i, d),
                    ("down_proj", q[2 * i * d :], s[2 * i :], d, i),
                ]:
                    source_weights[p + "." + name + ".weight"] = (
                        weights.reshape(rows, cols).float() * scales[:, None]
                    )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            output = root / "output"
            (source / "config.json").write_text(json.dumps(config))
            save_file(source_weights, str(source / "weights.safetensors"))

            convert(source, output)
            self.assertFalse((output / "INCOMPLETE").exists())

            actual = {}
            for path in output.glob("*.safetensors"):
                actual.update(load_file(str(path)))

            # Verify all expected tensors are present
            self.assertEqual(set(actual), set(tensors))

            for layer in range(config["num_hidden_layers"]):
                for expert in range(config["num_experts"]):
                    p = f"model.layers.{layer}.mlp.experts.{expert}"
                    weight_key = p + ".merged_weight"
                    scale_key = p + ".qs"

                    # In INT4, packed weights are uint8 and exactly half the number of elements
                    expected_int8_len = 3 * d * i
                    self.assertEqual(actual[weight_key].dtype, torch.uint8)
                    self.assertEqual(actual[weight_key].shape, torch.Size([expected_int8_len // 2]))

                    # Scales are FP32 per row
                    self.assertEqual(actual[scale_key].dtype, torch.float32)
                    self.assertEqual(actual[scale_key].shape, torch.Size([2 * i + d]))

            # Non-expert tensors must match identically
            for name in source_weights:
                if ".mlp.experts." not in name:
                    torch.testing.assert_close(actual[name], tensors[name])

            # Output directory guard: must fail if already exists
            with self.assertRaises(FileExistsError):
                convert(source, output)


if __name__ == "__main__":
    unittest.main()

