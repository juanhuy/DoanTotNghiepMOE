"""Converter regression test; uses only fixtures within this project."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import load_file, save_file
from prepare_olmoe import convert

class ConverterTest(unittest.TestCase):
    def test_round_trip_and_no_overwrite(self):
        fixture = Path(__file__).resolve().parents[1] / 'tests/fixtures/olmoe/unnormalized-mha'
        config = json.loads((fixture / 'config.json').read_text())
        tensors = load_file(str(fixture / 'model.safetensors'))
        source_weights = {k:v for k,v in tensors.items() if '.mlp.experts.' not in k}
        d, i = config['hidden_size'], config['intermediate_size']
        for layer in range(config['num_hidden_layers']):
            for expert in range(config['num_experts']):
                p = f'model.layers.{layer}.mlp.experts.{expert}'
                q, s = tensors[p+'.merged_weight'], tensors[p+'.qs']
                for name, weights, scales, rows, cols in [
                    ('gate_proj',q[:i*d],s[:i],i,d),
                    ('up_proj',q[i*d:2*i*d],s[i:2*i],i,d),
                    ('down_proj',q[2*i*d:],s[2*i:],d,i)]:
                    source_weights[p+'.'+name+'.weight'] = weights.reshape(rows,cols).float()*scales[:,None]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source'; source.mkdir(); output = root/'output'
            (source/'config.json').write_text(json.dumps(config))
            save_file(source_weights,str(source/'weights.safetensors'))
            convert(source,output)
            self.assertFalse((output/'INCOMPLETE').exists())
            actual = {}
            for path in output.glob('*.safetensors'): actual.update(load_file(str(path)))
            self.assertEqual(set(actual),set(tensors))
            for name in tensors:
                torch.testing.assert_close(actual[name],tensors[name],rtol=1e-6,atol=1e-8)
            with self.assertRaises(FileExistsError): convert(source,output)

if __name__=='__main__': unittest.main()
