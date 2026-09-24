use moe_tier_engine::{int8_expert::Int8Expert, safetensors::TensorIndex};
use serde_json::json;
use std::{fs, path::Path};

fn shard(path: &Path, header: serde_json::Value, payload: &[u8]) {
    raw_shard(path, &serde_json::to_string(&header).unwrap(), payload);
}
fn raw_shard(path: &Path, header: &str, payload: &[u8]) {
    let mut header = header.as_bytes().to_vec();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(payload);
    fs::write(path, bytes).unwrap();
}
fn fixture(path: &Path, scales: &[f32]) {
    // hidden=2, intermediate=1: gate=[1,-1], up=[2,1], down=[1,-2].
    // Non-unit row scales exercise all three projection offsets.
    shard(
        &path.join("a.safetensors"),
        json!({
            "model.layers.3.mlp.experts.7.merged_weight": {
                "dtype":"I8", "shape":[6], "data_offsets":[0,6]}
        }),
        &[1, 255, 2, 1, 1, 254],
    );
    let data: Vec<_> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    shard(
        &path.join("b.safetensors"),
        json!({
            "model.layers.3.mlp.experts.7.qs": {
                "dtype":"F32", "shape":[4], "data_offsets":[0,16]}
        }),
        &data,
    );
}

#[test]
fn merged_expert_across_shards_matches_dense_reference() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path(), &[0.5, 0.25, 2., 0.75]);
    let index = TensorIndex::open(dir.path()).unwrap();
    assert_eq!(index.tensors().len(), 2);
    let expert = Int8Expert::load(&index, 3, 7, 2, 1, 22).unwrap();
    assert_eq!(expert.resident_bytes(), 22);
    let actual = expert.forward(&[2., -1.]).unwrap();
    let gate = 1.5f32;
    let up = 0.75;
    let activation = gate / (1. + (-gate).exp()) * up;
    assert!((actual[0] - 2. * activation).abs() < 1e-6);
    assert!((actual[1] + 1.5 * activation).abs() < 1e-6);
    assert!(expert.forward(&[1.]).is_err());
    assert!(expert.forward(&[f32::NAN, 1.]).is_err());
    assert!(Int8Expert::load(&index, 3, 7, 2, 1, 21).is_err());
    assert!(Int8Expert::load(&index, 2, 7, 2, 1, 100).is_err());
    assert!(Int8Expert::load(&index, 3, 7, 3, 1, 100).is_err());
    assert!(index
        .read_f32("model.layers.3.mlp.experts.7.merged_weight", 100)
        .is_err());
}

#[test]
fn reads_float_formats_and_enforces_decoded_budget() {
    let dir = tempfile::tempdir().unwrap();
    let mut payload = Vec::new();
    for v in [0x3c00u16, 0xc000, 1] {
        payload.extend(v.to_le_bytes());
    }
    for v in [0x3f80u16, 0xbf80] {
        payload.extend(v.to_le_bytes());
    }
    payload.extend(3.5f32.to_le_bytes());
    shard(
        &dir.path().join("f.safetensors"),
        json!({
            "half": {"dtype":"F16","shape":[3],"data_offsets":[0,6]},
            "bf": {"dtype":"BF16","shape":[2],"data_offsets":[6,10]},
            "single": {"dtype":"F32","shape":[],"data_offsets":[10,14]},
            "__metadata__": {"format":"pt"}
        }),
        &payload,
    );
    let index = TensorIndex::open(dir.path()).unwrap();
    assert_eq!(
        index.read_f32("half", 12).unwrap(),
        vec![1., -2., 2f32.powi(-24)]
    );
    assert_eq!(index.read_f32("bf", 8).unwrap(), vec![1., -1.]);
    assert_eq!(index.read_f32("single", 4).unwrap(), vec![3.5]);
    assert!(index.read_f32("half", 11).is_err());
    assert!(index.read_raw("half", 5).is_err());
    // Indexing did not cache payload: a subsequent truncated read fails.
    fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join("f.safetensors"))
        .unwrap()
        .set_len(8)
        .unwrap();
    assert!(index.read_raw("half", 6).is_err());
}

#[test]
fn rejects_bad_offsets_lengths_duplicates_and_truncation() {
    for header in [
        json!({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}),
        json!({"x":{"dtype":"I8","shape":[4],"data_offsets":[1,5]}}),
        json!({"x":{"dtype":"I8","shape":[2],"data_offsets":[0,2]}, "y":{"dtype":"I8","shape":[2],"data_offsets":[1,3]}}),
        json!({"x":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}),
        json!({"x":{"dtype":"I8","shape":[usize::MAX,2],"data_offsets":[0,4]}}),
    ] {
        let dir = tempfile::tempdir().unwrap();
        shard(&dir.path().join("bad.safetensors"), header, &[0; 4]);
        assert!(TensorIndex::open(dir.path()).is_err());
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.safetensors");
    raw_shard(
        &path,
        r#"{"x":{"dtype":"I8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#,
        &[0],
    );
    assert!(TensorIndex::open(dir.path()).is_err());
    fs::write(&path, 1000u64.to_le_bytes()).unwrap();
    assert!(TensorIndex::open(dir.path()).is_err());
    shard(
        &path,
        json!({"x":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}),
        &[1],
    );
    fs::copy(&path, dir.path().join("duplicate.safetensors")).unwrap();
    assert!(TensorIndex::open(dir.path()).is_err());
}

#[test]
fn concurrent_shard_reads_preserve_tensor_offsets() {
    let dir = tempfile::tempdir().unwrap();
    let left = vec![17; 4096];
    let right = vec![239; 8192];
    let payload = [left.as_slice(), right.as_slice()].concat();
    shard(
        &dir.path().join("shared.safetensors"),
        json!({
            "left": {"dtype":"I8","shape":[4096],"data_offsets":[0,4096]},
            "right": {"dtype":"I8","shape":[8192],"data_offsets":[4096,12288]}
        }),
        &payload,
    );
    let index = TensorIndex::open(dir.path()).unwrap();
    let barrier = std::sync::Barrier::new(8);
    std::thread::scope(|scope| {
        for worker in 0..8 {
            let (index, barrier, left, right) = (&index, &barrier, &left, &right);
            scope.spawn(move || {
                barrier.wait();
                for iteration in 0..100 {
                    let (name, expected) = if (worker + iteration) % 2 == 0 {
                        ("left", left)
                    } else {
                        ("right", right)
                    };
                    assert!(index.read_raw(name, expected.len() - 1).is_err());
                    assert_eq!(index.read_raw(name, expected.len()).unwrap(), *expected);
                }
            });
        }
    });
}

#[test]
fn rejects_invalid_scales() {
    for value in [0., -1., f32::NAN, f32::INFINITY] {
        let dir = tempfile::tempdir().unwrap();
        fixture(dir.path(), &[value, 1., 1., 1.]);
        let index = TensorIndex::open(dir.path()).unwrap();
        assert!(Int8Expert::load(&index, 3, 7, 2, 1, 22).is_err());
    }
}
