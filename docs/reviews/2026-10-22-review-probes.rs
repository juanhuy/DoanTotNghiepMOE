
#[cfg(test)]
mod review_probes {
    use super::*;
    fn model() -> OlmoeModel {
        let dir=Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/olmoe/unnormalized-mha");
        OlmoeModel::load(dir, 100_000, 64, 10_000_000).unwrap()
    }
    fn generate(m: &mut OlmoeModel, prompt: &str, id: &str) -> Generation {
        m.generate_controlled(prompt,1,Some(id),&|| Ok(()),|| Ok(()),None).unwrap()
    }
    #[test]
    fn reset_disables_session_cache() {
        let mut m=model();
        generate(&mut m,"[BOS]token3 ","a");
        let before=generate(&mut m,"[BOS]token3 token4 ","a");
        m.reset_cache();
        generate(&mut m,"[BOS]token3 ","a");
        let after=generate(&mut m,"[BOS]token3 token4 ","a");
        println!("reset: cached_prompt_tokens before={}, after={}, remaining_budget={}",before.metrics.unwrap().cached_prompt_tokens,after.metrics.unwrap().cached_prompt_tokens,m.session_cache_bytes);
        assert_eq!(m.session_cache_bytes,0);
    }
    #[test]
    fn session_budget_counts_length_instead_of_capacity() {
        let mut m=model();
        m.session_cache_bytes=1024;
        for id in ["a","b"] { generate(&mut m,"[BOS]token3 ",id); }
        let logical:usize=m.sessions.values().map(session_bytes).sum();
        let capacity:usize=m.sessions.values().flat_map(|s| &s.kv).map(|k| (k.keys.capacity()+k.values.capacity())*4).sum();
        println!("session_memory: budget={}, sessions={}, accounted_bytes={}, vec_capacity_bytes={}",m.session_cache_bytes,m.sessions.len(),logical,capacity);
        assert!(capacity>m.session_cache_bytes);
        assert!(logical<=m.session_cache_bytes);
    }
    #[test]
    fn recycled_buffers_escape_l2_budget() {
        let mut m=model();
        m.cache_bytes_per_layer=900;
        for _ in 0..2 { m.generate("[BOS]token3 token4 token5 ",8).unwrap(); }
        let live:usize=m.layers.iter().map(|l|l.bytes).sum();
        let recycled:usize=m.layers.iter().flat_map(|l|&l.recycled_buffers).map(|(w,s)|w.capacity()+s.capacity()*4).sum();
        println!("expert_memory: total_l2_budget={}, accounted_cache={}, recycled_capacity={}",m.cache_bytes_per_layer*m.layers.len(),live,recycled);
        assert!(live+recycled>m.cache_bytes_per_layer*m.layers.len());
    }
}
