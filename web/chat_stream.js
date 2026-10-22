/* Shared by the browser and protocol tests. No UI or persistence side effects. */
'use strict';

async function consumeChatStream(reader, onDelta) {
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let buffer = '', text = '', terminal = null, sawDone = false;
  function consume(block) {
    const data = block.split(/\r?\n/).filter(line => line.startsWith('data:'))
      .map(line => line.slice(5).trimStart()).join('\n');
    if (!data) return;
    if (sawDone) throw new Error('SSE data after DONE');
    if (data === '[DONE]') {
      if (!terminal) throw new Error('SSE missing terminal result');
      sawDone = true;
      return;
    }
    const event = JSON.parse(data);
    if (event.error) throw new Error(event.error.message || String(event.error));
    if (terminal) throw new Error('SSE data after terminal result');
    const choice = event.choices?.[0];
    if (!choice) throw new Error('SSE missing choices');
    const delta = choice.delta?.content;
    if (delta !== undefined && typeof delta !== 'string') throw new Error('Invalid SSE content');
    if (delta) {
      text += delta;
      onDelta(delta);
    }
    if (choice.finish_reason != null) {
      if (!event.metrics || !event.usage) throw new Error('SSE missing metrics/usage');
      terminal = event;
    }
  }
  try {
    for (;;) {
      const chunk = await reader.read();
      buffer += decoder.decode(chunk.value || new Uint8Array(), { stream: !chunk.done });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() || '';
      for (const block of blocks) consume(block);
      if (chunk.done) break;
    }
    if (buffer.trim() || !terminal || !sawDone) throw new Error('SSE truncated: missing complete terminal/DONE');
    return { text, metrics: terminal.metrics, usage: terminal.usage,
      finishReason: terminal.choices[0].finish_reason };
  } catch (error) {
    try { await reader.cancel(); } catch (_) { /* Preserve the original failure. */ }
    throw error;
  } finally {
    reader.releaseLock();
  }
}

if (typeof module !== 'undefined') module.exports = { consumeChatStream };
