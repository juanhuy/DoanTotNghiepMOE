const test = require('node:test');
const assert = require('node:assert/strict');
const { consumeChatStream } = require('../web/chat_stream.js');

const terminal = { choices: [{ delta: {}, finish_reason: 'length' }],
  metrics: { total_ms: 1 }, usage: { completion_tokens: 1 } };
const delta = text => ({ choices: [{ delta: { content: text }, finish_reason: null }] });
function stream(events, bytewise = false) {
  const data = new TextEncoder().encode(events.map(e => 'data: ' +
    (typeof e === 'string' ? e : JSON.stringify(e)) + '\r\n\r\n').join(''));
  let index = 0, cancelled = false, released = false;
  return {
    async read() {
      if (index === data.length) return { done: true };
      const end = bytewise ? index + 1 : data.length;
      const value = data.slice(index, end); index = end;
      return { done: false, value };
    },
    async cancel() { cancelled = true; },
    releaseLock() { released = true; },
    get cancelled() { return cancelled; }, get released() { return released; }
  };
}

test('split UTF8 and CRLF preserves content and complete terminal result', async () => {
  const reader = stream([delta('Sao Hỏa ♥'), terminal, '[DONE]'], true);
  const chunks = [];
  const result = await consumeChatStream(reader, d => chunks.push(d));
  assert.equal(result.text, 'Sao Hỏa ♥');
  assert.equal(chunks.join(''), result.text);
  assert.deepEqual(result.metrics, terminal.metrics);
  assert.equal(result.finishReason, 'length');
  assert.equal(reader.released, true);
});
test('generation_error after partial content rejects and cancels reader', async () => {
  const reader = stream([delta('partial'), { error: { message: 'generation failed' } }, '[DONE]']);
  await assert.rejects(consumeChatStream(reader, () => {}), /generation failed/);
  assert.ok(reader.cancelled && reader.released);
});
test('EOF missing terminal or DONE cannot be a successful answer', async () => {
  for (const events of [[delta('partial')], [delta('partial'), terminal], ['[DONE]']]) {
    await assert.rejects(consumeChatStream(stream(events), () => {}), /missing|truncated/);
  }
});
test('post-terminal data and duplicate DONE are rejected', async () => {
  for (const events of [[terminal, delta('bad'), '[DONE]'], [terminal, '[DONE]', '[DONE]']]) {
    await assert.rejects(consumeChatStream(stream(events), () => {}), /after/);
  }
});
test('terminal without metrics cannot finish successfully', async () => {
  await assert.rejects(consumeChatStream(stream([{ choices: terminal.choices }, '[DONE]']), () => {}), /metrics/);
});
test('abort releases reader and propagates AbortError', async () => {
  const reader = stream([]);
  reader.read = async () => { throw new DOMException('Stopped', 'AbortError'); };
  await assert.rejects(consumeChatStream(reader, () => {}), { name: 'AbortError' });
  assert.ok(reader.cancelled && reader.released);
});
test('a subsequent request can complete after a failed request', async () => {
  await assert.rejects(consumeChatStream(stream([delta('bad')]), () => {}));
  const result = await consumeChatStream(stream([delta('retry'), terminal, '[DONE]']), () => {});
  assert.equal(result.text, 'retry');
});
