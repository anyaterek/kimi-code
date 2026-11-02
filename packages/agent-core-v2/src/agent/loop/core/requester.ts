import { estimateUsedContextTokens as usedContextTokens } from '#human/agent/context-usage';
import type { HistoryMessage } from '#human/agent/turn';
import { contentsOf, mergeFields, requesterContents, type LlmRequestPart } from '#human/llm/requester';

import { toLoopRequestMessages, type LoopLlmRequester } from './request';

type Fields = Readonly<Record<string, unknown>>;

export function createLoopRequester(
  requester: () => LoopLlmRequester,
  messages: () => readonly HistoryMessage[],
): LoopLlmRequester {
  const accumulate = (part: LlmRequestPart): LoopLlmRequester => {
    const resolve = (): LoopLlmRequester => {
      const history = messages();
      const content = toLoopRequestMessages(history);
      const current = requester();
      const prepared = current.request({ content: { messages: content } });
      const usedTokens = usedContextTokens(history, prepared.profile);
      return current.with({ content: { messages: content, usedTokens } }).with(part);
    };
    return {
      [requesterContents]: part,
      with: ((next: LlmRequestPart | LoopLlmRequester) =>
        accumulate(mergeFields(part as Fields, contentsOf(next) as Fields) as LlmRequestPart)) as LoopLlmRequester['with'],
      request: (rest) => resolve().request(rest),
      generate: (rest, control) => resolve().generate(rest, control),
    };
  };
  return accumulate({});
}
