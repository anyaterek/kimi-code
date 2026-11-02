import type { ServiceRegistration } from '#/_base/di/test';
import { buildContextCompactionShape, type ContextCompactionShape, type ContextCompactionShapeInput } from '#/agent/contextMemory/compactionHandoff';
import { computeUndoCut, type UndoCut } from '#/agent/contextMemory/contextOps';
import { ContextSpliced } from '#/agent/contextMemory/contextEvents';
import type { ContextMessage } from '#/agent/contextMemory/types';
import { contextFromHistoryEntry, historyFromContext } from '#/agent/loop/history';
import { IAgentLoopService } from '#/agent/loop/loop';
import { IEventBus, type ISessionEventBus } from '#/app/event/eventBus';
import { EventBusService } from '#/app/event/eventBusService';
import { IWireService } from '#/wire/wire';
import type { HistoryMessage } from '#human/agent/turn';

import { stubAgentWire } from '../../wire/stubs';
import { stubAgentContext } from '../agentContext/stubs';
import { stubLoopWithHooks, type StubLoop } from '../loop/stubs';

export function appendContext(loop: IAgentLoopService, ...messages: readonly ContextMessage[]): void {
  loop.remind(...historyFromContext(messages));
  loop.flushReminders();
}

export function compactContext(
  loop: IAgentLoopService,
  input: ContextCompactionShapeInput,
): Omit<ContextCompactionShape, 'messages'> {
  const shape = buildContextCompactionShape(loop.history(), input);
  loop.replaceHistory(historyFromContext(shape.messages), 'compaction');
  const { messages: _messages, ...result } = shape;
  void _messages;
  return result;
}

export interface StubContextMemory extends StubLoop {
  readonly messages: readonly ContextMessage[];
  undo(count: number): UndoCut;
}

function publishSplice(
  eventBus: IEventBus | undefined,
  input: {
    start: number;
    deleteCount: number;
    messages: readonly ContextMessage[];
    tokens?: number;
  },
): void {
  if (eventBus === undefined) return;
  const sessionBus = eventBus as Partial<ISessionEventBus>;
  if (typeof sessionBus.activateAgent === 'function') {
    const context = stubAgentContext('main', 1);
    sessionBus.activateAgent(context);
    sessionBus.publish?.(new ContextSpliced({ agentId: 'main', ...input }), context);
    return;
  }
  eventBus.publish(new ContextSpliced({ agentId: 'main', ...input }));
}

export function stubContextMemory(eventBus?: IEventBus): StubContextMemory {
  const messages: ContextMessage[] = [];
  let revision = 0;
  const base = stubLoopWithHooks();
  const replaceHistory = (entries: readonly HistoryMessage[]): boolean => {
    const previousLength = messages.length;
    const next = entries.map((entry) => contextFromHistoryEntry(entry));
    messages.splice(0, messages.length, ...next);
    revision += 1;
    publishSplice(eventBus, { start: 0, deleteCount: previousLength, messages: [...next] });
    return true;
  };
  return {
    ...base,
    get messages() {
      return messages;
    },
    history: () => messages,
    historyRevision: () => revision,
    replaceHistory,
    initializeHistory: async (entries) => {
      replaceHistory(entries);
    },
    remind: (...entries) => {
      const start = messages.length;
      const inserted = entries.map((entry) => contextFromHistoryEntry(entry));
      messages.push(...inserted);
      publishSplice(eventBus, { start, deleteCount: 0, messages: inserted });
    },
    undo: (count) => {
      const cut = computeUndoCut(messages, count);
      if (cut.cutIndex >= 0 && cut.removedCount >= count) {
        const deleteCount = messages.length - cut.cutIndex;
        messages.splice(cut.cutIndex, deleteCount);
        revision += 1;
        publishSplice(eventBus, { start: cut.cutIndex, deleteCount, messages: [] });
      }
      return cut;
    },
  };
}

export function registerContextMemoryServices(reg: ServiceRegistration): void {
  reg.defineInstance(IWireService, stubAgentWire());
  reg.define(IEventBus, EventBusService);
  reg.defineInstance(IAgentLoopService, stubContextMemory());
}
