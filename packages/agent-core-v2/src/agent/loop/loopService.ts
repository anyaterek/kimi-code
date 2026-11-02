import { randomUUID } from 'node:crypto';
import { EventEmitter } from 'node:events';

import { createControlledPromise } from '@antfu/utils';

import { Disposable, toDisposable, type IDisposable } from '#/_base/di/lifecycle';
import { IInstantiationService } from '#/_base/di/instantiation';
import { ILogService } from '#/_base/log/log';
import { LifecycleScope } from '#/app/scopes';
import { ScopeActivation, registerScopedService } from '#/_base/di/scope';
import { defineState } from '#/state/state';
import { abortError, isAbortError, isUserCancellation, userCancellationReason } from '#/_base/utils/abort';
import { toErrorMessage } from '#/_base/errors/errorMessage';
import { retryErrorFields } from '#/_base/utils/retry';
import { AgentLoopRequester } from './requester';
import { createAgentMediaPolicy } from './mediaPolicy';
import type { LoopRequester } from './requesterContract';
import { IAgentMediaResolverService } from '#/agent/media/mediaResolver';
import type { LLMRequestTrace } from '#/llm-adapter/contract/request-trace';
import type { ModelRequestTiming } from '#/llm-adapter/model/generate';
import { IAgentToolExecutorService } from '#/agent/toolExecutor/toolExecutor';
import { abortedToolOutput } from '#/agent/toolExecutor/toolExecutorService';
import type { ToolDidExecuteContext } from '#/agent/toolExecutor/toolHooks';
import type { ExecutableToolResult } from '#/tool/toolContract';
import { IAgentToolRegistryService } from '#/agent/toolRegistry/toolRegistry';
import { IConfigService } from '#/app/config/config';
import { AgentErrorEvent } from '#/agent/mcp/mcpEvents';
import { type FinishReason } from '#human/llm-requester/finish-reason';
import { mergeInPlace, type ContentPart, type UserMessage } from '#human/llm/message';
import { emptyUsage, type TokenUsage } from '#human/llm-requester/usage';
import { BugIndicatingError, ErrorCodes, Error2, isError2, toKimiErrorPayload } from '#/errors';
import { OrderedHookSlot } from '#/hooks';

import { isVacuousContentPart } from '#/agent/contextMemory/vacuousContent';
import { ContextSpliced } from '#/agent/contextMemory/contextEvents';
import { markInTurnOrigin } from '#/agent/contextMemory/conversationTime';
import { newMessageId } from '#/agent/contextMemory/messageId';
import { type ContextMessage, type PromptOrigin } from '#/agent/contextMemory/types';
import { gateImageFormatParts } from '#/agent/media/image-compress';
import { daemonFileRefFromPart } from '#/agent/media/mediaRef';
import { materializePromptDaemonRefs } from '#/agent/media/promptMediaIntake';
import { ISessionMediaStore } from '#/agent/media/sessionMediaStore';
import { IAgentProfileService } from '#/agent/profile/profile';
import { IAgentInterruptionReminderService } from '#/agent/interruptionReminder/interruptionReminder';
import { AgentSpaceImpl } from '#/agent/agentContext/agentSpace';
import { IAgentScopeContext } from '#/agent/scopeContext/scopeContext';
import { IAgentStateService } from '#/agent/state/agentState';
import { IFileService } from '#/app/file/fileService';
import { IPluginService } from '#/app/plugin/plugin';
import { ISessionRunGate } from '#/session/sessionActivity/sessionRunGate';
import type {
  TurnEndedEvent as TurnEndedTelemetryEvent,
  TurnInterruptedEvent,
  TurnStartedEvent as TurnStartedTelemetryEvent,
} from '#/app/telemetry/events';
import { ITelemetryService } from '#/app/telemetry/telemetry';
import { IEventDispatcher } from '#/state/eventDispatcher';
import { IWireService } from '#/wire/wire';
import {
  PromptAborted,
  PromptCompleted,
  PromptQueued,
  PromptStarted,
  PromptSteered,
  PromptSubmitted,
} from '#/agent/prompt/promptEvents';
import { LOOP_CONTROL_SECTION, type LoopControl } from './configSection';
import {
  createMaxStepsExceededError,
  IAgentLoopService,
  isMaxStepsExceededError,
  type AfterStepContext,
  type LoopCancelTarget,
  type LoopError,
  type LoopErrorContext,
  type LoopErrorHandler,
  type LoopErrorHandlerRegistrationOptions,
  type LoopNotify,
  type LoopNotifyHandle,
  type LoopRunResult,
  type LoopSnapshot,
  type LoopSubmitOptions,
  type LoopSubmitResult,
  type PromptCompletion,
  type PromptHandle,
  type PromptState,
  type PromptSubmitContext,
  type Turn,
  type TurnResult,
} from './loop';
import { mergeSteerMessages, stripBundledSkillBlocks } from '#human/agent/origin';
import { isUserPromptSubmitHookPart } from '#/agent/contextMemory/hookParts';
import {
  createSystemEntry,
  createUserEntry,
  type AssistantEntry,
  type HistoryMessage,
  type UserEntry,
} from '#human/agent/turn';
import {
  AssistantDelta,
  isDisplayablePromptOrigin,
  ThinkingDelta,
  ToolCallDelta,
  turnPromptAttachments,
  turnPromptText,
  TurnStarted,
  TurnStepCompleted,
  TurnStepInterrupted,
  TurnStepRetrying,
  TurnStepStarted,
  type TurnInterruptReason,
} from './turnEvents';
import { TurnCancel, TurnEnded, turnKey, TurnPrompt, TurnSteer } from './turnOps';
import { contextFromHistoryEntry, EMPTY_LOOP_PROMPT, historyEntryFromContext } from './history';
import {
  createLoop,
  createMaxStepsHook,
  MAX_STEPS_REASON,
  type Loop,
  type LoopEmitted,
  type MessageAppendingEvent,
  type StepContinuingEvent,
  type StepStartingEvent,
  type StepStoppingEvent,
  type TurnStartingEvent,
} from './core/index';
import { createCoreTools, type CoreTools } from './toolAdapter';
import { LoopJournal, type LoopJournalSnapshot } from './journal';

export type LoopInterruptReason = 'aborted' | 'max_steps' | 'error';

export const loopLastRequestIdKey = defineState<string | undefined>(
  'loop.lastRequestId',
  () => undefined as string | undefined,
);
export const loopDisposingKey = defineState<boolean>('loop.disposing', () => false);

const MAX_STEP_SIGNAL_LISTENERS = 64;

export class AgentLoopService extends Disposable implements IAgentLoopService {
  declare readonly _serviceBrand: undefined;

  readonly requester: LoopRequester;
  private readonly coreRequesterHost: AgentLoopRequester;

  readonly hooks: IAgentLoopService['hooks'] = {
    onWillBeginStep: new OrderedHookSlot(),
    onDidFinishStep: new OrderedHookSlot(),
    onBeforeSubmitPrompt: new OrderedHookSlot(),
  };

  private readonly errorHandlers: LoopErrorHandler[] = [];
  private readonly promptWaiters = new Map<string, PromptWaiter>();
  private readonly steered = new Map<string, SteeredPrompt>();
  private readonly steerCandidates = new Map<string, UserEntry>();
  private readonly terminalStates = new Map<string, PromptState>();
  private readonly pendingSubmissions: UserEntry[] = [];
  private readonly pendingReminders: HistoryMessage[] = [];
  private readonly nudges: Nudge[] = [];
  private nudgeCursor = 0;
  private active: ActiveTurn | undefined;
  private readonly pendingTurns = new Map<number, PendingTurn>();
  private readonly settleWaiters: Array<() => void> = [];
  private quiescenceDepth = 0;
  private historySnapshotDepth = 0;
  private historyVersion = 0;
  private historyView: { readonly source: readonly HistoryMessage[]; readonly messages: readonly ContextMessage[] } | undefined;
  private activeRequestTrace: LLMRequestTrace | undefined;
  private loop: Loop | undefined;
  private initialized = false;
  private closePromise: Promise<void> | undefined;
  private readonly tools: CoreTools;
  private readonly journal: LoopJournal;
  private turnWatch:
    | { timer: NodeJS.Timeout; signature: string; since: number; lastWarnAt: number }
    | undefined;

  constructor(
    @IAgentToolExecutorService private readonly toolExecutor: IAgentToolExecutorService,
    @IAgentToolRegistryService private readonly toolRegistry: IAgentToolRegistryService,
    @IConfigService private readonly config: IConfigService,
    @IEventDispatcher private readonly dispatcher: IEventDispatcher,
    @IAgentScopeContext private readonly scopeContext: IAgentScopeContext,
    @ITelemetryService private readonly telemetry: ITelemetryService,
    @IAgentStateService private readonly states: IAgentStateService,
    @IWireService private readonly wire: IWireService,
    @IInstantiationService private readonly instantiation: IInstantiationService,
    @IAgentProfileService private readonly profile: IAgentProfileService,
    @IPluginService private readonly plugins: IPluginService,
    @ISessionRunGate private readonly runGate: ISessionRunGate,
    @ILogService private readonly logger: ILogService,
    @IAgentMediaResolverService mediaResolver: IAgentMediaResolverService,
  ) {
    super();
    const space = this.scopeContext.agentContext.space;
    if (space instanceof AgentSpaceImpl) this._register(space._setHistoryProvider(() => this.history()));
    const requester = instantiation.createInstance(AgentLoopRequester);
    requester.registerMediaPolicy((model, request) =>
      createAgentMediaPolicy(mediaResolver, model, request),
    );
    requester.bindHistory(() => this.history(), () => this.historyRevision());
    this.requester = requester;
    this.coreRequesterHost = requester;
    this.journal = new LoopJournal(wire, scopeContext.agentId);
    this.tools = createCoreTools({
      toolExecutor,
      toolInfos: () => this.toolRegistry.list(),
      turnId: () => this.active?.id ?? 0,
      steerSignal: () => this.active?.steerController.signal,
      trace: () => this.activeRequestTrace,
      remind: (message) => this.remind(historyEntryFromContext(message)),
      onBatchError: (error) => {
        const turn = this.active;
        const step = turn?.current;
        if (turn === undefined || step === undefined || step.signal.aborted) return;
        turn.failedStep ??= { number: step.number, uuid: step.uuid, error };
      },
    });
    this.tools.sync();
    this.states.contributeState(turnKey);
    this.states.contributeState(loopLastRequestIdKey);
    this.states.contributeState(loopDisposingKey);
    this.toolExecutor.hooks.onDidExecuteTool.register('prompt-service-delivery', async (ctx, next) => {
      await this.deliverToolResult(ctx);
      await next();
    });
    const restore = this.dispatcher.hooks.onDidRestore.register('loop.core', async (_ctx, next) => {
      await this.initializeLoop();
      await next();
    });
    this._register(restore);
    this._register(toDisposable(() => this.disarmTurnWatch()));
  }

  private get lastRequestId(): string | undefined {
    return this.states.get(loopLastRequestIdKey);
  }

  private set lastRequestId(value: string | undefined) {
    this.states.set(loopLastRequestIdKey, value);
  }

  private get disposing(): boolean {
    return this.states.get(loopDisposingKey);
  }

  private set disposing(value: boolean) {
    this.states.set(loopDisposingKey, value);
  }

  history(): readonly ContextMessage[] {
    this.ensureLoop();
    const source = this.loop?.state.messages ?? [];
    const previous = this.historyView;
    if (previous?.source === source) return previous.messages;
    const messages = source.map((entry, index) =>
      previous?.source[index] === entry ? previous.messages[index]! : contextFromHistoryEntry(entry),
    );
    this.historyView = { source, messages };
    return messages;
  }

  historyRevision(): number {
    return this.historyVersion;
  }

  replaceHistory(messages: readonly HistoryMessage[], reason?: string): boolean {
    const previous = this.history();
    const ack = this.loop?.setContext([...messages], reason);
    if (ack?.type !== 'context.settled') return false;
    this.historyVersion += 1;
    void this.dispatcher.dispatch(new ContextSpliced({
      agentId: this.scopeContext.agentId,
      start: 0,
      deleteCount: previous.length,
      messages: this.history(),
    }));
    return true;
  }

  private ensureLoop(): void {
    if (this.loop !== undefined) return;
    this.createLoopFromSnapshot(this.journal.snapshot(), false);
    this.initialized = false;
  }

  private async initializeLoop(
    reset = false,
    seededMessages?: readonly HistoryMessage[],
    minimumNextTurnId?: number,
  ): Promise<void> {
    if (!reset && this.initialized) return;
    if (this.dispatcher.restorePhase === 'new') {
      this.ensureLoop();
      this.initialized = true;
      return;
    }
    this.loop = undefined;
    await this.wire.drainPersisted().catch(() => undefined);
    this.createLoopFromSnapshot(this.journal.snapshot(), reset, seededMessages, minimumNextTurnId);
  }

  private createLoopFromSnapshot(
    snapshot: LoopJournalSnapshot,
    reset: boolean,
    seededMessages?: readonly HistoryMessage[],
    minimumNextTurnId?: number,
  ): void {
    this.initialized = true;
    if (reset) this.historyVersion += 1;
    this.tools.sync();
    const loop = createLoop({
      requester: this.coreRequesterHost.coreRequester({
        source: () => ({
          type: 'turn',
          turnId: this.active?.id ?? snapshot.input.nextTurnId ?? 0,
          step: this.active?.gatedSteps ?? 0,
        }),
        onTrace: (trace) => {
          this.activeRequestTrace = trace;
        },
        onOutcome: (outcome) => {
          const step = this.active?.current;
          if (step !== undefined) step.timing = outcome.timing;
        },
      }),
      executor: this.tools.executor,
      input: {
        ...snapshot.input,
        profile: { systemPrompt: this.profile.getSystemPrompt(), tools: [] },
        messages: seededMessages === undefined ? snapshot.input.messages : [...seededMessages],
        nextTurnId: Math.max(snapshot.input.nextTurnId ?? 0, minimumNextTurnId ?? 0),
      },
      hooks: [
        createMaxStepsHook(
          () => this.config.get<LoopControl>(LOOP_CONTROL_SECTION)?.maxStepsPerTurn,
          (event) => this.active?.bypassStep === event.step,
        ),
        { point: 'tool.batch-starting', hook: (event) => this.tools.beginBatch(event.toolCalls) },
        { point: 'turn.starting', hook: (event, signal) => this.turnStarting(event, signal) },
        { point: 'message.appending', hook: (event, signal) => this.messageAppending(event, signal) },
        { point: 'step.starting', hook: (event, signal) => this.stepStarting(event, signal) },
        { point: 'step.stopping', hook: (event, signal) => this.stepStopping(event, signal) },
        { point: 'step.continuing', hook: (event, signal) => this.stepContinuing(event, signal) },
      ],
    });
    this.loop = loop;
    loop.subscribe((event) => {
      const journalEvent = this.journalEvent(event);
      this.projectLoopEvent(event);
      this.journal.append(journalEvent);
    });
    this.syncProfile();
    this.drainPendingToLoop();
    this.maybeSettle();
  }

  private syncProfile(): void {
    const loop = this.loop;
    if (loop === undefined) return;
    const systemPrompt = this.profile.getSystemPrompt();
    if (loop.state.profile.systemPrompt !== systemPrompt) loop.setSystemPrompt(systemPrompt);
    this.tools.sync();
  }

  async initializeHistory(messages: readonly HistoryMessage[]): Promise<void> {
    if (this.history().length > 0 || this.hasPendingRequests()) {
      throw new BugIndicatingError('Loop history initialization requires an empty quiescent loop');
    }
    await this.resetLoop(messages);
    if (!this.replaceHistory(messages, 'fork')) {
      throw new BugIndicatingError('Loop history initialization was rejected');
    }
  }

  async resetLoop(seededMessages?: readonly HistoryMessage[]): Promise<void> {
    if (this.disposing) return;
    if (this.active !== undefined || this.pendingTurns.size > 0) {
      throw new BugIndicatingError('Loop reset requires a quiescent loop');
    }
    const previous = this.loop;
    this.loop = undefined;
    if (previous !== undefined) await previous.close();
    this.pendingSubmissions.length = 0;
    this.nudgeCursor = 0;
    this.nudges.length = 0;
    await this.initializeLoop(true, seededMessages, previous?.state.nextTurnId);
  }

  close(): Promise<void> {
    if (this.closePromise !== undefined) return this.closePromise;
    this.disposing = true;
    const reason = abortError('Agent loop disposed');
    for (const waiter of this.promptWaiters.values()) {
      this.settleWaiterCancelled(waiter);
      this.terminalStates.set(waiter.id, 'cancelled');
    }
    this.promptWaiters.clear();
    this.steered.clear();
    this.pendingSubmissions.length = 0;
    for (const nudge of this.nudges.slice(this.nudgeCursor)) {
      if (nudge.dropped) continue;
      nudge.dropped = true;
      nudge.onDrop?.();
    }
    this.nudgeCursor = this.nudges.length;
    const active = this.active;
    active?.controller.abort(reason);
    active?.steerController.abort(reason);
    this.closePromise = (async () => {
      this.loop?.flushReminders();
      await this.loop?.close();
      this.pendingTurns.clear();
      this.maybeSettle();
      await this.settled();
      await this.journal.settled();
      await this.wire.drainPersisted();
    })();
    return this.closePromise;
  }

  override dispose(): void {
    void this.close();
    super.dispose();
  }

  submit(input: UserEntry, options?: LoopSubmitOptions): LoopSubmitResult {
    if (this.disposing) throw abortError('Agent loop disposed');
    this.runGate.begin();
    const meta = input.meta;
    const id = meta?.promptId ?? newMessageId();
    const origin = (meta?.origin as PromptOrigin | undefined) ?? { kind: 'user' };
    const tracked = meta?.tracked === true;
    const createdAt = meta?.createdAt ?? (tracked ? new Date().toISOString() : '');
    const userMessageId = meta?.userMessageId ?? (tracked ? id : '');
    const waiter = this.createWaiter(id, meta?.promptId, options?.onMaterialize);
    this.terminalStates.delete(id);
    this.promptWaiters.set(id, waiter);
    const message: ContextMessage = {
      role: 'user',
      content: [...input.message.content],
      id,
      origin: meta?.origin as PromptOrigin | undefined,
    };
    if (tracked) {
      const snapshot = this.snapshot();
      const queued = snapshot.state === 'running' || snapshot.paused || snapshot.queue.length > 0;
      this.publishPromptSubmitted(
        { promptId: id, origin, userMessageId, createdAt, message },
        queued ? 'queued' : 'running',
      );
      if (queued) this.publishPromptQueued({ promptId: id, origin, message });
    }
    const entry: UserEntry = {
      message: { role: 'user', content: [...input.message.content] },
      meta: { promptId: id, origin, tracked, createdAt, userMessageId },
    };
    if (this.loop !== undefined && this.quiescenceDepth === 0) {
      this.syncProfile();
      this.loop.submit(entry);
    } else this.pendingSubmissions.push(entry);
    if (options?.steerIfActive === true && this.active?.prompt.tracked === true && this.loop !== undefined) {
      this.loop.steer([id]);
    }
    this.maybeSettle();
    return { id };
  }

  async steer(promptIds: readonly string[]): Promise<void> {
    if (this.disposing) throw abortError('Agent loop disposed');
    if (promptIds.length === 0) throw new Error2(ErrorCodes.REQUEST_INVALID, 'prompt_ids must not be empty');
    const active = this.active;
    if (active === undefined || !active.prompt.tracked) {
      throw new Error2(ErrorCodes.PROMPT_NOT_FOUND, 'no active prompt to steer into');
    }
    const loop = this.loop;
    if (loop === undefined) throw new Error2(ErrorCodes.PROMPT_NOT_FOUND, 'one or more prompts are not pending');
    const queued = new Map(loop.state.inbox.normal.map((entry) => [entry.meta?.promptId, entry] as const));
    const ids = new Set(promptIds);
    if (ids.size !== promptIds.length || ![...ids].every((id) => queued.has(id))) {
      throw new Error2(ErrorCodes.PROMPT_NOT_FOUND, 'one or more prompts are not pending');
    }
    for (const id of ids) {
      const entry = queued.get(id);
      if (entry !== undefined) {
        this.steerCandidates.set(id, entry);
        await this.materializeDaemonRefs(entry.message);
      }
    }
    if (this.active !== active) throw new Error2(ErrorCodes.PROMPT_NOT_FOUND, 'one or more prompts are no longer pending');
    loop.steer(promptIds);
  }

  promptHandle(id: string): PromptHandle | undefined {
    const waiter = this.promptWaiters.get(id);
    if (waiter === undefined) return undefined;
    const projection = this.promptProjection(id);
    const state = (): PromptState => this.promptStateOf(id);
    const handle: PromptHandle = {
      id,
      userMessageId: projection?.userMessageId ?? '',
      createdAt: projection?.createdAt ?? '',
      get state() {
        return state();
      },
      message: projection?.message ?? EMPTY_HANDLE_MESSAGE,
      launched: waiter.launched,
      completion: waiter.completion,
    };
    return handle;
  }

  private promptStateOf(id: string): PromptState {
    if (this.active?.prompt.id === id) return 'running';
    if (this.steered.has(id)) return 'steered';
    return this.terminalStates.get(id) ?? 'pending';
  }

  private promptProjection(id: string): PromptProjection | undefined {
    const failedEntry = this.promptWaiters.get(id)?.failedEntry;
    if (failedEntry !== undefined) return projectionFromEntry(failedEntry);
    const active = this.active;
    if (active !== undefined && active.prompt.id === id) return active.prompt;
    const steered = this.steered.get(id);
    if (steered !== undefined) return steered;
    const queued = this.loop?.state.inbox.normal.find((item) => item.meta?.promptId === id);
    if (queued !== undefined) return projectionFromEntry(queued);
    const parked = this.pendingSubmissions.find((item) => item.meta?.promptId === id);
    if (parked !== undefined) return projectionFromEntry(parked);
    return undefined;
  }

  notify(note: LoopNotify = {}): LoopNotifyHandle {
    return this.enqueueNudge(note, 'notify');
  }

  remind(...entries: readonly HistoryMessage[]): void {
    if (this.disposing) throw abortError('Agent loop disposed');
    this.ensureLoop();
    if (this.loop !== undefined && this.quiescenceDepth === 0) this.loop.remind(...entries);
    else this.pendingReminders.push(...entries);
  }

  flushReminders(): void {
    if (this.pendingReminders.length > 0) this.loop?.remind(...this.pendingReminders.splice(0));
    this.loop?.flushReminders();
  }

  private enqueueNudge(note: LoopNotify, kind: Nudge['kind']): LoopNotifyHandle {
    if (this.disposing) throw abortError('Agent loop disposed');
    const nudge: Nudge = {
      kind,
      contextMessage: note.message,
      bypassMaxSteps: note.bypassMaxSteps ?? false,
      turnScoped: note.turnScoped ?? true,
      onConsume: note.onConsume,
      onDrop: note.onDrop,
    };
    this.nudges.push(nudge);
    this.sendNudge(nudge);
    return {
      get dropped() {
        return nudge.dropped === true;
      },
      drop: () => {
        if (nudge.dropped === true || nudge.consumed === true) return;
        nudge.dropped = true;
        nudge.onDrop?.();
        this.maybeSettle();
      },
    };
  }

  private sendNudge(nudge: Nudge): void {
    if (this.loop === undefined || this.quiescenceDepth > 0 || nudge.sentToMachine === true) return;
    nudge.sentToMachine = true;
    if (nudge.kind === 'remind') this.loop.remind(machineReminderEntry(nudge.contextMessage));
    else this.loop.notify(createUserEntry(machineUserMessage(nudge.contextMessage), {
      promptId: nudge.contextMessage?.id,
      origin: nudge.contextMessage?.origin,
    }));
  }

  private createWaiter(
    id: string,
    dispatchPromptId?: string,
    onMaterialize?: () => void,
  ): PromptWaiter {
    return {
      id,
      dispatchPromptId,
      launched: createControlledPromise<Turn | undefined>(),
      completion: createControlledPromise<PromptCompletion>(),
      onMaterialize,
    };
  }

  snapshot(): LoopSnapshot {
    const loop = this.loop;
    const machineQueue = loop?.state.inbox.normal ?? [];
    const parked = this.pendingSubmissions.filter(
      (entry) => !machineQueue.some((item) => item.meta?.promptId === entry.meta?.promptId),
    );
    const queue = [...machineQueue, ...parked];
    const active = this.active;
    const current = active?.current;
    const retry = active?.retry;
    return {
      state: active === undefined ? 'idle' : 'running',
      activeTurnId: active?.id,
      activePromptId: active !== undefined && active.prompt.tracked ? active.prompt.id : undefined,
      queue,
      notificationCount:
        (loop === undefined ? 0 : loop.state.inbox.notify.length + loop.state.inbox.remind.length) +
        this.nudges.slice(this.nudgeCursor).filter((nudge) => !nudge.dropped && !nudge.sentToMachine).length,
      paused: this.quiescenceDepth > 0,
      hasPendingRequests: this.hasPendingRequests(),
      turn:
        active === undefined
          ? undefined
          : {
              turnId: active.id,
              phase: retry !== undefined ? 'retrying' : (current?.pendingToolIds.size ?? 0) > 0 ? 'tool_call' : 'running',
              step: active.gatedSteps,
              ending: active.controller.signal.aborted,
              endingReason: active.controller.signal.aborted ? 'aborted' : undefined,
              retry,
              activeToolCalls: (loop?.state.calls ?? []).map((call) => ({ toolCallId: call.toolCall.id, name: call.toolCall.name })),
              since: active.startedAt,
            },
      activeRequestId: this.activeRequestTrace?.requestId,
    };
  }

  private settlePromptLaunched(waiter: PromptWaiter, active: ActiveTurn): void {
    waiter.launched.resolve(active.turn);
    void active.turn.result.then((result) => this.settlePromptCompletion(waiter, active.prompt, result));
    if (!active.prompt.tracked) return;
    this.publishPromptStarted(active.prompt.id, active.prompt.origin);
  }

  private settlePromptCompletion(
    waiter: PromptWaiter,
    prompt: ActivePrompt,
    result: TurnResult,
  ): void {
    const state = result.type === 'cancelled' ? 'cancelled' : result.type === 'failed' ? 'failed' : 'completed';
    waiter.completion.resolve({ promptId: waiter.id, result, state });
    for (const [childId, steeredEntry] of this.steered) {
      if (steeredEntry.parentId !== waiter.id) continue;
      const child = this.promptWaiters.get(childId);
      if (child !== undefined) {
        child.completion.resolve({ promptId: childId, result, state });
      }
      this.terminalStates.set(childId, state);
      this.steered.delete(childId);
    }
    if (prompt.tracked) {
      if (state === 'cancelled') this.publishPromptAborted(waiter.id);
      else this.publishPromptCompleted(waiter.id, state);
    }
    this.terminalStates.set(waiter.id, state);
  }

  private async materializeDaemonRefs(message: { readonly content: readonly ContentPart[] }): Promise<void> {
    if (!message.content.some((part) => daemonFileRefFromPart(part) !== undefined)) return;
    const files = this.instantiation.invokeFunction((accessor) => accessor.get(IFileService));
    const mediaStore = this.instantiation.invokeFunction((accessor) => accessor.get(ISessionMediaStore));
    await materializePromptDaemonRefs(message.content, { files, mediaStore });
  }

  private async turnStarting(event: TurnStartingEvent, signal: AbortSignal) {
    const normal = event.pending.find((item) => item.kind === 'normal');
    if (normal === undefined) return undefined;
    const entry = normal.entry;
    const waiter = entry.meta?.promptId === undefined ? undefined : this.promptWaiters.get(entry.meta.promptId);
    const pending: PendingTurn = { entry, waiter, message: entry.message };
    this.pendingTurns.set(event.turnId, pending);
    if (entry.meta?.tracked !== true || waiter === undefined) return undefined;
    const promptMessage: ContextMessage = {
      role: 'user',
      content: [...entry.message.content],
      id: waiter.id,
      origin: entry.meta?.origin as PromptOrigin | undefined,
    };
    const ctx: PromptSubmitContext = {
      promptMessage,
      isSteer: false,
      hookParts: [],
      block: false,
    };
    try {
      await this.hooks.onBeforeSubmitPrompt.run(ctx);
      if (ctx.block) return { type: 'blocked' as const, cause: { by: 'prompt', reason: 'blocked' } };
      await this.materializeDaemonRefs(promptMessage);
      pending.message = {
        role: 'user',
        content: [
          ...ctx.hookParts,
          ...gateImageFormatParts(promptMessage.content, this.profile.getModelProviderType()),
        ],
      };
      return undefined;
    } catch (error) {
      if (signal.aborted) return { type: 'blocked' as const, cause: { by: 'prompt', reason: 'aborted', message: toErrorMessage(error) } };
      return { type: 'blocked' as const, cause: { by: 'prompt', reason: 'gate_failed', message: toErrorMessage(error) } };
    }
  }

  private messageAppending(event: MessageAppendingEvent, _signal: AbortSignal) {
    if (event.item.kind === 'normal') {
      const pending = this.pendingTurns.get(event.turnId);
      if (pending !== undefined) return { entry: { ...event.item.entry, message: pending.message } };
      return undefined;
    }
    if (event.item.kind === 'notify' && this.steered.size > 0) {
      return {
        entry: {
          ...event.item.entry,
          message: {
            ...event.item.entry.message,
            content: gateImageFormatParts(event.item.entry.message.content, this.profile.getModelProviderType()),
          },
        },
      };
    }
    return undefined;
  }

  private async stepStarting(event: StepStartingEvent, _signal: AbortSignal) {
    const turn = this.active;
    if (turn === undefined) return undefined;
    turn.gatedSteps = event.step;
    turn.steps = Math.max(turn.steps, event.step);
    turn.retry = undefined;
    this.activeRequestTrace = undefined;
    this.telemetry.setContext({ request_id: undefined });
    const step = turn.current ?? createStepState(event.step, turn.controller.signal);
    turn.current = step;
    turn.interruptStep = event.step;
    EventEmitter.setMaxListeners(MAX_STEP_SIGNAL_LISTENERS, turn.controller.signal);
    try {
      await this.hooks.onWillBeginStep.run({
        turnId: turn.id,
        step: event.step,
        firstStepOfTurn: event.step === 1,
        signal: step.signal,
      });
    } catch (error) {
      if (isAbortError(error) || step.signal.aborted) {
        turn.abortReason = turn.controller.signal.aborted ? turn.controller.signal.reason : error;
      } else {
        turn.failedStep = { number: step.number, uuid: step.uuid, error };
      }
      return { blocked: { by: 'step', reason: 'step_hook_failed', message: toErrorMessage(error) } };
    }
    if (step.signal.aborted) {
      turn.abortReason = step.signal.reason ?? abortError('Step aborted');
      return { blocked: { by: 'step', reason: 'aborted' } };
    }
    return undefined;
  }

  private async stepStopping(event: StepStoppingEvent, _signal: AbortSignal) {
    const turn = this.active;
    if (turn === undefined) return undefined;
    const step = turn.current;
    if (step === undefined) return undefined;
    if (event.draft.type === 'done') {
      const finishReason = step.providerFinishReason ?? 'completed';
      const context: AfterStepContext = {
        turnId: turn.id,
        step: step.number,
        firstStepOfTurn: step.number === 1,
        signal: step.signal,
        usage: step.usage ?? emptyUsage(),
        finishReason,
        stopTurn: false,
      };
      try {
        await this.hooks.onDidFinishStep.run(context);
      } catch (error) {
        if (isAbortError(error) || step.signal.aborted) {
          turn.abortReason = turn.controller.signal.aborted ? turn.controller.signal.reason : error;
        } else {
          turn.failedStep = { number: step.number, uuid: step.uuid, error };
        }
      }
      turn.interruptStep = undefined;
      if (context.stopTurn) turn.stopRequested = true;
      if (finishReason === 'filtered') turn.filtered = true;
      if (
        !turn.stopRequested &&
        !turn.toolStopRequested &&
        this.loop !== undefined &&
        (this.loop.state.inbox.notify.length > 0 || this.loop.state.inbox.remind.length > 0)
      ) {
        return { type: 'continue' as const, cause: { by: 'loop', reason: 'drain' } };
      }
      return undefined;
    }
    if (event.draft.type !== 'failed') return undefined;
    const failure = turn.failedStep ?? { number: step.number, uuid: step.uuid, error: event.draft.error };
    turn.failedStep = undefined;
    const context: LoopErrorContext = {
      turnId: turn.id,
      step: failure.number,
      stepId: failure.uuid,
      signal: turn.controller.signal,
      error: failure.error,
      retry: () => {
        turn.retryRequested = true;
      },
    };
    const handler = this.errorHandlers.find((entry) => entry.match(context));
    if (handler !== undefined) {
      try {
        if (await handler.handle(context)) {
          turn.interruptStep = undefined;
          if (turn.retryRequested) {
            turn.retryRequested = false;
            return { type: 'continue' as const, cause: { by: handler.id, reason: 'retry' } };
          }
          return undefined;
        }
      } catch (handlerError) {
        if (isAbortError(handlerError) || turn.controller.signal.aborted) {
          turn.abortReason = turn.controller.signal.aborted ? turn.controller.signal.reason : handlerError;
        } else {
          turn.failedStep = { number: failure.number, uuid: failure.uuid, error: handlerError };
        }
        return undefined;
      }
    }
    turn.failedStep = failure;
    return undefined;
  }

  private stepContinuing(_event: StepContinuingEvent, _signal: AbortSignal) {
    return undefined;
  }

  private settleGateRejectedPrompt(turnId: number, state: 'blocked' | 'failed', error?: unknown): void {
    const pending = this.pendingTurns.get(turnId);
    this.pendingTurns.delete(turnId);
    const waiter = pending?.waiter;
    if (waiter === undefined) return;
    waiter.launched.resolve(undefined);
    waiter.completion.resolve({ promptId: waiter.id, result: undefined, state });
    if (state === 'failed') waiter.failedEntry = pending?.entry;
    this.publishPromptCompleted(waiter.id, state);
    this.terminalStates.set(waiter.id, state);
    this.promptWaiters.delete(waiter.id);
    this.maybeSettle();
    if (error !== undefined) this.logger.warn('prompt gate failed', { error });
  }

  private async deliverToolResult(ctx: ToolDidExecuteContext): Promise<void> {
    const delivery = ctx.result.delivery;
    if (delivery === undefined) return;
    const { delivery: _delivery, ...rest } = ctx.result;
    ctx.result = rest as ExecutableToolResult;
    if (delivery.kind === 'steer') {
      const message = delivery.message as ContextMessage;
      this.submit(
        { message: machineUserMessage(message), meta: { origin: message.origin } },
        { steerIfActive: true },
      );
    }
  }

  private publishPromptCompleted(promptId: string, reason: 'completed' | 'failed' | 'blocked'): void {
    void this.dispatcher.dispatch(
      new PromptCompleted({ agentId: this.scopeContext.agentId, promptId, finishedAt: new Date().toISOString(), reason }),
    );
  }

  private publishPromptQueued(input: {
    readonly promptId: string;
    readonly origin: PromptOrigin;
    readonly message: ContextMessage;
  }): void {
    if (input.origin.kind !== 'user') return;
    void this.dispatcher.dispatch(
      new PromptQueued({
        agentId: this.scopeContext.agentId,
        promptId: input.promptId,
        content: stripBundledSkillBlocks(input.message),
        clientMetadata: input.origin.clientMetadata,
        queueLength: this.snapshot().queue.length,
      }),
    );
  }

  private publishPromptSubmitted(
    input: {
      readonly promptId: string;
      readonly origin: PromptOrigin;
      readonly userMessageId: string;
      readonly createdAt: string;
      readonly message: ContextMessage;
    },
    status: 'running' | 'queued',
  ): void {
    if (input.origin.kind !== 'user') return;
    void this.dispatcher.dispatch(
      new PromptSubmitted({
        agentId: this.scopeContext.agentId,
        promptId: input.promptId,
        userMessageId: input.userMessageId,
        status,
        content: stripBundledSkillBlocks(input.message),
        clientMetadata: input.origin.clientMetadata,
        createdAt: input.createdAt,
      }),
    );
  }

  private publishPromptStarted(promptId: string, origin: PromptOrigin): void {
    if (origin.kind !== 'user') return;
    void this.dispatcher.dispatch(new PromptStarted({ agentId: this.scopeContext.agentId, promptId }));
  }

  private publishPromptAborted(promptId: string): void {
    void this.dispatcher.dispatch(
      new PromptAborted({ agentId: this.scopeContext.agentId, promptId, abortedAt: new Date().toISOString() }),
    );
  }

  cancel(target?: LoopCancelTarget, reason?: unknown): boolean {
    const cancellation = reason ?? userCancellationReason();
    if (target?.promptId !== undefined) {
      const active = this.active;
      if (active !== undefined && active.prompt.tracked && active.prompt.id === target.promptId) {
        return this.cancelActiveTurn(undefined, cancellation);
      }
      const waiter = this.promptWaiters.get(target.promptId);
      if (waiter === undefined) throw new Error2(ErrorCodes.PROMPT_NOT_FOUND, `prompt ${target.promptId} not found`);
      return this.cancelWaiter(waiter, cancellation);
    }
    return this.cancelActiveTurn(target?.turnId, cancellation);
  }

  private cancelWaiter(waiter: PromptWaiter, cancellation: unknown): boolean {
    const active = this.active;
    if (active !== undefined && active.prompt.id === waiter.id) return this.cancelActiveTurn(undefined, cancellation);
    const tracked = this.promptProjection(waiter.id)?.tracked === true;
    this.loop?.cancel(waiter.id);
    const parked = this.pendingSubmissions.findIndex((entry) => entry.meta?.promptId === waiter.id);
    if (parked >= 0) this.pendingSubmissions.splice(parked, 1);
    this.settleWaiterCancelled(waiter);
    if (tracked) this.publishPromptAborted(waiter.id);
    this.terminalStates.set(waiter.id, 'cancelled');
    this.promptWaiters.delete(waiter.id);
    this.steered.delete(waiter.id);
    return true;
  }

  private settleWaiterCancelled(waiter: PromptWaiter): void {
    waiter.launched.resolve(undefined);
    waiter.completion.resolve({ promptId: waiter.id, result: undefined, state: 'cancelled' });
    this.maybeSettle();
  }

  tryAcquireQuiescence(): IDisposable | undefined {
    if (this.disposing) throw abortError('Agent loop disposed');
    if (this.quiescenceDepth > 0 || this.historySnapshotDepth > 0 || this.active !== undefined || this.hasPendingRequests()) {
      return undefined;
    }
    this.quiescenceDepth += 1;
    return toDisposable(() => this.releaseQuiescence());
  }

  tryAcquireHistorySnapshot(): IDisposable | undefined {
    if (this.disposing) throw abortError('Agent loop disposed');
    if (this.quiescenceDepth > 0 || this.historySnapshotDepth > 0) return undefined;
    this.historySnapshotDepth += 1;
    return toDisposable(() => {
      this.historySnapshotDepth -= 1;
    });
  }

  private releaseQuiescence(): void {
    if (this.quiescenceDepth === 0) return;
    this.quiescenceDepth -= 1;
    if (this.quiescenceDepth > 0 || this.disposing) return;
    this.drainPendingToLoop();
    this.maybeSettle();
  }

  private drainPendingToLoop(): void {
    const loop = this.loop;
    if (loop === undefined) return;
    const queued = new Set(loop.state.inbox.normal.map((item) => item.meta?.promptId));
    for (const entry of this.pendingSubmissions.splice(0)) {
      const id = entry.meta?.promptId;
      if (id === undefined || queued.has(id) || !this.promptWaiters.has(id)) continue;
      loop.submit(entry);
    }
    if (this.quiescenceDepth > 0) return;
    if (this.pendingReminders.length > 0) loop.remind(...this.pendingReminders.splice(0));
    for (const nudge of this.nudges.slice(this.nudgeCursor)) this.sendNudge(nudge);
    if (loop.state.status === 'interrupted' || (loop.state.held && this.hasPendingRequests())) loop.continue();
  }

  private cancelActiveTurn(turnId: number | undefined, cancellation: unknown): boolean {
    const active = this.active;
    if (active === undefined || (turnId !== undefined && active.id !== turnId) || this.loop === undefined) return false;
    if (active.controller.signal.aborted) {
      this.loop.abort(active.controller.signal.reason);
      return true;
    }
    void this.dispatcher.dispatch(
      new TurnCancel({
        agentId: this.scopeContext.agentId,
        turnId: active.id,
        target: 'active',
        reason: cancelReasonFor(cancellation),
      }),
    );
    active.controller.abort(cancellation);
    this.loop.abort(cancellation);
    return true;
  }

  private hasPendingRequests(): boolean {
    if (this.disposing) return false;
    return (
      this.pendingSubmissions.length > 0 ||
      (this.loop?.state.inbox.normal.length ?? 0) > 0 ||
      this.nudges.slice(this.nudgeCursor).some((nudge) => !nudge.dropped) ||
      this.loop?.state.status === 'interrupted'
    );
  }

  settled(): Promise<void> {
    if (this.active === undefined && !this.hasPendingRequests() && this.pendingTurns.size === 0) {
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      this.settleWaiters.push(resolve);
    });
  }

  private maybeSettle(): void {
    if (this.active !== undefined || this.pendingTurns.size > 0 || this.hasPendingRequests()) return;
    if (this.settleWaiters.length === 0) return;
    const waiters = this.settleWaiters.splice(0);
    for (const resolve of waiters) resolve();
  }

  registerLoopErrorHandler(
    handler: LoopErrorHandler,
    options: LoopErrorHandlerRegistrationOptions = {},
  ): IDisposable {
    if (options.before !== undefined && options.after !== undefined) {
      throw new BugIndicatingError('Loop error handler registration cannot specify both before and after');
    }
    this.deleteErrorHandler(handler.id);
    const target = options.before ?? options.after;
    if (target === undefined) {
      this.errorHandlers.push(handler);
    } else {
      const targetIndex = this.errorHandlers.findIndex((entry) => entry.id === target);
      if (targetIndex < 0) throw new BugIndicatingError(`Loop error handler target "${target}" is not registered`);
      const insertAt = options.before !== undefined ? targetIndex : targetIndex + 1;
      this.errorHandlers.splice(insertAt, 0, handler);
    }
    return toDisposable(() => {
      this.deleteErrorHandler(handler.id);
    });
  }

  private deleteErrorHandler(id: string): boolean {
    const index = this.errorHandlers.findIndex((entry) => entry.id === id);
    if (index < 0) return false;
    this.errorHandlers.splice(index, 1);
    return true;
  }

  private beginActiveTurn(waiter: PromptWaiter, entry: UserEntry, id: number): ActiveTurn {
    const origin = (entry.meta?.origin as PromptOrigin | undefined) ?? { kind: 'user' };
    const tracked = entry.meta?.tracked === true;
    const prompt: ActivePrompt = {
      id: waiter.id,
      promptId: tracked ? waiter.id : waiter.dispatchPromptId,
      tracked,
      origin,
      message: {
        role: 'user',
        content: [...entry.message.content],
        id: waiter.id,
        origin: entry.meta?.origin as PromptOrigin | undefined,
      },
      userMessageId: entry.meta?.userMessageId ?? '',
      createdAt: entry.meta?.createdAt ?? '',
    };
    const controller = new AbortController();
    const ready = createControlledPromise<void>();
    const result = createControlledPromise<TurnResult>();
    void ready.catch(() => undefined);
    const turn: MutableTurn = {
      id,
      state: 'queued',
      signal: controller.signal,
      ready,
      result,
      cancel: (reason) => {
        if (this.active?.turn === turn) return this.cancelActiveTurn(undefined, reason ?? userCancellationReason());
        return true;
      },
    };
    const active: ActiveTurn = {
      id,
      prompt,
      controller,
      steerController: new AbortController(),
      turn,
      ready,
      result,
      startedAt: Date.now(),
      steps: 0,
      gatedSteps: 0,
      nudgeCursor: this.nudgeCursor,
      current: undefined,
      interruptStep: undefined,
      failedStep: undefined,
      stopRequested: false,
      toolStopRequested: false,
      forcedStopReason: undefined,
      lastStopReason: undefined,
      filtered: false,
      maxStepsError: undefined,
      bypassStep: undefined,
      abortReason: undefined,
      retryRequested: false,
      retry: undefined,
      partials: [],
      streamToolIds: new Map(),
      forceContentPartBoundary: false,
      readyResolved: false,
      mode: undefined,
      providerType: undefined,
      protocol: undefined,
    };
    this.active = active;
    this.armTurnWatch(active);
    active.readyResolved = true;
    ready.resolve();
    active.mode = this.telemetry.getContext().mode;
    const { provider_type, protocol } = this.telemetry.getContext();
    active.providerType = provider_type;
    active.protocol = protocol;
    this.telemetry.setContext({ turn_id: id });
    const thinkingEffort = this.requester.prepareTurnConfig(id)?.thinkingEffort;
    this.telemetry.setContext({ thinking_effort: thinkingEffort });
    void this.dispatcher.dispatch(
      new TurnPrompt({
        agentId: this.scopeContext.agentId,
        input: prompt.message.content,
        origin: prompt.origin,
        promptId: prompt.promptId,
        turnId: id,
      }),
    );
    turn.state = 'running';
    void this.dispatcher.dispatch(
      new TurnStarted({
        agentId: this.scopeContext.agentId,
        turnId: id,
        promptId: prompt.promptId,
        origin: prompt.origin,
        prompt: isDisplayablePromptOrigin(prompt.origin)
          ? turnPromptText(prompt.message.content.filter((part) => !isUserPromptSubmitHookPart(part)), prompt.origin)
          : undefined,
        promptAttachments: turnPromptAttachments(prompt.message.content, prompt.origin),
      }),
    );
    const started: TurnStartedTelemetryEvent = {
      turn_id: id,
      mode: active.mode ?? 'agent',
      provider_type,
      protocol,
      enabled_plugins: this.plugins.enabledPluginIds()?.join(','),
    };
    this.telemetry.track2('turn_started', started);
    return active;
  }

  private consumeMessage(event: Extract<LoopEmitted, { type: 'message.appended' }>): void {
    if (event.source !== 'notify' && event.source !== 'remind') return;
    const kind = event.source;
    const nudge = this.nudges.slice(this.nudgeCursor).find(
      (item) => !item.dropped && !item.consumed && item.kind === kind,
    );
    if (nudge === undefined) return;
    nudge.consumed = true;
    this.nudgeCursor = Math.max(this.nudgeCursor, this.nudges.indexOf(nudge) + 1);
    if (nudge.bypassMaxSteps && this.active !== undefined) {
      this.active.bypassStep = this.active.gatedSteps + 1;
    }
    const message = nudge.contextMessage;
    if (message !== undefined && nudge.promptIds !== undefined && nudge.promptIds.length > 0) {
      void this.dispatcher.dispatch(
        new TurnSteer({
          agentId: this.scopeContext.agentId,
          input: message.content,
          origin: message.origin ?? { kind: 'user' },
          messageId: message.id,
          promptIds: [...nudge.promptIds],
          turnId: this.active?.id,
        }),
      );
    }
    nudge.onConsume?.();
  }

  private activateTurn(turnId: number): void {
    if (this.active !== undefined) return;
    const pending = this.pendingTurns.get(turnId);
    this.pendingTurns.delete(turnId);
    if (pending?.waiter !== undefined && pending.entry !== undefined) {
      const active = this.beginActiveTurn(pending.waiter, { ...pending.entry, message: pending.message }, turnId);
      pending.waiter.onMaterialize?.();
      this.settlePromptLaunched(pending.waiter, active);
      return;
    }
    const seeded = this.nudges.find(
      (nudge) => !nudge.dropped && nudge.contextMessage !== undefined && nudge.contextMessage.content.length > 0,
    );
    const seededMessage = seeded?.contextMessage;
    const id = seededMessage?.id ?? newMessageId();
    const waiter = this.createWaiter(id, seededMessage?.id);
    this.promptWaiters.set(id, waiter);
    const entry: UserEntry = {
      message: { role: 'user', content: [...(seededMessage?.content ?? [])] },
      meta: { promptId: id, origin: seededMessage?.origin, tracked: false },
    };
    const active = this.beginActiveTurn(waiter, entry, turnId);
    this.settlePromptLaunched(waiter, active);
  }

  private journalEvent(event: LoopEmitted): LoopEmitted {
    switch (event.type) {
      case 'tool.started':
        return { ...event, display: this.tools.displays.get(event.toolCall.id) };
      case 'tool.ended': {
        const extras = this.tools.extras.get(event.toolCallId);
        const durationMs = this.tools.durations.get(event.toolCallId);
        if (extras === undefined && durationMs === undefined && event.outcome.type !== 'aborted') return event;
        const entry = event.outcome.entry;
        const toolName = this.active?.current?.entry?.message.toolCalls.find((call) => call.id === event.toolCallId)?.name ?? event.toolCallId;
        return {
          ...event,
          result: {
            output: extras?.output ?? (event.outcome.type === 'aborted'
              ? abortedToolOutput(toolName, event.outcome.reason)
              : entry.message.content),
            isError: extras?.isError ?? (event.outcome.type === 'aborted' ? true : entry.meta?.isError),
            note: extras?.note ?? entry.meta?.note,
            durationMs: durationMs ?? entry.meta?.durationMs,
          },
        };
      }
      case 'step.ended':
        return { ...event, timing: this.active?.current?.timing };
      case 'turn.ended': {
        const active = this.active;
        return {
          ...event,
          durationMs: active === undefined ? undefined : Date.now() - active.startedAt,
          stopReason: active?.lastStopReason ?? undefined,
          requestId: event.outcome.type === 'done' ? this.lastRequestId : this.activeRequestTrace?.requestId,
        };
      }
      default:
        return event;
    }
  }

  private projectLoopEvent(event: LoopEmitted): void {
    if (event.type === 'message.appended') {
      void this.dispatcher.dispatch(new ContextSpliced({
        agentId: this.scopeContext.agentId,
        start: Math.max(0, this.history().length - 1),
        deleteCount: 0,
        messages: [contextFromHistoryEntry(event.entry)],
      }));
    }
    switch (event.type) {
      case 'turn.started':
        return;
      case 'turn.resumed': {
        if (this.active !== undefined) return;
        const entry = this.loop?.state.messages.findLast(
          (item): item is UserEntry => item.message.role === 'user',
        );
        const promptId = entry?.meta?.promptId ?? newMessageId();
        const waiter = this.promptWaiters.get(promptId) ?? this.createWaiter(promptId);
        this.promptWaiters.set(promptId, waiter);
        const active = this.beginActiveTurn(waiter, (entry as UserEntry | undefined) ?? createUserEntry(EMPTY_LOOP_PROMPT), event.turnId);
        this.settlePromptLaunched(waiter, active);
        return;
      }
      case 'step.started': {
        this.activateTurn(event.turnId);
        const turn = this.active;
        if (turn === undefined) return;
        const step = createStepState(event.step, turn.controller.signal);
        turn.current = step;
        turn.partials = [];
        turn.forceContentPartBoundary = false;
        if (!turn.readyResolved) {
          turn.readyResolved = true;
          turn.ready.resolve();
        }
        void this.dispatcher.dispatch(
          new TurnStepStarted({ agentId: this.scopeContext.agentId, turnId: turn.id, step: step.number, stepId: step.uuid }),
        );
        return;
      }
      case 'llm.streaming.part': {
        const turn = this.active;
        if (turn === undefined) return;
        const part = event.part;
        switch (part.type) {
          case 'text':
            this.accumulatePart(turn, part);
            void this.dispatcher.dispatch(new AssistantDelta({ agentId: this.scopeContext.agentId, turnId: turn.id, delta: part.text }));
            return;
          case 'think': {
            const accumulated = this.accumulatePart(turn, part);
            if (accumulated?.type === 'think' && accumulated.hidden === true) return;
            void this.dispatcher.dispatch(new ThinkingDelta({ agentId: this.scopeContext.agentId, turnId: turn.id, delta: part.think }));
            return;
          }
          case 'function':
            turn.forceContentPartBoundary = true;
            turn.streamToolIds.set(part._streamIndex, part.id);
            void this.dispatcher.dispatch(
              new ToolCallDelta({
                agentId: this.scopeContext.agentId,
                turnId: turn.id,
                toolCallId: part.id,
                name: part.name,
                argumentsPart: part.arguments ?? undefined,
              }),
            );
            return;
          case 'tool_call_part': {
            const toolCallId = turn.streamToolIds.get(part.index);
            if (toolCallId === undefined) return;
            void this.dispatcher.dispatch(
              new ToolCallDelta({
                agentId: this.scopeContext.agentId,
                turnId: turn.id,
                toolCallId,
                argumentsPart: part.argumentsPart ?? undefined,
              }),
            );
            return;
          }
          default:
            return;
        }
      }
      case 'llm.streaming.usage': {
        const step = this.active?.current;
        if (step !== undefined) step.usage = mergeUsage(step.usage, event.usage);
        return;
      }
      case 'llm.streaming.finish': {
        const step = this.active?.current;
        if (step !== undefined) {
          step.providerFinishReason = event.finish.finishReason ?? undefined;
          step.rawFinishReason = event.finish.rawFinishReason ?? undefined;
        }
        return;
      }
      case 'llm.streaming.message_id': {
        const step = this.active?.current;
        if (step !== undefined) step.messageId = event.messageId;
        return;
      }
      case 'llm.retrying': {
        const turn = this.active;
        if (turn === undefined) return;
        const step = turn.current;
        if (event.reason === 'recovery') {
          turn.current = undefined;
          return;
        }
        const fields = retryErrorFields(event.error);
        turn.retry = {
          failedAttempt: event.failedAttempt,
          nextAttempt: event.nextAttempt,
          maxAttempts: event.maxAttempts,
          delayMs: event.delayMs,
          errorName: fields.errorName,
          statusCode: fields.statusCode,
        };
        void this.dispatcher.dispatch(
          new TurnStepRetrying({
            agentId: this.scopeContext.agentId,
            turnId: turn.id,
            step: step?.number ?? turn.gatedSteps,
            stepId: step?.uuid,
            failedAttempt: event.failedAttempt,
            nextAttempt: event.nextAttempt,
            maxAttempts: event.maxAttempts,
            delayMs: event.delayMs,
            errorName: fields.errorName,
            errorMessage: fields.errorMessage,
            statusCode: fields.statusCode,
          }),
        );
        turn.current = undefined;
        return;
      }
      case 'message.appended': {
        this.consumeMessage(event);
        if (event.source !== 'llm') return;
        const turn = this.active;
        const step = turn?.current;
        if (turn === undefined || step === undefined) return;
        step.entry = event.entry as AssistantEntry;
        const meta = step.entry.meta;
        step.usage = meta?.usage ?? step.usage;
        step.providerFinishReason = meta?.finish?.finishReason ?? step.providerFinishReason;
        step.rawFinishReason = meta?.finish?.rawFinishReason ?? step.rawFinishReason;
        step.messageId = meta?.messageId ?? step.messageId;
        turn.partials = [];
        this.lastRequestId = this.activeRequestTrace?.requestId;
        const toolCalls = step.entry.message.toolCalls;
        if (toolCalls.length === 0) this.endOrInterruptStep(turn, step, step.providerFinishReason ?? 'completed');
        else step.pendingToolIds = new Set(toolCalls.map((call) => call.id));
        return;
      }
      case 'tool.ended': {
        const turn = this.active;
        const step = turn?.current;
        if (turn === undefined || step === undefined) return;
        step.pendingToolIds.delete(event.toolCallId);
        const extras = this.tools.extras.get(event.toolCallId);
        if (event.outcome.type === 'succeeded' && event.outcome.stopTurn === true) {
          step.toolStopTurn = true;
          turn.toolStopRequested = true;
          turn.forcedStopReason ??= extras?.stopTurnReason;
        }
        if (step.pendingToolIds.size === 0) {
          this.endOrInterruptStep(turn, step, step.toolStopTurn ? 'completed' : 'tool_calls');
        }
        return;
      }
      case 'step.ended': {
        const turn = this.active;
        const step = turn?.current;
        if (turn === undefined) return;
        const outcome = event.outcome;
        if (outcome.type === 'blocked' && outcome.cause.reason === MAX_STEPS_REASON) {
          const maxSteps = this.config.get<LoopControl>(LOOP_CONTROL_SECTION)?.maxStepsPerTurn ?? 0;
          turn.maxStepsError = createMaxStepsExceededError(maxSteps);
        }
        if (outcome.type === 'failed') {
          turn.failedStep ??= { number: event.step, uuid: step?.uuid ?? randomUUID(), error: outcome.error };
        }
        if (step !== undefined && turn.current !== undefined) turn.current = undefined;
        return;
      }
      case 'turn.ended': {
        const outcome = event.outcome;
        if (outcome.type === 'blocked' && this.active === undefined) {
          this.settleGateRejectedPrompt(event.turnId, outcome.cause.reason === 'blocked' ? 'blocked' : 'failed', outcome.cause);
          return;
        }
        const turn = this.active;
        if (turn === undefined) return;
        void this.finishTurn(turn, outcome);
        return;
      }
      case 'prompt.steered': {
        this.projectSteered(event);
        return;
      }
      default:
        return;
    }
  }

  private projectSteered(event: Extract<LoopEmitted, { type: 'prompt.steered' }>): void {
    const active = this.active;
    if (active === undefined || event.entry === undefined) return;
    const children: { readonly waiter: PromptWaiter; readonly projection: SteeredPrompt }[] = [];
    for (const id of event.promptIds) {
      const waiter = this.promptWaiters.get(id);
      if (waiter === undefined) continue;
      const entry = this.steerCandidates.get(id);
      this.steerCandidates.delete(id);
      const projection = waiter === undefined ? undefined : this.promptProjection(id) ?? (entry === undefined ? undefined : projectionFromEntry(entry));
      if (projection === undefined) continue;
      children.push({ waiter, projection: { ...projection, parentId: active.prompt.id } });
    }
    if (children.length === 0) return;
    for (const { waiter, projection } of children) {
      this.steered.set(waiter.id, projection);
      waiter.launched.resolve(active.turn);
    }
    active.steerController.abort(abortError('Steered by new input'));
    const merged = children.length === 1
      ? { content: children[0]!.projection.message.content, origin: children[0]!.projection.origin }
      : mergeSteerMessages(children.map((child) => ({ content: child.projection.message.content, origin: child.projection.origin })));
    const messageId = children.length === 1 ? children[0]!.waiter.id : newMessageId();
    const promptIds = children.map((child) => child.waiter.id);
    this.nudges.push({
      kind: 'notify',
      contextMessage: {
        role: 'user',
        content: gateImageFormatParts(merged.content, this.profile.getModelProviderType()),
        origin: markInTurnOrigin(merged.origin),
        id: messageId,
      },
      promptIds,
      bypassMaxSteps: false,
      turnScoped: false,
      sentToMachine: true,
    });
    void this.dispatcher.dispatch(
      new PromptSteered({
        agentId: this.scopeContext.agentId,
        activePromptId: active.prompt.id,
        promptIds,
        messageId,
        content: children.flatMap((child) => stripBundledSkillBlocks(child.projection.message)),
        steeredAt: new Date().toISOString(),
      }),
    );
  }

  private accumulatePart(turn: ActiveTurn, part: ContentPart): ContentPart | undefined {
    const last = turn.partials.at(-1);
    if (part.type === 'think' && last?.type === 'text' && isVacuousContentPart(part)) return undefined;
    if (!turn.forceContentPartBoundary && last !== undefined && mergeInPlace(last, part)) return last;
    turn.forceContentPartBoundary = false;
    turn.partials.push({ ...part });
    return turn.partials.at(-1);
  }

  private endOrInterruptStep(turn: ActiveTurn, step: CoreStepState, finishReason: FinishReason): void {
    if (turn.controller.signal.aborted) {
      turn.current = undefined;
      return;
    }
    this.completeStepProjection(turn, step, finishReason);
  }

  private completeStepProjection(turn: ActiveTurn, step: CoreStepState, finishReason: FinishReason): void {
    const normalized = normalizeFinishReason(finishReason);
    const usage = step.usage ?? emptyUsage();
    turn.lastStopReason = finishReason;
    turn.current = step;
    void this.dispatcher.dispatch(
      new TurnStepCompleted({
        agentId: this.scopeContext.agentId,
        turnId: turn.id,
        step: step.number,
        stepId: step.uuid,
        usage,
        finishReason: normalized,
        llmFirstTokenLatencyMs: step.timing?.firstTokenLatencyMs,
        llmStreamDurationMs: step.timing?.streamDurationMs,
        llmRequestBuildMs: step.timing?.requestBuildMs,
        llmServerFirstTokenMs: step.timing?.serverFirstTokenMs,
        llmServerDecodeMs: step.timing?.serverDecodeMs,
        llmClientConsumeMs: step.timing?.clientConsumeMs,
        llmClientBlockedMs: step.timing?.clientBlockedMs,
        providerFinishReason: step.providerFinishReason,
        rawFinishReason: step.rawFinishReason,
      }),
    );
  }

  private async finishTurn(turn: ActiveTurn, outcome: Extract<LoopEmitted, { type: 'turn.ended' }>['outcome']): Promise<void> {
    if (this.active !== turn) return;
    if (turn.abortReason !== undefined || turn.controller.signal.aborted || outcome.type === 'aborted') {
      const reason = turn.abortReason ?? (turn.controller.signal.aborted ? turn.controller.signal.reason : undefined) ?? abortError('Turn aborted');
      this.interruptRunForCancel(turn, reason);
      await this.endTurn(turn, { type: 'cancelled', steps: turn.steps, reason });
      if (
        this.loop !== undefined &&
        this.quiescenceDepth === 0 &&
        !this.disposing &&
        (this.pendingSubmissions.length > 0 || this.loop.state.inbox.normal.length > 0 || this.loop.state.inbox.notify.length > 0)
      ) {
        this.loop.continue();
      }
      return;
    }
    if (turn.filtered) {
      await this.endTurn(turn, {
        type: 'failed',
        steps: turn.steps,
        error: new Error2(ErrorCodes.PROVIDER_FILTERED, 'Provider safety policy blocked the response.', {
          name: 'ProviderFilteredError',
          details: { finishReason: 'filtered' },
        }),
      });
      return;
    }
    if (turn.maxStepsError !== undefined || (outcome.type !== 'failed' && outcome.cause?.reason === MAX_STEPS_REASON)) {
      const error = turn.maxStepsError ?? createMaxStepsExceededError(this.config.get<LoopControl>(LOOP_CONTROL_SECTION)?.maxStepsPerTurn ?? 0);
      this.failStep(turn.id, turn.interruptStep, error);
      await this.endTurn(turn, { type: 'failed', steps: turn.steps, error });
      return;
    }
    if (outcome.type === 'failed' || turn.failedStep !== undefined) {
      const error = turn.failedStep?.error ?? (outcome.type === 'failed' ? outcome.error : undefined) ?? new Error('Turn failed');
      this.failStep(turn.id, turn.interruptStep, error);
      await this.endTurn(turn, { type: 'failed', steps: turn.steps, error });
      return;
    }
    if (outcome.type === 'blocked') {
      const error = new Error2(ErrorCodes.INTERNAL, outcome.cause.message ?? outcome.cause.reason);
      this.failStep(turn.id, turn.interruptStep, error);
      await this.endTurn(turn, { type: 'failed', steps: turn.steps, error });
      return;
    }
    await this.endTurn(turn, this.completedResult(turn));
  }

  private failStep(turnId: number, step: number | undefined, error: unknown): void {
    const reason: LoopInterruptReason = isMaxStepsExceededError(error) ? 'max_steps' : 'error';
    const interruptedError = isError2(error) && error.code === ErrorCodes.INTERNAL && error.cause !== undefined ? error.cause : error;
    this.emitStepInterrupted(turnId, step, reason, toErrorMessage(interruptedError));
  }

  private interruptRunForCancel(turn: ActiveTurn, reason: unknown): void {
    if (!this.disposing) {
      this.instantiation.invokeFunction((accessor) => accessor.get(IAgentInterruptionReminderService).notifyUserCancellation(reason));
    }
    const current = turn.current;
    if (current !== undefined) turn.current = undefined;
    if (turn.interruptStep !== undefined) {
      this.emitStepInterrupted(
        turn.id,
        turn.interruptStep,
        'aborted',
        isUserCancellation(reason) ? undefined : toErrorMessage(reason),
      );
      turn.interruptStep = undefined;
    }
  }

  private completedResult(turn: ActiveTurn): LoopRunResult {
    return {
      type: 'completed',
      steps: turn.steps,
      truncated: turn.lastStopReason === 'truncated',
      stopReason: turn.forcedStopReason,
    };
  }

  private async endTurn(turn: ActiveTurn, result: TurnResult): Promise<void> {
    if (this.active !== turn) return;
    this.active = undefined;
    this.disarmTurnWatch();
    for (const nudge of this.nudges.slice(this.nudgeCursor)) {
      if (nudge.turnScoped && !nudge.dropped) {
        nudge.dropped = true;
        nudge.onDrop?.();
      }
    }
    turn.turn.state = result.type;
    if (!turn.readyResolved) {
      if (result.type === 'failed') turn.ready.reject(result.error);
      else if (result.type === 'cancelled') turn.ready.reject(result.reason instanceof Error ? result.reason : abortError('Turn cancelled'));
      else turn.ready.reject(new Error2(ErrorCodes.INTERNAL, 'Turn ended before first step'));
    }
    const durationMs = Date.now() - turn.startedAt;
    const requestId = result.type === 'completed' ? this.lastRequestId : this.activeRequestTrace?.requestId;
    const error = result.type === 'failed' ? toKimiErrorPayload(result.error) : undefined;
    const interruptReason = result.type === 'completed' ? undefined : interruptReasonFor(result);
    await this.wire.drainPersisted().catch(() => undefined);
    this.flushReminders();
    const turnEnded = this.dispatcher.dispatch(
      new TurnEnded({
        agentId: this.scopeContext.agentId,
        turnId: turn.id,
        reason: result.type,
        error,
        durationMs,
        interruptReason,
        stopReason: result.type === 'completed' ? result.stopReason : undefined,
        requestId,
      }),
    );
    if (error !== undefined) void this.dispatcher.dispatch(new AgentErrorEvent({ ...error, agentId: this.scopeContext.agentId }));
    await turnEnded;
    await this.dispatcher.flush();
    if (interruptReason !== undefined) {
      const interrupted: TurnInterruptedEvent = {
        turn_id: turn.id,
        at_step: result.steps,
        mode: turn.mode ?? 'agent',
        interrupt_reason: interruptReason,
        provider_type: turn.providerType,
        protocol: turn.protocol,
        request_id: requestId,
      };
      this.telemetry.track2('turn_interrupted', interrupted);
    }
    const ended: TurnEndedTelemetryEvent = {
      turn_id: turn.id,
      reason: result.type,
      duration_ms: durationMs,
      mode: turn.mode ?? 'agent',
      error_type: error?.code,
      provider_type: turn.providerType,
      protocol: turn.protocol,
      request_id: requestId,
      enabled_plugins: this.plugins.enabledPluginIds()?.join(','),
    };
    this.telemetry.track2('turn_ended', ended);
    this.telemetry.setContext({ turn_id: undefined, request_id: undefined, thinking_effort: undefined });
    this.activeRequestTrace = undefined;
    this.lastRequestId = undefined;
    await this.wire.drainPersisted().catch(() => undefined);
    turn.result.resolve(result);
    this.maybeSettle();
  }

  private emitStepInterrupted(
    turnId: number,
    activeStep: number | undefined,
    reason: LoopInterruptReason,
    message?: string,
  ): void {
    if (activeStep === undefined) return;
    void this.dispatcher.dispatch(
      new TurnStepInterrupted({ agentId: this.scopeContext.agentId, turnId, step: activeStep, reason, message }),
    );
  }

  private armTurnWatch(active: ActiveTurn): void {
    this.disarmTurnWatch();
    const signatureOf = (): string => [
      active.gatedSteps,
      active.steps,
      active.partials.length,
      active.current?.pendingToolIds.size ?? 0,
      active.retry?.nextAttempt ?? 0,
    ].join(':');
    const watch = {
      timer: undefined as unknown as NodeJS.Timeout,
      signature: signatureOf(),
      since: Date.now(),
      lastWarnAt: 0,
    };
    watch.timer = setInterval(() => {
      if (this.active !== active) {
        this.disarmTurnWatch();
        return;
      }
      const signature = signatureOf();
      if (signature !== watch.signature) {
        watch.signature = signature;
        watch.since = Date.now();
        return;
      }
      const silentMs = Date.now() - watch.since;
      if (silentMs < 10 * 60_000 || Date.now() - watch.lastWarnAt < 10 * 60_000) return;
      watch.lastWarnAt = Date.now();
      this.logger.warn('turn watchdog: no new events from the running turn', {
        agentId: this.scopeContext.agentId,
        turnId: active.id,
        silentMinutes: Math.round(silentMs / 60_000),
        steps: active.steps,
        queuedPrompts: this.loop?.state.inbox.normal.length ?? 0,
      });
    }, 60_000);
    watch.timer.unref();
    this.turnWatch = watch;
  }

  private disarmTurnWatch(): void {
    if (this.turnWatch === undefined) return;
    clearInterval(this.turnWatch.timer);
    this.turnWatch = undefined;
  }
}

function normalizeFinishReason(reason: FinishReason): string {
  if (reason === 'tool_calls') return 'tool_use';
  if (reason === 'completed') return 'end_turn';
  if (reason === 'truncated') return 'max_tokens';
  return reason;
}

function machineReminderEntry(message: ContextMessage | undefined): ReturnType<typeof createSystemEntry> | UserEntry {
  const meta = { promptId: message?.id, origin: message?.origin };
  if (message?.role === 'system') return createSystemEntry({ ...message, content: [...message.content] }, meta);
  return createUserEntry(machineUserMessage(message), meta);
}

function machineUserMessage(message: ContextMessage | undefined): UserMessage {
  if (message === undefined) return EMPTY_LOOP_PROMPT;
  return { role: 'user', content: [...message.content] };
}

function mergeUsage(base: TokenUsage | undefined, patch: Partial<TokenUsage>): TokenUsage {
  return {
    inputOther: patch.inputOther ?? base?.inputOther ?? 0,
    output: patch.output ?? base?.output ?? 0,
    inputCacheRead: patch.inputCacheRead ?? base?.inputCacheRead,
    inputCacheCreation: patch.inputCacheCreation ?? base?.inputCacheCreation ?? 0,
  };
}

type MutableTurn = {
  -readonly [K in keyof Turn]: Turn[K];
};

interface PromptWaiter {
  readonly id: string;
  readonly dispatchPromptId?: string;
  readonly launched: ReturnType<typeof createControlledPromise<Turn | undefined>>;
  readonly completion: ReturnType<typeof createControlledPromise<PromptCompletion>>;
  readonly onMaterialize?: () => void;
  failedEntry?: UserEntry;
}

interface PendingTurn {
  readonly entry: UserEntry;
  readonly waiter?: PromptWaiter;
  message: UserMessage;
}

interface PromptProjection {
  readonly tracked: boolean;
  readonly origin: PromptOrigin;
  readonly message: ContextMessage;
  readonly userMessageId: string;
  readonly createdAt: string;
}

interface ActivePrompt extends PromptProjection {
  readonly id: string;
  readonly promptId?: string;
}

interface SteeredPrompt extends PromptProjection {
  readonly parentId: string;
}

const EMPTY_HANDLE_MESSAGE: ContextMessage = {
  role: 'user',
  content: [],
};

function projectionFromEntry(entry: UserEntry): PromptProjection {
  const origin = (entry.meta?.origin as PromptOrigin | undefined) ?? { kind: 'user' };
  return {
    tracked: entry.meta?.tracked === true,
    origin,
    message: {
      role: 'user',
      content: [...entry.message.content],
      id: entry.meta?.promptId,
      origin: entry.meta?.origin as PromptOrigin | undefined,
    },
    userMessageId: entry.meta?.userMessageId ?? '',
    createdAt: entry.meta?.createdAt ?? '',
  };
}

interface Nudge {
  readonly kind: 'notify' | 'remind';
  readonly contextMessage?: ContextMessage;
  readonly promptIds?: readonly string[];
  readonly bypassMaxSteps: boolean;
  readonly turnScoped: boolean;
  readonly onConsume?: () => void;
  readonly onDrop?: () => void;
  dropped?: boolean;
  consumed?: boolean;
  sentToMachine?: boolean;
}

interface CoreStepState {
  readonly number: number;
  readonly uuid: string;
  readonly signal: AbortSignal;
  entry?: AssistantEntry;
  usage?: TokenUsage;
  timing?: ModelRequestTiming;
  providerFinishReason?: FinishReason;
  rawFinishReason?: string;
  messageId?: string;
  pendingToolIds: Set<string>;
  toolStopTurn: boolean;
}

function createStepState(number: number, signal: AbortSignal): CoreStepState {
  return {
    number,
    uuid: randomUUID(),
    signal,
    pendingToolIds: new Set(),
    toolStopTurn: false,
  };
}

interface MachineFailedStep {
  readonly number: number;
  readonly uuid: string;
  readonly error: unknown;
}

interface ActiveTurn {
  readonly id: number;
  readonly prompt: ActivePrompt;
  readonly controller: AbortController;
  steerController: AbortController;
  readonly turn: MutableTurn;
  readonly ready: ReturnType<typeof createControlledPromise<void>>;
  readonly result: ReturnType<typeof createControlledPromise<TurnResult>>;
  readonly startedAt: number;
  steps: number;
  gatedSteps: number;
  nudgeCursor: number;
  current: CoreStepState | undefined;
  interruptStep: number | undefined;
  failedStep: MachineFailedStep | undefined;
  stopRequested: boolean;
  toolStopRequested: boolean;
  forcedStopReason: string | undefined;
  lastStopReason: FinishReason | undefined;
  filtered: boolean;
  maxStepsError: LoopError | undefined;
  bypassStep: number | undefined;
  abortReason: unknown;
  retryRequested: boolean;
  retry: { failedAttempt: number; nextAttempt: number; maxAttempts: number; delayMs: number; errorName?: string; statusCode?: number } | undefined;
  partials: ContentPart[];
  streamToolIds: Map<number | string | undefined, string>;
  forceContentPartBoundary: boolean;
  readyResolved: boolean;
  mode: 'agent' | 'plan' | undefined;
  providerType: string | undefined;
  protocol: string | undefined;
}

function cancelReasonFor(cancellation: unknown): 'user_cancelled' | 'aborted' {
  return isUserCancellation(cancellation) ? 'user_cancelled' : 'aborted';
}

function interruptReasonFor(
  result: Extract<TurnResult, { readonly type: 'cancelled' | 'failed' }>,
): TurnInterruptReason {
  if (result.type === 'cancelled') return isUserCancellation(result.reason) ? 'user_cancelled' : 'aborted';
  if (isMaxStepsExceededError(result.error)) return 'max_steps';
  if (isError2(result.error) && result.error.code === ErrorCodes.PROVIDER_FILTERED) return 'filtered';
  return 'error';
}

registerScopedService(
  LifecycleScope.Agent,
  IAgentLoopService,
  AgentLoopService,
  ScopeActivation.OnScopeCreated,
  'loop',
);
