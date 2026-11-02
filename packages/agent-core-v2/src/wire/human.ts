import { AGENT_SWITCHED_TYPE } from './tree/tree';

export const HUMAN_AGENT_DOMAIN = 'agent';

export const AGENT_WIRE_RECORD_TYPES: ReadonlySet<string> = new Set([
  AGENT_SWITCHED_TYPE,
  'agent.message.appended',
  'agent.turn.started',
  'agent.turn.ended',
  'agent.input.submitted',
  'agent.input.notified',
  'agent.input.reminded',
  'agent.input.steered',
  'agent.input.cancelled',
  'agent.queue.drained',
  'agent.input.drained',
  'agent.notifications.drained',
  'context.append_message',
  'context.append_loop_event',
  'context.apply_compaction',
  'context.clear',
  'context.undo',
  'context.undone',
  'turn.prompt',
  'turn.steer',
  'turn.cancel',
  'turn.ended',
]);

const LEGACY_HUMAN_RECORD_PREFIX = 'human.';

export function isHumanRecordType(type: string): boolean {
  return AGENT_WIRE_RECORD_TYPES.has(type) || type.startsWith(LEGACY_HUMAN_RECORD_PREFIX);
}

export function humanRecordType(domain: string, type: string): string {
  return `${domain}.${type}`;
}

export function humanEventType(recordType: string, domain: string): string | undefined {
  if (recordType === AGENT_SWITCHED_TYPE) return undefined;
  for (const prefix of [`${domain}.`, `${LEGACY_HUMAN_RECORD_PREFIX}${domain}.`]) {
    if (!recordType.startsWith(prefix)) continue;
    const type = recordType.slice(prefix.length);
    return type.length === 0 ? undefined : type;
  }
  return undefined;
}
