// OpenCode Plugin: Per-Subagent Session Headers for KV Cache Pinning
//
// Sends X-Session-ID for session-pinned routing and X-Agent-Type
// for subagent identification.
//
// Install:
//   cp opencode-plugin-session-headers.ts ~/.config/opencode/plugins/session-headers.ts
//   Add to opencode.json: "plugin": ["./plugins/session-headers.ts"]

export const SessionHeadersPlugin = async () => {
  return {
    "chat.headers": async (
      input: { sessionID: string; agent?: string },
      output: { headers: Record<string, string> },
    ) => {
      output.headers["X-Session-ID"] ??= input.sessionID;
      if (input.agent) output.headers["X-Agent-Type"] ??= input.agent;
    },
  };
};

export default SessionHeadersPlugin;
