// OpenCode Plugin: Per-Subagent Session Headers for KV Cache Pinning
//
// Each (session, subagent) pair gets a unique stable ID appended
// to the base session UUID (e.g. "uuid-general", "uuid-explore").
// This pins each subagent to its own worker for KV cache reuse.
//
// Install:
//   cp opencode-plugin-session-headers.ts ~/.config/opencode/plugins/session-headers.ts
//   Add to opencode.json: "plugin": ["./plugins/session-headers.ts"]

export const SessionHeadersPlugin = async () => {
  return {
    "chat.headers": async (
      input: { sessionID: string; agent: string },
      output: { headers: Record<string, string> },
    ) => {
      const suffix = input.agent && input.agent !== "default" ? `-${input.agent}` : "";
      output.headers["X-Session-ID"] = `${input.sessionID}${suffix}`;
    },
  };
};

export default SessionHeadersPlugin;
