#!/bin/bash
# TetraMem MCP launcher - stdio mode for OpenClaw
# Each invocation is a separate stdio process, no locking needed
exec node /root/.openclaw/mcp-tetramem/index.js
