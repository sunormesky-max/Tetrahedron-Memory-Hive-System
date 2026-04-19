import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

const TETRA_API_URL = process.env.TETRA_API_URL || 'http://127.0.0.1:8000';

const server = new Server(
  { name: 'mcp-tetramem', version: '3.0.0' },
  { capabilities: { tools: {} } }
);

async function callApi(endpoint, method = 'GET', data = null) {
  const url = `${TETRA_API_URL}${endpoint}`;
  const options = { method, headers: { 'Content-Type': 'application/json' } };
  if (data) options.body = JSON.stringify(data);
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API ${response.status}: ${text}`);
  }
  return response.json();
}

const TOOLS = [
  {
    name: 'tetramem_store',
    description: 'Store a memory into the tetrahedral memory hive. Every memory is eternal - never deleted or decayed. It attaches to a tetrahedron (3-simplex) in the geometric mesh.',
    inputSchema: {
      type: 'object',
      properties: {
        content: { type: 'string', description: 'Memory content text' },
        labels: { type: 'array', items: { type: 'string' }, description: 'Semantic tags for categorization and retrieval' },
        weight: { type: 'number', description: 'Importance weight 0.1-10.0 (default 1.0, increases with integration)', default: 1.0 },
        metadata: { type: 'object', description: 'Extra metadata (source, context, etc.)' }
      },
      required: ['content']
    }
  },
  {
    name: 'tetramem_query',
    description: 'Query memories using pure topological navigation (BFS along shared faces/edges/vertices). No vector embeddings used. Returns memories topologically closest to the query seed.',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Query text' },
        k: { type: 'number', description: 'Number of results (default 5)', default: 5 },
        labels: { type: 'array', items: { type: 'string' }, description: 'Filter by labels' }
      },
      required: ['query']
    }
  },
  {
    name: 'tetramem_associate',
    description: 'Find associated memories by traversing topological connections from a given memory. Returns memories connected via shared faces, edges, or vertices.',
    inputSchema: {
      type: 'object',
      properties: {
        tetra_id: { type: 'string', description: 'Source memory ID' },
        max_depth: { type: 'number', description: 'Traversal depth (default 2)', default: 2 }
      },
      required: ['tetra_id']
    }
  },
  {
    name: 'tetramem_navigate',
    description: 'Navigate the tetrahedral mesh from a seed memory, following topological BFS path. Shows the geometric neighborhood of a memory.',
    inputSchema: {
      type: 'object',
      properties: {
        seed_id: { type: 'string', description: 'Seed tetrahedron ID' },
        max_steps: { type: 'number', description: 'Max navigation steps (default 30)', default: 30 },
        strategy: { type: 'string', enum: ['bfs', 'dfs'], description: 'Navigation strategy (default bfs)', default: 'bfs' }
      },
      required: ['seed_id']
    }
  },
  {
    name: 'tetramem_dream',
    description: 'Trigger a dream cycle - the system autonomously synthesizes new concepts from existing memories via random walk + cross-cluster fusion. This is the self-emergence mechanism.',
    inputSchema: {
      type: 'object',
      properties: {
        force: { type: 'boolean', description: 'Force dream regardless of conditions', default: false }
      }
    }
  },
  {
    name: 'tetramem_self_organize',
    description: 'Trigger topological self-organization. Uses Persistent Homology to detect structural issues (caves, redundant clusters) and performs geometric surgery.',
    inputSchema: {
      type: 'object',
      properties: {
        max_iterations: { type: 'number', description: 'Max iterations (default 5)', default: 5 }
      }
    }
  },
  {
    name: 'tetramem_abstract_reorganize',
    description: 'Trigger abstract reorganization of dense tetrahedra. Memories with many secondary entries are fused into higher-order abstractions. Cross-fusion creates bridging concepts.',
    inputSchema: {
      type: 'object',
      properties: {
        min_density: { type: 'number', description: 'Min secondary memories to trigger reorg (default 2)', default: 2 },
        max_operations: { type: 'number', description: 'Max reorg operations (default 20)', default: 20 }
      }
    }
  },
  {
    name: 'tetramem_closed_loop',
    description: 'Execute the full cognitive closed loop: Dream (self-emergent synthesis) + Self-Organization (PH-driven geometric surgery) + Abstract Reorganization (concept fusion). The complete eternal memory lifecycle.',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  {
    name: 'tetramem_export',
    description: 'Export all memories to markdown file for OpenClaw memorySearch integration. Generates tetramem_export.md with all tetrahedra.',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  {
    name: 'tetramem_stats',
    description: 'Get TetraMem-XL system statistics: total memories, labels, mesh topology info.',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  {
    name: 'tetramem_topology_health',
    description: 'Get topology health report: persistent homology status, entropy, mesh integrity.',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  {
    name: 'tetramem_seed_by_label',
    description: 'Find a tetrahedron seed by labels. Useful for starting topological navigation from a known topic.',
    inputSchema: {
      type: 'object',
      properties: {
        labels: { type: 'array', items: { type: 'string' }, description: 'Labels to search for' }
      },
      required: ['labels']
    }
  },
  {
    name: 'tetramem_agent_context',
    description: 'Assemble a rich memory context for a given topic. Returns core memories, dream insights, associations, and a reasoning summary. This is the primary tool for agents to understand a topic.',
    inputSchema: {
      type: 'object',
      properties: {
        topic: { type: 'string', description: 'The topic to gather context for' },
        max_memories: { type: 'number', description: 'Max memories to return (default 15)', default: 15 }
      },
      required: ['topic']
    }
  },
  {
    name: 'tetramem_agent_reasoning',
    description: 'Find a multi-hop reasoning chain from a source memory to a target concept. Traverses the topological network to find connected reasoning paths.',
    inputSchema: {
      type: 'object',
      properties: {
        source_id: { type: 'string', description: 'Source memory ID (first 12 chars)' },
        target_query: { type: 'string', description: 'Target concept to reason towards' },
        max_hops: { type: 'number', description: 'Max reasoning hops (default 5)', default: 5 }
      },
      required: ['source_id', 'target_query']
    }
  },
  {
    name: 'tetramem_agent_suggest',
    description: 'Get proactive action suggestions based on the current state of the memory field. The system analyzes connectivity, pulse activity, and clusters to recommend actions.',
    inputSchema: {
      type: 'object',
      properties: {
        context: { type: 'string', description: 'Optional context string for contextual suggestions', default: '' }
      }
    }
  },
  {
    name: 'tetramem_agent_navigate',
    description: 'Navigate the memory topology from one node to another using cost-optimal pathfinding. Useful for understanding how two memories are topologically connected.',
    inputSchema: {
      type: 'object',
      properties: {
        source_id: { type: 'string', description: 'Source memory ID' },
        target_id: { type: 'string', description: 'Target memory ID' },
        max_hops: { type: 'number', description: 'Max hops (default 6)', default: 6 }
      },
      required: ['source_id', 'target_id']
    }
  },
  {
    name: 'tetramem_feedback_record',
    description: 'Record the outcome of an agent action. The system learns from feedback to adjust memory priority and strengthen associations. Positive outcomes boost weight, negative outcomes tag as low-priority (never deletes).',
    inputSchema: {
      type: 'object',
      properties: {
        action: { type: 'string', description: 'The action taken (e.g. "query", "navigate", "suggest")' },
        context_id: { type: 'string', description: 'The memory node ID involved' },
        outcome: { type: 'string', enum: ['positive', 'negative', 'neutral'], description: 'The outcome of the action' },
        confidence: { type: 'number', description: 'Confidence in the outcome (0-1, default 0.5)', default: 0.5 },
        reasoning: { type: 'string', description: 'Why this outcome was chosen', default: '' }
      },
      required: ['action', 'context_id', 'outcome']
    }
  },
  {
    name: 'tetramem_feedback_insights',
    description: 'Get learning insights from the feedback loop. Shows which memories have been most effective, which paths have been validated, and overall feedback statistics.',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  {
    name: 'tetramem_session_create',
    description: 'Create a new conversation session. Memories added during the session are ephemeral (temporary) and can be consolidated into permanent memories when the session ends.',
    inputSchema: {
      type: 'object',
      properties: {
        agent_id: { type: 'string', description: 'Agent identifier (default "default")', default: 'default' },
        metadata: { type: 'object', description: 'Optional session metadata' }
      }
    }
  },
  {
    name: 'tetramem_session_add',
    description: 'Add a conversation turn to an active session. The content is stored as an ephemeral memory tied to the session.',
    inputSchema: {
      type: 'object',
      properties: {
        session_id: { type: 'string', description: 'Session ID' },
        role: { type: 'string', enum: ['user', 'agent', 'system'], description: 'Who is speaking (default "user")', default: 'user' },
        content: { type: 'string', description: 'The message content' }
      },
      required: ['session_id', 'content']
    }
  },
  {
    name: 'tetramem_session_recall',
    description: 'Recall conversation history from a session. Returns the most recent conversation turns.',
    inputSchema: {
      type: 'object',
      properties: {
        session_id: { type: 'string', description: 'Session ID' },
        n: { type: 'number', description: 'Number of recent turns to recall (default 20)', default: 20 }
      },
      required: ['session_id']
    }
  },
  {
    name: 'tetramem_session_consolidate',
    description: 'End a session and consolidate important ephemeral memories into permanent ones. Memories above weight threshold are promoted, others are softly kept with reduced activation.',
    inputSchema: {
      type: 'object',
      properties: {
        session_id: { type: 'string', description: 'Session ID to consolidate' }
      },
      required: ['session_id']
    }
  }
];

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  try {
    let result;
    switch (name) {
      case 'tetramem_store':
        result = await callApi('/api/v1/store', 'POST', {
          content: args.content,
          labels: args.labels || [],
          weight: args.weight || 1.0,
          metadata: args.metadata
        });
        return { content: [{ type: 'text', text: `Stored in TetraMem-XL (id: ${result.id}). Memory is eternal and will never be deleted.` }] };

      case 'tetramem_query':
        result = await callApi('/api/v1/query', 'POST', {
          query: args.query,
          k: args.k || 5,
          labels: args.labels
        });
        return { content: [{ type: 'text', text: JSON.stringify(result.results.map(r => ({
          id: r.id, content: r.content, distance: r.distance, weight: r.weight, labels: r.labels
        })), null, 2) }] };

      case 'tetramem_associate':
        result = await callApi('/api/v1/associate', 'POST', {
          tetra_id: args.tetra_id,
          max_depth: args.max_depth || 2
        });
        return { content: [{ type: 'text', text: JSON.stringify(result.associations, null, 2) }] };

      case 'tetramem_navigate':
        result = await callApi('/api/v1/navigate', 'POST', {
          seed_id: args.seed_id,
          max_steps: args.max_steps || 30,
          strategy: args.strategy || 'bfs'
        });
        return { content: [{ type: 'text', text: JSON.stringify(result.path, null, 2) }] };

      case 'tetramem_dream':
        result = await callApi('/api/v1/dream', 'POST', { force: args.force || false });
        return { content: [{ type: 'text', text: `Dream cycle completed: ${JSON.stringify(result.result)}` }] };

      case 'tetramem_self_organize':
        result = await callApi('/api/v1/self-organize', 'POST', { max_iterations: args.max_iterations || 5 });
        return { content: [{ type: 'text', text: `Self-organization completed: ${JSON.stringify(result.stats)}` }] };

      case 'tetramem_abstract_reorganize':
        result = await callApi('/api/v1/abstract-reorganize', 'POST', {
          min_density: args.min_density || 2,
          max_operations: args.max_operations || 20
        });
        return { content: [{ type: 'text', text: `Abstract reorganization: ${JSON.stringify(result.result)}` }] };

      case 'tetramem_closed_loop':
        result = await callApi('/api/v1/closed-loop', 'POST', {});
        return { content: [{ type: 'text', text: `Closed loop cycle: Dream + Self-Org + Abstract Reorg completed.\n${JSON.stringify(result.result, null, 2)}` }] };

      case 'tetramem_export':
        result = await callApi('/api/v1/export', 'POST', {});
        return { content: [{ type: 'text', text: `Exported ${result.size} bytes to ${result.path}` }] };

      case 'tetramem_stats':
        result = await callApi('/api/v1/stats');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_topology_health':
        result = await callApi('/api/v1/topology-health');
        return { content: [{ type: 'text', text: JSON.stringify(result.result, null, 2) }] };

      case 'tetramem_seed_by_label':
        result = await callApi('/api/v1/seed-by-label', 'POST', { labels: args.labels });
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_agent_context':
        result = await callApi('/api/v1/agent/context', 'POST', {
          topic: args.topic,
          max_memories: args.max_memories || 15
        });
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_agent_reasoning':
        result = await callApi('/api/v1/agent/reasoning', 'POST', {
          source_id: args.source_id,
          target_query: args.target_query,
          max_hops: args.max_hops || 5
        });
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_agent_suggest':
        result = await callApi('/api/v1/agent/suggest', 'POST', {
          context: args.context || ''
        });
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_agent_navigate':
        result = await callApi('/api/v1/navigate', 'POST', {
          source_id: args.source_id,
          target_id: args.target_id,
          max_hops: args.max_hops || 6
        });
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_feedback_record':
        result = await callApi('/api/v1/feedback/record', 'POST', {
          action: args.action,
          context_id: args.context_id,
          outcome: args.outcome,
          confidence: args.confidence || 0.5,
          reasoning: args.reasoning || ''
        });
        return { content: [{ type: 'text', text: `Feedback recorded: ${JSON.stringify(result)}` }] };

      case 'tetramem_feedback_insights':
        result = await callApi('/api/v1/feedback/insights');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_session_create':
        result = await callApi('/api/v1/session/create', 'POST', {
          agent_id: args.agent_id || 'default',
          metadata: args.metadata
        });
        return { content: [{ type: 'text', text: `Session created: ${result.session_id}` }] };

      case 'tetramem_session_add':
        result = await callApi(`/api/v1/session/${args.session_id}/add`, 'POST', {
          role: args.role || 'user',
          content: args.content
        });
        return { content: [{ type: 'text', text: `Added to session: ${JSON.stringify(result)}` }] };

      case 'tetramem_session_recall':
        result = await callApi(`/api/v1/session/${args.session_id}/recall?n=${args.n || 20}`);
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_session_consolidate':
        result = await callApi(`/api/v1/session/${args.session_id}/consolidate`, 'POST');
        return { content: [{ type: 'text', text: `Session consolidated: ${JSON.stringify(result)}` }] };

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    return { content: [{ type: 'text', text: `Error: ${error.message}` }] };
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('TetraMem-XL MCP server v3.0 started', { url: TETRA_API_URL });
}

main().catch((error) => {
  console.error('Server failed:', error);
  process.exit(1);
});
