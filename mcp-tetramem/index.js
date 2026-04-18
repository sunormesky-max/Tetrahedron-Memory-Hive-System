import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

const TETRA_API_URL = process.env.TETRA_API_URL || 'http://127.0.0.1:8000';

const server = new Server(
  { name: 'mcp-tetramem', version: '4.1.0' },
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
    description: 'Store a memory into TetraMem-XL honeycomb neural field. Memories are eternal — never deleted, only integrated. Content is mapped to BCC lattice nodes with PCNN pulse reinforcement.',
    inputSchema: {
      type: 'object',
      properties: {
        content: { type: 'string', description: 'Memory content text' },
        labels: { type: 'array', items: { type: 'string' }, description: 'Semantic tags for categorization' },
        weight: { type: 'number', description: 'Importance weight 0.1-10.0 (default 1.0)', default: 1.0 },
        metadata: { type: 'object', description: 'Extra metadata (source, context, etc.)' }
      },
      required: ['content']
    }
  },
  {
    name: 'tetramem_query',
    description: 'Query memories using token-overlap scoring with activation/weight/Hebbian bias. Returns top-k matching memories from the BCC lattice.',
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
    description: 'Find associated memories by traversing BCC lattice connections (face/edge neighbors) from a given memory.',
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
    name: 'tetramem_timeline',
    description: 'Browse memories chronologically with optional label filter and minimum weight threshold.',
    inputSchema: {
      type: 'object',
      properties: {
        direction: { type: 'string', enum: ['newest', 'oldest'], description: 'Sort direction (default newest)', default: 'newest' },
        limit: { type: 'number', description: 'Max results (default 20)', default: 20 },
        labels: { type: 'array', items: { type: 'string' }, description: 'Filter by labels' },
        min_weight: { type: 'number', description: 'Minimum weight filter (default 0.0)', default: 0.0 }
      }
    }
  },
  {
    name: 'tetramem_list',
    description: 'List all occupied memory nodes with full details including position, labels, weight, activation, PCNN state.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_get',
    description: 'Get a single memory node by ID with full details.',
    inputSchema: {
      type: 'object',
      properties: { id: { type: 'string', description: 'Memory node ID' } },
      required: ['id']
    }
  },
  {
    name: 'tetramem_dream',
    description: 'Trigger a dream cycle. The PCNN pulse engine runs continuously — this returns current pulse/bridge status.',
    inputSchema: {
      type: 'object',
      properties: {
        force: { type: 'boolean', description: 'Force dream regardless of conditions', default: false }
      }
    }
  },
  {
    name: 'tetramem_self_organize',
    description: 'Trigger convergence bridge detection and global decay cycle. Detects high-accumulator empty nodes and creates bridge memories.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_closed_loop',
    description: 'Execute the full cognitive closed loop: Bridge detection + Global decay cycle.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_stats',
    description: 'Get TetraMem-XL system statistics: nodes, edges, pulse counts, PCNN config, Hebbian path memory, self-check status.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_export',
    description: 'Export all memories to markdown file.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_topology',
    description: 'Get topology graph of occupied nodes and their connections. Used for visualization and analysis.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_lattice_info',
    description: 'Get BCC lattice info including resolution, spacing, and all memory positions for 3D visualization.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_pulse_snapshot',
    description: 'Get current pulse snapshot — nodes with high pulse accumulator or high activation.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_pulse_status',
    description: 'Get PCNN pulse engine status: pulse count, bridge count, hot nodes, Hebbian path memory, adaptive interval.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_tension',
    description: 'Get tension map — nodes with high weight variance relative to neighbors. Identifies structural stress points.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_tension_detailed',
    description: 'Get detailed PCNN tension map with per-node weight variance analysis.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_pcnn_states',
    description: 'Get PCNN node states (feeding, linking, internal activity, threshold, fired) for top active nodes.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_hebbian',
    description: 'Get Hebbian path memory statistics and top reinforced pathways.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_pcnn_config',
    description: 'Get PCNN configuration parameters: decay rates, coupling coefficients, hop limits.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_phase_status',
    description: 'Get phase transition detector status: global tension, tension trend, transition count.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_phase_trigger',
    description: 'Manually trigger phase transition. Detects tension clusters and executes structural reorganization.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_self_check_status',
    description: 'Get self-check engine status: total checks, anomalies found, repairs done, engine running state.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_self_check_run',
    description: 'Manually trigger a full self-check: isolation scan, duplicate detection, vitality scan. Auto-repairs detected issues.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_self_check_history',
    description: 'Get self-check history — previous check results with anomaly details.',
    inputSchema: {
      type: 'object',
      properties: { n: { type: 'number', description: 'Number of recent checks (default 10)', default: 10 } }
    }
  },
  {
    name: 'tetramem_duplicates',
    description: 'Detect duplicate memories using Jaccard token overlap. Returns pairs with similarity >= 70%.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_isolated',
    description: 'Detect isolated memory nodes — occupied nodes with no occupied neighbors. Indicates connectivity issues.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_topology_health',
    description: 'Get topology health report equivalent to stats.',
    inputSchema: { type: 'object', properties: {} }
  },
  {
    name: 'tetramem_health',
    description: 'Get API health check: version, uptime.',
    inputSchema: { type: 'object', properties: {} }
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
          content: args.content, labels: args.labels || [],
          weight: args.weight || 1.0, metadata: args.metadata
        });
        return { content: [{ type: 'text', text: `Stored in TetraMem-XL (id: ${result.id}). Memory is eternal and will never be deleted.` }] };

      case 'tetramem_query':
        result = await callApi('/api/v1/query', 'POST', {
          query: args.query, k: args.k || 5, labels: args.labels
        });
        return { content: [{ type: 'text', text: JSON.stringify(result.results.map(r => ({
          id: r.id, content: r.content, distance: r.distance, weight: r.weight, labels: r.labels
        })), null, 2) }] };

      case 'tetramem_associate':
        result = await callApi('/api/v1/associate', 'POST', {
          tetra_id: args.tetra_id, max_depth: args.max_depth || 2
        });
        return { content: [{ type: 'text', text: JSON.stringify(result.associations, null, 2) }] };

      case 'tetramem_timeline':
        result = await callApi('/api/v1/timeline', 'POST', {
          direction: args.direction || 'newest', limit: args.limit || 20,
          labels: args.labels, min_weight: args.min_weight || 0.0
        });
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_list':
        result = await callApi('/api/v1/tetrahedra');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_get':
        result = await callApi(`/api/v1/tetrahedra/${args.id}`);
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_dream':
        result = await callApi('/api/v1/dream', 'POST', { force: args.force || false });
        return { content: [{ type: 'text', text: `Dream cycle: ${JSON.stringify(result.result)}` }] };

      case 'tetramem_self_organize':
        result = await callApi('/api/v1/self-organize', 'POST', {});
        return { content: [{ type: 'text', text: `Self-organize: ${JSON.stringify(result.stats)}` }] };

      case 'tetramem_closed_loop':
        result = await callApi('/api/v1/closed-loop', 'POST', {});
        return { content: [{ type: 'text', text: `Closed loop: ${JSON.stringify(result.result, null, 2)}` }] };

      case 'tetramem_stats':
        result = await callApi('/api/v1/stats');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_export':
        result = await callApi('/api/v1/export', 'POST', {});
        return { content: [{ type: 'text', text: `Exported ${result.size} bytes to ${result.path}` }] };

      case 'tetramem_topology':
        result = await callApi('/api/v1/topology-graph');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_lattice_info':
        result = await callApi('/api/v1/lattice-info');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_pulse_snapshot':
        result = await callApi('/api/v1/pulse-snapshot');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_pulse_status':
        result = await callApi('/api/v1/pulse-status');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_tension':
        result = await callApi('/api/v1/tension');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_tension_detailed':
        result = await callApi('/api/v1/pcnn/tension-map');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_pcnn_states':
        result = await callApi('/api/v1/pcnn/states');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_hebbian':
        result = await callApi('/api/v1/pcnn/hebbian');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_pcnn_config':
        result = await callApi('/api/v1/pcnn/config');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_phase_status':
        result = await callApi('/api/v1/phase-transition/status');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_phase_trigger':
        result = await callApi('/api/v1/phase-transition/trigger', 'POST', {});
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_self_check_status':
        result = await callApi('/api/v1/self-check/status');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_self_check_run':
        result = await callApi('/api/v1/self-check/run', 'POST', {});
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_self_check_history':
        result = await callApi(`/api/v1/self-check/history?n=${args.n || 10}`);
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_duplicates':
        result = await callApi('/api/v1/duplicates');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_isolated':
        result = await callApi('/api/v1/isolated');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_topology_health':
        result = await callApi('/api/v1/topology-health');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

      case 'tetramem_health':
        result = await callApi('/api/v1/health');
        return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };

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
  console.error('TetraMem-XL MCP server v4.1 started', { url: TETRA_API_URL });
}

main().catch((error) => {
  console.error('Server failed:', error);
  process.exit(1);
});
