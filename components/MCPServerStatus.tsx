import React, { useEffect, useState } from 'react';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import { Button } from './ui/button';
import { toast } from './ui/use-toast';

interface ServerConfig {
  command: string;
  args: string[];
  env: {
    PYTHONPATH: string;
  };
}

interface MCPConfig {
  mcpServers: {
    'fast-markdown': ServerConfig;
  };
}

interface ServerStatus {
  status: 'running' | 'stopped' | 'partial' | 'error';
  pid?: number;
  details?: string;
}

const MCPServerStatus = () => {
  const [status, setStatus] = useState<ServerStatus>({ status: 'stopped' });
  const [logs, setLogs] = useState<string[]>([]);
  const [config, setConfig] = useState<MCPConfig | null>(null);

  useEffect(() => {
    // Check if MCP server is enabled
    const mcpEnabled = process.env.NEXT_PUBLIC_ENABLE_MCP_SERVER === 'true';
    if (!mcpEnabled) {
      setStatus({ status: 'stopped', details: 'MCP server is disabled by configuration' });
      return;
    }

    // Use environment variables or fallback to default paths
    const rootDir = process.env.NEXT_PUBLIC_ROOT_DIR || '/Users/cyberagi/Documents/GitHub/DevDocs';
    
    // Fetch the configuration from the backend
    const getConfig = async () => {
      try {
        const response = await fetch('http://localhost:24125/api/mcp/config');
        if (response.ok) {
          const data = await response.json();
          setConfig(data);
        }
      } catch (error) {
        console.error('Error fetching config:', error);
        // Fallback to default config if API fails
        const defaultConfig: MCPConfig = {
          mcpServers: {
            'fast-markdown': {
              command: `${rootDir}/fast-markdown-mcp/venv/bin/python`,
              args: [
                '-m', 'fast_markdown_mcp.server',
                `${rootDir}/storage/markdown`
              ],
              env: {
                PYTHONPATH: `${rootDir}/fast-markdown-mcp/src`
              }
            }
          }
        };
        setConfig(defaultConfig);
      }
    };

    const checkStatus = async () => {
      try {
        const response = await fetch('http://localhost:24125/api/mcp/status');
        const data = await response.json();
        setStatus({
          status: data.status,
          pid: data.pid,
          details: data.details
        });
      } catch (error) {
        setStatus({ 
          status: 'error',
          details: error instanceof Error ? error.message : 'Failed to check status'
        });
      }
    };

    const getLogs = async () => {
      try {
        const response = await fetch('http://localhost:24125/api/mcp/logs');
        if (response.ok) {
          const data = await response.json();
          setLogs(data.logs);
        }
      } catch (error) {
        console.error('Error fetching logs:', error);
      }
    };

    // Initial fetch
    getConfig();
    checkStatus();
    getLogs();

    // Set up polling
    const interval = setInterval(() => {
      checkStatus();
      getLogs();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getStatusBadge = () => {
    switch (status.status) {
      case 'running':
        return <Badge className="bg-green-500">●&nbsp;Running</Badge>;
      case 'partial':
        return <Badge className="bg-yellow-500">◐&nbsp;Partial</Badge>;
      case 'error':
        return <Badge variant="destructive">✕&nbsp;Error</Badge>;
      default:
        return <Badge variant="destructive">○&nbsp;Stopped</Badge>;
    }
  };

  const copyConfig = () => {
    if (config) {
      navigator.clipboard.writeText(JSON.stringify(config, null, 2))
        .then(() => {
          toast({
            title: "Configuration Copied",
            description: "Configuration copied to clipboard",
          });
        })
        .catch((error) => {
          console.error('Error copying config:', error);
          toast({
            title: "Error",
            description: "Failed to copy configuration",
            variant: "destructive",
          });
        });
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">MCP Server Status</h3>
        {getStatusBadge()}
      </div>

      {status.details && (
        <div className="text-sm text-gray-300 bg-gray-900/50 rounded-xl p-3 border border-gray-700/50">
          {status.details}
        </div>
      )}

      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-white">Configuration</h3>
          <Button
            variant="outline"
            size="sm"
            onClick={copyConfig}
            className="text-xs"
          >
            Copy Config
          </Button>
        </div>
        <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-700/50">
          <pre className="text-sm font-mono text-gray-300 whitespace-pre-wrap overflow-x-auto">
            {config ? JSON.stringify(config, null, 2) : 'Loading configuration...'}
          </pre>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-3 text-white">Server Logs</h3>
        <ScrollArea className="h-[200px] bg-gray-900/50 rounded-xl p-4 border border-gray-700/50">
          {logs.length > 0 ? (
            logs.map((log, index) => (
              <div key={index} className="text-sm text-gray-300 mb-1 font-mono">
                {log}
              </div>
            ))
          ) : (
            <div className="text-sm text-gray-500 italic">No logs available</div>
          )}
        </ScrollArea>
      </div>
    </div>
  );
};

export default MCPServerStatus;