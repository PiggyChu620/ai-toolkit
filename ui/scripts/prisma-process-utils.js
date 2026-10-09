const { spawnSync } = require('node:child_process');

function runCapture(command, args) {
  return spawnSync(command, args, {
    cwd: process.cwd(),
    shell: true,
    encoding: 'utf8',
  });
}

function runCommand(command, args) {
  const result = spawnSync(command, args, {
    cwd: process.cwd(),
    stdio: 'inherit',
    shell: true,
  });

  return result.status ?? 1;
}

function isLockedPrismaEngineError(output) {
  if (!output) {
    return false;
  }

  const normalized = String(output).toLowerCase();
  return (
    normalized.includes('eperm: operation not permitted, rename') &&
    normalized.includes('query_engine-windows.dll.node')
  );
}

function normalizeForCompare(value) {
  return String(value || '')
    .replace(/\//g, '\\')
    .toLowerCase();
}

function getNodeProcesses() {
  if (process.platform !== 'win32') {
    return [];
  }

  const script = [
    '$ErrorActionPreference = "Stop"',
    '$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "node.exe" } | Select-Object ProcessId, CommandLine',
    '$procs | ConvertTo-Json -Compress',
  ].join('; ');

  const result = spawnSync('powershell.exe', ['-NoProfile', '-Command', script], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });

  if ((result.status ?? 1) !== 0 || !result.stdout || !result.stdout.trim()) {
    return [];
  }

  try {
    const parsed = JSON.parse(result.stdout.trim());
    if (Array.isArray(parsed)) {
      return parsed;
    }
    return parsed ? [parsed] : [];
  } catch {
    return [];
  }
}

function findBlockingUiWorkerProcesses() {
  const cwdNorm = normalizeForCompare(process.cwd());

  const processes = getNodeProcesses();
  const uiProcesses = [];
  const workerProcesses = [];

  for (const proc of processes) {
    const commandLine = normalizeForCompare(proc.CommandLine);
    if (!commandLine) {
      continue;
    }

    const isUiProcess =
      commandLine.includes(cwdNorm) &&
      commandLine.includes('next\\dist\\bin\\next') &&
      (commandLine.includes(' start --port 8675') ||
        commandLine.includes(' dev --turbopack') ||
        commandLine.includes(' next start --port 8675') ||
        commandLine.includes(' next dev --turbopack'));

    if (isUiProcess) {
      uiProcesses.push({
        pid: proc.ProcessId,
        role: 'UI',
        commandLine: proc.CommandLine || '',
      });
      continue;
    }

    const isWorkerProcess =
      commandLine.includes('dist\\cron\\worker.js') ||
      commandLine.includes('cron\\worker.ts') ||
      commandLine.includes('tsconfig.worker.json');

    if (!isWorkerProcess) {
      continue;
    }

    const looksLikeThisWorkspaceWorker =
      commandLine.includes(cwdNorm) ||
      commandLine.trim() === 'node  dist\\cron\\worker.js' ||
      commandLine.trim() === 'node dist\\cron\\worker.js';

    if (looksLikeThisWorkspaceWorker) {
      workerProcesses.push({
        pid: proc.ProcessId,
        role: 'Worker',
        commandLine: proc.CommandLine || '',
      });
    }
  }

  if (uiProcesses.length > 0 && workerProcesses.length === 0) {
    for (const proc of processes) {
      const commandLine = normalizeForCompare(proc.CommandLine);
      if (
        commandLine.trim() === 'node  dist\\cron\\worker.js' ||
        commandLine.trim() === 'node dist\\cron\\worker.js'
      ) {
        workerProcesses.push({
          pid: proc.ProcessId,
          role: 'Worker',
          commandLine: proc.CommandLine || '',
        });
      }
    }
  }

  const combined = [...uiProcesses, ...workerProcesses];
  const seen = new Set();
  return combined.filter(proc => {
    if (seen.has(proc.pid)) {
      return false;
    }
    seen.add(proc.pid);
    return true;
  });
}

function terminateProcesses(processes) {
  const uniqueProcesses = [];
  const seen = new Set();
  for (const proc of processes) {
    if (!proc || seen.has(proc.pid)) {
      continue;
    }
    seen.add(proc.pid);
    uniqueProcesses.push(proc);
  }

  for (const proc of uniqueProcesses) {
    spawnSync('taskkill', ['/PID', String(proc.pid), '/T', '/F'], {
      cwd: process.cwd(),
      stdio: 'ignore',
      shell: true,
    });
  }
}

function sleepMs(milliseconds) {
  const script = `Start-Sleep -Milliseconds ${Math.max(0, Math.floor(milliseconds))}`;
  spawnSync('powershell.exe', ['-NoProfile', '-Command', script], {
    cwd: process.cwd(),
    stdio: 'ignore',
  });
}

module.exports = {
  findBlockingUiWorkerProcesses,
  isLockedPrismaEngineError,
  runCapture,
  runCommand,
  sleepMs,
  terminateProcesses,
};
