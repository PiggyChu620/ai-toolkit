import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';

const isWindows = process.platform === 'win32';

export interface KillResult {
  killed: boolean;
  reason: 'killed' | 'not-alive' | 'ownership-unverified' | 'error';
}

export const getPidPath = (trainingRoot: string, jobName: string): string =>
  path.join(trainingRoot, jobName, 'pid.txt');

export const readJobPid = (trainingRoot: string, jobName: string): number | null => {
  const pidPath = getPidPath(trainingRoot, jobName);
  try {
    if (!fs.existsSync(pidPath)) {
      return null;
    }
    const rawPid = fs.readFileSync(pidPath, 'utf-8').trim();
    const parsedPid = Number.parseInt(rawPid, 10);
    return Number.isFinite(parsedPid) && parsedPid > 0 ? parsedPid : null;
  } catch {
    return null;
  }
};

export const cleanupPidFile = (trainingRoot: string, jobName: string): void => {
  const pidPath = getPidPath(trainingRoot, jobName);
  try {
    if (fs.existsSync(pidPath)) {
      fs.unlinkSync(pidPath);
    }
  } catch (error) {
    console.error(`Failed to remove pid file ${pidPath}`, error);
  }
};

export const isProcessAlive = (pid: number): boolean => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};

/** Best-effort read of a PID's full command line, or null if unavailable. */
export const getProcessCommandLine = (pid: number): string | null => {
  try {
    if (isWindows) {
      const out = execFileSync(
        'powershell.exe',
        [
          '-NoProfile',
          '-NonInteractive',
          '-Command',
          `(Get-CimInstance Win32_Process -Filter "ProcessId=${pid}" -ErrorAction SilentlyContinue).CommandLine`,
        ],
        { encoding: 'utf-8', windowsHide: true, timeout: 10000 },
      );
      const trimmed = (out ?? '').trim();
      return trimmed.length > 0 ? trimmed : null;
    }
    const out = execFileSync('ps', ['-o', 'command=', '-p', String(pid)], {
      encoding: 'utf-8',
      timeout: 10000,
    });
    const trimmed = (out ?? '').trim();
    return trimmed.length > 0 ? trimmed : null;
  } catch {
    return null;
  }
};

/**
 * Verify the PID really is THIS job's training process before we kill it.
 * Windows recycles PIDs aggressively, so a stale pid.txt can point at an
 * unrelated process. The trainer is always launched with its job folder's
 * `.job_config.json` (and run.py) on the command line — see startJob.ts — so we
 * require both to appear before trusting the PID.
 */
export const processBelongsToJob = (pid: number, trainingRoot: string, jobName: string): boolean => {
  const cmd = getProcessCommandLine(pid);
  if (!cmd) {
    return false;
  }
  const cmdNorm = cmd.toLowerCase().replace(/\\/g, '/');
  if (!cmdNorm.includes('run.py')) {
    return false;
  }
  const jobFolderNorm = path
    .join(trainingRoot, jobName)
    .toLowerCase()
    .replace(/\\/g, '/');
  // Require a trailing separator so a job named "test" doesn't match "test2"'s
  // command line (substring collision on prefix-named jobs).
  return cmdNorm.includes(`${jobFolderNorm}/`);
};

const rawForceKillProcessTree = (pid: number): void => {
  if (isWindows) {
    execFileSync('taskkill', ['/PID', String(pid), '/T', '/F'], {
      stdio: 'ignore',
      windowsHide: true,
    });
    return;
  }
  try {
    process.kill(-pid, 'SIGKILL');
    return;
  } catch {}
  process.kill(pid, 'SIGKILL');
};

/**
 * Force-kill the process tree, but ONLY after confirming the PID still belongs
 * to this job. Refuses (without killing) if ownership can't be verified, so we
 * never kill a recycled/foreign PID.
 */
export const killJobProcessIfOwned = (pid: number, trainingRoot: string, jobName: string): KillResult => {
  if (!isProcessAlive(pid)) {
    return { killed: false, reason: 'not-alive' };
  }
  if (!processBelongsToJob(pid, trainingRoot, jobName)) {
    console.warn(
      `Refusing to kill pid ${pid}: it does not look like job "${jobName}"'s training process ` +
        `(possible recycled PID). Leaving it alone.`,
    );
    return { killed: false, reason: 'ownership-unverified' };
  }
  try {
    rawForceKillProcessTree(pid);
    return { killed: true, reason: 'killed' };
  } catch (error) {
    console.error(`Failed to kill job "${jobName}" pid ${pid}`, error);
    return { killed: false, reason: 'error' };
  }
};

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/** Poll until the process is gone or the timeout elapses. Returns true if gone. */
export const waitForProcessExit = async (pid: number, timeoutMs: number, pollMs = 250): Promise<boolean> => {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!isProcessAlive(pid)) {
      return true;
    }
    await sleep(pollMs);
  }
  return !isProcessAlive(pid);
};

/**
 * Make sure a job has no live process before we wipe/overwrite its folder. Sets
 * the cooperative DB stop flag first (caller's job), waits, then validated-kills
 * as a last resort. Returns true if no owned process is alive afterwards.
 */
export const ensureJobProcessStopped = async (
  trainingRoot: string,
  jobName: string,
  graceMs = 8000,
): Promise<boolean> => {
  const pid = readJobPid(trainingRoot, jobName);
  if (pid === null) {
    return true;
  }
  if (!isProcessAlive(pid)) {
    return true;
  }
  // Give a cooperative stop a brief chance, then validated force-kill.
  await waitForProcessExit(pid, graceMs);
  if (!isProcessAlive(pid)) {
    return true;
  }
  const result = killJobProcessIfOwned(pid, trainingRoot, jobName);
  if (result.reason === 'ownership-unverified') {
    // Not ours — treat the (foreign) PID as "not our problem"; the folder wipe
    // below may still fail if something really holds a handle, which the caller
    // handles via retry.
    return true;
  }
  await waitForProcessExit(pid, 5000);
  return !isProcessAlive(pid);
};

/**
 * Recursively remove a directory, retrying a few times. On Windows a just-killed
 * process can briefly hold file handles, causing EPERM/EBUSY on the first try.
 */
export const removeDirWithRetry = async (dir: string, attempts = 5, delayMs = 300): Promise<void> => {
  for (let i = 0; i < attempts; i++) {
    try {
      if (!fs.existsSync(dir)) {
        return;
      }
      fs.rmSync(dir, { recursive: true, force: true });
      return;
    } catch (error) {
      if (i === attempts - 1) {
        throw error;
      }
      // let any just-released file handles settle before retrying (non-blocking)
      await sleep(delayMs);
    }
  }
};
