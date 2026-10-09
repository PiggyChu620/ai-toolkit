import fs from 'fs';
import path from 'path';

import { Job } from '@prisma/client';

import prisma from './prisma';

const ACTIVE_JOB_STATUSES = new Set<Job['status']>(['running', 'stopping']);
const PROCESS_START_GRACE_MS = 2 * 60 * 1000;

export const isProcessAlive = (pid: number) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};

const getPidPath = (trainingRoot: string, jobName: string) =>
  path.join(trainingRoot, jobName, 'pid.txt');

const readPid = (pidPath: string): number | null => {
  if (!fs.existsSync(pidPath)) {
    return null;
  }

  const rawPid = fs.readFileSync(pidPath, 'utf-8').trim();
  const parsedPid = Number.parseInt(rawPid, 10);
  return Number.isFinite(parsedPid) && parsedPid > 0 ? parsedPid : null;
};

const cleanupPidFile = (pidPath: string) => {
  try {
    if (fs.existsSync(pidPath)) {
      fs.unlinkSync(pidPath);
    }
  } catch (error) {
    console.error(`Failed to remove stale pid file ${pidPath}`, error);
  }
};

const isWithinStartGrace = (job: Job) => {
  if (job.status !== 'running' || job.info !== 'Starting job...') {
    return false;
  }

  const updatedAt = new Date(job.updated_at).getTime();
  return Number.isFinite(updatedAt) && Date.now() - updatedAt < PROCESS_START_GRACE_MS;
};

export async function reconcileJobProcessState(job: Job, trainingRoot: string): Promise<Job> {
  if (!ACTIVE_JOB_STATUSES.has(job.status)) {
    return job;
  }

  const pidPath = getPidPath(trainingRoot, job.name);
  const pid = readPid(pidPath);
  if (pid === null && isWithinStartGrace(job)) {
    return job;
  }
  if (pid !== null && isProcessAlive(pid)) {
    return job;
  }

  let status: Job['status'];
  let info: string;
  let stop = job.stop;
  let returnToQueue = job.return_to_queue;

  if (job.stop || job.status === 'stopping') {
    // an explicit Stop wins over a queue pause
    status = 'stopped';
    info = 'Job stopped';
    returnToQueue = false;
  } else if (job.return_to_queue) {
    status = 'queued';
    info = 'Job queued after worker exit';
    stop = false;
    returnToQueue = false;
  } else {
    status = 'error';
    info = 'Job process exited unexpectedly';
    stop = false;
  }

  const updatedJob = await prisma.job.update({
    where: { id: job.id },
    data: {
      status,
      info,
      stop,
      return_to_queue: returnToQueue,
      speed_string: '',
      cache_speed_string: '',
    },
  });

  cleanupPidFile(pidPath);
  return updatedJob;
}

export async function reconcileJobProcessStates(jobs: Job[], trainingRoot: string): Promise<Job[]> {
  const reconciledJobs: Job[] = [];
  for (const job of jobs) {
    reconciledJobs.push(await reconcileJobProcessState(job, trainingRoot));
  }
  return reconciledJobs;
}
