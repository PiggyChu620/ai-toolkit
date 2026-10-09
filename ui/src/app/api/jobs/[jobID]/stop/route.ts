import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';
import fs from 'fs';
import path from 'path';
import { isProcessAlive, readJobPid, killJobProcessIfOwned } from '@/server/processControl';
import { isSavingBeforeStop } from '@/utils/stopping';

// How long to wait for the trainer to honor the cooperative DB `stop` flag and
// exit at a step boundary before force-killing. Force-killing mid-step can
// corrupt an in-progress checkpoint, so this is deliberately generous. Raise
// AITK_STOP_GRACE_MS for very long video / large-model steps.
const STOP_GRACE_MS = (() => {
  const parsed = Number.parseInt(process.env.AITK_STOP_GRACE_MS ?? '', 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 30000;
})();
// A trainer stopped during training first finishes and saves the step it is on (it reports
// "Saving before stopping..." in job.info); that takes a step plus a checkpoint write, so the
// force-kill then waits this long instead. Longer than the trainer's own limit (20 min).
const STOP_SAVE_MAX_MS = (() => {
  const parsed = Number.parseInt(process.env.AITK_STOP_SAVE_MAX_MS ?? '', 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 25 * 60 * 1000;
})();
const STOP_POLL_MS = 500;

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const sendSoftStopSignal = (pid: number) => {
  // Cooperative stop. On Windows the trainer cooperates via the DB `stop` flag
  // (set below); POSIX signals to a detached child are unreliable there.
  if (process.platform === 'win32') {
    return;
  }
  try {
    process.kill(-pid, 'SIGINT');
    return;
  } catch {}
  try {
    process.kill(pid, 'SIGINT');
  } catch {}
};

const forceKill = (jobID: string, pid: number, trainingRoot: string, jobName: string) => {
  // validated force-kill, so a recycled PID can't make us kill an unrelated process
  const result = killJobProcessIfOwned(pid, trainingRoot, jobName);
  if (!result.killed && result.reason === 'ownership-unverified') {
    console.warn(`Job ${jobID}: stop could not verify pid ${pid}; not force-killing.`);
  }
};

// one last-resort watchdog per job while its trainer saves before stopping
const savingWatchdogs: Set<string> =
  ((globalThis as any).__aitkSavingStopWatchdogs ??= new Set<string>()) as Set<string>;

const watchSavingStop = async (jobID: string, pid: number, trainingRoot: string, jobName: string) => {
  if (savingWatchdogs.has(jobID)) return;
  savingWatchdogs.add(jobID);
  try {
    const deadline = Date.now() + STOP_SAVE_MAX_MS;
    while (Date.now() < deadline && isProcessAlive(pid)) {
      await sleep(2000);
    }
    if (isProcessAlive(pid)) {
      console.warn(`Job ${jobID}: still saving before stopping after ${STOP_SAVE_MAX_MS} ms; force-killing.`);
      forceKill(jobID, pid, trainingRoot, jobName);
    }
  } finally {
    savingWatchdogs.delete(jobID);
  }
};

export async function GET(request: NextRequest, context: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await context.params;
  // ?force=1: stop right away, without letting the trainer save the step it is on
  const force = request.nextUrl.searchParams.get('force') === '1';

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  if (job.status !== 'running' && job.status !== 'stopping') {
    if (job.status === 'queued') {
      // e.g. the trainer sent the job back to the queue just as Stop was pressed: honor the Stop,
      // or the queue would start the job again
      const stopped = await prisma.job.update({
        where: { id: jobID },
        data: { status: 'stopped', stop: false, return_to_queue: false, info: 'Job stopped' },
      });
      return NextResponse.json(stopped);
    }
    // already stopped / completed / errored: keep the final status the trainer wrote
    return NextResponse.json(job);
  }

  // Cooperative stop signal: the trainer polls this flag and stops at the next
  // safe step boundary (flushing / saving as configured). Keep its "saving" text if it is
  // already saving before a stop (e.g. the queue was paused first).
  const alreadySaving = !force && isSavingBeforeStop(job.info);
  await prisma.job.update({
    where: { id: jobID },
    data: alreadySaving
      ? { stop: true }
      : {
          stop: true,
          info: force ? 'Stopping job now...' : 'Stopping job...',
        },
  });

  const trainingRoot = await getTrainingFolder();
  const pidPath = path.join(trainingRoot, job.name, 'pid.txt');
  const pid = readJobPid(trainingRoot, job.name);

  let processExited = false;
  let savingBeforeStop = false;

  if (pid !== null) {
    if (!isProcessAlive(pid)) {
      processExited = true;
    } else if (force) {
      forceKill(jobID, pid, trainingRoot, job.name);
      const deadline = Date.now() + 5000;
      while (Date.now() < deadline && isProcessAlive(pid)) {
        await sleep(STOP_POLL_MS);
      }
      processExited = !isProcessAlive(pid);
    } else {
      const deadline = Date.now() + STOP_GRACE_MS;
      while (Date.now() < deadline) {
        if (!isProcessAlive(pid)) {
          break;
        }
        const current = await prisma.job.findUnique({ where: { id: jobID }, select: { info: true } });
        if (isSavingBeforeStop(current?.info)) {
          savingBeforeStop = true;
          break;
        }
        await sleep(STOP_POLL_MS);
      }

      if (savingBeforeStop) {
        // it stops by itself once the step is saved; force-kill only if that never happens
        void watchSavingStop(jobID, pid, trainingRoot, job.name);
      } else if (isProcessAlive(pid)) {
        // no response to the DB flag: a soft signal (POSIX only; Windows has none), then the force-kill
        if (process.platform !== 'win32') {
          sendSoftStopSignal(pid);
          const softDeadline = Date.now() + 3000;
          while (Date.now() < softDeadline && isProcessAlive(pid)) {
            await sleep(STOP_POLL_MS);
          }
        }
        if (isProcessAlive(pid)) {
          // Last resort after the cooperative grace
          forceKill(jobID, pid, trainingRoot, job.name);
        }
      }

      processExited = !isProcessAlive(pid);
    }
  }

  // a trainer that exited by itself already set its final status ("Job stopped (saved at step N)",
  // or "queued"); keep that rather than overwriting it
  const exitedJob = processExited ? await prisma.job.findUnique({ where: { id: jobID } }) : null;
  const trainerSetStatus = exitedJob !== null && ['stopped', 'queued'].includes(exitedJob.status);
  const stoppedJob = processExited
    ? await prisma.job.update({
        where: { id: jobID },
        data: trainerSetStatus
          ? { speed_string: '', cache_speed_string: '' }
          : {
              status: 'stopped',
              info: 'Job stopped',
              speed_string: '',
              cache_speed_string: '',
            },
      })
    : await prisma.job.findUnique({
        where: { id: jobID },
      });

  try {
    if (pid !== null && !isProcessAlive(pid) && fs.existsSync(pidPath)) {
      fs.unlinkSync(pidPath);
    }
  } catch (error) {
    console.error(`Failed to clean pid file for job ${jobID}`, error);
  }

  if (!processExited) {
    return NextResponse.json(
      {
        ...(stoppedJob ?? job),
        info: savingBeforeStop ? (stoppedJob?.info ?? 'Stopping job...') : 'Stopping job...',
      },
      { status: 202 },
    );
  }

  return NextResponse.json(stoppedJob);
}
