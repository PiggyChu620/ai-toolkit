import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';
import path from 'path';
import fs from 'fs';
import { readJobPid, isProcessAlive, processBelongsToJob, removeDirWithRetry, cleanupPidFile } from '@/server/processControl';

export async function GET(request: NextRequest, context: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await context.params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingRoot = await getTrainingFolder();
  const trainingFolder = path.join(trainingRoot, job.name);

  // Restart wipes the folder back to step 0. Refuse while a real training
  // process is still alive — otherwise the wipe EPERMs on Windows and orphans
  // the process. Validated against pid.txt so a recycled/dead PID with a stale
  // 'running' status doesn't block a legitimate restart.
  const pid = readJobPid(trainingRoot, job.name);
  if (pid !== null && isProcessAlive(pid) && processBelongsToJob(pid, trainingRoot, job.name)) {
    return NextResponse.json(
      { error: 'Stop the job before restarting it from step 0' },
      { status: 409 },
    );
  }
  cleanupPidFile(trainingRoot, job.name);

  try {
    if (fs.existsSync(trainingFolder)) {
      await removeDirWithRetry(trainingFolder);
    }
  } catch (error) {
    console.error(`Failed to clear job folder for restart ${jobID}`, error);
    return NextResponse.json(
      { error: 'Failed to clear job folder (a file may still be in use). Try again in a moment.' },
      { status: 500 },
    );
  }

  const highestQueuePosition = await prisma.job.aggregate({
    _max: {
      queue_position: true,
    },
  });
  const newQueuePosition = (highestQueuePosition._max.queue_position || 0) + 1000;

  const queue = await prisma.queue.findFirst({
    where: {
      gpu_ids: job.gpu_ids,
    },
  });

  if (!queue) {
    await prisma.queue.create({
      data: {
        gpu_ids: job.gpu_ids,
        is_running: true,
      },
    });
  } else if (!queue.is_running) {
    await prisma.queue.update({
      where: { id: queue.id },
      data: { is_running: true },
    });
  }

  const restartedJob = await prisma.job.update({
    where: { id: jobID },
    data: {
      queue_position: newQueuePosition,
      status: 'queued',
      stop: false,
      return_to_queue: false,
      step: 0,
      info: 'Job queued from step 0',
      speed_string: '',
      cache_speed_string: '',
    },
  });

  return NextResponse.json(restartedJob);
}
