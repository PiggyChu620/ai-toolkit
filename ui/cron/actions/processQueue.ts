import prisma from '../prisma';
import { getTrainingFolder } from '../paths';

import { Job, Queue } from '@prisma/client';
import startJob from './startJob';
import { reconcileJobProcessStates } from '../../src/server/jobProcess';

export default async function processQueue() {
  const queues: Queue[] = await prisma.queue.findMany({
    orderBy: {
      id: 'asc',
    },
  });
  const trainingRoot = await getTrainingFolder();

  for (const queue of queues) {
    const activeJobs = await prisma.job.findMany({
      where: {
        status: { in: ['running', 'stopping'] },
        gpu_ids: queue.gpu_ids,
      },
    });
    const reconciledActiveJobs = await reconcileJobProcessStates(activeJobs, trainingRoot);

    if (!queue.is_running) {
      // stop any running jobs first
      // only once per job: the trainer may be reporting that it is saving before it stops
      // (and not over an explicit Stop already in progress)
      const runningJobs = reconciledActiveJobs.filter(
        job => job.status === 'running' && !job.return_to_queue && !job.stop,
      );

      for (const job of runningJobs) {
        console.log(`Stopping job ${job.id} on GPU(s) ${job.gpu_ids}`);
        await prisma.job.update({
          where: { id: job.id },
          data: {
            return_to_queue: true,
            info: 'Stopping job...',
          },
        });
      }
    }
    if (queue.is_running) {
      // first see if one is already running, status of running or stopping
      const runningJob = reconciledActiveJobs.find(job => ['running', 'stopping'].includes(job.status)) ?? null;

      if (runningJob) {
        // already running, nothing to do
        continue; // skip to next queue
      } else {
        // find the next job in the queue
        const nextJob: Job | null = await prisma.job.findFirst({
          where: {
            status: 'queued',
            gpu_ids: queue.gpu_ids,
          },
          orderBy: {
            queue_position: 'asc',
          },
        });
        if (nextJob) {
          console.log(`Starting job ${nextJob.id} on GPU(s) ${nextJob.gpu_ids}`);
          await startJob(nextJob.id);
        } else {
          // no more jobs, stop the queue
          console.log(`No more jobs in queue for GPU(s) ${queue.gpu_ids}, stopping queue`);
          await prisma.queue.update({
            where: { id: queue.id },
            data: { is_running: false },
          });
        }
      }
    }
  }
}
