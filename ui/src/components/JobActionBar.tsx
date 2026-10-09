import Link from 'next/link';
import { Eye, Trash2, Pen, Play, Pause, Cog, X, RotateCcw, CircleStop } from 'lucide-react';
import { Button } from '@headlessui/react';
import { openConfirm } from '@/components/ConfirmModal';
import { Job } from '@prisma/client';
import { startJob, stopJob, restartJob, deleteJob, getAvaliableJobActions, markJobAsStopped } from '@/utils/jobs';
import { startQueue } from '@/utils/queue';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';

interface JobActionBarProps {
  job: Job;
  onRefresh?: () => void;
  afterDelete?: () => void;
  hideView?: boolean;
  className?: string;
  autoStartQueue?: boolean;
}

export default function JobActionBar({
  job,
  onRefresh,
  afterDelete,
  className,
  hideView,
  autoStartQueue = false,
}: JobActionBarProps) {
  const { canStart, canStop, canStopNow, canDelete, canEdit, canRestart, canRemoveFromQueue } =
    getAvaliableJobActions(job);

  if (!afterDelete) afterDelete = onRefresh;

  return (
    <div className={`${className}`}>
      {canStart && (
        <Button
          onClick={async () => {
            if (!canStart) return;
            await startJob(job.id);
            // start the queue as well
            if (autoStartQueue) {
              await startQueue(job.gpu_ids);
            }
            if (onRefresh) onRefresh();
          }}
          className={`ml-2 opacity-100`}
        >
          <Play />
        </Button>
      )}
      {canRemoveFromQueue && (
        <Button
          onClick={async () => {
            if (!canRemoveFromQueue) return;
            await markJobAsStopped(job.id);
            if (onRefresh) onRefresh();
          }}
          className={`ml-2 opacity-100`}
        >
          <X />
        </Button>
      )}
      {canStop && (
        <Button
          onClick={() => {
            if (!canStop) return;
            openConfirm({
              title: 'Stop Job',
              message: `Are you sure you want to stop the job "${job.name}"? You CAN resume later.`,
              type: 'info',
              confirmText: 'Stop',
              onConfirm: async () => {
                await stopJob(job.id);
                if (onRefresh) onRefresh();
              },
            });
          }}
          className={`ml-2 opacity-100`}
        >
          <Pause />
        </Button>
      )}
      {canStopNow && (
        <Button
          onClick={() => {
            if (!canStopNow) return;
            openConfirm({
              title: 'Stop Now',
              message: `"${job.name}" is saving the step it was on before it stops. Stop it right now instead? The training since its last saved checkpoint will be lost.`,
              type: 'warning',
              confirmText: 'Stop now',
              onConfirm: async () => {
                await stopJob(job.id, true);
                if (onRefresh) onRefresh();
              },
            });
          }}
          className={`ml-2 opacity-100 text-red-400`}
          title="Stop now, without saving"
        >
          <CircleStop />
        </Button>
      )}
      {canRestart && (
        <Button
          onClick={() => {
            if (!canRestart) return;
            openConfirm({
              title: 'Restart Job',
              message: `Are you sure you want to restart the job "${job.name}" from step 0? This will permanently remove its current checkpoints, optimizer state, samples, and logs before queueing it again.`,
              type: 'warning',
              confirmText: 'Restart',
              onConfirm: async () => {
                await restartJob(job.id);
                if (autoStartQueue) {
                  await startQueue(job.gpu_ids);
                }
                if (onRefresh) onRefresh();
              },
            });
          }}
          className={`ml-2 opacity-100`}
          title="Restart from step 0"
        >
          <RotateCcw />
        </Button>
      )}
      {!hideView && (
        <Link href={`/jobs/${job.id}`} className="ml-2 text-gray-200 hover:text-gray-100 inline-block">
          <Eye />
        </Link>
      )}
      {canEdit && (
        <Link href={`/jobs/new?id=${job.id}`} className="ml-2 hover:text-gray-100 inline-block">
          <Pen />
        </Link>
      )}
      <Button
        onClick={() => {
          let message = `Are you sure you want to delete the job "${job.name}"? This will also permanently remove it from your disk.`;
          if (job.status === 'running') {
            message += ' WARNING: The job is currently running. You should stop it first if you can.';
          }
          openConfirm({
            title: 'Delete Job',
            message: message,
            type: 'warning',
            confirmText: 'Delete',
            onConfirm: async () => {
              if (job.status === 'running') {
                try {
                  await stopJob(job.id, true); // it is being deleted: no point saving it first
                } catch (e) {
                  console.error('Error stopping job before deleting:', e);
                }
              }
              await deleteJob(job.id);
              if (afterDelete) afterDelete();
            },
          });
        }}
        className={`ml-2 opacity-100`}
      >
        <Trash2 />
      </Button>
      <div className="border-r border-1 border-gray-700 ml-2 inline"></div>
      <Menu>
        <MenuButton className={'ml-2'}>
          <Cog />
        </MenuButton>
        <MenuItems anchor="bottom" className="bg-gray-900 border border-gray-700 rounded shadow-lg w-48 px-2 py-2 mt-4">
          <MenuItem>
            <Link href={`/jobs/new?cloneId=${job.id}`} className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded block">
              Clone Job
            </Link>
          </MenuItem>
          <MenuItem>
            <div
              className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded"
              onClick={() => {
                let message = `Are you sure you want to mark this job as stopped? This will set the job status to 'stopped' if the status is hung. Only do this if you are 100% sure the job is stopped. This will NOT stop the job.`;
                openConfirm({
                  title: 'Mark Job as Stopped',
                  message: message,
                  type: 'warning',
                  confirmText: 'Mark as Stopped',
                  onConfirm: async () => {
                    await markJobAsStopped(job.id);
                    onRefresh && onRefresh();
                  },
                });
              }}
            >
              Mark as Stopped
            </div>
          </MenuItem>
        </MenuItems>
      </Menu>
    </div>
  );
}
