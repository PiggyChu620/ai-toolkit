import { JobConfig } from '@/types';
import { isSavingBeforeStop } from '@/utils/stopping';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';

export const startJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/start`)
      .then(res => res.data)
      .then(data => {
        console.log('Job started:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error starting job:', error);
        reject(error);
      });
  });
};

// force: stop at once, without letting the trainer save the step it is on first
export const stopJob = (jobID: string, force: boolean = false) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/stop${force ? '?force=1' : ''}`)
      .then(res => res.data)
      .then(data => {
        console.log('Job stopped:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error stopping job:', error);
        reject(error);
      });
  });
};

export const restartJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/restart`)
      .then(res => res.data)
      .then(data => {
        console.log('Job restarted:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error restarting job:', error);
        reject(error);
      });
  });
};

export const deleteJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/delete`)
      .then(res => res.data)
      .then(data => {
        console.log('Job deleted:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error deleting job:', error);
        reject(error);
      });
  });
};

export const markJobAsStopped = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/mark_stopped`)
      .then(res => res.data)
      .then(data => {
        console.log('Job marked as stopped:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error marking job as stopped:', error);
        reject(error);
      });
  });
};

export const getJobConfig = (job: Job) => {
  return JSON.parse(job.job_config) as JobConfig;
};

export const getAvaliableJobActions = (job: Job) => {
  const jobConfig = getJobConfig(job);
  const isStopping = job.stop && job.status === 'running';
  const canDelete = ['queued', 'completed', 'stopped', 'error'].includes(job.status) && !isStopping;
  const canEdit = ['queued','completed', 'stopped', 'error'].includes(job.status) && !isStopping;
  const canRestart = ['queued', 'completed', 'stopped', 'error'].includes(job.status) && !isStopping;
  const canRemoveFromQueue = job.status === 'queued';
  const canStop = job.status === 'running' && !isStopping;
  // saving the step it was on before stopping: offer to skip that and stop at once
  const canStopNow = isStopping && isSavingBeforeStop(job.info);
  let canStart = ['stopped', 'error'].includes(job.status) && !isStopping;
  // can resume if more steps were added
  if (job.status === 'completed' && jobConfig.config.process[0].train.steps > job.step && !isStopping) {
    canStart = true;
  }
  return { canDelete, canEdit, canStop, canStopNow, canStart, canRestart, canRemoveFromQueue };
};

export const getNumberOfSamples = (job: Job) => {
  const jobConfig = getJobConfig(job);
  return jobConfig.config.process[0].sample?.prompts?.length || 0;
};

export const getTotalSteps = (job: Job) => {
  // Prefer the runtime-resolved target the trainer reports to the DB. It reflects
  // Train-by-Epoch and train-on-top-of-loaded-LoRA, which only resolve once the
  // dataset is loaded; the static config steps does not. Fall back to the config
  // for jobs that have not reported a resolved total yet (total_steps defaults to 0).
  if (job.total_steps && job.total_steps > 0) {
    return job.total_steps;
  }
  const jobConfig = getJobConfig(job);
  return jobConfig.config.process[0].train.steps;
};
