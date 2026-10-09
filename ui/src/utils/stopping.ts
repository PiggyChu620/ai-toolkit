// While a stopped job saves the step it was training, the trainer reports this in job.info
// (DiffusionTrainer.SAVING_BEFORE_STOP has the same text). The stop route then holds off its
// force-kill, and the job's actions offer "Stop now" to skip the save.
export const SAVING_BEFORE_STOP_PREFIX = 'Saving before stopping';

export const isSavingBeforeStop = (info: string | null | undefined): boolean =>
  !!info && info.startsWith(SAVING_BEFORE_STOP_PREFIX);
