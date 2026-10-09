import prisma from '../prisma';
import { Job } from '@prisma/client';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT, getTrainingFolder, getHFToken } from '../paths';
import { reconcileJobProcessState } from '../../src/server/jobProcess';
import { cleanupPidFile, isProcessAlive, processBelongsToJob, readJobPid } from '../../src/server/processControl';
const isWindows = process.platform === 'win32';

const startAndWatchJob = (job: Job) => {
  // starts and watches the job asynchronously
  return new Promise<void>(async (resolve, reject) => {
    const jobID = job.id;

    // setup the training
    const trainingRoot = await getTrainingFolder();

    const trainingFolder = path.join(trainingRoot, job.name);
    if (!fs.existsSync(trainingFolder)) {
      fs.mkdirSync(trainingFolder, { recursive: true });
    }

    // make the config file
    const configPath = path.join(trainingFolder, '.job_config.json');

    //log to path
    const logPath = path.join(trainingFolder, 'log.txt');

    try {
      // if the log path exists, move it to a folder called logs and rename it {num}_log.txt, looking for the highest num
      // if the log path does not exist, create it
      if (fs.existsSync(logPath)) {
        const logsFolder = path.join(trainingFolder, 'logs');
        if (!fs.existsSync(logsFolder)) {
          fs.mkdirSync(logsFolder, { recursive: true });
        }

        let num = 0;
        while (fs.existsSync(path.join(logsFolder, `${num}_log.txt`))) {
          num++;
        }

        fs.renameSync(logPath, path.join(logsFolder, `${num}_log.txt`));
      }
    } catch (e) {
      console.error('Error moving log file:', e);
    }

    // the job page's step-images panel must not show the previous run's last step
    for (const stale of ['.current_step.json', '.current_step.json.tmp']) {
      try {
        fs.rmSync(path.join(trainingFolder, stale), { force: true });
      } catch (e) {
        console.error(`Error removing ${stale}:`, e);
      }
    }

    // update the config dataset path
    const jobConfig = JSON.parse(job.job_config);
    jobConfig.config.process[0].sqlite_db_path = path.join(TOOLKIT_ROOT, 'aitk_db.db');

    // write the config file
    fs.writeFileSync(configPath, JSON.stringify(jobConfig, null, 2));

    let pythonPath = 'python';
    const embeddedPythonPath = isWindows
      ? path.join(TOOLKIT_ROOT, '..', 'python_embeded', 'python.exe')
      : path.join(TOOLKIT_ROOT, '..', 'python_embeded', 'bin', 'python');
    // use .venv or venv if it exists
    if (fs.existsSync(path.join(TOOLKIT_ROOT, '.venv'))) {
      if (isWindows) {
        pythonPath = path.join(TOOLKIT_ROOT, '.venv', 'Scripts', 'python.exe');
      } else {
        pythonPath = path.join(TOOLKIT_ROOT, '.venv', 'bin', 'python');
      }
    } else if (fs.existsSync(path.join(TOOLKIT_ROOT, 'venv'))) {
      if (isWindows) {
        pythonPath = path.join(TOOLKIT_ROOT, 'venv', 'Scripts', 'python.exe');
      } else {
        pythonPath = path.join(TOOLKIT_ROOT, 'venv', 'bin', 'python');
      }
    } else if (fs.existsSync(embeddedPythonPath)) {
      pythonPath = embeddedPythonPath;
    }

    const runFilePath = path.join(TOOLKIT_ROOT, 'run.py');
    if (!fs.existsSync(runFilePath)) {
      console.error(`run.py not found at path: ${runFilePath}`);
      await prisma.job.update({
        where: { id: jobID },
        data: {
          status: 'error',
          info: `Error launching job: run.py not found`,
        },
      });
      return;
    }

    const additionalEnv: any = {
      AITK_JOB_ID: jobID,
      CUDA_DEVICE_ORDER: 'PCI_BUS_ID',
      CUDA_VISIBLE_DEVICES: `${job.gpu_ids}`,
      IS_AI_TOOLKIT_UI: '1',
      PYTHONFAULTHANDLER: '1',
      TORCH_DISABLE_ADDR2LINE: '1',
      TORCH_SHOW_CPP_STACKTRACES: '1',
    };

    // HF_TOKEN
    const hfToken = await getHFToken();
    if (hfToken && hfToken.trim() !== '') {
      additionalEnv.HF_TOKEN = hfToken;
    }

    // Add the --log argument to the command
    const args = [runFilePath, configPath, '--log', logPath];
    const pidPath = path.join(trainingFolder, 'pid.txt');
    const processOutputPath = path.join(trainingFolder, 'process_output.txt');
    let didFinalizeProcess = false;
    let processOutputFd: number | null = null;

    const cleanupPidFile = () => {
      try {
        if (fs.existsSync(pidPath)) {
          fs.unlinkSync(pidPath);
        }
      } catch (e) {
        console.error('Error removing pid file:', e);
      }
    };

    const closeProcessOutputFd = () => {
      if (processOutputFd === null) {
        return;
      }
      try {
        fs.closeSync(processOutputFd);
      } catch (e) {
        console.error('Error closing process output file descriptor:', e);
      }
      processOutputFd = null;
    };

    const finalizeUnexpectedExit = async () => {
      if (didFinalizeProcess) {
        return;
      }
      didFinalizeProcess = true;
      closeProcessOutputFd();
      try {
        const currentJob = await prisma.job.findUnique({
          where: { id: jobID },
        });
        if (currentJob) {
          await reconcileJobProcessState(currentJob, trainingRoot);
        }
      } catch (error) {
        console.error('Error reconciling unexpected job exit:', error);
      }
    };

    const finalizeLaunchError = async (error: Error) => {
      if (didFinalizeProcess) {
        return;
      }
      didFinalizeProcess = true;
      closeProcessOutputFd();
      cleanupPidFile();
      try {
        await prisma.job.update({
          where: { id: jobID },
          data: {
            status: 'error',
            info: `Error launching job: ${error?.message || 'Unknown error'}`,
            speed_string: '',
            cache_speed_string: '',
          },
        });
      } catch (updateError) {
        console.error('Error updating job launch failure state:', updateError);
      }
    };

    try {
      let subprocess;
      processOutputFd = fs.openSync(processOutputPath, 'w');

      if (isWindows) {
        // Spawn Python directly on Windows so the process can survive parent exit
        subprocess = spawn(pythonPath, args, {
          env: {
            ...process.env,
            ...additionalEnv,
          },
          cwd: TOOLKIT_ROOT,
          detached: true,
          windowsHide: true,
          stdio: ['ignore', processOutputFd, processOutputFd],
        });
      } else {
        // For non-Windows platforms, fully detach and ignore stdio so it survives daemon-like
        subprocess = spawn(pythonPath, args, {
          detached: true,
          stdio: ['ignore', processOutputFd, processOutputFd],
          env: {
            ...process.env,
            ...additionalEnv,
          },
          cwd: TOOLKIT_ROOT,
        });
      }

      // Important: let the child run independently of this Node process.
      if (subprocess.unref) {
        subprocess.unref();
      }
      closeProcessOutputFd();

      subprocess.once('error', (error) => {
        console.error('Training subprocess emitted an error:', error);
        void finalizeLaunchError(error);
      });

      subprocess.once('exit', (code, signal) => {
        if (didFinalizeProcess) {
          return;
        }
        console.error(`Training subprocess exit observed with code=${code} signal=${signal}`);
        void finalizeUnexpectedExit();
      });

      // Optionally write a pid file for future management (stop/inspect) without keeping streams open
      try {
        fs.writeFileSync(pidPath, String(subprocess.pid ?? ''), { flag: 'w' });
      } catch (e) {
        console.error('Error writing pid file:', e);
      }

      // No stdout/stderr listeners; output is redirected to process_output.txt.
      // No monitoring loop; the whole point is to let it live past this worker.
    } catch (error: any) {
      // Handle any exceptions during process launch
      closeProcessOutputFd();
      console.error('Error launching process:', error);

      await prisma.job.update({
        where: { id: jobID },
        data: {
          status: 'error',
          info: `Error launching job: ${error?.message || 'Unknown error'}`,
        },
      });
      return;
    }
    // Resolve the promise immediately after starting the process
    resolve();
  });
};

export default async function startJob(jobID: string) {
  const job: Job | null = await prisma.job.findUnique({
    where: { id: jobID },
  });
  if (!job) {
    console.error(`Job with ID ${jobID} not found`);
    return;
  }
  // its previous run may still be exiting (finishing its last checkpoint, freeing the GPU): wait
  // for it; the queue tries again on its next pass
  const trainingRoot = await getTrainingFolder();
  const previousPid = readJobPid(trainingRoot, job.name);
  if (previousPid !== null) {
    if (isProcessAlive(previousPid) && processBelongsToJob(previousPid, trainingRoot, job.name)) {
      console.log(`Job ${jobID}: its previous process (${previousPid}) is still exiting; starting it later`);
      return;
    }
    // left by a finished run (or the PID now belongs to another program): not ours, drop it
    cleanupPidFile(trainingRoot, job.name);
  }
  // update job status to 'running', this will run sync so we don't start multiple jobs.
  await prisma.job.update({
    where: { id: jobID },
    data: {
      status: 'running',
      stop: false,
      // a job the trainer sent back to the queue still has the flag set; it would stop again
      return_to_queue: false,
      info: 'Starting job...',
      speed_string: '',
      cache_speed_string: '',
    },
  });
  // start and watch the job asynchronously so the cron can continue
  startAndWatchJob(job);
}
