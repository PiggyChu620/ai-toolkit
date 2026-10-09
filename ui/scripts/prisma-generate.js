const {
  findBlockingUiWorkerProcesses,
  isLockedPrismaEngineError,
  runCapture,
  sleepMs,
  terminateProcesses,
} = require('./prisma-process-utils');

function emitProcessTerminationMessage(processes) {
  const summary = processes
    .map(proc => `${proc.role} PID ${proc.pid}`)
    .join(', ');
  process.stdout.write(
    `Detected blocking AI-Toolkit session (${summary}). Terminating it and retrying Prisma generate.\n`
  );
}

function runPrismaGenerate() {
  return runCapture('npx', ['prisma', 'generate']);
}

function writeResultOutput(result, includeStderr = true) {
  if (result.stdout) {
    process.stdout.write(result.stdout);
  }
  if (includeStderr && result.stderr) {
    process.stderr.write(result.stderr);
  }
}

let result = runPrismaGenerate();
let combinedOutput = `${result.stdout || ''}\n${result.stderr || ''}`;

if ((result.status ?? 1) !== 0 && isLockedPrismaEngineError(combinedOutput)) {
  const blockingProcesses = findBlockingUiWorkerProcesses();
  if (blockingProcesses.length > 0) {
    writeResultOutput(result, false);
    emitProcessTerminationMessage(blockingProcesses);
    terminateProcesses(blockingProcesses);
    sleepMs(1500);
    result = runPrismaGenerate();
    combinedOutput = `${result.stdout || ''}\n${result.stderr || ''}`;
  }
}

if ((result.status ?? 1) === 0) {
  writeResultOutput(result);
  process.exit(0);
}

if (isLockedPrismaEngineError(combinedOutput)) {
  if (result.stdout) {
    process.stdout.write(result.stdout);
  }
  process.stderr.write(
    'Prisma client regeneration failed because the Prisma engine DLL is still locked.\n'
  );
  process.stderr.write(
    'If this keeps happening, stop the UI and worker manually, then rerun `npm run prisma_generate`.\n'
  );
  process.exit(1);
}

writeResultOutput(result);
process.exit(result.status ?? 1);
