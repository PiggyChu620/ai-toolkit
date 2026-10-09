const {
  findBlockingUiWorkerProcesses,
  isLockedPrismaEngineError,
  runCapture,
  runCommand,
  sleepMs,
  terminateProcesses,
} = require('./prisma-process-utils');

function writeResultOutput(result, includeStderr = true) {
  if (result.stdout) {
    process.stdout.write(result.stdout);
  }
  if (includeStderr && result.stderr) {
    process.stderr.write(result.stderr);
  }
}

function runPrismaGenerate() {
  return runCapture('npx', ['prisma', 'generate']);
}

const dbPushStatus = runCommand('npx', ['prisma', 'db', 'push', '--skip-generate']);
if (dbPushStatus !== 0) {
  process.exit(dbPushStatus);
}

let generateResult = runPrismaGenerate();
let combinedOutput = `${generateResult.stdout || ''}\n${generateResult.stderr || ''}`;

if ((generateResult.status ?? 1) !== 0 && isLockedPrismaEngineError(combinedOutput)) {
  const blockingProcesses = findBlockingUiWorkerProcesses();
  if (blockingProcesses.length > 0) {
    writeResultOutput(generateResult, false);
    const summary = blockingProcesses
      .map(proc => `${proc.role} PID ${proc.pid}`)
      .join(', ');
    process.stdout.write(
      `Detected blocking AI-Toolkit session (${summary}). Terminating it and retrying Prisma generate.\n`
    );
    terminateProcesses(blockingProcesses);
    sleepMs(1500);
    generateResult = runPrismaGenerate();
    combinedOutput = `${generateResult.stdout || ''}\n${generateResult.stderr || ''}`;
  }
}

if ((generateResult.status ?? 1) === 0) {
  writeResultOutput(generateResult);
  process.exit(0);
}

if (isLockedPrismaEngineError(combinedOutput)) {
  if (generateResult.stdout) {
    process.stdout.write(generateResult.stdout);
  }
  process.stdout.write(
    '\nPrisma client regeneration was skipped because the Prisma engine DLL is still locked.\n'
  );
  process.stdout.write(
    'The database schema push already succeeded. Stop the remaining lock holder and rerun `npm run prisma_generate` if you need a fresh generated client.\n'
  );
  process.exit(0);
}

writeResultOutput(generateResult);
process.exit(generateResult.status ?? 1);
