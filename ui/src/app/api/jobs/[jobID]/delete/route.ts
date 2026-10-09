import { NextRequest, NextResponse } from 'next/server';
import { getTrainingFolder } from '@/server/settings';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { ensureJobProcessStopped, removeDirWithRetry, cleanupPidFile } from '@/server/processControl';

export async function GET(request: NextRequest, context: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await context.params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  // Set the cooperative stop flag, then make sure no live trainer is still
  // holding handles in this folder before we wipe it. Otherwise rmSync throws
  // EPERM on Windows and the Python process is orphaned mid-delete.
  await prisma.job.update({
    where: { id: jobID },
    data: { stop: true },
  });

  const trainingRoot = await getTrainingFolder();
  const trainingFolder = path.join(trainingRoot, job.name);

  await ensureJobProcessStopped(trainingRoot, job.name);
  cleanupPidFile(trainingRoot, job.name);

  try {
    if (fs.existsSync(trainingFolder)) {
      await removeDirWithRetry(trainingFolder);
    }
  } catch (error) {
    console.error(`Failed to remove job folder for ${jobID}`, error);
    return NextResponse.json(
      { error: 'Failed to remove job folder (a file may still be in use). Stop the job and try again.' },
      { status: 500 },
    );
  }

  await prisma.job.delete({
    where: { id: jobID },
  });

  return NextResponse.json(job);
}
