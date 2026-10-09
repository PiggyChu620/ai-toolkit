import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';

export async function GET(request: NextRequest, context: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await context.params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  // update job status to 'running'
  await prisma.job.update({
    where: { id: jobID },
    data: {
      stop: true,
      status: 'stopped',
      info: 'Job stopped',
    },
  });

  console.log(`Job ${jobID} marked as stopped`);

  return NextResponse.json(job);
}
