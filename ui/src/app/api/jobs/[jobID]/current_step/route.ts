import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import { getTrainingFolder } from '@/server/settings';
import { currentStepEventStream } from '@/server/currentStepStream';

export const dynamic = 'force-dynamic';

// Streams the images the trainer is on (job page, under the log) as server-sent events.
export async function GET(request: NextRequest, context: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await context.params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);

  return new Response(currentStepEventStream(jobFolder, request.signal), {
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      // no-transform keeps Next's gzip from buffering the stream
      'Cache-Control': 'no-cache, no-transform',
      'X-Accel-Buffering': 'no',
    },
  });
}
