import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

const MAX_LOG_RESPONSE_BYTES = 256 * 1024;

const parsePositiveInt = (value: string | null): number | null => {
  if (!value) {
    return null;
  }

  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed < 0) {
    return null;
  }

  return parsed;
};

const readLogChunk = (logPath: string, startOffset: number, length: number): string => {
  if (length <= 0) {
    return '';
  }

  let fd: number | null = null;
  try {
    fd = fs.openSync(logPath, 'r');
    const buffer = Buffer.alloc(length);
    const bytesRead = fs.readSync(fd, buffer, 0, length, startOffset);
    return buffer.subarray(0, bytesRead).toString('utf-8');
  } finally {
    if (fd !== null) {
      fs.closeSync(fd);
    }
  }
};

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
  const logPath = path.join(jobFolder, 'log.txt');
  const requestedOffset = parsePositiveInt(request.nextUrl.searchParams.get('offset')) ?? 0;
  const requestedMaxBytes = parsePositiveInt(request.nextUrl.searchParams.get('maxBytes')) ?? MAX_LOG_RESPONSE_BYTES;
  const maxBytes = Math.min(requestedMaxBytes, MAX_LOG_RESPONSE_BYTES);

  if (!fs.existsSync(logPath)) {
    return NextResponse.json({ log: '', startOffset: 0, nextOffset: 0, reset: true });
  }

  try {
    const fileSize = fs.statSync(logPath).size;
    let reset = requestedOffset <= 0 || requestedOffset > fileSize;
    let startOffset = requestedOffset;

    if (reset || fileSize - startOffset > maxBytes) {
      reset = true;
      startOffset = Math.max(fileSize - maxBytes, 0);
    }

    const log = readLogChunk(logPath, startOffset, fileSize - startOffset);
    return NextResponse.json({
      log,
      startOffset,
      nextOffset: fileSize,
      reset,
    });
  } catch (error) {
    console.error('Error reading log file:', error);
    return NextResponse.json({ error: 'Error reading log file' }, { status: 500 });
  }
}
