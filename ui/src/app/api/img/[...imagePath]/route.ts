/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder, getDataRoot } from '@/server/settings';
import { resolveInsideAllowedDir } from '@/server/safePath';
import { makeThumbnail, MAX_THUMBNAIL_SIZE, THUMBNAIL_EXTENSIONS } from '@/server/thumbnail';

export async function GET(request: NextRequest, context: { params: Promise<{ imagePath: string[] }> }) {
  const { imagePath } = await context.params;
  try {
    // Decode the path
    const filepath = decodeURIComponent(imagePath.join('/'));

    // Get allowed directories
    const datasetRoot = await getDatasetsRoot();
    const trainingRoot = await getTrainingFolder();
    const dataRoot = await getDataRoot();

    const allowedDirs = [datasetRoot, trainingRoot, dataRoot];

    // Security check: resolve and ensure the path is genuinely inside an allowed dir
    const safePath = resolveInsideAllowedDir(filepath, allowedDirs);
    if (!safePath) {
      console.warn(`Access denied: ${filepath} not in ${allowedDirs.join(', ')}`);
      return new NextResponse('Access denied', { status: 403 });
    }

    // Check if file exists
    if (!fs.existsSync(safePath)) {
      console.warn(`File not found: ${safePath}`);
      return new NextResponse('File not found', { status: 404 });
    }

    // Get file info
    const stat = fs.statSync(safePath);
    if (!stat.isFile()) {
      return new NextResponse('Not a file', { status: 400 });
    }

    // Determine content type
    const ext = path.extname(safePath).toLowerCase();
    const contentTypeMap: { [key: string]: string } = {
      // Images
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.png': 'image/png',
      '.gif': 'image/gif',
      '.webp': 'image/webp',
      '.svg': 'image/svg+xml',
      '.bmp': 'image/bmp',
      // Videos
      '.mp4': 'video/mp4',
      '.avi': 'video/x-msvideo',
      '.mov': 'video/quicktime',
      '.mkv': 'video/x-matroska',
      '.wmv': 'video/x-ms-wmv',
      '.m4v': 'video/x-m4v',
      '.flv': 'video/x-flv',
      '.webm': 'video/webm',
      // Audio
      '.mp3': 'audio/mpeg',
      '.wav': 'audio/wav',
    };

    const contentType = contentTypeMap[ext] || 'application/octet-stream';

    // ?w=N: a copy shrunk to fit inside N x N, made in memory (falls back to the original)
    const requestedSize = Number.parseInt(request.nextUrl.searchParams.get('w') ?? '', 10);
    const wantsThumbnail = Number.isFinite(requestedSize) && requestedSize > 0;
    if (wantsThumbnail && !THUMBNAIL_EXTENSIONS.has(ext) && !contentType.startsWith('image/')) {
      // e.g. a video: there is no small version, and the whole file is not what was asked for
      return new NextResponse('No thumbnail for this file type', { status: 415 });
    }
    if (wantsThumbnail && THUMBNAIL_EXTENSIONS.has(ext)) {
      const thumbnail = await makeThumbnail(safePath, Math.min(requestedSize, MAX_THUMBNAIL_SIZE));
      if (thumbnail) {
        return new NextResponse(new Uint8Array(thumbnail), {
          headers: {
            'Content-Type': 'image/webp',
            'Content-Length': String(thumbnail.length),
            'Cache-Control': 'public, max-age=86400',
          },
        });
      }
    }

    // Read file as buffer
    const fileBuffer = fs.readFileSync(safePath);

    // Return file with appropriate headers
    return new NextResponse(fileBuffer, {
      headers: {
        'Content-Type': contentType,
        'Content-Length': String(stat.size),
        'Cache-Control': 'public, max-age=86400',
      },
    });
  } catch (error) {
    console.error('Error serving image:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
