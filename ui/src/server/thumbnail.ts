// Downscaled copies of images, made in memory per request (nothing is written to disk).
// sharp ships with Next.js; if it cannot load, callers fall back to the original file.

// formats sharp can read; anything else is served as the original
export const THUMBNAIL_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.gif', '.avif', '.tif', '.tiff']);
export const MAX_THUMBNAIL_SIZE = 2048;

type Sharp = typeof import('sharp');
let sharpPromise: Promise<Sharp | null> | null = null;

const loadSharp = (): Promise<Sharp | null> => {
  if (!sharpPromise) {
    sharpPromise = import('sharp')
      .then(module => {
        const sharp = ((module as any).default ?? module) as Sharp;
        // one libvips thread per image and no operation cache; with the queue below that is one
        // core at most, so thumbnails never compete with a running trainer for CPU or RAM
        sharp.concurrency(1);
        sharp.cache(false);
        return sharp;
      })
      .catch(error => {
        console.warn('Image downscaling unavailable, serving originals:', error);
        return null;
      });
  }
  return sharpPromise;
};

// One image at a time: sharp would otherwise run up to four at once (Node's thread pool).
let queue: Promise<unknown> = Promise.resolve();

/** The image shrunk to fit inside size x size (never enlarged), as WebP; null if it cannot be made. */
export function makeThumbnail(filePath: string, size: number): Promise<Buffer | null> {
  const result = queue.then(() => makeThumbnailNow(filePath, size));
  queue = result.catch(() => undefined);
  return result;
}

async function makeThumbnailNow(filePath: string, size: number): Promise<Buffer | null> {
  const sharp = await loadSharp();
  if (!sharp) return null;
  try {
    return await sharp(filePath, { failOn: 'none' })
      .rotate() // apply the EXIF orientation, as the trainer does
      .resize({ width: size, height: size, fit: 'inside', withoutEnlargement: true })
      .webp({ quality: 82 })
      .toBuffer();
  } catch (error) {
    console.warn(`Could not downscale ${filePath}, serving the original:`, error);
    return null;
  }
}
