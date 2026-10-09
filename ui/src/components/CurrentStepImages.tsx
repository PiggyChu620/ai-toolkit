import { Job } from '@prisma/client';
import { Film, Images } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { justifiedRows, targetTileHeight, tileAspect } from '@/utils/tileLayout';

// What the trainer publishes (toolkit/current_step_feed.py), streamed by /api/jobs/[jobID]/current_step.
interface StepSlot {
  pass: number;
  width: number | null;
  height: number | null;
}

interface StepImage {
  path: string;
  caption: string;
  pass: number;
  width: number | null;
  height: number | null;
  bucket: [number | null, number | null];
  flip_x: boolean;
  flip_y: boolean;
  is_reg: boolean;
}

interface CurrentStepState {
  step: number | null;
  total_steps: number | null;
  pass: number; // forward pass in progress, 1-based; 0 between steps
  passes: number; // forward passes in this step (gradient accumulation)
  slots: StepSlot[]; // every image of the step, in training order
  items: StepImage[]; // the images whose forward pass has started (a prefix of slots)
}

const GAP = 8;
const THUMBNAIL_SIZES = [256, 384, 512, 768, 1024, 1536, 2048];
// the trainer's video extensions (toolkit/data_loader.py): no image to show, only the name
const VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.wmv', '.m4v', '.flv'];

function usePageVisible(): boolean {
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    const update = () => setVisible(document.visibilityState !== 'hidden');
    update();
    document.addEventListener('visibilitychange', update);
    return () => document.removeEventListener('visibilitychange', update);
  }, []);
  return visible;
}

// Event-driven: one long-lived request; the server pushes a new state before each forward pass
// and when the step ends. It only reconnects if the connection drops. A hidden browser tab lets go
// of its connection (browsers allow only six per server, shared by every tab) and gets the
// current state again the moment it is shown.
function useCurrentStep(jobID: string, enabled: boolean): CurrentStepState | null {
  const [state, setState] = useState<CurrentStepState | null>(null);
  const visible = usePageVisible();

  useEffect(() => setState(null), [jobID]);

  useEffect(() => {
    if (!enabled) {
      setState(null);
      return;
    }
    if (!visible) return; // keep the last state: nobody sees it, and it is replaced on return
    const controller = new AbortController();
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = async () => {
      let retryMs = 3000;
      try {
        const headers: Record<string, string> = {};
        const token = localStorage.getItem('AI_TOOLKIT_AUTH');
        if (token) headers['Authorization'] = `Bearer ${token}`;
        const response = await fetch(`/api/jobs/${encodeURIComponent(jobID)}/current_step`, {
          headers,
          cache: 'no-store',
          signal: controller.signal,
        });
        if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
        const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
        let buffer = '';
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += value;
          let end: number;
          while ((end = buffer.indexOf('\n\n')) !== -1) {
            const block = buffer.slice(0, end);
            buffer = buffer.slice(end + 2);
            const data: string[] = [];
            for (const line of block.split('\n')) {
              if (line.startsWith('data:')) data.push(line.slice(line.startsWith('data: ') ? 6 : 5));
              else if (line.startsWith('retry:')) retryMs = Number.parseInt(line.slice(6), 10) || retryMs;
            }
            if (data.length > 0 && !controller.signal.aborted) {
              try {
                setState(JSON.parse(data.join('\n')));
              } catch {
                // malformed event: keep the last good state
              }
            }
          }
        }
      } catch {
        if (controller.signal.aborted) return;
      }
      if (!controller.signal.aborted) retryTimer = setTimeout(connect, retryMs);
    };

    void connect();
    return () => {
      controller.abort();
      if (retryTimer) clearTimeout(retryTimer);
    };
  }, [jobID, enabled, visible]);

  return state;
}

function useElementWidth(element: HTMLElement | null): number {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    if (!element) return;
    const observer = new ResizeObserver(entries => {
      setWidth(Math.floor(entries[0]?.contentRect.width ?? 0));
    });
    observer.observe(element);
    setWidth(Math.floor(element.clientWidth));
    return () => observer.disconnect();
  }, [element]);
  return width;
}

// encodeURIComponent throws on a lone UTF-16 surrogate (possible in a file name); such a name
// cannot be served anyway, so it becomes U+FFFD and the tile falls back to showing the name
const imageUrl = (filePath: string) =>
  `/api/img/${encodeURIComponent(
    filePath.replace(/[\uD800-\uDBFF][\uDC00-\uDFFF]|[\uD800-\uDFFF]/g, pair => (pair.length === 2 ? pair : '\uFFFD')),
  )}`;

// downscaled on the server to roughly the tile's size; snapped so the browser cache keeps hitting
const thumbnailUrl = (filePath: string, tileWidth: number, tileHeight: number) => {
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
  const wanted = Math.max(tileWidth, tileHeight) * dpr;
  const size = THUMBNAIL_SIZES.find(s => s >= wanted) ?? THUMBNAIL_SIZES[THUMBNAIL_SIZES.length - 1];
  return `${imageUrl(filePath)}?w=${size}`;
};

const fileName = (filePath: string) => filePath.split(/[\\/]/).pop() || filePath;


const isVideo = (filePath: string) => VIDEO_EXTENSIONS.some(ext => filePath.toLowerCase().endsWith(ext));

interface StepTileProps {
  image: StepImage | undefined;
  slot: StepSlot;
  width: number;
  height: number;
  isCurrentPass: boolean;
}

function StepTile({ image, slot, width, height, isCurrentPass }: StepTileProps) {
  const [failed, setFailed] = useState(false); // tiles are keyed by path, so this resets per image

  if (!image) {
    // reserved for a forward pass that has not started yet
    return (
      <div
        className="flex-none overflow-hidden rounded-md border border-dashed border-gray-700 flex items-center justify-center text-xs text-gray-600"
        style={{ width, height }}
      >
        pass {slot.pass}
      </div>
    );
  }

  const [bucketW, bucketH] = image.bucket ?? [null, null];
  const flip = image.flip_x || image.flip_y ? `scale(${image.flip_x ? -1 : 1}, ${image.flip_y ? -1 : 1})` : undefined;
  return (
    <a
      href={imageUrl(image.path)}
      target="_blank"
      rel="noreferrer"
      title={`${image.path}\n\n${image.caption}`}
      className={`group relative flex-none overflow-hidden rounded-md bg-gray-950 ${
        isCurrentPass ? 'ring-2 ring-blue-500' : 'ring-1 ring-gray-800'
      }`}
      style={{ width, height }}
    >
      {isVideo(image.path) ? (
        <div className="flex h-full w-full flex-col items-center justify-center gap-1 p-2 text-center text-xs text-gray-500 break-all">
          <Film className="h-5 w-5 flex-none" />
          {fileName(image.path)}
        </div>
      ) : failed ? (
        <div className="flex h-full w-full items-center justify-center p-2 text-center text-xs text-gray-500 break-all">
          {fileName(image.path)}
        </div>
      ) : (
        <img
          src={thumbnailUrl(image.path, width, height)}
          alt={fileName(image.path)}
          decoding="async"
          className="h-full w-full object-contain"
          style={flip ? { transform: flip } : undefined}
          onError={() => setFailed(true)}
        />
      )}
      <div className="pointer-events-none absolute left-1 top-1 flex gap-1 text-[10px] leading-4">
        <span className="rounded bg-black/70 px-1.5 text-gray-200">#{image.pass}</span>
        {image.flip_x || image.flip_y ? <span className="rounded bg-black/70 px-1.5 text-amber-300">mirrored</span> : null}
        {image.is_reg ? <span className="rounded bg-black/70 px-1.5 text-purple-300">reg</span> : null}
      </div>
      <div className="pointer-events-none absolute inset-x-0 bottom-0 translate-y-full bg-black/80 p-2 text-[11px] leading-snug text-gray-200 transition-transform duration-150 group-hover:translate-y-0">
        <p className="line-clamp-5 break-words">{image.caption || <span className="text-gray-500">(no caption)</span>}</p>
        <p className="mt-1 truncate text-gray-400">
          {fileName(image.path)}
          {bucketW && bucketH ? ` · trained at ${bucketW}×${bucketH}` : ''}
        </p>
      </div>
    </a>
  );
}

interface CurrentStepImagesProps {
  job: Job;
}

/** The images of the training step in progress; each appears just before its forward pass. */
export default function CurrentStepImages({ job }: CurrentStepImagesProps) {
  const enabled = job.status === 'running';
  const state = useCurrentStep(job.id, enabled);
  const [gridElement, setGridElement] = useState<HTMLDivElement | null>(null);
  const width = useElementWidth(gridElement);
  const [lastGridHeight, setLastGridHeight] = useState(0);

  const slots = state?.slots ?? [];
  const rows = useMemo(() => {
    const target = targetTileHeight(slots.length);
    const aspects = slots.map(slot => tileAspect(slot.width, slot.height));
    return justifiedRows(aspects, width, GAP, target, Math.round(target * 1.35));
  }, [slots, width]);
  const gridHeight = rows.reduce((sum, row) => sum + row.height, 0) + GAP * Math.max(0, rows.length - 1);

  // between steps the panel keeps its height, so the page does not jump every step
  useEffect(() => {
    if (gridHeight > 0) setLastGridHeight(gridHeight);
  }, [gridHeight]);

  if (!enabled || !state || state.step === null) return null;

  const items = state.items ?? [];
  const betweenSteps = slots.length === 0;
  return (
    // a full-width row of JobOverview's grid, under the log card and the widgets
    <div className="col-span-full bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800">
      <div className="bg-gray-800 px-4 py-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-gray-100">
          <Images className="w-5 h-5 mr-2 -mt-1 text-sky-400 inline-block" /> Images in this step
        </h2>
        <span className="text-sm text-gray-400">
          Step {state.step}
          {state.total_steps ? ` of ${state.total_steps}` : ''}
          {betweenSteps
            ? ' · finished'
            : ` · forward pass ${state.pass} of ${state.passes} · ${items.length} of ${slots.length} image${
                slots.length === 1 ? '' : 's'
              }`}
        </span>
      </div>
      <div className="p-4">
        <div ref={setGridElement} className="w-full" style={{ minHeight: betweenSteps ? lastGridHeight : undefined }}>
          {betweenSteps ? (
            <p className="text-sm text-gray-500">The next step&apos;s images appear as its forward passes start.</p>
          ) : (
            <div className="flex flex-col" style={{ gap: GAP }}>
              {rows.map(row => (
                <div key={row.start} className="flex" style={{ gap: GAP }}>
                  {slots.slice(row.start, row.end).map((slot, offset) => {
                    const index = row.start + offset;
                    const image = items[index];
                    return (
                      <StepTile
                        key={`${index}:${image?.path ?? ''}`}
                        image={image}
                        slot={slot}
                        width={tileAspect(slot.width, slot.height) * row.height}
                        height={row.height}
                        isCurrentPass={slot.pass === state.pass}
                      />
                    );
                  })}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
