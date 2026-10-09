import fs from 'fs';
import path from 'path';
import { Worker } from 'worker_threads';

// Written by the trainer (toolkit/current_step_feed.py) before each forward pass and when a step ends.
export const CURRENT_STEP_FILE = '.current_step.json';

const EMPTY_STATE = JSON.stringify({ version: 1, step: null, pass: 0, passes: 0, slots: [], items: [] });
const READ_DELAY_MS = 30; // one replace is several directory events; read once, shortly after the first
const KEEPALIVE_MS = 25_000; // also how quickly a vanished client or a moved job folder is noticed
const RETRY_MS = 5_000; // reconnect delay, used only when the job folder cannot be watched

// The folder watch runs on its own thread. On Windows, libuv resolves each event's file name on
// the thread that owns the watcher, and on a slow disk (the USB drive the job folders are on)
// that blocked the UI server's only event loop for up to hundreds of ms per event, holding up
// every request. Messages: 'ready' (watching), 'missing' (no such folder), 'changed' (the feed
// file may have changed), 'dead' (the watch broke).
const WATCHER_SOURCE = `
const { parentPort, workerData } = require('worker_threads');
const fs = require('fs');
const path = require('path');
let watcher = null;
const dead = () => {
  if (watcher) watcher.close();
  watcher = null;
  parentPort.postMessage('dead');
};
try {
  watcher = fs.watch(workerData.folder, (_event, filename) => {
    if (!watcher) return;
    const raw = filename ? String(filename) : '';
    // On Windows a watcher whose folder was deleted raises no 'error': libuv reports, in an
    // endless burst, the folder's own full path (ordinary events carry bare names). The watch is
    // dead even if a new folder already sits at the same path.
    if (raw && path.isAbsolute(raw)) return dead();
    // the trainer writes .current_step.json.tmp and renames it over .current_step.json; some
    // platforms leave filename empty, so treat that as "maybe ours"
    const name = path.basename(raw).toLowerCase();
    if (!name || name.startsWith(workerData.stem)) parentPort.postMessage('changed');
  });
  watcher.on('error', dead);
  parentPort.postMessage('ready');
} catch {
  parentPort.postMessage('missing');
}
`;

/**
 * Server-sent events for the job page's "images in this step" panel.
 *
 * Sends the current state as soon as the folder is watched, then a new event whenever the
 * trainer replaces the file (file-system notifications; nothing is polled). Every event is the
 * complete state, so a dropped or merged notification is repaired by the next one.
 */
export function currentStepEventStream(jobFolder: string, signal?: AbortSignal): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const filePath = path.join(jobFolder, CURRENT_STEP_FILE);
  let worker: Worker | null = null;
  let keepalive: ReturnType<typeof setInterval> | null = null;
  let readTimer: ReturnType<typeof setTimeout> | null = null;
  let closed = false;
  let lastSent: string | null = null;
  let reading = false;
  let readAgain = false;
  let folderMissing = false;
  let streamController: ReadableStreamDefaultController<Uint8Array> | null = null;

  const cleanup = () => {
    if (closed) return;
    closed = true;
    void worker?.terminate();
    worker = null;
    if (keepalive) clearInterval(keepalive);
    if (readTimer) clearTimeout(readTimer);
    keepalive = null;
    readTimer = null;
    signal?.removeEventListener('abort', cleanup);
    try {
      streamController?.close();
    } catch {
      // already closed or errored by the client going away
    }
  };

  const send = (text: string) => {
    if (closed || !streamController) return;
    try {
      streamController.enqueue(encoder.encode(text));
    } catch {
      cleanup();
    }
  };

  const sendState = async () => {
    if (reading) {
      readAgain = true; // a change landed mid-read: read once more when this one finishes
      return;
    }
    reading = true;
    try {
      do {
        readAgain = false;
        let state: string;
        try {
          // re-serialized: one line, whatever the file's formatting (SSE data cannot hold newlines)
          state = JSON.stringify(JSON.parse(await fs.promises.readFile(filePath, 'utf-8')));
        } catch (error: any) {
          if (error?.code !== 'ENOENT') continue; // torn or locked mid-write: the write's own notification brings us back
          state = EMPTY_STATE; // no run has written one yet
        }
        if (state !== lastSent && !closed) {
          lastSent = state;
          send(`data: ${state}\n\n`);
        }
      } while (readAgain && !closed);
    } finally {
      reading = false;
    }
  };

  // Throttled, not debounced: the timer is not pushed back by later events, so a trainer that
  // updates faster than the delay still gets read every READ_DELAY_MS instead of never.
  const scheduleRead = () => {
    if (closed || readTimer) return;
    readTimer = setTimeout(() => {
      readTimer = null;
      void sendState();
    }, READ_DELAY_MS);
  };

  const retryLater = () => {
    if (closed) return;
    send(`retry: ${RETRY_MS}\n\n`);
    cleanup();
  };

  return new ReadableStream<Uint8Array>({
    start(controller) {
      streamController = controller;
      if (signal?.aborted) {
        cleanup();
        return;
      }
      signal?.addEventListener('abort', cleanup, { once: true });
      try {
        worker = new Worker(WATCHER_SOURCE, {
          eval: true,
          workerData: { folder: jobFolder, stem: CURRENT_STEP_FILE.toLowerCase() },
        });
      } catch {
        retryLater();
        return;
      }
      worker.on('message', message => {
        if (message === 'changed') scheduleRead();
        else if (message === 'ready') void sendState(); // read only once watching, so no change slips between
        else if (message === 'missing') {
          folderMissing = true; // not created yet: report "nothing yet", then let the client come back
          void sendState().finally(retryLater);
        }
        else if (message === 'dead') retryLater();
      });
      worker.on('error', retryLater);
      worker.on('exit', () => {
        if (!folderMissing) retryLater(); // after terminate() this is a no-op (already closed)
      });
      keepalive = setInterval(() => {
        send(': keepalive\n\n');
        // a renamed or moved job folder sends no event at all
        fs.promises.access(jobFolder).catch(retryLater);
      }, KEEPALIVE_MS);
    },
    cancel() {
      cleanup();
    },
  });
}
