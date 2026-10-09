'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';

interface JobLogResponse {
  log?: string;
  nextOffset?: number;
  reset?: boolean;
}

const INITIAL_LOG_BYTES = 256 * 1024;
const MAX_LOG_CHARS = 500_000;

const clean = (text: string): string => {
  // remove \x1B[A\x1B[A
  text = text.replace(/\x1B\[A/g, '');
  return text;
};

const trimLog = (text: string): string => {
  if (text.length <= MAX_LOG_CHARS) {
    return text;
  }

  return text.slice(text.length - MAX_LOG_CHARS);
};

export default function useJobLog(jobID: string, reloadInterval: null | number = null) {
  const [log, setLog] = useState<string>('');
  const didInitialLoadRef = useRef(false);
  const offsetRef = useRef(0);
  const inFlightRef = useRef(false);
  const requestGenerationRef = useRef(0);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const refresh = useCallback(() => {
    if (inFlightRef.current) {
      return;
    }

    let loadStatus: 'loading' | 'refreshing' = 'loading';
    if (didInitialLoadRef.current) {
      loadStatus = 'refreshing';
    }
    setStatus(loadStatus);
    inFlightRef.current = true;
    const requestGeneration = requestGenerationRef.current;

    apiClient
      .get<JobLogResponse>(`/api/jobs/${jobID}/log`, {
        params: {
          offset: offsetRef.current,
          maxBytes: INITIAL_LOG_BYTES,
        },
      })
      .then(res => res.data)
      .then(data => {
        if (requestGeneration !== requestGenerationRef.current) {
          return;
        }

        const nextOffset = typeof data.nextOffset === 'number' ? data.nextOffset : offsetRef.current;
        offsetRef.current = nextOffset;

        const cleanLog = clean(data.log ?? '');
        setLog(currentLog => {
          if (data.reset || !didInitialLoadRef.current) {
            return trimLog(cleanLog);
          }

          if (!cleanLog) {
            return currentLog;
          }

          return trimLog(currentLog + cleanLog);
        });
        setStatus('success');
        didInitialLoadRef.current = true;
      })
      .catch(error => {
        if (requestGeneration !== requestGenerationRef.current) {
          return;
        }

        console.error('Error fetching log:', error);
        setStatus('error');
      })
      .finally(() => {
        if (requestGeneration === requestGenerationRef.current) {
          inFlightRef.current = false;
        }
      });
  }, [jobID]);

  useEffect(() => {
    requestGenerationRef.current += 1;
    didInitialLoadRef.current = false;
    offsetRef.current = 0;
    inFlightRef.current = false;
    setLog('');
    setStatus('idle');

    refresh();

    if (reloadInterval) {
      const interval = setInterval(() => {
        refresh();
      }, reloadInterval);

      return () => {
        clearInterval(interval);
      };
    }
  }, [jobID, reloadInterval, refresh]);

  return { log, setLog, status, refresh };
}
