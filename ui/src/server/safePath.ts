import path from 'path';

const isWindows = process.platform === 'win32';

/**
 * Returns the resolved absolute path if `target` is genuinely contained within
 * one of `allowedDirs`, otherwise null.
 *
 * This replaces the old `decoded.startsWith(allowedDir) && !decoded.includes('..')`
 * check, which had two problems on Windows:
 *   1. A bare string prefix lets a sibling directory escape — e.g. allowed
 *      "C:\out" would match a request for "C:\out_evil\secret".
 *   2. URL segments are joined with "/" while the allowed dirs use "\" , so a
 *      legitimate file could be rejected (false 403) purely on separator/case
 *      mismatch.
 *
 * `path.resolve` normalizes separators, `.` and `..`, and `path.relative` gives
 * us real containment (rel must not climb out with ".." and must not be an
 * absolute path, which happens across drives on Windows). Comparison is
 * case-insensitive on Windows.
 */
export const resolveInsideAllowedDir = (target: string, allowedDirs: string[]): string | null => {
  if (!target) {
    return null;
  }

  const resolvedTarget = path.resolve(target);
  const targetForCompare = isWindows ? resolvedTarget.toLowerCase() : resolvedTarget;

  for (const dir of allowedDirs) {
    if (!dir) {
      continue;
    }
    const resolvedDir = path.resolve(dir);
    const dirForCompare = isWindows ? resolvedDir.toLowerCase() : resolvedDir;

    const rel = path.relative(dirForCompare, targetForCompare);
    // Inside when: same dir (rel === '') OR rel stays within (no leading '..'
    // and not absolute — an absolute rel means a different drive on Windows).
    if (rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel))) {
      return resolvedTarget;
    }
  }

  return null;
};
