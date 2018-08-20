export function scaleToRange(val, start, end) {
  return start + (val * Math.max(0, end - start));
}
