/** Marker color for a detection, by its confidence (the backend's detection score:
 * 50% at the detection threshold, up to 100%), on the map and in alerts. */
export function detectionColor(probability: number): string {
  if (probability >= 0.8) return "#dc2626";
  if (probability >= 0.65) return "#f97316";
  return "#facc15";
}

/** Parses the backend's naive-UTC ISO timestamps as UTC. */
export function parseUtc(iso: string): Date {
  return new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
}

/** The map's default minimum confidence filter, in percent. Every detection scores
 * at least 50%; only rows saved before the detection score, which hold a raw model
 * probability, can be lower. Users can lower it in the filters. */
export const DEFAULT_MIN_CONFIDENCE = 50;
