import { typesafeKey } from "../env";

const ENDPOINT = "https://api.typesafe.ai/v1/systemone";

/**
 * One request to Jev. Returns the parsed answer, or null when Jev cannot be reached: the caller then carries on
 * without it and marks the search as degraded. A failure that comes back quickly (a dropped connection, a 429 or a
 * 5xx) is tried once more; a timeout is not, because waiting twice as long is worse than answering without Jev.
 */
export async function systemOne<T>(body: Record<string, unknown>, timeoutMs: number): Promise<T | null> {
  const key = typesafeKey();
  const questions = Object.keys((body.questions as Record<string, unknown> | undefined) ?? {});
  const label = questions.slice(0, 4).join(",") || "jev";
  if (!key) {
    console.log(`[jev] ${label} skipped: no key`);
    return null;
  }
  for (let attempt = 0; attempt < 2; attempt++) {
    const started = performance.now();
    try {
      const res = await fetch(ENDPOINT, {
        method: "POST",
        headers: { authorization: `Bearer ${key}`, "content-type": "application/json" },
        body: JSON.stringify({ model: "jev-latest", ...body }),
        signal: AbortSignal.timeout(timeoutMs),
      });
      const ms = Math.round(performance.now() - started);
      if (res.ok) {
        console.log(`[jev] ${label} ${res.status} in ${ms}ms`);
        return (await res.json()) as T;
      }
      const detail = (await res.text()).slice(0, 180);
      console.log(`[jev] ${label} attempt ${attempt + 1} ${res.status} in ${ms}ms ${detail}`);
      if (res.status !== 429 && res.status < 500) return null;
    } catch (e) {
      const ms = Math.round(performance.now() - started);
      const why = e instanceof Error ? e.name : "error";
      console.log(`[jev] ${label} attempt ${attempt + 1} ${why} after ${ms}ms (limit ${timeoutMs}ms)`);
      if (performance.now() - started > 2000) return null;
    }
  }
  console.log(`[jev] ${label} gave up`);
  return null;
}