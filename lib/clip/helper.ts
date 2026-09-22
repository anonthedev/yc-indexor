import { AutoProcessor, AutoTokenizer, RawImage, SiglipTextModel, SiglipVisionModel } from "@huggingface/transformers";
import sharp from "sharp";

const ID = "Xenova/siglip-base-patch16-224";

const tokenizer = await AutoTokenizer.from_pretrained(ID);
const text = await SiglipTextModel.from_pretrained(ID);
const processor = await AutoProcessor.from_pretrained(ID);
const vision = await SiglipVisionModel.from_pretrained(ID);

export type ImageResult = { embedding: number[]; colorText: string; colors: string[]; ms: number };

function l2(data: ArrayLike<number>): number[] {
  let n = 0;
  for (let i = 0; i < data.length; i++) n += data[i] * data[i];
  n = Math.sqrt(n) || 1;
  const out = new Array<number>(data.length);
  for (let i = 0; i < data.length; i++) out[i] = Math.round((data[i] / n) * 1e5) / 1e5;
  return out;
}

/** Same flatten the index script uses, so a new logo lands in the same vector space. */
async function readLogo(file: string): Promise<RawImage> {
  const { data, info } = await sharp(file)
    .rotate()
    .flatten({ background: "#ffffff" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  if (info.channels !== 3) throw new Error(`${file}: expected 3 channels, got ${info.channels}`);
  return new RawImage(new Uint8ClampedArray(data), info.width, info.height, 3);
}

const HUES: [number, string][] = [
  [12, "red"], [38, "orange"], [52, "yellow-orange"], [66, "yellow"], [88, "yellow-green (lime)"],
  [160, "green"], [178, "green-blue (teal)"], [198, "light blue (cyan)"], [236, "blue"],
  [268, "blue-purple (indigo)"], [295, "purple"], [335, "pink"], [350, "pink-red"], [361, "red"],
];

function colorName(r: number, g: number, b: number): string {
  const mx = Math.max(r, g, b);
  const mn = Math.min(r, g, b);
  const d = mx - mn;
  const s = mx === 0 ? 0 : d / mx;
  let h = 0;
  if (d > 0) {
    if (mx === r) h = ((g - b) / d) % 6;
    else if (mx === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  if (mx < 0.2) return "black";
  if (s < 0.14) {
    if (mx > 0.75 && s >= 0.045 && h >= 15 && h <= 70) return "cream or beige";
    return mx > 0.84 ? "white" : mx > 0.6 ? "light gray" : mx > 0.35 ? "gray" : "dark gray";
  }
  const base = HUES.find(([edge]) => h < edge)![1];
  if ((base === "orange" || base === "yellow-orange") && mx < 0.55) return "brown";
  return (mx < 0.5 ? "dark " : s < 0.38 && mx > 0.78 ? "pale " : "") + base;
}

/** The color sentence the old Swift helper wrote, measured on a 48×48 white-backed square. */
async function colorWords(file: string): Promise<{ colorText: string; colors: string[] }> {
  const n = 48;
  const { data, info } = await sharp(file)
    .rotate()
    .flatten({ background: "#ffffff" })
    .resize(n, n, { fit: "cover" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const counts = new Map<string, number>();
  const ring = new Map<string, number>();
  const center = new Map<string, number>();
  let brightness = 0;
  const bump = (map: Map<string, number>, name: string) => map.set(name, (map.get(name) ?? 0) + 1);
  for (let y = 0; y < info.height; y++) {
    for (let x = 0; x < info.width; x++) {
      const o = (y * info.width + x) * info.channels;
      const r = data[o] / 255;
      const g = data[o + 1] / 255;
      const b = data[o + 2] / 255;
      const name = colorName(r, g, b);
      bump(counts, name);
      brightness += (r + g + b) / 3;
      if (x === 2 || x === n - 3 || y === 2 || y === n - 3) bump(ring, name);
      if (x >= 16 && x < 32 && y >= 16 && y < 32) bump(center, name);
    }
  }
  const total = info.width * info.height;
  const main = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6).filter(([, c]) => c / total >= 0.04);
  const tone = brightness / total < 0.3 ? "dark overall" : brightness / total > 0.72 ? "light overall" : "medium brightness";
  const background = [...ring.entries()].sort((a, b) => a[1] - b[1]).at(-1)?.[0] ?? "unknown";
  const middle = [...center.entries()].sort((a, b) => b[1] - a[1]).slice(0, 2).map(([name]) => name).join(" and ");
  const colorText =
    "Colors by area: " +
    main.map(([name, c]) => `${name} ${Math.round((c / total) * 100)}%`).join(", ") +
    `. The background (outer edge) is ${background}. The middle of the image is mostly ${middle}. The image is ${tone}.`;
  return { colorText, colors: main.slice(0, 3).map(([name]) => name) };
}

export async function embedText(sentence: string): Promise<number[]> {
  const inputs = tokenizer([sentence], { padding: "max_length", truncation: true });
  const { pooler_output } = await text(inputs);
  return l2(pooler_output.data);
}

export async function embedImage(file: string): Promise<ImageResult> {
  const started = performance.now();
  const image = await readLogo(file);
  const { pooler_output } = await vision(await processor(image));
  const { colorText, colors } = await colorWords(file);
  return { embedding: l2(pooler_output.data), colorText, colors, ms: performance.now() - started };
}