import { AutoProcessor, SiglipVisionModel, RawImage } from "@huggingface/transformers";
import sharp from "sharp";
import fs from "node:fs";
import path from "node:path";

const ID = "Xenova/siglip-base-patch16-224";
const DIM = 768;

function l2(data) {
  let n = 0;
  for (let i = 0; i < data.length; i++) n += data[i] * data[i];
  n = Math.sqrt(n) || 1;
  const out = new Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = Math.round((data[i] / n) * 1e5) / 1e5;
  return out;
}

/** RGBA and gray-with-alpha logos flattened onto white, which is what the Core ML indexer did. */
async function readLogo(file) {
  const { data, info } = await sharp(file)
    .rotate()
    .flatten({ background: "#ffffff" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  if (info.channels !== 3) throw new Error(`${file}: expected 3 channels, got ${info.channels}`);
  return new RawImage(new Uint8ClampedArray(data), info.width, info.height, 3);
}

console.log("loading", ID);
const tLoad = performance.now();
const processor = await AutoProcessor.from_pretrained(ID);
const vision = await SiglipVisionModel.from_pretrained(ID);
console.log(`model ready in ${((performance.now() - tLoad) / 1000).toFixed(1)}s`);

const library = JSON.parse(fs.readFileSync("data/library.json", "utf8"));
console.log(`${library.length} logos`);
const started = performance.now();

for (let i = 0; i < library.length; i++) {
  const item = library[i];
  const file = path.join("public/icons", item.file);
  const t0 = performance.now();
  try {
    const image = await readLogo(file);
    const { pooler_output } = await vision(await processor(image));
    item.vector = l2(pooler_output.data);
  } catch (err) {
    console.error(`failed ${i}/${library.length} ${item.file}: ${err.message}`);
    throw err;
  }
  if (item.vector.length !== DIM) throw new Error(`${item.file}: expected ${DIM}, got ${item.vector.length}`);
  if (i % 25 === 0 || i === library.length - 1) {
    const done = i + 1;
    const avg = (performance.now() - started) / done;
    const left = ((library.length - done) * avg) / 1000;
    console.log(`${done}/${library.length} ${item.file} ${(performance.now() - t0).toFixed(0)}ms  avg ${avg.toFixed(0)}ms  ~${left.toFixed(0)}s left`);
  }
}

fs.writeFileSync("data/library.siglip.json", JSON.stringify(library));
console.log(`wrote data/library.siglip.json in ${((performance.now() - started) / 1000).toFixed(1)}s`);