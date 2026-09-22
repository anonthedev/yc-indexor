# YC-indexor

Describe a YC startup in any words you like and the logos that match float up out of a physics pile, each with how likely it is.
All 6,241 companies are searchable by what they do, what their logo looks like, and what they say inside it.

You need Node 20 or newer, and a TypeSafe key. Looks are SigLIP (`Xenova/siglip-base-patch16-224`), run in Node. Meaning search is bge-small, the same way.

## Run it

```bash
git clone https://github.com/Aayan-DEV/aayans-yc-indexor && cd aayans-yc-indexor
npm install
cp .env.example .env.local
npm run dev
```

Then open http://localhost:3000.

## The key

Put a TypeSafe key in `.env.local` as `TYPE_SAFE_KEY`. Get one at [typesafe.ai](https://typesafe.ai).

`.env.local` is gitignored and is not in this repo, which is the point: it holds your real key. `.env.example` is the
template you copy. If this sits next to sibling projects that share one key file, a `.env` one directory up is read as a
fallback.

Without a key the app still starts and the pile still works, but Jev never gets asked and ranking falls back to a local
scorer. You can see it in the API response: `degraded: true` and `costUsd: null`. Results get noticeably worse.

First start takes about sixteen seconds while the sentence model loads. After that a search is roughly one second and
costs about a fifth of a cent.

## How it finds things

Retrieval is ordinary vector maths over all 6,241 companies, four signals at once:

- **meaning**, bge-small sentence embeddings of each company's write-up
- **looks**, SigLIP image vectors of the logo itself
- **letters**, the words already read inside each logo, stored in `data/letters.json`
- **tags**, YC's own 337 tags, re-applied to every company by Jev because YC's own tagging is patchy

That narrows 6,241 down to at most 320 finalists. Jev then scores every finalist in one parallel request and returns a
calibrated probability rather than generated text, which is why the percentages on screen are real and why it lands in
about a second.

## Adding companies

Everything the app reads is committed, so you never need this to run it. `tools/yc-data/` has the scripts that built
`data/companies.json` and `public/icons/` in the first place, with a README giving the order to run them in.

## Everything else

`NOTES.md` is the long version: the retrieval pipeline in detail, the physics and frame-pacing work, the sprite atlas, and
the measurements behind all of it.

## Licence

The code is [MIT](LICENSE). The logos and the YC data bundled with it are not mine to license; see [NOTICE](NOTICE).
