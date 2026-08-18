# Architecture figure — generation prompt

For the FedChain system architecture diagram (paper §3.1). The palette below is
the **same Okabe-Ito set the result figures use**, so the architecture figure
and the plots read as one system rather than two unrelated visual languages.

---

## Read this before you generate

Diffusion image models cannot render technical text reliably. Every label in a
generated architecture diagram will come back misspelled, duplicated, or
rendered as glyph-soup — `logUpdate(round, clientId, H(θ), cid)` has no chance,
and reviewers notice immediately. So use the prompt in one of two ways:

- **Layout-and-style reference.** Generate the composition, then rebuild it in
  vector (TikZ, draw.io, Figma, Illustrator) with real text. This is what the
  prompt is genuinely good for.
- **Background/frame only.** Generate the panel structure and connectors, then
  set every label in vector on top.

If you want the figure to be submission-ready without a redraw step, skip the
image model and go straight to TikZ — I can write it from the same spec. That
is what essentially every A\* systems paper actually uses, and it has the
advantages that matter here: text is real text, it scales without resampling,
it embeds as vector PDF, and the fonts match the paper body.

---

## Primary prompt

> A clean, flat, two-dimensional system architecture diagram for an academic
> computer-science paper, drawn in the visual language of an IEEE or ACM
> conference figure. Pure vector illustration style, white background, no
> photorealism, no 3D extrusion, no isometric perspective, no drop shadows, no
> gradients, no glow, no texture, no bevels, no skeuomorphic icons.
>
> **Composition.** A horizontal figure, roughly 16:7, organised as three
> stacked horizontal bands separated by thin light-grey rules, with a narrow
> label at the left edge of each band:
>
> - **Top band — "Participants".** Three identical rounded-rectangle node cards
>   in a row, evenly spaced. Each card contains a small database-cylinder
>   glyph, a small neural-network glyph made of three columns of dots and thin
>   connecting lines, and two lines of label text. Cards outlined in a medium
>   blue, filled with a very pale tint of the same blue.
>
> - **Middle band — "Verifiable transport".** Two wide rounded rectangles side
>   by side. The left one holds a hexagonal-lattice glyph of connected nodes
>   (content-addressed storage), outlined and tinted in amber-orange. The right
>   one holds a horizontal chain of four small linked blocks (a blockchain),
>   outlined and tinted in teal-green. Between the two bands, a small circular
>   badge with a fingerprint or hash glyph, outlined in a strong
>   orange-red — this is the integrity check and should be the visual accent of
>   the figure.
>
> - **Bottom band — "Aggregator".** One wide rounded-rectangle card containing
>   a merge glyph: three thin arrows converging into one, then a single
>   rounded-square output. Outlined in medium blue, pale blue fill. A small
>   shield or checkmark badge sits at its input edge in the same orange-red as
>   the integrity badge.
>
> **Flow.** Thin directed arrows, 1.5 pt, dark grey, with small solid
> arrowheads and generous clearance from every box. Each arrow carries a small
> numbered circular badge, 1 through 5, running clockwise: participants down to
> storage, storage across to the chain, chain and storage down to the
> aggregator, aggregator back up to the participants as a broadcast. The return
> broadcast arrow is dashed to distinguish it from the forward path.
>
> **Typography.** All labels in a clean grotesque sans-serif, near-black,
> generously spaced, small relative to the boxes, horizontal only — never
> rotated, never overlapping a rule or an arrow. Ample white space; the diagram
> should look uncrowded and precise.
>
> **Colour.** Exactly four accents on white: medium blue `#0072B2`,
> amber-orange `#E69F00`, teal-green `#009E73`, orange-red `#D55E00`. Neutrals:
> near-black `#1A1A1A` for text, mid-grey `#5C5C5C` for arrows, light grey
> `#D8D8D8` for rules and band separators. Fills are 8–12% tints of their
> outline colour. No other hues anywhere.
>
> Precise, restrained, publication-quality, engineered rather than decorative.

## Negative prompt

> 3D, isometric, perspective, drop shadow, gradient, glossy, neon, glow,
> hand-drawn, sketchy, watercolour, photorealistic, stock-photo icons, clip
> art, cartoon mascots, servers as physical racks, cloud puns, dark background,
> rainbow palette, purple-and-cyan tech aesthetic, dense text, paragraphs,
> watermark, signature, border frame, UI chrome, drop pins, gradients on
> arrows, crowded layout, overlapping labels

## Compact variant

For models with short prompt limits:

> Flat 2D vector system architecture diagram, IEEE conference paper style,
> white background. Three horizontal bands: top, three identical client node
> cards with database and small neural-net glyphs, blue outline on pale blue
> fill; middle, two panels — a hexagonal peer-to-peer storage lattice in
> amber-orange and a four-block blockchain in teal-green — with a circular
> hash-fingerprint badge in orange-red between them; bottom, one aggregator
> card with three arrows merging into one, blue, with an orange-red checkmark
> badge at its input. Thin grey directed arrows with small numbered circular
> badges 1–5; the return broadcast arrow dashed. Small horizontal sans-serif
> labels, lots of white space. Palette strictly `#0072B2`, `#E69F00`,
> `#009E73`, `#D55E00` on white with grey neutrals. No 3D, no shadows, no
> gradients, no clip art.

---

## The labels to set in vector afterwards

The image model will not get these right. Set them yourself:

| Element | Label |
|---|---|
| Band 1 | **Participants** (honest; data never leaves) |
| Client cards | `Client k` · `shard D_k` · `QLoRA fine-tune → θ_k` |
| Band 2 | **Verifiable transport** (untrusted) |
| Storage panel | `IPFS` · `pin(pack(θ_k)) → CID` |
| Chain panel | `FedChainAudit` · `logUpdate(r, k, H(θ_k), CID)` |
| Integrity badge | `H(·) = SHA-256 over canonical serialization` |
| Band 3 | **Aggregator** (honest) |
| Aggregator card | `fetch by CID → re-hash → compare to chain` · `FedAvg over LoRA factors → θ_g` |
| Verify badge | `mismatch ⇒ exclude from round` |
| Step 1 | train locally |
| Step 2 | pack + pin → CID |
| Step 3 | anchor `H(θ_k)`, CID |
| Step 4 | retrieve + verify against commitment |
| Step 5 | aggregate, anchor `θ_g`, broadcast (dashed) |

**One detail worth keeping visible**, because it is a contribution of the paper
rather than boilerplate: the hash badge should be annotated *canonical
serialization* somewhere, not just "SHA-256". §3.2 argues that hashing raw bytes
makes the commitment irreproducible across honest re-runs, and the benign-control
column of Table 4 is the measurement that backs it. A reader who takes only the
figure away should still take that away.

**And one thing the figure must not imply:** no arrow should suggest weights
travel to the chain. The chain receives a 32-byte digest and a CID, never an
adapter — that is what makes gas flat in model size (Fig. 5). Keep the
weight-bearing arrows strictly participant → storage → aggregator.
