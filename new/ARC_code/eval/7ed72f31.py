from __future__ import annotations
import numpy as np
import random
from scipy import ndimage
from arc_utils import STRUCT8, get_bg, label_mask

def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using red (2) markers as reflection axes.

    Input structure:
    - bg = most common color. Red (2) = marker color.
    - Colored sprites appear near red markers.

    Rule:
    - Red line within a sprite (vertical or horizontal) → mirror payload across that axis.
    - Single red pixel in a mixed sprite → point-reflect payload around it.
    - Isolated red pixel (alone, no payload) → point-reflect each adjacent 8-connected color component.
    - Originals remain; mirrored pixels overlay.

    Key insight: Red marker geometry (line vs single pixel) determines reflection type; isolated vs embedded red determines what gets reflected.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    MARK = 2

    # 4-connected components for the mixed (red + payload) sprites.
    lab, n = label_mask(g != int(bg), connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        comps.append({"coords": coords, "vals": vals, "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs)))})

    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    # --- 1) Mirror payload for marker-containing components ---
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue

        rs = {int(r) for r, _c in reds}
        cs = {int(c) for _r, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue  # ambiguous marker shape

        r0, c0, r1, c1 = map(int, comp["bbox"])
        payload = {int(v) for v in vals if int(v) != int(MARK)}
        if not payload:
            continue

        if axis[0] == "row":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(c0) <= int(c) <= int(c1)

        elif axis[0] == "col":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(r0) <= int(r) <= int(r1)

        else:
            rr0 = max(0, int(r0 - 1))
            rr1 = min(int(H - 1), int(r1 + 1))
            cc0 = max(0, int(c0 - 1))
            cc1 = min(int(W - 1), int(c1 + 1))

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)

        for r in range(int(H)):
            for c in range(int(W)):
                v = int(g[int(r), int(c)])
                if v == int(bg) or v == int(MARK):
                    continue
                if not eligible(int(r), int(c), int(v)):
                    continue

                if axis[0] == "row":
                    rr, cc = refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = refl_col(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = refl_point(axis[1], int(r), int(c))

                if inb(int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(v)

    # --- 2) Isolated red pixels mirror adjacent color-components by point reflection ---
    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        pr = int(pr)
        pc = int(pc)
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _nv = label_mask(g == int(v), connectivity=8)  # 8-connected within one color
            labs = {int(labv[int(rr), int(cc)]) for rr, cc, vv in neigh_pixels if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0}
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    if inb(rr, cc):
                        out[int(rr), int(cc)] = int(v)

    return out

# --------------- generator (Claude attempt 1) ---------------

"""Generator for task 7ed72f31 -- mirror payload across red markers.

Approach (output-first):
  1. Pick grid size, bg color, color palette.
  2. For each marker, construct the FULL OUTPUT sprite (payload + red + mirror).
  3. Place sprites on an output canvas with padding (prevents component merging).
  4. Derive input by erasing the mirrored halves.
  5. Oracle check main(input) == output.
"""

# ── CFG: tunable parameters ────────────────────────────────────────────────
MARK = 2

# Canvas
GRID_MIN, GRID_MAX = 18, 28                # square grid side length range

# Marker counts
TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX = 3, 7
MAX_EMBEDDED, MAX_ISOLATED = 4, 4          # per-type caps

# Embedded payload shape
PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX = 2, 6  # bbox side range
PAYLOAD_PX_MIN, PAYLOAD_PX_MAX = 3, 11    # absolute pixel count bounds
PAYLOAD_DENSITY_LO, PAYLOAD_DENSITY_HI = 0.50, 0.80

# Isolated payload
ISOLATED_PX_MIN, ISOLATED_PX_MAX = 1, 8   # pixel count range
ISOLATED_BBOX_MAX = 4                      # max side of isolated bbox

# Color palette
PALETTE_COLORS_MIN, PALETTE_COLORS_MAX = 3, 4  # distinct non-bg, non-red colors

# Layout
PLACEMENT_PADDING = 1                      # cells between sprites (prevents merging)
BG_COVERAGE_MIN = 0.88                     # input must have >= this fraction bg

# Retries
MASK_RETRIES = 40                          # attempts for isolated bbox search
GEN_RETRIES = 300                          # outer generate_input attempts

# 8-neighbor offsets (for isolated red placement search)
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ── Sprite builders ────────────────────────────────────────────────────────

def _random_connected_mask(h: int, w: int, target_pixels: int, *, rng: random.Random) -> np.ndarray | None:
    """Grow a random 4-connected mask of exactly *target_pixels* within an h x w bbox.

    Returns None if growth stalls or the bbox is not tight (every row/col has >= 1 pixel).
    """
    if target_pixels < 1 or target_pixels > h * w:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[rng.randrange(h), rng.randrange(w)] = True
    count = 1
    while count < target_pixels:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                for dr, dc in _DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            return None
        r, c = rng.choice(frontier)
        mask[r, c] = True
        count += 1
    if not mask.any(axis=1).all() or not mask.any(axis=0).all():
        return None
    return mask


def _build_embedded_sprite(rng: random.Random) -> dict | None:
    """Build output sprite for an embedded (row/col red line) marker.

    Returns dict with keys payload, mirror (bool arrays) and red_pixels, or None.
    """
    ph = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    pw = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    max_px = min(PAYLOAD_PX_MAX, int(PAYLOAD_DENSITY_HI * ph * pw))
    min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * ph * pw)))
    if min_px > max_px:
        return None
    pmask = _random_connected_mask(ph, pw, rng.randint(min_px, max_px), rng=rng)
    if pmask is None:
        return None

    is_row = rng.choice([True, False])
    side_flip = rng.random() < 0.5

    # Assemble symmetric sprite: payload on one side, mirror on the other, red line between.
    if is_row:
        H, W = 2 * ph + 1, pw
        payload = np.zeros((H, W), dtype=bool)
        mirror = np.zeros((H, W), dtype=bool)
        red_pixels = [(ph, c) for c in range(pw)]
        orig_slice, mirr_slice = (np.s_[:ph], np.s_[ph + 1:]) if not side_flip else (np.s_[ph + 1:], np.s_[:ph])
        payload[orig_slice] = pmask
        mirror[mirr_slice] = pmask[::-1]
    else:
        H, W = pw, 2 * ph + 1
        payload = np.zeros((H, W), dtype=bool)
        mirror = np.zeros((H, W), dtype=bool)
        red_pixels = [(r, ph) for r in range(pw)]
        orig_slice, mirr_slice = (np.s_[:, :ph], np.s_[:, ph + 1:]) if not side_flip else (np.s_[:, ph + 1:], np.s_[:, :ph])
        payload[orig_slice] = pmask.T
        mirror[mirr_slice] = pmask.T[:, ::-1]

    return {'payload': payload, 'mirror': mirror, 'red_pixels': red_pixels}


def _is_4connected_to(mask: np.ndarray, r: int, c: int) -> bool:
    """Return True if (r, c) is 4-adjacent to any True cell in *mask*."""
    H, W = mask.shape
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
            return True
    return False


def _build_isolated_sprite(rng: random.Random) -> dict | None:
    """Build output sprite for an isolated red marker (single red pixel, payload is 8-adjacent).

    Returns dict with keys payload, mirror (bool arrays) and red_pixels, or None.
    """
    n_px = rng.randint(ISOLATED_PX_MIN, ISOLATED_PX_MAX)
    pmask = None
    for _ in range(MASK_RETRIES):
        ph = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        pw = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        if ph * pw < n_px or n_px < max(ph, pw):
            continue
        pmask = (np.ones((ph, pw), dtype=bool) if n_px == ph * pw
                 else _random_connected_mask(ph, pw, n_px, rng=rng))
        if pmask is not None:
            break
    if pmask is None:
        return None

    pcoords = list(zip(*np.where(pmask)))
    rng.shuffle(pcoords)

    for pr, pc in pcoords:
        dirs = list(_DIRS8)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            red_r, red_c = pr + dr, pc + dc
            refl = [(2 * red_r - r, 2 * red_c - c) for r, c in pcoords]
            all_pts = pcoords + refl + [(red_r, red_c)]
            all_rs = [r for r, _ in all_pts]
            all_cs = [c for _, c in all_pts]
            minr, minc = min(all_rs), min(all_cs)
            H = max(all_rs) - minr + 1
            W = max(all_cs) - minc + 1

            payload = np.zeros((H, W), dtype=bool)
            mirror = np.zeros((H, W), dtype=bool)
            for r, c in pcoords:
                payload[r - minr, c - minc] = True
            for r, c in refl:
                mirror[r - minr, c - minc] = True
            red_pos = (red_r - minr, red_c - minc)

            # Validate: red must not overlap payload/mirror, no payload-mirror overlap,
            # red must not be 4-connected to any colored pixel.
            if payload[red_pos] or mirror[red_pos]:
                continue
            if (payload & mirror).any():
                continue
            if _is_4connected_to(payload | mirror, *red_pos):
                continue
            return {'payload': payload, 'mirror': mirror, 'red_pixels': [red_pos]}
    return None


# ── Placement ──────────────────────────────────────────────────────────────

def _random_free_location(canvas: np.ndarray, patch_mask: np.ndarray, bg: int,
                          rng: random.Random, *, padding: int = PLACEMENT_PADDING):
    """Find a random position where *patch_mask* fits on *canvas* without overlap."""
    H, W = canvas.shape
    h, w = patch_mask.shape
    if h > H or w > W:
        return None
    occ = (canvas != bg)
    if padding > 0:
        occ = ndimage.binary_dilation(occ, structure=STRUCT8, iterations=padding)
    cands = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if not (occ[r0:r0 + h, c0:c0 + w] & patch_mask).any():
                cands.append((r0, c0))
    if not cands:
        return None
    return rng.choice(cands)


# ── Stamp helpers ──────────────────────────────────────────────────────────

def _stamp_sprite(canvas: np.ndarray, spr: dict, r0: int, c0: int) -> None:
    """Paint a sprite's payload, mirror, and red pixels onto *canvas* at (r0, c0)."""
    h, w = spr['payload'].shape
    colored = spr['payload'] | spr['mirror']
    canvas[r0:r0 + h, c0:c0 + w][colored] = spr['color']
    for rr, cc in spr['red_pixels']:
        canvas[r0 + rr, c0 + cc] = MARK


def _erase_mirror(canvas: np.ndarray, spr: dict, bg: int) -> None:
    """Erase the mirror half of a sprite back to *bg*."""
    r0, c0 = spr['r0'], spr['c0']
    h, w = spr['mirror'].shape
    canvas[r0:r0 + h, c0:c0 + w][spr['mirror']] = bg


# ── Main attempt ───────────────────────────────────────────────────────────

def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    N = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice([c for c in range(10) if c != MARK])
    palette = [c for c in range(10) if c != MARK and c != bg]

    total_markers = rng.randint(TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX)
    n_embedded = rng.randint(max(1, total_markers - MAX_ISOLATED), min(MAX_EMBEDDED, total_markers))
    n_isolated = total_markers - n_embedded

    n_colors = rng.randint(PALETTE_COLORS_MIN, min(PALETTE_COLORS_MAX, len(palette), total_markers))
    chosen_colors = rng.sample(palette, n_colors)

    # Build all sprites.
    sprites: list[dict] = []
    for builder, count in [(_build_embedded_sprite, n_embedded), (_build_isolated_sprite, n_isolated)]:
        for _ in range(count):
            s = builder(rng)
            if s is None:
                return None
            sprites.append(s)

    # Assign colors: first n_colors sprites each get a unique color, rest get random picks.
    assignment = list(chosen_colors) + [rng.choice(chosen_colors) for _ in range(len(sprites) - n_colors)]
    rng.shuffle(assignment)
    for spr, c in zip(sprites, assignment):
        spr['color'] = c

    # Place sprites on output canvas.
    output = np.full((N, N), bg, dtype=int)
    rng.shuffle(sprites)
    for spr in sprites:
        footprint = spr['payload'] | spr['mirror']
        for r, c in spr['red_pixels']:
            footprint[r, c] = True
        loc = _random_free_location(output, footprint, bg, rng)
        if loc is None:
            return None
        r0, c0 = loc
        spr['r0'], spr['c0'] = r0, c0
        _stamp_sprite(output, spr, r0, c0)

    # Derive input by erasing mirror halves.
    inp = output.copy()
    for spr in sprites:
        _erase_mirror(inp, spr, bg)

    if (inp == bg).sum() / (N * N) < BG_COVERAGE_MIN:
        return None

    # Oracle check.
    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after %d attempts" % GEN_RETRIES)
