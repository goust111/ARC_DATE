from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location
import math
import random

def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill rectangular cavities in a template shape by placing source sprites into matching holes.

    Input structure:
    - bg = most common color. A dominant "base" color forms a large frame/template with rectangular gaps.
    - Small colored sprites sit outside the template region.

    Rule:
    - The base color (largest connected component excluding bg) defines the template bbox.
    - Each rectangular hole inside the template matches exactly one outside sprite by size.
    - Place each sprite into its matching cavity; output is the filled template bbox.

    Key insight: Sprite bounding boxes match cavity dimensions exactly — it's a jigsaw-fit puzzle where each piece has a unique slot.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- pick base color by largest CC size (excluding bg) ---
    base_candidates: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(int(max_area), int(coords.shape[0]))
        base_candidates.append((int(max_area), int(col)))

    assert base_candidates
    base_candidates.sort(reverse=True)
    if len(base_candidates) > 1:
        assert base_candidates[0][0] > base_candidates[1][0]
    base_col = int(base_candidates[0][1])

    # --- destination bbox = bbox of all base-color pixels ---
    base_coords = np.argwhere(g == int(base_col))
    assert not (base_coords.size == 0)
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst_bbox = (int(r0), int(c0), int(r1), int(c1))

    dst = crop_bbox(g, dst_bbox)
    dh, dw = map(int, dst.shape)
    hole_mask = dst != int(base_col)

    out = np.full((dh, dw), int(base_col), dtype=int)

    # --- source sprites = connected components of non-bg pixels outside dst_bbox ---
    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        b = tuple(map(int, comp["bbox"]))
        pr0, pc0, pr1, pc1 = b
        patch = crop_bbox(g, b)
        ph, pw = map(int, patch.shape)
        m = np.zeros((ph, pw), dtype=bool)
        coords = comp["coords"]
        m[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        sprites.append({"h": int(ph), "w": int(pw), "mask": m, "patch": patch})

    # Larger sprites first; search for a non-overlapping packing.
    sprites.sort(key=lambda s: (-int(s["h"] * s["w"]), -int(s["mask"].sum())))

    # --- candidate placements per sprite: any all-hole rectangle of same bbox ---
    candidates: list[list[tuple[int, int]]] = []
    areas = [int(s["h"] * s["w"]) for s in sprites]
    assert sum(areas) == int(hole_mask.sum())
    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        pos: list[tuple[int, int]] = []
        if h <= dh and w <= dw:
            for rr0 in range(int(dh - h + 1)):
                for cc0 in range(int(dw - w + 1)):
                    if bool(hole_mask[rr0 : rr0 + h, cc0 : cc0 + w].all()):
                        pos.append((int(rr0), int(cc0)))
        candidates.append(pos)

    assert all(candidates)
    used = ~hole_mask  # base pixels are treated as already occupied
    placement: list[tuple[int, int] | None] = [None] * len(sprites)

    def dfs(i: int) -> bool:
        if i == len(sprites):
            return True

        h, w = int(sprites[i]["h"]), int(sprites[i]["w"])
        for rr0, cc0 in candidates[i]:
            if bool(used[rr0 : rr0 + h, cc0 : cc0 + w].any()):
                continue
            used[rr0 : rr0 + h, cc0 : cc0 + w] = True
            placement[i] = (int(rr0), int(cc0))
            if dfs(i + 1):
                return True
            placement[i] = None
            used[rr0 : rr0 + h, cc0 : cc0 + w] = False
        return False

    assert dfs(0)

    # --- paste according to the exact cover placement ---
    for i, pos in enumerate(placement):
        assert pos is not None
        rr0, cc0 = map(int, pos)
        sp = sprites[i]
        paste(out, sp["patch"], rr0, cc0, mask=sp["mask"], overwrite=True)

    return out

# --------------- generator (att2) ---------------

# ── Configuration ────────────────────────────────────────────────────────
GRID_MIN, GRID_MAX = 15, 25           # grid dimension range
THICK_MIN, THICK_MAX = 7, 10          # template strip thickness
OUTER_GAP = 8                         # min outer-region width for sprite placement
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 7, 15
DOT_MIN, DOT_MAX = 1, 4               # 1x1 black marker sprites
DOT_NON_DOT_BUFFER = 6                # total - dots >= this
DOT_LONG_BUFFER = 4                   # total - dots <= long - this
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.39, 0.50  # hole area as fraction of template
CORRIDOR_COVERAGE = 0.78              # min fraction of long dim covered by holes
MAX_RECT_LEN = 8                      # max single rect dimension along long axis
MAX_RECT_WIDTH = 6                    # max rect width (cross axis)
MIN_SHAPE_GROUPS = 3                  # reject if fewer distinct (h,w) groups
MIN_PALETTE_SIZE = 5                  # min sprite colors (excluding bg, base)
BICHROMATIC_FRAC = 0.3                # fraction of shape-groups with 2 colors
WIDTH_TOLERANCE = 2                   # near-optimal tolerance for width selection
MAX_RETRIES = 300                     # outer retry limit


def _split_lengths(total: int, count: int) -> list[int] | None:
    """Partition *total* into *count* positive integers, each <= MAX_RECT_LEN."""
    if count <= 0 or count > total or count * MAX_RECT_LEN < total:
        return None
    lengths = [1] * count
    remaining = total - count
    while remaining:
        choices = [i for i, v in enumerate(lengths) if v < MAX_RECT_LEN]
        if not choices:
            return None
        i = random.choice(choices)
        add = random.randint(1, min(remaining, MAX_RECT_LEN - lengths[i]))
        lengths[i] += add
        remaining -= add
    random.shuffle(lengths)
    return lengths


def _choose_widths(lengths: list[int], thick: int, dots: int, full_area: int) -> list[int] | None:
    """Pick a width for each length so total rect area falls within hole-fraction bounds."""
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    lo = max(0, math.ceil(HOLE_FRAC_LO * full_area) - dots)
    hi = max(lo, math.floor(HOLE_FRAC_HI * full_area) - dots)
    # DP over widths: paths[area_so_far] = list of chosen widths
    paths: dict[int, list[int]] = {0: []}
    for h in lengths:
        opts = list(range(1, max_w + 1))
        if h == 1 and max_w >= 2:
            opts = list(range(2, max_w + 1))
        random.shuffle(opts)
        items = list(paths.items())
        random.shuffle(items)
        nxt: dict[int, list[int]] = {}
        for cur, path in items:
            for w in opts:
                s = cur + h * w
                if s <= hi and s not in nxt:
                    nxt[s] = path + [w]
        paths = nxt
        if not paths:
            return None
    valid = [(s, p) for s, p in paths.items() if lo <= s <= hi]
    if not valid:
        return None
    target = random.randint(lo, hi)
    best = min(abs(s - target) for s, _ in valid)
    near = [p for s, p in valid if abs(s - target) <= best + WIDTH_TOLERANCE]
    return random.choice(near)


def _place_dot_markers(rects: list[tuple[int, int, int, int, bool]],
                       occupied: np.ndarray, thick: int, dots: int,
                       ) -> list[tuple[int, int, int, int, bool]] | None:
    """Attach *dots* 1x1 markers adjacent to existing rects. Returns None on failure."""
    for _ in range(dots):
        choices: list[tuple[int, int]] = []
        for r, c, h, w, _ in rects:
            for rr in range(r, r + h):
                for cc in (c - 1, c + w):
                    if 0 <= cc < thick and not occupied[rr, cc]:
                        choices.append((rr, cc))
        if not choices:
            return None
        rr, cc = random.choice(choices)
        occupied[rr, cc] = True
        rects.append((rr, cc, 1, 1, True))
    return rects


def _corridor_rects(long: int, thick: int, count: int, dots: int) -> list[tuple[int, int, int, int, bool]] | None:
    """Generate connected stacked rectangles along the long axis, plus dot markers."""
    min_len = max(count + 2, round(CORRIDOR_COVERAGE * long))
    if min_len > long - 1:
        min_len = long - 1
    hole_len = random.randint(min_len, long - 1)
    lengths = _split_lengths(hole_len, count)
    if lengths is None:
        return None
    widths = _choose_widths(lengths, thick, dots, long * thick)
    if widths is None:
        return None

    # Place rects sequentially, ensuring overlap with previous for connectivity
    start = random.randint(0, long - hole_len)
    r = start
    prev_c = prev_w = None
    rects: list[tuple[int, int, int, int, bool]] = []
    for h, w in zip(lengths, widths):
        c_hi = thick - w - 1
        if c_hi < 1:
            return None
        if prev_c is None:
            c = random.randint(1, c_hi)
        else:
            c_lo = max(1, prev_c - w + 1)
            c_hi = min(c_hi, prev_c + prev_w - 1)
            if c_lo > c_hi:
                return None
            c = random.randint(c_lo, c_hi)
        rects.append((r, c, h, w, False))
        r += h
        prev_c, prev_w = c, w

    occupied = np.zeros((long, thick), dtype=bool)
    for r, c, h, w, _ in rects:
        occupied[r : r + h, c : c + w] = True

    return _place_dot_markers(rects, occupied, thick, dots)


def _make_patch(h: int, w: int, cols: list[int]) -> np.ndarray:
    """Fill an (h, w) rectangle with 1 or 2 colors using a random pattern."""
    out = np.full((h, w), cols[0], dtype=int)
    if len(cols) == 1:
        return out

    b = cols[1]
    pattern = random.choice(["checker", "dots", "split"])
    rr, cc = np.indices((h, w))
    if pattern == "checker":
        out[(rr + cc + random.randint(0, 1)) % 2 == 0] = b
    elif pattern == "dots":
        step = random.choice([2, 3])
        mask = (rr * 2 + cc + random.randint(0, step - 1)) % step == 0
        if not mask.any() or mask.all():
            mask = np.zeros((h, w), dtype=bool)
            mask[random.randrange(h), random.randrange(w)] = True
        out[mask] = b
    elif h > 1 and (w == 1 or random.random() < 0.5):
        out[random.randint(1, h - 1):, :] = b
    else:
        out[:, random.randint(1, w - 1):] = b
    return out


def _make_pieces(rects: list[tuple[int, int, int, int, bool]],
                 bg: int, base: int) -> dict[int, np.ndarray] | None:
    """Assign colors and build patch arrays for each rect. Returns None on palette failure."""
    # Group non-dot rects by (h, w) shape
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (_, _, h, w, is_dot) in enumerate(rects):
        if not is_dot:
            groups.setdefault((h, w), []).append(i)
    if len(groups) < MIN_SHAPE_GROUPS:
        return None

    available = [c for c in range(1, 10) if c not in (bg, base)]
    max_colors = min(len(available), 2 * len(groups))
    if max_colors < MIN_PALETTE_SIZE:
        return None
    palette = random.sample(available, random.randint(MIN_PALETTE_SIZE, max_colors))

    # Decide which shape-groups get bichromatic fills
    keys = list(groups)
    random.shuffle(keys)
    needed_bi = max(0, len(palette) - len(keys))
    bi_count = min(len(keys), max(needed_bi, round(BICHROMATIC_FRAC * len(keys))))
    bi_keys = set(random.sample(keys, bi_count))

    # Assign colors: one primary per group, optional secondary for bichromatic
    unused = palette[:]
    random.shuffle(unused)
    group_cols: dict[tuple[int, int], list[int]] = {}
    for key in keys:
        group_cols[key] = [unused.pop() if unused else random.choice(palette)]
    for key in bi_keys:
        primary = group_cols[key][0]
        second = unused.pop() if unused else random.choice([c for c in palette if c != primary])
        group_cols[key].append(second)

    pieces: dict[int, np.ndarray] = {}
    for i, (_, _, h, w, is_dot) in enumerate(rects):
        pieces[i] = np.array([[0]], dtype=int) if is_dot else _make_patch(h, w, group_cols[(h, w)])
    return pieces


def _render_output(dh: int, dw: int, base: int,
                   rects: list[tuple[int, int, int, int, bool]],
                   pieces: dict[int, np.ndarray]) -> np.ndarray:
    """Build the expected output: base-colored template with pieces pasted in."""
    out = np.full((dh, dw), base, dtype=int)
    for i, (r, c, _, _, _) in enumerate(rects):
        paste(out, pieces[i], r, c)
    return out


def _scatter_sprites(H: int, W: int, bg: int, base: int,
                     tr0: int, tc0: int, dh: int, dw: int,
                     rects: list[tuple[int, int, int, int, bool]],
                     pieces: dict[int, np.ndarray]) -> np.ndarray:
    """Build the input grid: template frame with holes + scattered sprites outside."""
    inp = np.full((H, W), bg, dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = base
    for r, c, h, w, _ in rects:
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = bg

    # Place sprites in the outer region, largest first (shuffle for random tie-breaking)
    placement = np.full((H, W), bg, dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = base
    order = list(range(len(rects)))
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i].size, reverse=True)
    for i in order:
        patch = pieces[i]
        r, c = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, r, c)
        paste(placement, patch, r, c)
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: sample layout, build pieces, scatter sprites, oracle-verify."""
    vertical = random.random() < 0.5
    thick = random.randint(THICK_MIN, THICK_MAX)
    if vertical:
        H = random.randint(GRID_MIN, GRID_MAX)
        W = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        dh, dw = H, thick
        tr0, tc0 = 0, (0 if random.random() < 0.5 else W - thick)
    else:
        H = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        W = random.randint(GRID_MIN, GRID_MAX)
        dh, dw = thick, W
        tr0, tc0 = (0 if random.random() < 0.5 else H - thick), 0
    long = dh if vertical else dw

    # Sprite / dot count selection
    total = random.randint(SPRITE_COUNT_MIN, SPRITE_COUNT_MAX)
    min_dots = max(DOT_MIN, total - (long - DOT_LONG_BUFFER))
    max_dots = min(DOT_MAX, total - DOT_NON_DOT_BUFFER)
    if min_dots > max_dots:
        return None
    dots = random.randint(min_dots, max_dots)

    canon = _corridor_rects(long, thick, total - dots, dots)
    if canon is None:
        return None
    rects = canon if vertical else [(c, r, w, h, d) for r, c, h, w, d in canon]

    bg, base = random.sample(range(1, 10), 2)
    pieces = _make_pieces(rects, bg, base)
    if pieces is None:
        return None

    out = _render_output(dh, dw, base, rects, pieces)
    try:
        inp = _scatter_sprites(H, W, bg, base, tr0, tc0, dh, dw, rects, pieces)
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5dbc8537 input")
