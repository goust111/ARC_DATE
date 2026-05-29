# Variant of edb79dae
# Dimension: D6 — 第二步变换：重着色规则 (transform)
# Change: 放置后的所有精灵根据图例中的形状-颜色映射进行统一重着色（s2.color = d2.color），同形状精灵获得相同输出颜色
#         → 多色重着色（图例指定颜色图案）

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, components_mask, find_components, get_bg, paste, random_free_location


# Design analysis for the D6 variant
# ----------------------------------
# spec_variant:
#   unchanged_core:
#     - Keep the original framed layout, placeholder grid, one outside sprite per placeholder color,
#       and one outside legend per placeholder color.
#     - Step 1 remains "find same-color sprite outside the frame and stamp its shape into the matching
#       placeholder bbox".
#   d6_change:
#     - The legend no longer maps a source color to one flat target color.
#     - Each legend is now a full rectangle of size N x (N+1):
#         leftmost column: all source color
#         right N x N block: a two-color pattern sample
#     - The pattern sample has the same N x N bbox size as the sprite/placeholder.
#     - During stamping, sprite cells inherit the legend's sample color at the same relative (dr, dc)
#       position, so the sprite becomes a cut-out of the legend pattern rather than a monochrome blob.
#   concrete_flow_trace:
#     - Original:
#         placeholder color 3 -> external sprite mask of color 3 -> legend says 3 maps to 8
#         -> every occupied sprite cell inside the frame becomes 8.
#     - Variant:
#         placeholder color 3 -> external sprite mask of color 3 -> legend is
#           [3 | 8 2 8
#            3 | 2 8 2
#            3 | 8 2 8]
#         -> occupied sprite cells become 8/2/8... according to their relative coordinates.
#     - Difference:
#         step 1 still chooses shape by source color, but step 2 now copies a visible per-pixel color
#         arrangement instead of a single color id.
#   complexity_calibration:
#     - Patterns are restricted to a small motif family (checker / horizontal stripes / vertical stripes / diagonal bands)
#       using exactly two colors. This is more informative than a flat recolor but still learnable from 2-4 examples.
#     - Each source color appears in multiple placeholder slots whenever possible, so the same patterned recoloring
#       repeats and becomes visually inferable.
#   implicit_assumption_audit:
#     - The original legend parser assumed exactly two colors and inferred src/tgt by area; that breaks once a legend
#       contains more than two colors. The new solver therefore detects multicolor connected components whose bbox is
#       a full N x (N+1) rectangle and whose first column is a uniform placeholder color.
#     - The original repaint loop wrote one scalar tgt color for every sprite pixel; that would overwrite any intended
#       pattern. The new loop instead indexes the legend tile at (dr, dc).
#     - Sprite extraction still relies on there being exactly one non-legend source-colored sprite bbox of size N x N.
#       The generator preserves that by excluding legend cells and keeping sprite templates separate.
#   placement_legality:
#     - Legends remain opaque rectangular patches with no background cells, so each one is a single multicolor
#       connected component and is easy to detect.
#     - Outside patches are placed with the same reserve grid and one-cell padding as the original task, preventing
#       overlap or accidental merging.
#   observability:
#     - Because output sprites are no longer solid blocks of one color, the D6 change is visible immediately.
#     - The legend explicitly shows the pattern tile next to a source-color tag column, so a human can align the
#       pattern with the sprite bbox and infer that the tile is being clipped by the sprite mask.
#   cleanliness:
#     - No extra overlays are introduced inside the frame; only the sprite interiors become patterned.
#     - The legend encoding stays compact and rectangular, which keeps the outside area readable.
#   new_solver_assumptions_backed_by_generator:
#     - Every placeholder color has one legend whose tile shape equals that placeholder's NxN size.
#     - Legend tiles use at least two colors on occupied sprite cells, preventing degeneration to the original
#       monochrome recolor.
#     - Legend tiles do not reuse the source color, so the source tag column is visually and programmatically distinct.


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    """
    Find a rectangular border component:
      - all bbox-border cells are this color,
      - bbox-interior is not entirely this color.
    Return (frame_color, (r0,c0,r1,c1)).
    """
    candidates = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        for pts in components_mask(g == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == col):
                continue
            if np.all(sub[~border] == col):
                continue
            candidates.append((col, (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected exactly one rectangular frame component, found {len(candidates)}"
    frame_col, bbox = candidates[0]
    return int(frame_col), bbox


def _bbox_h(b: tuple[int, int, int, int]) -> int:
    return int(b[2] - b[0] + 1)


def _bbox_w(b: tuple[int, int, int, int]) -> int:
    return int(b[3] - b[1] + 1)


def _extract_legend_patterns(
    g: np.ndarray,
    outside: np.ndarray,
    bg: int,
    frame_col: int,
    placeholder_colors: set[int],
    phw: dict[int, tuple[int, int]],
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    masked = np.full_like(g, bg)
    masked[outside] = g[outside]
    comps = find_components(masked, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    pattern_map: dict[int, np.ndarray] = {}
    legend_mask = np.zeros(g.shape, dtype=bool)

    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = _bbox_h((r0, c0, r1, c1))
        w = _bbox_w((r0, c0, r1, c1))
        if w != h + 1:
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if comp["area"] != sub.size:
            continue
        if bg in sub or frame_col in sub:
            continue

        src_col = int(sub[0, 0])
        if src_col not in placeholder_colors:
            continue
        if not np.all(sub[:, 0] == src_col):
            continue

        tile = sub[:, 1:]
        if tile.shape != phw[src_col]:
            continue
        tile_colors = set(map(int, np.unique(tile)))
        if src_col in tile_colors:
            continue
        if len(tile_colors) < 2:
            continue

        assert src_col not in pattern_map, f"duplicate pattern legend for placeholder color {src_col}"
        pattern_map[src_col] = tile.copy()
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    assert set(pattern_map) == placeholder_colors, (
        f"legend mappings {sorted(pattern_map)} do not match placeholders {sorted(placeholder_colors)}"
    )
    return pattern_map, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp patterned sprites into a framed region using placeholder positions and multicolor legends.

    Input structure:
    - bg background, a rectangular single-color frame, and solid placeholder blocks inside the frame.
    - Outside the frame: one monochrome sprite template per placeholder color, plus one rectangular legend
      whose left column is the placeholder color and whose right NxN block is a multicolor pattern sample.

    Rule:
    - Find the frame and the placeholder blocks inside it.
    - For each placeholder color, locate the matching-color sprite shape outside the frame.
    - Locate that color's legend and read its NxN pattern sample.
    - Stamp the sprite at every matching placeholder position, coloring each occupied sprite cell with the
      legend sample at the same relative coordinate.

    Key insight: The outside legend no longer names a single destination color; it provides a whole color tile,
    and the output is formed by clipping that tile with the matched sprite mask.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    inner_mask = np.zeros((H, W), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    placeholder_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    assert placeholder_colors, "no placeholder colors found inside frame"

    placeholders: list[tuple[int, tuple[int, int, int, int]]] = []
    for col in sorted(placeholder_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            placeholders.append((col, (r0, c0, r1, c1)))

    phw: dict[int, tuple[int, int]] = {}
    for col in placeholder_colors:
        dims = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for c, b in placeholders if c == col]
        assert dims, f"no placeholder component found for color {col}"
        unique_dims = sorted(set(dims))
        assert len(unique_dims) == 1, f"ambiguous placeholder sizes for color {col}: {unique_dims}"
        phw[col] = unique_dims[0]
        assert phw[col][0] == phw[col][1], f"expected square placeholder for color {col}, got {phw[col]}"

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    pattern_map, legend_mask = _extract_legend_patterns(g, outside, bg, frame_col, placeholder_colors, phw)

    comps = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    cid = 0
    for col in colors:
        for pts in components_mask((g == col) & outside, connectivity=4):
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            if all(bool(legend_mask[r, c]) for r, c in pts_tuples):
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(pts_tuples),
                }
            )
            cid += 1

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = [c for c in comps if c["col"] == src_col]
        assert all_candidates, f"no non-legend sprite found for color {src_col}"
        want_h, want_w = phw[src_col]
        sized = [
            c
            for c in all_candidates
            if _bbox_h(c["bbox"]) == want_h and _bbox_w(c["bbox"]) == want_w
        ]
        assert len(sized) == 1, f"expected one size-matched sprite for color {src_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]

        pts_set = set(base["pts"])
        for c in all_candidates:
            if c["id"] == base["id"]:
                continue
            r0, c0, r1, c1 = c["bbox"]
            assert sr0 <= r0 and sr1 >= r1 and sc0 <= c0 and sc1 >= c1, (
                f"extra sprite component for color {src_col} is outside the base bbox"
            )
            pts_set.update(c["pts"])

        coords = tuple(sorted((r - sr0, c - sc0) for r, c in pts_set))
        sprite_coords[src_col] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        assert src_col in sprite_coords and src_col in pattern_map, (
            f"placeholder color {src_col} missing sprite or legend pattern"
        )
        tile = pattern_map[src_col]
        coords, sh, sw = sprite_coords[src_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        assert tile.shape == (sh, sw), f"legend tile shape does not match sprite for color {src_col}"
        assert pr0 + sh <= H and pc0 + sw <= W, f"sprite for color {src_col} does not fit at placeholder"
        for dr, dc in coords:
            rr, cc = pr0 + dr, pc0 + dc
            out[rr, cc] = int(tile[dr, dc])

    return out


# --------------- generator (o5x-d6) ---------------

# === CFG: tunables ===
FRAME_COL = 5
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 5)
SPRITE_FILL_MIN = 0.5
SPRITE_FILL_MAX = 0.9
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 300
PATTERN_NAMES = ("checker", "vertical", "horizontal", "diagonal")


def _legend(src: int, pattern: np.ndarray) -> np.ndarray:
    """Build an N x (N+1) legend block: source tag column + N x N pattern tile."""
    src_col = np.full((pattern.shape[0], 1), int(src), dtype=int)
    return np.hstack([src_col, np.asarray(pattern, dtype=int)])


def _sprite_mask(n: int) -> np.ndarray:
    """Generate a random NxN binary mask with fill ratio in [SPRITE_FILL_MIN, SPRITE_FILL_MAX]."""
    lo = max((n * n + 1) // 2, int(np.ceil(SPRITE_FILL_MIN * n * n)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)
    target = random.randint(lo, hi)

    mask = np.zeros((n, n), dtype=bool)
    c = random.randrange(n)
    for r in range(n):
        c = max(0, min(n - 1, c + random.choice([-1, 0, 1])))
        mask[r, c] = True
    mask[random.randrange(n), :] = True

    if random.random() < 0.5:
        mask = mask.T.copy()
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)

    while int(mask.sum()) < target:
        dilated = np.zeros_like(mask)
        dilated[1:, :] |= mask[:-1, :]
        dilated[:-1, :] |= mask[1:, :]
        dilated[:, 1:] |= mask[:, :-1]
        dilated[:, :-1] |= mask[:, 1:]
        frontier = list(zip(*np.where(dilated & ~mask)))
        r, c = random.choice(frontier)
        mask[r, c] = True
    return mask


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    """Render a binary mask as a colored sprite on bg."""
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _patterned_sprite(mask: np.ndarray, tile: np.ndarray, bg: int) -> np.ndarray:
    """Render a binary mask by clipping an N x N color tile."""
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = np.asarray(tile, dtype=int)[mask]
    return out


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    """Content area needed for grid_rows x grid_cols of NxN blocks with 1-cell gaps."""
    return (grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1)


def _choose_structure() -> tuple[int, int, int, int, int] | None:
    """Pick (n, rows, cols, empty_count, total) satisfying frame size constraints."""
    total = random.randint(*TOTAL_BLOCKS_RANGE)
    options = []
    for n in range(*SPRITE_N_RANGE):
        for rows in range(*GRID_ROWS_RANGE):
            for cols in range(*GRID_COLS_RANGE):
                ch, cw = _content_extent(n, rows, cols)
                base_fh = ch + 4
                base_fw = cw + 4
                slots = rows * cols
                empty = slots - total
                if base_fh <= MAX_FRAME_DIM and base_fw <= MAX_FRAME_DIM and 0 <= empty <= MAX_EMPTY_SLOTS:
                    options.append((n, rows, cols, empty, total))
    return random.choice(options) if options else None


def _choose_empty_slots(
    rows: int, cols: int, empty_count: int
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    """Choose empty slots, ensuring every row and column keeps at least one filled slot."""
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        filled = [s for s in slots if s not in empty]
        if len({r for r, _ in filled}) == rows and len({c for _, c in filled}) == cols:
            valid.append(empty)
    empty_slots = random.choice(valid)
    filled_slots = [s for s in slots if s not in empty_slots]
    random.shuffle(filled_slots)
    return empty_slots, filled_slots


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    """Create an h x w frame filled with bg and a 1-cell FRAME_COL border."""
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _place_patch(
    grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int,
    *, transparent: bool,
) -> bool:
    """Find a free location for patch and paste it onto grid/reserve. Returns success."""
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return True


def _make_pattern_tile(mask: np.ndarray, src: int, usable: list[int]) -> tuple[np.ndarray, str] | None:
    """Pick a visible two-color pattern tile that uses both colors on occupied sprite cells."""
    n = int(mask.shape[0])
    color_pool = [c for c in usable if c != src]
    if len(color_pool) < 2:
        return None

    def build(name: str, a: int, b: int) -> np.ndarray:
        rr, cc = np.indices((n, n))
        if name == "checker":
            selector = (rr + cc) % 2
        elif name == "vertical":
            selector = cc % 2
        elif name == "horizontal":
            selector = rr % 2
        else:
            selector = ((rr // 2) + cc) % 2
        return np.where(selector == 0, int(a), int(b)).astype(int)

    tries = [(random.sample(color_pool, 2), random.sample(PATTERN_NAMES, len(PATTERN_NAMES))) for _ in range(10)]
    for (a, b), names in tries:
        for name in names:
            tile = build(name, a, b)
            if len(set(map(int, np.unique(tile[mask])))) >= 2:
                return tile, name
    return None


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, _ = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    masks = {src: _sprite_mask(n) for src in src_cols}

    patterns: dict[int, np.ndarray] = {}
    pattern_names: dict[int, str] = {}
    for src in src_cols:
        built = _make_pattern_tile(masks[src], src, usable)
        if built is None:
            return None
        tile, name = built
        patterns[src] = tile
        pattern_names[src] = name

    if len(set(pattern_names.values())) == 1 and k > 1 and random.random() < 0.7:
        return None

    ch, cw = _content_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + 4)
    min_fw = max(MIN_FRAME_W, cw + 4)
    frame_h = random.randint(min_fh, min(MAX_FRAME_DIM, max(min_fh, ch + 2 + FRAME_MARGIN_SLACK)))
    frame_w = random.randint(min_fw, min(MAX_FRAME_DIM, max(min_fw, cw + 2 + FRAME_MARGIN_SLACK)))
    y_margin = frame_h - 2 - ch
    x_margin = frame_w - 2 - cw
    top_margin = random.randint(1, y_margin - 1)
    left_margin = random.randint(1, x_margin - 1)

    _, filled_slots = _choose_empty_slots(rows, cols, empty_count)
    cell_colors = {}
    for slot, col in zip(filled_slots[:k], src_cols):
        cell_colors[slot] = col
    for slot in filled_slots[k:]:
        cell_colors[slot] = random.choice(src_cols)
    if any(sum(1 for s in cell_colors.values() if s == src) >= 2 for src in src_cols):
        pass
    elif len(filled_slots) > k:
        return None

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        paste(output_frame, _patterned_sprite(masks[src], patterns[src], bg), r0, c0, transparent=bg)

    ext_area = k * (n * n + n * (n + 1) + 12)
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h and w >= frame_w and h * w - frame_h * frame_w >= ext_area
    ]
    if not possible_dims:
        return None
    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    fr0 = random.choice([0, h - frame_h])
    fc0 = random.choice([0, w - frame_w])
    if h - frame_h > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fr0 = random.randrange(0, h - frame_h + 1)
    if w - frame_w > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fc0 = random.randrange(0, w - frame_w + 1)
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    patches: list[tuple[np.ndarray, bool]] = []
    for src in src_cols:
        patches.append((_colored_sprite(masks[src], src, bg), True))
        patches.append((_legend(src, patterns[src]), False))
    random.shuffle(patches)
    patches.sort(key=lambda item: item[0].size, reverse=True)
    for patch, transparent in patches:
        if not _place_patch(grid, reserve, patch, bg, transparent=transparent):
            return None

    try:
        if not np.array_equal(main(grid), output_frame):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae D6 input")
