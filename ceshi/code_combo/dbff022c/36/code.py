# Combo variant of dbff022c (index 36)
# combo_spec: D4/3+D11/2
# Name: 矩形框架精灵 + 网格排列

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior:
#    Original dbff022c can contain irregular hollow sprites scattered around the grid.
#    Here every sprite is a strict rectangular frame. Example: with a top palette
#    [2 3 4] / [7 8 6], the body of the grid may contain a 2x3 lattice of cells; inside
#    each cell sits one frame such as 3x5 or 4x4. A border-2 frame gets its whole inner
#    rectangle filled with 7, a border-3 frame gets 8, and one distractor frame uses a
#    color absent from the palette so its interior stays 0. The original flood-fill style
#    "any enclosed hole" logic is therefore narrowed to "interiors of detected rectangle
#    frames in a regular array".
# b) Complexity calibration:
#    The rule is easier to localize because objects are aligned, so the generator increases
#    object count to 4-6 and reuses a small number of palette keys across multiple cells.
#    This keeps the visual effect obvious without exploding the color vocabulary.
# c) Implicit assumptions changed from the original solver:
#    - Border-adjacency voting over arbitrary enclosed background regions is no longer used.
#    - Each relevant non-palette component must be a single-color rectangle border whose
#      bbox interior is all background in the input.
#    - Frames are expected to form multiple row/column bands; ordering uses those bands,
#      but the fill rule itself only depends on border color -> palette mapping.
# d) Placement legality:
#    The generator uses fixed-size lattice cells with at least one background cell between
#    neighboring cells and between the lattice and the palette strip, so frames never
#    overlap, touch each other, or touch the palette.
# e/f/g) Visual observability and cleanliness:
#    Repeated row/column alignment makes the table-like structure immediately visible.
#    Rectangles vary in size inside the cells so the D4/3 change is visible, while palette
#    keys are reused across multiple frames so the color mapping can be inferred from only
#    a few examples.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 2xN or Nx2 palette block on a grid edge.

    Returns (side, start, end) where:
      - side in {"top","bottom","left","right"}
      - for top/bottom: start/end are column indices [start,end)
      - for left/right: start/end are row indices [start,end)
    """
    h, w = g.shape
    best = None  # (score_tuple, side, start, end)

    def isolated(side: str, start: int, end: int) -> bool:
        pal = np.zeros((h, w), dtype=bool)
        if side == "top":
            pal[0:2, start:end] = True
        elif side == "bottom":
            pal[h - 2 : h, start:end] = True
        elif side == "left":
            pal[start:end, 0:2] = True
        else:
            pal[start:end, w - 2 : w] = True

        for r, c in zip(*np.where(pal), strict=False):
            v = int(g[int(r), int(c)])
            if v == bg:
                continue
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc]:
                    if int(g[rr, cc]) == v:
                        return False
        return True

    def consider_segment(side: str, seg_a: np.ndarray, seg_b: np.ndarray, start: int, end: int) -> None:
        nonlocal best
        da = set(map(int, np.unique(seg_a)))
        db = set(map(int, np.unique(seg_b)))
        da.discard(bg)
        db.discard(bg)
        if len(da) < 2 or len(db) < 2:
            return
        merged = da | db
        if len(merged) < 3:
            return
        border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
        if not border_keys or len(border_keys) != len(set(border_keys)):
            return
        if not isolated(side, start, end):
            return
        nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
        score = (len(set(border_keys)), len(merged), -(end - start), nonbg)
        cand = (score, side, start, end)
        if best is None or cand[0] > best[0]:
            best = cand

    if h >= 2:
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                consider_segment("top", g[0, c0:c1], g[1, c0:c1], c0, c1)
                consider_segment("bottom", g[h - 1, c0:c1], g[h - 2, c0:c1], c0, c1)
    if w >= 2:
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                consider_segment("left", g[r0:r1, 0], g[r0:r1, 1], r0, r1)
                consider_segment("right", g[r0:r1, w - 1], g[r0:r1, w - 2], r0, r1)

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """Return (mapping border->fill, palette_mask)."""
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}

    if side == "top":
        pal_mask[0:2, start:end] = True
        border_line = g[0, start:end]
        fill_line = g[1, start:end]
    elif side == "bottom":
        pal_mask[h - 2 : h, start:end] = True
        border_line = g[h - 1, start:end]
        fill_line = g[h - 2, start:end]
    elif side == "left":
        pal_mask[start:end, 0:2] = True
        border_line = g[start:end, 0]
        fill_line = g[start:end, 1]
    else:
        pal_mask[start:end, w - 2 : w] = True
        border_line = g[start:end, w - 1]
        fill_line = g[start:end, w - 2]

    for b, f in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        b = int(b)
        f = int(f)
        if b != bg:
            mapping[b] = f

    return mapping, pal_mask


def _frame_border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    mask = np.ones((h, w), dtype=bool)
    mask[1:-1, 1:-1] = False
    return mask


def _find_rect_frames(g: np.ndarray, bg: int, pal_mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """
    Find monochrome rectangle frames outside the palette.

    Returns [(r0, c0, r1, c1, color), ...] sorted later in grid order.
    """
    comps = components_mask((g != bg) & (~pal_mask), connectivity=4)
    frames: list[tuple[int, int, int, int, int]] = []
    for pts in comps:
        rows = pts[:, 0]
        cols = pts[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        colors = {int(g[int(r), int(c)]) for r, c in pts}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        box = g[r0 : r1 + 1, c0 : c1 + 1]
        local_pal = pal_mask[r0 : r1 + 1, c0 : c1 + 1]
        if local_pal.any():
            continue
        border_mask = _frame_border_mask(box.shape)
        if int(pts.shape[0]) != int(border_mask.sum()):
            continue
        if not np.all(box[border_mask] == color):
            continue
        if not np.all(box[~border_mask] == bg):
            continue
        frames.append((r0, c0, r1, c1, color))
    return frames


def _cluster_axis(vals2: list[int], max_gap: int = 2) -> list[int]:
    """Cluster doubled centers into regular row/column bands."""
    order = sorted(range(len(vals2)), key=lambda i: vals2[i])
    groups: list[list[int]] = []
    current: list[int] = []
    current_vals: list[int] = []
    for idx in order:
        val = int(vals2[idx])
        if not current or val - current_vals[-1] <= max_gap:
            current.append(idx)
            current_vals.append(val)
        else:
            groups.append(current)
            current = [idx]
            current_vals = [val]
    if current:
        groups.append(current)

    band_of = [0] * len(vals2)
    for band, ids in enumerate(groups):
        for idx in ids:
            band_of[idx] = band
    return band_of


def _ordered_frames(frames: list[tuple[int, int, int, int, int]]) -> list[tuple[int, int, int, int, int]]:
    """Sort frames by inferred lattice row then lattice column."""
    if not frames:
        return []
    row_centers2 = [r0 + r1 for r0, _, r1, _, _ in frames]
    col_centers2 = [c0 + c1 for _, c0, _, c1, _ in frames]
    row_band = _cluster_axis(row_centers2)
    col_band = _cluster_axis(col_centers2)
    order = sorted(
        range(len(frames)),
        key=lambda i: (row_band[i], col_band[i], frames[i][0], frames[i][1]),
    )
    return [frames[i] for i in order]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the edge legend and fill the interiors of rectangular frame sprites.

    Input structure:
    - A 2xN or Nx2 palette block sits on one outer edge. The outer line lists unique
      frame-border colors and the inner line lists their fill colors.
    - Away from that palette, all relevant objects are single-color hollow rectangles.
      They are arranged on a regular row/column lattice, with at least one background
      cell separating neighboring frames and the palette strip.

    Rule:
    - Detect the palette and convert it into a border-color -> fill-color lookup.
    - Find every monochrome rectangular frame outside the palette.
    - For each frame whose border color appears in the palette, paint its entire inner
      rectangle with the mapped fill color. Frames whose border colors are absent from
      the palette are distractors and remain hollow.

    Key insight: once the objects are constrained to clean rectangle frames in a visible
    grid arrangement, there is no need to reason about arbitrary enclosed holes. The
    deciding signal is simply each frame's border color matched against the edge legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    for r0, c0, r1, c1, color in _ordered_frames(_find_rect_frames(g, bg, pal_mask)):
        fill = mapping.get(color)
        if fill is None:
            continue
        out[r0 + 1 : r1, c0 + 1 : c1] = int(fill)
    return out


# --------------- generator ---------------

_LAYOUTS = ((2, 2), (2, 3), (3, 2))
_PALETTE_SIDES = ("top", "bottom", "left", "right")
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 4
_CELL_H_MIN, _CELL_H_MAX = 4, 6
_CELL_W_MIN, _CELL_W_MAX = 4, 6
_GAP_MIN, _GAP_MAX = 1, 2
_MARGIN = 1
_PALETTE_SEP = 1
_MAX_RETRIES = 200


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    n = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = border_keys
        pal[inner] = fill_values
    else:
        pal = np.zeros((n, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = border_keys
        pal[:, inner] = fill_values
    return pal


def _place_palette(grid: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> np.ndarray:
    h, w = grid.shape
    ph, pw = pal.shape
    pal_mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        r0, c0 = 0, rng.randint(0, w - pw)
    elif side == "bottom":
        r0, c0 = h - ph, rng.randint(0, w - pw)
    elif side == "left":
        r0, c0 = rng.randint(0, h - ph), 0
    else:
        r0, c0 = rng.randint(0, h - ph), w - pw
    grid[r0 : r0 + ph, c0 : c0 + pw] = pal
    pal_mask[r0 : r0 + ph, c0 : c0 + pw] = True
    return pal_mask


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, border: int, fill: int | None) -> None:
    grid[r0, c0 : c0 + w] = border
    grid[r0 + h - 1, c0 : c0 + w] = border
    grid[r0 : r0 + h, c0] = border
    grid[r0 : r0 + h, c0 + w - 1] = border
    if fill is not None and h >= 3 and w >= 3:
        grid[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = fill


def _choose_layout(rng: random.Random) -> tuple[int, int, int, int, int, int]:
    rows, cols = rng.choice(_LAYOUTS)
    cell_h = rng.randint(_CELL_H_MIN, _CELL_H_MAX)
    cell_w = rng.randint(_CELL_W_MIN, _CELL_W_MAX)
    gap_y = rng.randint(_GAP_MIN, _GAP_MAX)
    gap_x = rng.randint(_GAP_MIN, _GAP_MAX)
    return rows, cols, cell_h, cell_w, gap_y, gap_x


def _layout_bounds(
    side: str, rows: int, cols: int, cell_h: int, cell_w: int, gap_y: int, gap_x: int
) -> tuple[int, int, int, int]:
    layout_h = rows * cell_h + (rows - 1) * gap_y
    layout_w = cols * cell_w + (cols - 1) * gap_x
    if side in ("top", "bottom"):
        h = layout_h + 2 * _MARGIN + 2 + _PALETTE_SEP
        w = layout_w + 2 * _MARGIN
        r0 = _MARGIN + (2 + _PALETTE_SEP if side == "top" else 0)
        c0 = _MARGIN
    else:
        h = layout_h + 2 * _MARGIN
        w = layout_w + 2 * _MARGIN + 2 + _PALETTE_SEP
        r0 = _MARGIN
        c0 = _MARGIN + (2 + _PALETTE_SEP if side == "left" else 0)
    return h, w, r0, c0


def _frame_sizes(rng: random.Random, count: int, cell_h: int, cell_w: int) -> list[tuple[int, int]]:
    sizes = [(rng.randint(3, cell_h), rng.randint(3, cell_w)) for _ in range(count)]
    if len(set(sizes)) == 1:
        i = rng.randrange(count)
        h, w = sizes[i]
        if cell_h > 3 and h == cell_h:
            h -= 1
        elif cell_h > h:
            h += 1
        elif cell_w > 3 and w == cell_w:
            w -= 1
        elif cell_w > w:
            w += 1
        sizes[i] = (max(3, min(cell_h, h)), max(3, min(cell_w, w)))
    return sizes


def _choose_colors(
    rng: random.Random, n_palette: int, n_frames: int
) -> tuple[list[int], list[int], dict[int, int], int, list[int]]:
    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_keys = colors[:n_palette]
    fill_values = colors[n_palette : 2 * n_palette]
    distractor = colors[2 * n_palette]
    mapping = dict(zip(border_keys, fill_values))

    used_key_count = 2
    used_keys = rng.sample(border_keys, used_key_count)

    repeated = list(used_keys)
    while len(repeated) < n_frames - 1:
        repeated.append(rng.choice(used_keys))
    rng.shuffle(repeated)
    frame_colors = repeated + [distractor]
    rng.shuffle(frame_colors)
    return border_keys, fill_values, mapping, distractor, frame_colors


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rng = random.Random()
    bg = 0
    side = rng.choice(_PALETTE_SIDES)
    rows, cols, cell_h, cell_w, gap_y, gap_x = _choose_layout(rng)
    h, w, layout_r0, layout_c0 = _layout_bounds(side, rows, cols, cell_h, cell_w, gap_y, gap_x)

    span = w if side in ("top", "bottom") else h
    max_n = min(_PALETTE_N_MAX, span)
    if max_n < _PALETTE_N_MIN:
        return None
    n_palette = rng.randint(_PALETTE_N_MIN, max_n)

    border_keys, fill_values, mapping, distractor, frame_colors = _choose_colors(rng, n_palette, rows * cols)
    palette = _build_palette(side, border_keys, fill_values)

    output = np.full((h, w), bg, dtype=int)
    pal_mask = _place_palette(output, palette, side, rng)

    sizes = _frame_sizes(rng, rows * cols, cell_h, cell_w)
    placed: list[tuple[int, int, int, int, int]] = []
    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            fh, fw = sizes[idx]
            cell_r0 = layout_r0 + rr * (cell_h + gap_y)
            cell_c0 = layout_c0 + cc * (cell_w + gap_x)
            fr0 = cell_r0 + (cell_h - fh) // 2
            fc0 = cell_c0 + (cell_w - fw) // 2
            border = frame_colors[idx]
            fill = mapping.get(border)
            _draw_frame(output, fr0, fc0, fh, fw, border, fill)
            placed.append((fr0, fc0, fh, fw, border))
            idx += 1

    inp = output.copy()
    for r0, c0, fh, fw, color in placed:
        if color in mapping:
            inp[r0 + 1 : r0 + fh - 1, c0 + 1 : c0 + fw - 1] = bg

    frames = _find_rect_frames(inp, bg, pal_mask)
    if len(frames) != rows * cols:
        return None

    border = np.concatenate([inp[0, :], inp[-1, :], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None
    return inp, output


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        generated = _gen_attempt()
        if generated is not None:
            inp, _ = generated
            return inp
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
