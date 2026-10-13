import numpy as np
from collections import defaultdict


BG = 0
FILL_COLORS = (1, 2, 3, 4)
BORDER_COLORS = (5, 6, 7, 8, 9)
MAX_GRID_SIZE = 30
MAX_GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _crop_component(grid: np.ndarray, pts: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    mask = np.zeros_like(patch, dtype=bool)
    for r, c in pts:
        mask[r - r0, c - c0] = True
    return {
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "mask": mask,
    }


def _side_index_arrays(shape: tuple[int, int], side: str) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    if side == "top":
        return np.zeros(w, dtype=int), np.arange(w, dtype=int)
    if side == "bottom":
        return np.full(w, h - 1, dtype=int), np.arange(w, dtype=int)
    if side == "left":
        return np.arange(h, dtype=int), np.zeros(h, dtype=int)
    if side == "right":
        return np.arange(h, dtype=int), np.full(h, w - 1, dtype=int)
    raise ValueError(side)


def _infer_fragment(comp: dict) -> dict:
    patch = comp["patch"]
    mask = comp["mask"]
    colors = np.unique(patch[mask])
    assert len(colors) == 2, f"expected exactly two colors per fragment, got {colors}"

    candidates = []
    for side in ("top", "bottom", "left", "right"):
        rr, cc = _side_index_arrays(mask.shape, side)
        if not np.all(mask[rr, cc]):
            continue
        side_vals = patch[rr, cc]
        if len(np.unique(side_vals)) != 1:
            continue
        border_color = int(side_vals[0])
        rest = mask.copy()
        rest[rr, cc] = False
        if not rest.any():
            continue
        rest_colors = np.unique(patch[rest])
        if len(rest_colors) != 1:
            continue
        fill_color = int(rest_colors[0])
        if fill_color == border_color:
            continue
        candidates.append(
            {
                "side": side,
                "border_color": border_color,
                "fill_color": fill_color,
            }
        )

    assert len(candidates) == 1, f"could not infer unique border side: {candidates}"
    return {**comp, **candidates[0]}


def _paste_fragment(canvas: np.ndarray, fragment: dict, top: int, left: int) -> None:
    patch = fragment["patch"]
    mask = fragment["mask"]
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        v = int(patch[r, c])
        if canvas[rr, cc] not in (BG, v):
            raise AssertionError("pair overlap mismatch")
        canvas[rr, cc] = v


def _stitch_pair(a: dict, b: dict) -> np.ndarray:
    sa, sb = a["side"], b["side"]
    if {sa, sb} == {"left", "right"}:
        left = a if sa == "right" else b
        right = b if left is a else a
        h1, w1 = left["patch"].shape
        h2, w2 = right["patch"].shape
        assert h1 == h2, "left/right fragments must have the same height"
        out = np.zeros((h1, w1 + w2 - 1), dtype=int)
        _paste_fragment(out, left, 0, 0)
        _paste_fragment(out, right, 0, w1 - 1)
    elif {sa, sb} == {"top", "bottom"}:
        top = a if sa == "bottom" else b
        bottom = b if top is a else a
        h1, w1 = top["patch"].shape
        h2, w2 = bottom["patch"].shape
        assert w1 == w2, "top/bottom fragments must have the same width"
        out = np.zeros((h1 + h2 - 1, w1), dtype=int)
        _paste_fragment(out, top, 0, 0)
        _paste_fragment(out, bottom, h1 - 1, 0)
    else:
        raise AssertionError(f"invalid fragment side pairing: {sa}, {sb}")

    assert np.all(out != BG), "merged pair must form a solid rectangle"
    return out


def _pack_rectangles(rectangles: list[np.ndarray], max_width: int = MAX_GRID_SIZE) -> np.ndarray:
    assert rectangles, "at least one rectangle is required"
    placements: list[tuple[int, int, np.ndarray]] = []
    row_y = 0
    row_width = 0
    row_height = 0
    total_width = 0

    for rect in rectangles:
        h, w = rect.shape
        assert w <= max_width, f"rectangle width exceeds packing width: {w}"
        needed = w if row_width == 0 else row_width + 1 + w
        if row_width != 0 and needed > max_width:
            row_y += row_height + 1
            row_width = 0
            row_height = 0
        x = 0 if row_width == 0 else row_width + 1
        placements.append((row_y, x, rect))
        row_width = x + w
        row_height = max(row_height, h)
        total_width = max(total_width, row_width)

    total_height = max(y + rect.shape[0] for y, _, rect in placements)
    out = np.zeros((total_height, total_width), dtype=int)
    for y, x, rect in placements:
        h, w = rect.shape
        out[y : y + h, x : x + w] = rect
    return out


def _pair_rectangles_from_grid(grid: np.ndarray) -> list[tuple[tuple[int, int], np.ndarray]]:
    comps = [_crop_component(grid, pts) for pts in _components(grid != BG)]
    fragments = [_infer_fragment(comp) for comp in comps]

    by_border: dict[int, list[dict]] = defaultdict(list)
    for frag in fragments:
        by_border[frag["border_color"]].append(frag)

    pairs: list[tuple[tuple[int, int], np.ndarray]] = []
    for border_color, members in by_border.items():
        assert len(members) == 2, f"border color {border_color} must appear on exactly two fragments"
        merged = _stitch_pair(members[0], members[1])
        a_box = members[0]["bbox"]
        b_box = members[1]["bbox"]
        anchor = (min(a_box[0], b_box[0]), min(a_box[1], b_box[1]))
        pairs.append((anchor, merged))

    pairs.sort(key=lambda item: item[0])
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair fragments by their shared border color and stitch each pair into a rectangle.

    Input structure:
    - The background is black.
    - Several disconnected colored fragments are scattered on the canvas.
    - Each fragment has one fill color plus a one-cell border color that appears on exactly one full side: top, bottom, left, or right.
    - Every border color appears on exactly two fragments, marking one complementary pair.

    Rule:
    - Find each non-black fragment as a connected component.
    - Detect which side carries its special border color, then group the two fragments with the same border color.
    - Stitch right-to-left or bottom-to-top pairs so their border row or column overlaps into a single border seam and the union becomes a solid rectangle.
    - Sort the finished rectangles by the reading order of their source pairs and pack them tightly left-to-right with one black gap, wrapping to the next row when needed.

    Key insight:
    - The border color simultaneously identifies the matching partner and tells you which two opposite sides must be overlapped during the merge.
    """
    g = np.asarray(grid, dtype=int)
    if g.size == 0 or not np.any(g != BG):
        return g.copy()

    rectangles = [rect for _, rect in _pair_rectangles_from_grid(g)]
    return _pack_rectangles(rectangles)


def _sample_fill_pair(rng: np.random.Generator) -> tuple[int, int]:
    if rng.random() < 0.75:
        vals = rng.choice(FILL_COLORS, size=2, replace=False)
        return int(vals[0]), int(vals[1])
    color = int(rng.choice(FILL_COLORS))
    return color, color


def _make_vertical_pair(rng: np.random.Generator, border_color: int) -> dict:
    height = int(rng.integers(3, 7))
    left_w = int(rng.integers(3, 6))
    right_w = int(rng.integers(3, 6))
    fill_left, fill_right = _sample_fill_pair(rng)

    left = np.full((height, left_w), fill_left, dtype=int)
    right = np.full((height, right_w), fill_right, dtype=int)
    left[:, -1] = border_color
    right[:, 0] = border_color

    merged = np.full((height, left_w + right_w - 1), fill_left, dtype=int)
    merged[:, left_w - 1 :] = fill_right
    merged[:, left_w - 1] = border_color
    return {
        "fragments": [left, right],
        "merged": merged,
    }


def _make_horizontal_pair(rng: np.random.Generator, border_color: int) -> dict:
    width = int(rng.integers(3, 7))
    top_h = int(rng.integers(3, 6))
    bottom_h = int(rng.integers(3, 6))
    fill_top, fill_bottom = _sample_fill_pair(rng)

    top = np.full((top_h, width), fill_top, dtype=int)
    bottom = np.full((bottom_h, width), fill_bottom, dtype=int)
    top[-1, :] = border_color
    bottom[0, :] = border_color

    merged = np.full((top_h + bottom_h - 1, width), fill_top, dtype=int)
    merged[top_h - 1 :, :] = fill_bottom
    merged[top_h - 1, :] = border_color
    return {
        "fragments": [top, bottom],
        "merged": merged,
    }


def _slot_layouts(n_items: int, cell_h: int, cell_w: int) -> list[tuple[int, int]]:
    layouts = []
    for rows in range(2, 4):
        for cols in range(2, 4):
            if rows * cols < n_items:
                continue
            if rows * cell_h <= MAX_GRID_SIZE and cols * cell_w <= MAX_GRID_SIZE:
                layouts.append((rows, cols))
    return layouts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_pairs = int(rng.integers(2, 5))
    border_colors = rng.choice(BORDER_COLORS, size=n_pairs, replace=False)

    pair_specs = []
    fragments: list[dict] = []
    max_h = 0
    max_w = 0
    for border_color in border_colors:
        if rng.random() < 0.5:
            spec = _make_vertical_pair(rng, int(border_color))
        else:
            spec = _make_horizontal_pair(rng, int(border_color))
        pair_index = len(pair_specs)
        pair_specs.append(spec)
        for frag_patch in spec["fragments"]:
            h, w = frag_patch.shape
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            fragments.append({"pair_index": pair_index, "patch": frag_patch})

    cell_h = max_h + 3
    cell_w = max_w + 3
    layouts = _slot_layouts(len(fragments), cell_h, cell_w)
    if not layouts:
        return None

    rows, cols = layouts[int(rng.integers(0, len(layouts)))]
    base_h = rows * cell_h
    base_w = cols * cell_w
    max_top = MAX_GRID_SIZE - base_h
    max_left = MAX_GRID_SIZE - base_w
    top_pad = int(rng.integers(0, max_top + 1))
    left_pad = int(rng.integers(0, max_left + 1))
    grid = np.zeros((base_h + top_pad, base_w + left_pad), dtype=int)

    slot_ids = list(rng.permutation(rows * cols)[: len(fragments)])
    placed_pairs: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    merged_by_pair: dict[int, np.ndarray] = {}

    for frag, slot_id in zip(fragments, slot_ids):
        slot_r = slot_id // cols
        slot_c = slot_id % cols
        patch = frag["patch"]
        h, w = patch.shape
        r0 = top_pad + slot_r * cell_h + int(rng.integers(1, cell_h - h))
        c0 = left_pad + slot_c * cell_w + int(rng.integers(1, cell_w - w))
        if np.any(grid[r0 : r0 + h, c0 : c0 + w] != BG):
            return None
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        placed_pairs[frag["pair_index"]].append((r0, c0, r0 + h - 1, c0 + w - 1))
        merged_by_pair[frag["pair_index"]] = pair_specs[frag["pair_index"]]["merged"]

    if any(len(boxes) != 2 for boxes in placed_pairs.values()):
        return None

    ordered = []
    for pair_index, boxes in placed_pairs.items():
        anchor = (min(box[0] for box in boxes), min(box[1] for box in boxes))
        ordered.append((anchor, merged_by_pair[pair_index]))
    ordered.sort(key=lambda item: item[0])
    expected = _pack_rectangles([rect for _, rect in ordered])

    if expected.shape[0] > MAX_GRID_SIZE or expected.shape[1] > MAX_GRID_SIZE:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_RETRIES} attempts")
