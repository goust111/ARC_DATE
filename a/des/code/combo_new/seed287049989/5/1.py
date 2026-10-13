from __future__ import annotations

import numpy as np


Rect = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> Rect:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_area(box: Rect) -> int:
    r0, c0, r1, c1 = box
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _contains(outer: Rect, inner: Rect) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comps.append({"color": color, "coords": arr, "bbox": _bbox(arr)})
    return comps


def _is_rect_frame(coords: np.ndarray, box: Rect) -> bool:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if int(coords.shape[0]) != expected:
        return False
    mask = np.zeros((h, w), dtype=bool)
    rel = coords - np.asarray([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return bool(
        mask[0].all()
        and mask[-1].all()
        and mask[:, 0].all()
        and mask[:, -1].all()
        and not mask[1:-1, 1:-1].any()
    )


def _frame_forest(frames: list[dict]) -> tuple[list[int], dict[int, list[int]]]:
    parent = [-1] * len(frames)
    children = {i: [] for i in range(len(frames))}
    for i, frame in enumerate(frames):
        containers = [j for j, other in enumerate(frames) if i != j and _contains(other["bbox"], frame["bbox"])]
        if containers:
            parent_idx = min(containers, key=lambda j: _bbox_area(frames[j]["bbox"]))
            parent[i] = int(parent_idx)
            children[int(parent_idx)].append(i)
    return parent, children


def _subtree(root: int, children: dict[int, list[int]]) -> list[int]:
    order = [root]
    for child in children[root]:
        order.extend(_subtree(child, children))
    return order


def _paint_scaled(
    out: np.ndarray,
    region_mask: np.ndarray,
    origin_r: int,
    origin_c: int,
    template_mask: np.ndarray,
    scale: int,
    color: int,
) -> None:
    scaled = np.kron(template_mask.astype(np.uint8), np.ones((scale, scale), dtype=np.uint8)).astype(bool)
    h, w = scaled.shape
    r1 = min(out.shape[0], origin_r + h)
    c1 = min(out.shape[1], origin_c + w)
    if r1 <= origin_r or c1 <= origin_c:
        return
    region = region_mask[origin_r:r1, origin_c:c1]
    sub = out[origin_r:r1, origin_c:c1]
    sub[region & scaled[: r1 - origin_r, : c1 - origin_c]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each nested rectangular frame structure by painting clipped scaled copies of its innermost template.

    Input structure:
    - The background is the most common color.
    - Several non-overlapping structures consist of nested one-cell-thick rectangular frames, with different colors from inner to outer.
    - The innermost frame encloses a small asymmetric seed pattern drawn in one foreground color on the background.

    Rule:
    - For each structure, sort its rectangular frame layers from inner to outer.
    - Read the non-background pixels inside the innermost frame as a binary template mask.
    - For layer k counted from the inside, clear that layer's interior region to the background, then paint a top-left-aligned scale-(k+1) copy of the template in the k-th frame's color, clipping it to that layer's region.
    - Keep every original frame border unchanged.

    Key insight:
    - The nesting depth determines the scale factor, while each layer inherits its paint color directly from the matching frame border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    comps = _components_by_color(g, bg)
    frames = [comp for comp in comps if _is_rect_frame(comp["coords"], comp["bbox"])]
    assert frames, "expected at least one rectangular frame"

    parent, children = _frame_forest(frames)
    roots = [i for i, p in enumerate(parent) if p == -1]
    assert roots, "expected at least one outermost frame"

    out = g.copy()
    for root in roots:
        chain = sorted(_subtree(root, children), key=lambda idx: _bbox_area(frames[idx]["bbox"]))
        for inner_idx, outer_idx in zip(chain, chain[1:]):
            assert _contains(frames[outer_idx]["bbox"], frames[inner_idx]["bbox"]), "frame subtree must be a chain"

        inner_box = frames[chain[0]]["bbox"]
        ir0, ic0, ir1, ic1 = inner_box
        seed_patch = g[ir0 + 1 : ir1, ic0 + 1 : ic1]
        seed_colors = set(map(int, np.unique(seed_patch))) - {bg}
        assert len(seed_colors) == 1, "innermost frame must contain exactly one template color"
        template_mask = seed_patch != bg
        assert bool(template_mask.any()), "template must contain foreground pixels"

        previous_box: Rect | None = None
        for k, idx in enumerate(chain, start=1):
            r0, c0, r1, c1 = frames[idx]["bbox"]
            region = np.zeros_like(g, dtype=bool)
            region[r0 + 1 : r1, c0 + 1 : c1] = True
            if previous_box is not None:
                pr0, pc0, pr1, pc1 = previous_box
                region[pr0 : pr1 + 1, pc0 : pc1 + 1] = False
            out[region] = bg
            _paint_scaled(out, region, r0 + 1, c0 + 1, template_mask, k + 1, int(frames[idx]["color"]))
            previous_box = (r0, c0, r1, c1)

    return out


MAX_GENERATION_ATTEMPTS = 300
MAX_LAYERS = 4
MIN_INNER = 3
MAX_INNER = 6
MIN_GAP = 2
MAX_GAP = 4
MAX_OUTER_PAD = 6


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _ring_mask(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return _is_rect_frame(coords, _bbox(coords))


def _grow_connected_mask(rng: np.random.Generator, h: int, w: int, target: int) -> np.ndarray:
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {start}
    frontier: set[tuple[int, int]] = set()
    while len(cells) < target:
        frontier.clear()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        pick = list(frontier)[int(rng.integers(0, len(frontier)))]
        cells.add(pick)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_template_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    for _ in range(120):
        ph = int(rng.integers(2, h + 1))
        pw = int(rng.integers(2, w + 1))
        max_area = ph * pw - 1
        if max_area < 2:
            continue
        target = int(rng.integers(2, max_area + 1))
        raw = _grow_connected_mask(rng, ph, pw, target)
        rr0, cc0, rr1, cc1 = _bbox(np.argwhere(raw))
        core = raw[rr0 : rr1 + 1, cc0 : cc1 + 1]
        canvas = np.zeros((h, w), dtype=bool)
        ch, cw = core.shape
        canvas[:ch, :cw] = core
        if _ring_mask(canvas):
            continue
        if not canvas.any() or canvas.all():
            continue
        if np.array_equal(canvas, np.flipud(canvas)):
            continue
        if np.array_equal(canvas, np.fliplr(canvas)):
            continue
        if np.array_equal(canvas, np.rot90(canvas, 2)):
            continue
        return canvas
    raise RuntimeError("failed to build an asymmetric template")


def _apply_dihedral(arr: np.ndarray, op: int) -> np.ndarray:
    result = np.array(arr, copy=True)
    if op >= 4:
        result = np.fliplr(result)
        op -= 4
    return np.rot90(result, op)


def _build_structure(rng: np.random.Generator, bg: int) -> np.ndarray:
    layers = int(rng.integers(1, MAX_LAYERS + 1))
    max_inner_h = min(MAX_INNER, 30 - (4 * layers - 2))
    max_inner_w = min(MAX_INNER, 30 - (4 * layers - 2))
    inner_h = int(rng.integers(MIN_INNER, max_inner_h + 1))
    inner_w = int(rng.integers(MIN_INNER, max_inner_w + 1))
    template = _make_template_mask(rng, inner_h, inner_w)

    palette = np.array([c for c in range(10) if c != bg], dtype=int)
    colors = list(map(int, rng.choice(palette, size=layers + 1, replace=False)))
    border_colors = colors[:layers]
    seed_color = colors[-1]

    outer_h = inner_h + 4 * layers - 2
    outer_w = inner_w + 4 * layers - 2
    patch = np.full((outer_h, outer_w), bg, dtype=int)

    for k in range(1, layers + 1):
        pad = 2 * (layers - k)
        frame_h = inner_h + 4 * k - 2
        frame_w = inner_w + 4 * k - 2
        _draw_frame(patch, pad, pad, pad + frame_h - 1, pad + frame_w - 1, border_colors[k - 1])

    inner_pad = 2 * (layers - 1)
    seed_area = patch[inner_pad + 1 : inner_pad + 1 + inner_h, inner_pad + 1 : inner_pad + 1 + inner_w]
    seed_area[template] = int(seed_color)

    return _apply_dihedral(patch, int(rng.integers(0, 8)))


def _row_groups(count: int, rng: np.random.Generator) -> list[list[int]]:
    order = list(range(count))
    rng.shuffle(order)
    if count == 1:
        return [order]
    if count == 2:
        return [order] if bool(rng.integers(0, 2)) else [[order[0]], [order[1]]]
    split = int(rng.integers(0, 3))
    if split == 0:
        return [order]
    if split == 1:
        return [order[:2], order[2:]]
    return [order[:1], order[1:]]


def _layout_structures(
    rng: np.random.Generator, structures: list[np.ndarray]
) -> tuple[list[tuple[int, int, np.ndarray]], int, int]:
    rows = _row_groups(len(structures), rng)
    row_specs = []
    max_width = 0
    for row in rows:
        row_h = max(int(structures[idx].shape[0]) for idx in row)
        entries = []
        x = 0
        for j, idx in enumerate(row):
            patch = structures[idx]
            ph, pw = patch.shape
            yoff = int(rng.integers(0, row_h - ph + 1))
            entries.append((idx, yoff, x))
            x += pw
            if j + 1 < len(row):
                x += int(rng.integers(MIN_GAP, MAX_GAP + 1))
        row_specs.append((row_h, x, entries))
        max_width = max(max_width, x)

    placements: list[tuple[int, int, np.ndarray]] = []
    y = 0
    for i, (row_h, row_w, entries) in enumerate(row_specs):
        row_x = int(rng.integers(0, max_width - row_w + 1))
        for idx, yoff, xoff in entries:
            placements.append((y + yoff, row_x + xoff, structures[idx]))
        y += row_h
        if i + 1 < len(row_specs):
            y += int(rng.integers(MIN_GAP, MAX_GAP + 1))
    return placements, y, max_width


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.integers(0, 10))
    structure_count = int(rng.integers(1, 4))
    structures = [_build_structure(rng, bg) for _ in range(structure_count)]
    placements, layout_h, layout_w = _layout_structures(rng, structures)
    if layout_h > 30 or layout_w > 30:
        return None

    extra_h = int(rng.integers(0, min(MAX_OUTER_PAD, 30 - layout_h) + 1))
    extra_w = int(rng.integers(0, min(MAX_OUTER_PAD, 30 - layout_w) + 1))
    top = int(rng.integers(0, extra_h + 1))
    left = int(rng.integers(0, extra_w + 1))
    grid = np.full((layout_h + extra_h, layout_w + extra_w), bg, dtype=int)

    for r0, c0, patch in placements:
        ph, pw = patch.shape
        region = grid[top + r0 : top + r0 + ph, left + c0 : left + c0 + pw]
        if np.any(region != bg):
            return None
        region[:, :] = patch

    values, counts = np.unique(grid, return_counts=True)
    if int(values[np.argmax(counts)]) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame scaling input")
