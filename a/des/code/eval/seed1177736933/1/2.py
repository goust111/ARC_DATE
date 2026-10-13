import numpy as np
from collections import Counter, deque


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_shape(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _first_scan_index(grid: np.ndarray, color: int) -> int:
    idx = np.flatnonzero(grid.ravel() == int(color))
    assert idx.size > 0
    return int(idx[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay same-color fragment pairs inside the canvas-sized bbox and paint the
    uncovered cells onto the solid grey canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    color_to_comps: dict[int, list[np.ndarray]] = {color: _components(g == color) for color in colors}

    canvas_candidates = []
    for color, comps in color_to_comps.items():
        if len(comps) != 1:
            continue
        coords = comps[0]
        ch, cw = _bbox_shape(coords)
        if int(coords.shape[0]) == int(ch * cw):
            canvas_candidates.append((int(color), _bbox(coords), ch, cw))

    assert len(canvas_candidates) == 1
    canvas_color, _canvas_bbox, canvas_h, canvas_w = canvas_candidates[0]

    ordered_pairs: list[tuple[int, int, np.ndarray]] = []
    for color, comps in color_to_comps.items():
        if int(color) == int(canvas_color):
            continue
        assert len(comps) == 2
        merged = np.zeros((canvas_h, canvas_w), dtype=bool)
        for coords in comps:
            h, w = _bbox_shape(coords)
            assert (h, w) == (canvas_h, canvas_w)
            r0, c0, _, _ = _bbox(coords)
            local = coords - np.array([r0, c0], dtype=int)
            merged[local[:, 0], local[:, 1]] = True
        ordered_pairs.append((_first_scan_index(g, color), int(color), ~merged))

    ordered_pairs.sort(key=lambda item: item[0])
    out = np.full((canvas_h, canvas_w), int(canvas_color), dtype=int)
    for _, color, holes in ordered_pairs:
        out[holes] = int(color)
    return out


# --------------- generator ---------------

_GEN_CANVAS_COLOR = 5
_GEN_PAIR_OPTIONS = (2, 3, 4)
_GEN_CANVAS_H_RANGE = (5, 9)
_GEN_CANVAS_W_RANGE = (5, 9)
_GEN_GAP_RANGE = (1, 4)
_GEN_OUTER_PAD_RANGE = (1, 6)
_GEN_ATTEMPTS = 400


def _random_path(
    start: tuple[int, int],
    end: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    steps: list[tuple[int, int]] = []
    if r1 > r0:
        steps.extend([(1, 0)] * (r1 - r0))
    else:
        steps.extend([(-1, 0)] * (r0 - r1))
    if c1 > c0:
        steps.extend([(0, 1)] * (c1 - c0))
    else:
        steps.extend([(0, -1)] * (c0 - c1))
    if steps:
        order = rng.permutation(len(steps))
        steps = [steps[int(i)] for i in order]

    r, c = int(r0), int(c0)
    path = [(r, c)]
    for dr, dc in steps:
        r += dr
        c += dc
        path.append((int(r), int(c)))
    return path


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                out.add((int(nr), int(nc)))
    return list(out)


def _spanning_fragment(
    h: int,
    w: int,
    rng: np.random.Generator,
    density_range: tuple[float, float],
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    center = (int(rng.integers(h)), int(rng.integers(w)))
    anchors = [
        (0, int(rng.integers(w))),
        (h - 1, int(rng.integers(w))),
        (int(rng.integers(h)), 0),
        (int(rng.integers(h)), w - 1),
    ]
    for anchor in anchors:
        for cell in _random_path(anchor, center, rng):
            mask[cell] = True

    area = int(h * w)
    min_density, max_density = density_range
    target = int(rng.integers(max(int(mask.sum()), int(np.ceil(area * min_density))), max(int(mask.sum()) + 1, int(np.floor(area * max_density)) + 1)))
    target = min(target, area - 1)

    while int(mask.sum()) < target:
        cells = _frontier(mask)
        if not cells:
            break
        pick = cells[int(rng.integers(len(cells)))]
        mask[pick] = True

    assert _touches_all_sides(mask)
    assert not mask.all()
    return mask


def _make_pair(
    h: int,
    w: int,
    rng: np.random.Generator,
    overlap_mode: bool,
    num_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    area = int(h * w)
    if overlap_mode:
        density_range = (0.35, 0.68)
        max_holes = max(2, area // 3)
    else:
        density_range = (0.52, 0.82)
        max_holes = max(2, area // (2 * num_pairs))
    min_holes = 1 if max_holes == 1 else 2

    for _ in range(200):
        a = _spanning_fragment(h, w, rng, density_range)
        b = _spanning_fragment(h, w, rng, density_range)
        holes = ~(a | b)
        hole_count = int(holes.sum())
        if min_holes <= hole_count <= max_holes:
            return a, b, holes
    return None


def _layout_positions(
    count: int,
    box_h: int,
    box_w: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    layouts = [
        (rows, cols)
        for rows in range(1, min(count, 4) + 1)
        for cols in range(1, min(count, 4) + 1)
        if rows * cols >= count
    ]
    rows, cols = layouts[int(rng.integers(len(layouts)))]
    row_offsets = [0]
    for _ in range(1, rows):
        row_offsets.append(row_offsets[-1] + box_h + int(rng.integers(*_GEN_GAP_RANGE)))
    col_offsets = [0]
    for _ in range(1, cols):
        col_offsets.append(col_offsets[-1] + box_w + int(rng.integers(*_GEN_GAP_RANGE)))

    slots = np.arange(rows * cols)
    rng.shuffle(slots)
    chosen = [int(x) for x in slots[:count]]
    positions = []
    for slot in chosen:
        rr = slot // cols
        cc = slot % cols
        positions.append((int(row_offsets[rr]), int(col_offsets[cc])))
    rng.shuffle(positions)
    return positions


def _ensure_bg_majority(grid: np.ndarray, bg: int) -> np.ndarray:
    out = grid
    while True:
        counts = Counter(map(int, out.ravel()))
        bg_count = int(counts[int(bg)])
        max_other = max((count for color, count in counts.items() if int(color) != int(bg)), default=0)
        if bg_count > max_other:
            return out
        h, w = out.shape
        if h <= w:
            out = np.pad(out, ((0, 1), (0, 0)), constant_values=int(bg))
        else:
            out = np.pad(out, ((0, 0), (0, 1)), constant_values=int(bg))


def _structurally_valid(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        bg = _most_common_color(g)
        colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
        color_to_comps = {color: _components(g == color) for color in colors}

        canvas = []
        for color, comps in color_to_comps.items():
            if len(comps) != 1:
                continue
            coords = comps[0]
            h, w = _bbox_shape(coords)
            if int(coords.shape[0]) == int(h * w):
                canvas.append((int(color), h, w))
        if len(canvas) != 1:
            return False
        canvas_color, canvas_h, canvas_w = canvas[0]

        for color, comps in color_to_comps.items():
            if int(color) == int(canvas_color):
                continue
            if len(comps) != 2:
                return False
            for coords in comps:
                if _bbox_shape(coords) != (canvas_h, canvas_w):
                    return False
        return True
    except Exception:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    num_pairs = int(rng.choice(_GEN_PAIR_OPTIONS))
    overlap_mode = bool(rng.integers(2))
    canvas_h = int(rng.integers(*_GEN_CANVAS_H_RANGE))
    canvas_w = int(rng.integers(*_GEN_CANVAS_W_RANGE))
    if not overlap_mode and canvas_h * canvas_w < 4 * num_pairs:
        return None

    bg_choices = [c for c in range(10) if c != _GEN_CANVAS_COLOR]
    bg = int(rng.choice(bg_choices))
    frag_choices = [c for c in range(10) if c not in (bg, _GEN_CANVAS_COLOR)]
    rng.shuffle(frag_choices)
    frag_colors = [int(c) for c in frag_choices[:num_pairs]]

    hole_masks: dict[int, np.ndarray] = {}
    fragments: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    used_holes = np.zeros((canvas_h, canvas_w), dtype=bool)
    need_overlap = overlap_mode

    for color in frag_colors:
        accepted = None
        for _ in range(300):
            pair = _make_pair(canvas_h, canvas_w, rng, overlap_mode, num_pairs)
            if pair is None:
                continue
            a, b, holes = pair
            if overlap_mode:
                intersects = bool((holes & used_holes).any())
                if need_overlap and used_holes.any() and not intersects:
                    continue
            else:
                if bool((holes & used_holes).any()):
                    continue
            accepted = (a, b, holes)
            need_overlap = need_overlap and not bool((holes & used_holes).any())
            used_holes |= holes
            break
        if accepted is None:
            return None
        a, b, holes = accepted
        fragments[int(color)] = (a, b)
        hole_masks[int(color)] = holes

    all_holes = list(hole_masks.values())
    if overlap_mode and not any(bool((all_holes[i] & all_holes[j]).any()) for i in range(len(all_holes)) for j in range(i + 1, len(all_holes))):
        return None
    for i, holes in enumerate(all_holes):
        others = np.zeros_like(holes)
        for j, other in enumerate(all_holes):
            if i != j:
                others |= other
        if not bool((holes & ~others).any()):
            return None

    objects: list[tuple[str, int, np.ndarray]] = [("canvas", _GEN_CANVAS_COLOR, np.ones((canvas_h, canvas_w), dtype=bool))]
    for color in frag_colors:
        a, b = fragments[int(color)]
        objects.append(("frag", int(color), a))
        objects.append(("frag", int(color), b))

    positions = _layout_positions(len(objects), canvas_h, canvas_w, rng)
    top_pad = int(rng.integers(*_GEN_OUTER_PAD_RANGE))
    left_pad = int(rng.integers(*_GEN_OUTER_PAD_RANGE))
    object_positions = [(r + top_pad, c + left_pad) for r, c in positions]
    total_h = max(r for r, _ in object_positions) + canvas_h + int(rng.integers(*_GEN_OUTER_PAD_RANGE))
    total_w = max(c for _, c in object_positions) + canvas_w + int(rng.integers(*_GEN_OUTER_PAD_RANGE))
    grid = np.full((total_h, total_w), int(bg), dtype=int)

    for (kind, color, mask), (r0, c0) in zip(objects, object_positions):
        if kind == "canvas":
            grid[r0 : r0 + canvas_h, c0 : c0 + canvas_w] = int(color)
        else:
            patch = grid[r0 : r0 + canvas_h, c0 : c0 + canvas_w]
            patch[mask] = int(color)

    grid = _ensure_bg_majority(grid, bg)
    if not _structurally_valid(grid):
        return None

    order = sorted(frag_colors, key=lambda color: _first_scan_index(grid, color))
    expected = np.full((canvas_h, canvas_w), int(_GEN_CANVAS_COLOR), dtype=int)
    for color in order:
        expected[hole_masks[int(color)]] = int(color)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
