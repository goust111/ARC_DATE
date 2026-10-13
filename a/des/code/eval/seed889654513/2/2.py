from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


MARKER = 9
GEN_RETRIES = 400
PARTITION_RETRIES = 200
PLACEMENT_RETRIES = 120
PLACEMENT_GAP = 1

PAIR_COUNT_RANGE = (2, 4)
RECT_H_RANGE = (4, 7)
RECT_W_RANGE = (4, 7)
GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)

_RNG = np.random.default_rng()
_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = _N4 if connectivity == 4 else (
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    )
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _N4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(cells)


def _fragment_from_patch(patch: np.ndarray, bg: int, origin: tuple[int, int] = (0, 0)) -> dict:
    patch = np.asarray(patch, dtype=int)
    coords = np.argwhere(patch != bg)
    assert coords.size, "fragment patch must contain foreground"
    values = patch[coords[:, 0], coords[:, 1]]
    marker_coords = coords[values == MARKER]
    assert len(marker_coords) == 1, "each fragment must contain exactly one marker"
    base_values = {int(v) for v in values if int(v) != MARKER}
    assert len(base_values) == 1, "each fragment must have exactly one base color"
    base_color = int(next(iter(base_values)))
    r_off, c_off = origin
    abs_coords = coords + np.array([[r_off, c_off]])
    rel_values = {(int(r), int(c)): int(patch[int(r), int(c)]) for r, c in coords}
    return {
        "patch": patch.copy(),
        "shape": tuple(map(int, patch.shape)),
        "coords": abs_coords,
        "base_color": base_color,
        "rel_values": rel_values,
        "marker_rel": tuple(map(int, marker_coords[0])),
        "center": (
            float(abs_coords[:, 0].mean()),
            float(abs_coords[:, 1].mean()),
        ),
    }


def _extract_fragments(grid: np.ndarray, bg: int) -> list[dict]:
    parts = components_mask(grid != bg, connectivity=4)
    return [_fragment_from_patch(grid[r0:r1 + 1, c0:c1 + 1], bg, (r0, c0))
            for coords in parts
            for (r0, c0, r1, c1) in [bbox_coords(coords)]]


def _assembly_candidates(frag_a: dict, frag_b: dict) -> list[dict]:
    assert frag_a["base_color"] == frag_b["base_color"], "pair colors must match"
    cells_a = set(frag_a["rel_values"])
    cells_b = set(frag_b["rel_values"])
    h_a, w_a = frag_a["shape"]
    h_b, w_b = frag_b["shape"]
    base = int(frag_a["base_color"])
    candidates: list[dict] = []

    for dr in range(-h_b, h_a + 1):
        for dc in range(-w_b, w_a + 1):
            shifted_b = {(r + dr, c + dc) for (r, c) in cells_b}
            if cells_a & shifted_b:
                continue
            union = cells_a | shifted_b
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            min_r = min(rows)
            max_r = max(rows)
            min_c = min(cols)
            max_c = max(cols)
            rect_h = max_r - min_r + 1
            rect_w = max_c - min_c + 1
            if rect_h * rect_w != len(union):
                continue

            rect = np.full((rect_h, rect_w), base, dtype=int)
            markers = []
            for (r, c), value in frag_a["rel_values"].items():
                rr = r - min_r
                cc = c - min_c
                rect[rr, cc] = int(value)
                if int(value) == MARKER:
                    markers.append((int(rr), int(cc)))
            for (r, c), value in frag_b["rel_values"].items():
                rr = r + dr - min_r
                cc = c + dc - min_c
                rect[rr, cc] = int(value)
                if int(value) == MARKER:
                    markers.append((int(rr), int(cc)))

            candidates.append(
                {
                    "rect": rect,
                    "markers": tuple(markers),
                    "delta": (int(dr), int(dc)),
                }
            )
    return candidates


def _assemble_unique_pair(frag_a: dict, frag_b: dict) -> dict:
    candidates = _assembly_candidates(frag_a, frag_b)
    assert len(candidates) == 1, "pair must assemble into a rectangle in exactly one way"
    return candidates[0]


def _flip_rect_by_marker(rect: np.ndarray) -> np.ndarray:
    rect = np.asarray(rect, dtype=int)
    markers = np.argwhere(rect == MARKER)
    assert len(markers) >= 1, "assembled rectangle must contain at least one marker"
    center_r = (rect.shape[0] - 1) / 2.0
    center_c = (rect.shape[1] - 1) / 2.0
    mean_r = float(markers[:, 0].mean())
    mean_c = float(markers[:, 1].mean())
    off_r = mean_r - center_r
    off_c = mean_c - center_c
    assert abs(abs(off_r) - abs(off_c)) > 1e-9, "marker direction must be unambiguous"
    if abs(off_c) > abs(off_r):
        return np.fliplr(rect)
    return np.flipud(rect)


def _pack_row(rects: list[np.ndarray], bg: int) -> np.ndarray:
    assert rects
    height = max(int(rect.shape[0]) for rect in rects)
    width = sum(int(rect.shape[1]) for rect in rects) + max(0, len(rects) - 1)
    out = np.full((height, width), int(bg), dtype=int)
    c0 = 0
    for rect in rects:
        h, w = map(int, rect.shape)
        out[:h, c0:c0 + w] = rect
        c0 += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color fragments, reassemble each pair into a solid rectangle, then
    flip the rectangle according to the maroon marker's horizontal/vertical bias.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fragments = _extract_fragments(g, bg)
    assert fragments, "expected at least one fragment"

    groups: dict[int, list[dict]] = defaultdict(list)
    for frag in fragments:
        groups[int(frag["base_color"])].append(frag)

    solved: list[tuple[tuple[float, float], np.ndarray]] = []
    for color, pair in groups.items():
        assert len(pair) == 2, f"color {color} must appear in exactly two fragments"
        assembled = _assemble_unique_pair(pair[0], pair[1])
        rect = _flip_rect_by_marker(assembled["rect"])
        key = (
            (pair[0]["center"][0] + pair[1]["center"][0]) / 2.0,
            (pair[0]["center"][1] + pair[1]["center"][1]) / 2.0,
        )
        solved.append((key, rect))

    solved.sort(key=lambda item: (item[0][0], item[0][1]))
    return _pack_row([rect for _, rect in solved], bg)


def _random_connected_subset(h: int, w: int, target: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    start = (int(rng.integers(h)), int(rng.integers(w)))
    cells = {start}
    frontier = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        r, c = cell
        for dr, dc in _N4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(start)
    while len(cells) < target and frontier:
        choice = tuple(frontier.pop()) if len(frontier) == 1 else list(frontier)[int(rng.integers(len(frontier)))]
        frontier.discard(choice)
        if choice in cells:
            continue
        cells.add(choice)
        add_frontier(choice)
    return cells


def _make_patch_from_cells(
    cells: set[tuple[int, int]],
    full_rect: np.ndarray,
    bg: int,
) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = int(full_rect[r, c])
    return patch


def _sample_partition(
    h: int,
    w: int,
    axis: str,
    sign: int,
    rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
    all_cells = {(r, c) for r in range(h) for c in range(w)}
    interior = {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)}
    if len(interior) < 2:
        return None

    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    area = h * w
    low = max(2, area // 3)
    high = min(area - 2, area - low)

    for _ in range(PARTITION_RETRIES):
        target = int(rng.integers(low, high + 1))
        a_cells = _random_connected_subset(h, w, target, rng)
        b_cells = all_cells - a_cells
        if not _is_connected(a_cells) or not _is_connected(b_cells):
            continue

        interior_a = [cell for cell in a_cells if cell in interior]
        interior_b = [cell for cell in b_cells if cell in interior]
        if not interior_a or not interior_b:
            continue

        marker_pairs = []
        for ma in interior_a:
            for mb in interior_b:
                mean_r = (ma[0] + mb[0]) / 2.0
                mean_c = (ma[1] + mb[1]) / 2.0
                off_r = mean_r - center_r
                off_c = mean_c - center_c
                if axis == "horizontal":
                    if sign * off_c <= 0 or abs(off_c) <= abs(off_r):
                        continue
                else:
                    if sign * off_r <= 0 or abs(off_r) <= abs(off_c):
                        continue
                marker_pairs.append((ma, mb))
        if not marker_pairs:
            continue

        marker_a, marker_b = marker_pairs[int(rng.integers(len(marker_pairs)))]
        trial_rect = np.ones((h, w), dtype=int)
        trial_rect[marker_a] = MARKER
        trial_rect[marker_b] = MARKER
        frag_a = _fragment_from_patch(_make_patch_from_cells(a_cells, trial_rect, 0), 0)
        frag_b = _fragment_from_patch(_make_patch_from_cells(b_cells, trial_rect, 0), 0)
        if len(_assembly_candidates(frag_a, frag_b)) != 1:
            continue
        return a_cells, b_cells, marker_a, marker_b

    return None


def _build_pair_spec(base_color: int, bg: int, rng: np.random.Generator) -> dict | None:
    axis = "horizontal" if int(rng.integers(2)) == 0 else "vertical"
    sign = -1 if int(rng.integers(2)) == 0 else 1
    for _ in range(PARTITION_RETRIES):
        h = int(rng.integers(RECT_H_RANGE[0], RECT_H_RANGE[1] + 1))
        w = int(rng.integers(RECT_W_RANGE[0], RECT_W_RANGE[1] + 1))
        sampled = _sample_partition(h, w, axis, sign, rng)
        if sampled is None:
            continue
        a_cells, b_cells, marker_a, marker_b = sampled
        full_rect = np.full((h, w), int(base_color), dtype=int)
        full_rect[marker_a] = MARKER
        full_rect[marker_b] = MARKER
        patch_a = _make_patch_from_cells(a_cells, full_rect, bg)
        patch_b = _make_patch_from_cells(b_cells, full_rect, bg)
        frag_a = _fragment_from_patch(patch_a, bg)
        frag_b = _fragment_from_patch(patch_b, bg)
        assembled = _assemble_unique_pair(frag_a, frag_b)
        return {
            "axis": axis,
            "sign": sign,
            "patches": [patch_a, patch_b],
            "assembled": assembled["rect"],
        }
    return None


def _boxes_too_close(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (
        ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0
    )


def _place_patches(patches: list[np.ndarray], bg: int, rng: np.random.Generator) -> np.ndarray | None:
    max_h = max(int(patch.shape[0]) for patch in patches)
    max_w = max(int(patch.shape[1]) for patch in patches)
    h_lo = max(GRID_H_RANGE[0], max_h + 2)
    w_lo = max(GRID_W_RANGE[0], max_w + 2)
    if h_lo > GRID_H_RANGE[1] or w_lo > GRID_W_RANGE[1]:
        return None

    for _ in range(PLACEMENT_RETRIES):
        height = int(rng.integers(h_lo, GRID_H_RANGE[1] + 1))
        width = int(rng.integers(w_lo, GRID_W_RANGE[1] + 1))
        grid = np.full((height, width), int(bg), dtype=int)
        placed_boxes: list[tuple[int, int, int, int]] = []
        ordered = sorted(patches, key=lambda arr: int(arr.shape[0] * arr.shape[1]), reverse=True)

        ok = True
        for patch in ordered:
            ph, pw = map(int, patch.shape)
            row_choices = max(0, height - ph - 1)
            col_choices = max(0, width - pw - 1)
            if row_choices == 0 or col_choices == 0:
                ok = False
                break
            placed = False
            for _ in range(200):
                r0 = int(rng.integers(1, row_choices + 1))
                c0 = int(rng.integers(1, col_choices + 1))
                box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if any(_boxes_too_close(box, prev, PLACEMENT_GAP) for prev in placed_boxes):
                    continue
                mask = patch != bg
                grid[r0:r0 + ph, c0:c0 + pw][mask] = patch[mask]
                placed_boxes.append(box)
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok and get_bg(grid) == bg:
            return grid
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.integers(0, MARKER))
    pair_count = int(rng.integers(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1] + 1))
    palette = [color for color in range(10) if color not in {bg, MARKER}]
    if len(palette) < pair_count:
        return None
    rng.shuffle(palette)

    patches = []
    for idx in range(pair_count):
        spec = _build_pair_spec(int(palette[idx]), bg, rng)
        if spec is None:
            return None
        patches.extend(spec["patches"])

    grid = _place_patches(patches, bg, rng)
    if grid is None:
        return None

    fragments = _extract_fragments(grid, bg)
    if len(fragments) != 2 * pair_count:
        return None

    per_color: dict[int, list[dict]] = defaultdict(list)
    for frag in fragments:
        per_color[int(frag["base_color"])].append(frag)
    if len(per_color) != pair_count:
        return None
    if any(len(group) != 2 for group in per_color.values()):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.size == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
