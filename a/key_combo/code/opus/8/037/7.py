from __future__ import annotations

from collections import deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
RECT_COLORS = (1, 2, 3, 4, 6)
SEED_FALLBACK = 7
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _draw_frame(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _bbox_h(bbox: BBox) -> int:
    return bbox[2] - bbox[0] + 1


def _bbox_w(bbox: BBox) -> int:
    return bbox[3] - bbox[1] + 1


def _bbox_area(bbox: BBox) -> int:
    return _bbox_h(bbox) * _bbox_w(bbox)


def _contains(outer: BBox, inner: BBox) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _frame_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _detect_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    rects: list[dict[str, object]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        frame = _frame_mask((h, w))
        if np.array_equal(crop == color, frame):
            rects.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return rects


def _boundary_seeds(grid: np.ndarray, rect_colors: set[int]) -> list[Coord]:
    h, w = grid.shape
    seeds: set[Coord] = set()
    for c in range(w):
        if int(grid[0, c]) != BG and int(grid[0, c]) not in rect_colors:
            seeds.add((0, c))
        if int(grid[h - 1, c]) != BG and int(grid[h - 1, c]) not in rect_colors:
            seeds.add((h - 1, c))
    for r in range(h):
        if int(grid[r, 0]) != BG and int(grid[r, 0]) not in rect_colors:
            seeds.add((r, 0))
        if int(grid[r, w - 1]) != BG and int(grid[r, w - 1]) not in rect_colors:
            seeds.add((r, w - 1))
    return sorted(seeds)


def _reachable_background(grid: np.ndarray, seeds: list[Coord]) -> np.ndarray:
    seen = np.zeros(grid.shape, dtype=bool)
    q: deque[Coord] = deque()
    for r, c in seeds:
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == BG and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if not _in_bounds(grid.shape, rr, cc):
                continue
            if seen[rr, cc] or int(grid[rr, cc]) != BG:
                continue
            seen[rr, cc] = True
            q.append((rr, cc))
    return seen


def _outer_background_neighbors(grid: np.ndarray, bbox: BBox) -> list[Coord]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    coords: list[Coord] = []
    if r0 > 0:
        for c in range(c0, c1 + 1):
            if int(grid[r0 - 1, c]) == BG:
                coords.append((r0 - 1, c))
    if r1 + 1 < h:
        for c in range(c0, c1 + 1):
            if int(grid[r1 + 1, c]) == BG:
                coords.append((r1 + 1, c))
    if c0 > 0:
        for r in range(r0, r1 + 1):
            if int(grid[r, c0 - 1]) == BG:
                coords.append((r, c0 - 1))
    if c1 + 1 < w:
        for r in range(r0, r1 + 1):
            if int(grid[r, c1 + 1]) == BG:
                coords.append((r, c1 + 1))
    return coords


def _hidden_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    rects = _detect_rectangles(grid)
    rect_colors = {int(rect["color"]) for rect in rects}
    seeds = _boundary_seeds(grid, rect_colors)
    assert seeds, "expected edge seed pixels"
    reachable = _reachable_background(grid, seeds)

    hidden = []
    for rect in rects:
        bg_neighbors = _outer_background_neighbors(grid, rect["bbox"])
        if not any(reachable[r, c] for r, c in bg_neighbors):
            hidden.append(rect)
    return sorted(hidden, key=lambda rect: (rect["bbox"][1], rect["bbox"][0]))


def _pack_crops(grid: np.ndarray, rects: list[dict[str, object]]) -> np.ndarray:
    crops = [grid[r0 : r1 + 1, c0 : c1 + 1].copy() for r0, c0, r1, c1 in (rect["bbox"] for rect in rects)]
    out_h = max(crop.shape[0] for crop in crops)
    out_w = sum(crop.shape[1] for crop in crops)
    out = np.zeros((out_h, out_w), dtype=int)
    c = 0
    for crop in crops:
        h, w = crop.shape
        out[:h, c : c + w] = crop
        c += w
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood from edge seeds through black background, keep only unreachable hollow rectangles, then crop and pack them.

    Input structure:
    - A black background contains 3 to 5 complete one-pixel-thick hollow rectangles, each in a distinct non-black color.
    - Some rectangles are directly exposed to the outer background, while others are nested inside larger rectangles.
    - The grid boundary also contains one single-pixel seed on each side; seeds start the reachability flood but are not rectangle pixels.

    Rule:
    - Starting from all edge seed pixels, traverse only 4-connected black background cells and stop at every non-black pixel.
    - A rectangle is exposed if at least one black cell orthogonally adjacent to its outer wall is flood-reachable; otherwise it is hidden.
    - Crop every hidden rectangle by its own bounding box, preserving its interior contents, then sort the crops by original left column and row and concatenate them horizontally with no gap, top-aligned.

    Key insight:
    - The flood fill is not the output action itself; it is the visibility test that decides which hollow rectangles survive into the cropped packed output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    hidden = _hidden_rectangles(g)
    assert hidden, "expected at least one hidden rectangle"
    return _pack_crops(g, hidden)


def _choice(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


def _randint(rng: random.Random, low: int, high: int) -> int:
    return rng.randint(low, high)


def _build_length1(rng: random.Random, colors: list[int], idx: int) -> tuple[np.ndarray, list[dict[str, object]], int]:
    h = _randint(rng, 3, 7)
    w = _randint(rng, 3, 7)
    canvas = np.zeros((h, w), dtype=int)
    _draw_frame(canvas, (0, 0, h - 1, w - 1), colors[idx])
    rects = [{"color": colors[idx], "bbox": (0, 0, h - 1, w - 1), "depth": 0}]
    return canvas, rects, idx + 1


def _length2_candidates(require_moat: bool) -> list[tuple[int, int, int, int, int, int]]:
    out = []
    for outer_h in range(5, 8):
        for outer_w in range(5, 8):
            for inner_h in range(3, outer_h - 1):
                for inner_w in range(3, outer_w - 1):
                    for off_r in range(1, outer_h - inner_h):
                        for off_c in range(1, outer_w - inner_w):
                            has_bg_neighbor = (
                                off_r > 1
                                or off_c > 1
                                or off_r + inner_h < outer_h - 1
                                or off_c + inner_w < outer_w - 1
                            )
                            if require_moat and not has_bg_neighbor:
                                continue
                            out.append((outer_h, outer_w, inner_h, inner_w, off_r, off_c))
    return out


def _build_length2(
    rng: random.Random, colors: list[int], idx: int, require_moat: bool
) -> tuple[np.ndarray, list[dict[str, object]], int]:
    outer_h, outer_w, inner_h, inner_w, off_r, off_c = _choice(rng, _length2_candidates(require_moat))
    canvas = np.zeros((outer_h, outer_w), dtype=int)
    outer_bbox = (0, 0, outer_h - 1, outer_w - 1)
    inner_bbox = (off_r, off_c, off_r + inner_h - 1, off_c + inner_w - 1)
    _draw_frame(canvas, outer_bbox, colors[idx])
    _draw_frame(canvas, inner_bbox, colors[idx + 1])
    rects = [
        {"color": colors[idx], "bbox": outer_bbox, "depth": 0},
        {"color": colors[idx + 1], "bbox": inner_bbox, "depth": 1},
    ]
    return canvas, rects, idx + 2


def _build_length3(rng: random.Random, colors: list[int], idx: int) -> tuple[np.ndarray, list[dict[str, object]], int]:
    del rng
    canvas = np.zeros((7, 7), dtype=int)
    b0 = (0, 0, 6, 6)
    b1 = (1, 1, 5, 5)
    b2 = (2, 2, 4, 4)
    _draw_frame(canvas, b0, colors[idx])
    _draw_frame(canvas, b1, colors[idx + 1])
    _draw_frame(canvas, b2, colors[idx + 2])
    rects = [
        {"color": colors[idx], "bbox": b0, "depth": 0},
        {"color": colors[idx + 1], "bbox": b1, "depth": 1},
        {"color": colors[idx + 2], "bbox": b2, "depth": 2},
    ]
    return canvas, rects, idx + 3


def _structure_options(n_rects: int) -> list[tuple[int, ...]]:
    if n_rects == 3:
        return [(2, 1)]
    if n_rects == 4:
        return [(2, 2), (2, 1, 1)]
    assert n_rects == 5
    return [(3, 2), (2, 2, 1)]


def _build_roots(rng: random.Random, n_rects: int) -> tuple[list[dict[str, object]], int]:
    structure = _choice(rng, _structure_options(n_rects))
    colors = list(RECT_COLORS)
    rng.shuffle(colors)
    roots: list[dict[str, object]] = []
    idx = 0
    moat_given = False
    for length in structure:
        if length == 1:
            canvas, rects, idx = _build_length1(rng, colors, idx)
        elif length == 2:
            require_moat = not moat_given
            canvas, rects, idx = _build_length2(rng, colors, idx, require_moat=require_moat)
            moat_given = True
        else:
            canvas, rects, idx = _build_length3(rng, colors, idx)
        roots.append({"canvas": canvas, "rects": rects})
    assert idx == n_rects
    return roots, len(structure)


def _expanded_overlap(a: BBox, b: BBox, pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _candidate_positions(shape: tuple[int, int], used: list[BBox], h: int, w: int) -> list[Coord]:
    H, W = shape
    out: list[Coord] = []
    for r0 in range(1, H - h):
        for c0 in range(1, W - w):
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_expanded_overlap(bbox, other, pad=1) for other in used):
                continue
            out.append((r0, c0))
    return out


def _place_roots(rng: random.Random, roots: list[dict[str, object]]) -> tuple[np.ndarray, list[dict[str, object]]]:
    items = sorted(
        [(i, root["canvas"].shape[0], root["canvas"].shape[1]) for i, root in enumerate(roots)],
        key=lambda item: item[1] * item[2],
        reverse=True,
    )
    for _ in range(120):
        H = _randint(rng, 18, 28)
        W = _randint(rng, 18, 28)
        placements: dict[int, Coord] = {}
        used: list[BBox] = []
        for root_idx, h, w in items:
            candidates = _candidate_positions((H, W), used, h, w)
            if not candidates:
                break
            if placements and rng.random() < 0.35:
                existing_cols = [c for _, c in placements.values()]
                aligned = [pos for pos in candidates if pos[1] in existing_cols]
                if aligned:
                    candidates = aligned
            if placements and rng.random() < 0.35:
                existing_rows = [r for r, _ in placements.values()]
                aligned = [pos for pos in candidates if pos[0] in existing_rows]
                if aligned:
                    candidates = aligned
            r0, c0 = _choice(rng, candidates)
            placements[root_idx] = (r0, c0)
            used.append((r0, c0, r0 + h - 1, c0 + w - 1))
        if len(placements) != len(roots):
            continue

        grid = np.zeros((H, W), dtype=int)
        rects: list[dict[str, object]] = []
        for root_idx, root in enumerate(roots):
            canvas = root["canvas"]
            r0, c0 = placements[root_idx]
            h, w = canvas.shape
            grid[r0 : r0 + h, c0 : c0 + w] = canvas
            for rect in root["rects"]:
                rr0, cc0, rr1, cc1 = rect["bbox"]
                rects.append(
                    {
                        "color": int(rect["color"]),
                        "bbox": (r0 + rr0, c0 + cc0, r0 + rr1, c0 + cc1),
                        "depth": int(rect["depth"]),
                    }
                )
        return grid, rects
    raise RuntimeError("failed to place root rectangles")


def _seed_positions(rng: random.Random, shape: tuple[int, int]) -> list[Coord]:
    H, W = shape
    return [
        (0, _randint(rng, 1, W - 2)),
        (H - 1, _randint(rng, 1, W - 2)),
        (_randint(rng, 1, H - 2), 0),
        (_randint(rng, 1, H - 2), W - 1),
    ]


def _parent_map(rects: list[dict[str, object]]) -> dict[int, int | None]:
    parents: dict[int, int | None] = {}
    for i, rect in enumerate(rects):
        rb = rect["bbox"]
        candidates = [j for j, other in enumerate(rects) if i != j and _contains(other["bbox"], rb)]
        if not candidates:
            parents[i] = None
            continue
        parents[i] = min(candidates, key=lambda j: _bbox_area(rects[j]["bbox"]))
    return parents


def _semantic_check(grid: np.ndarray, planned_rects: list[dict[str, object]]) -> bool:
    detected = _detect_rectangles(grid)
    if len(detected) != len(planned_rects):
        return False
    detected_by_color = {int(rect["color"]): rect for rect in detected}
    planned_by_color = {int(rect["color"]): rect for rect in planned_rects}
    if set(detected_by_color) != set(planned_by_color):
        return False
    for color in detected_by_color:
        if tuple(detected_by_color[color]["bbox"]) != tuple(planned_by_color[color]["bbox"]):
            return False

    rect_colors = set(detected_by_color)
    seeds = _boundary_seeds(grid, rect_colors)
    if len(seeds) != 4:
        return False
    seed_colors = {int(grid[r, c]) for r, c in seeds}
    if len(seed_colors) != 1:
        return False

    parents = _parent_map(detected)
    expected_hidden = {i for i, parent in parents.items() if parent is not None}
    if not expected_hidden or len(expected_hidden) == len(detected):
        return False

    reachable = _reachable_background(grid, seeds)
    actual_hidden = set()
    has_hidden_with_bg_neighbor = False
    for i, rect in enumerate(detected):
        bg_neighbors = _outer_background_neighbors(grid, rect["bbox"])
        if bg_neighbors:
            has_hidden_with_bg_neighbor = has_hidden_with_bg_neighbor or (parents[i] is not None)
        if not any(reachable[r, c] for r, c in bg_neighbors):
            actual_hidden.add(i)
    if actual_hidden != expected_hidden:
        return False
    if not has_hidden_with_bg_neighbor:
        return False

    ordered_hidden = [detected[i] for i in sorted(expected_hidden, key=lambda i: (detected[i]["bbox"][1], detected[i]["bbox"][0]))]
    expected_out = _pack_crops(grid, ordered_hidden)
    try:
        actual_out = main(grid)
    except Exception:
        return False
    return np.array_equal(actual_out, expected_out)


def _attempt_generate() -> np.ndarray | None:
    rng = random.Random()
    n_rects = _choice(rng, [3, 4, 5])
    roots, _ = _build_roots(rng, n_rects)
    grid, rects = _place_roots(rng, roots)

    seed_color = SEED_FALLBACK if any(int(rect["color"]) == 1 for rect in rects) else 1
    for r, c in _seed_positions(rng, grid.shape):
        if int(grid[r, c]) != BG:
            return None
        grid[r, c] = seed_color

    if max(grid.shape) > 30:
        return None
    if not _semantic_check(grid, rects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 7")
