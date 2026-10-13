from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(10))

# Generator configuration.
MAX_GENERATION_ATTEMPTS = 400
PANEL_H_RANGE = (12, 18)
PANEL_W_RANGE = (12, 18)
OBJECT_COUNT_CHOICES = (2, 3, 4)
FINGERPRINT_CHOICES = (1, 2, 3, 4)
SMALL_SIDE_LIMIT_BY_COUNT = {2: 5, 3: 4, 4: 3}
PLACEMENT_GAP = 1


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    best = int(counts.max())
    winners = [int(v) for v, c in zip(values, counts) if int(c) == best]
    assert len(winners) == 1, "background must be uniquely dominant"
    return winners[0]


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(1, w - 1):
        if c != w - 1 - c:
            continue
        if not np.all(grid[:, c] == grid[0, c]):
            continue
        color = int(grid[0, c])
        if color == int(bg):
            continue
        if int(np.count_nonzero(grid == color)) != h:
            continue
        candidates.append(c)
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    return int(candidates[0])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _fingerprint_from_bbox(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    shorter = min(height, width)
    longer = max(height, width)
    return int((2 * longer + shorter) // (2 * shorter))


def _extract_objects(panel: np.ndarray, bg: int) -> list[dict]:
    h, w = panel.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []
    colors = []

    for r in range(h):
        for c in range(w):
            color = int(panel[r, c])
            if visited[r, c] or color == int(bg):
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(panel[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            bbox = _bbox_from_coords(coords)
            objects.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": bbox,
                    "fingerprint": _fingerprint_from_bbox(bbox),
                }
            )
            colors.append(color)

    assert objects, "each panel must contain at least one foreground object"
    assert len(colors) == len(set(colors)), "object colors must be unique within a panel"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match left/right panel objects by rounded bbox aspect ratio and overlay them in one panel.

    Input structure:
    - The dominant color is the background.
    - A single full-height vertical separator splits the grid into equal left/right panels.
    - Each panel contains separated monochrome 4-connected objects, with distinct colors per panel.

    Rule:
    - For every object, compute a fingerprint from its bbox aspect ratio:
      round(max(height, width) / min(height, width)) to the nearest integer.
    - Pair left-panel and right-panel objects with identical fingerprints.
    - Output one panel-sized canvas: draw all left objects in place, then draw all
      right-panel objects mirrored horizontally into the same panel space.
    - Right-panel pixels overwrite left-panel pixels on overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    sep = _find_separator_col(g, bg)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape[1] == right.shape[1] and left.shape[1] > 0, "panels must have equal positive width"

    left_objects = _extract_objects(left, bg)
    right_objects = _extract_objects(right, bg)
    assert len(left_objects) == len(right_objects), "left and right panel object counts must match"

    left_groups: dict[int, list[dict]] = defaultdict(list)
    right_groups: dict[int, list[dict]] = defaultdict(list)
    for obj in left_objects:
        left_groups[int(obj["fingerprint"])].append(obj)
    for obj in right_objects:
        right_groups[int(obj["fingerprint"])].append(obj)

    assert set(left_groups) == set(right_groups), "both panels must share the same fingerprint set"
    pairs: list[tuple[dict, dict]] = []
    for fp in sorted(left_groups):
        left_bucket = sorted(left_groups[fp], key=lambda obj: tuple(map(int, obj["bbox"])))
        right_bucket = sorted(right_groups[fp], key=lambda obj: tuple(map(int, obj["bbox"])))
        assert len(left_bucket) == len(right_bucket), "fingerprint multiplicities must match"
        pairs.extend(zip(left_bucket, right_bucket))
    assert len(pairs) == len(left_objects)

    out = np.full(left.shape, int(bg), dtype=int)
    for left_obj, _right_obj in pairs:
        coords = left_obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(left_obj["color"])

    panel_w = int(left.shape[1])
    for _left_obj, right_obj in pairs:
        for r, c in right_obj["coords"]:
            out[int(r), int(panel_w - 1 - c)] = int(right_obj["color"])

    return out


def _shape_seed(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if w >= h:
        row = random.randrange(h)
        top_col = random.randrange(w)
        bottom_col = random.randrange(w)
        mask[row, :] = True
        mask[: row + 1, top_col] = True
        mask[row:, bottom_col] = True
    else:
        col = random.randrange(w)
        left_row = random.randrange(h)
        right_row = random.randrange(h)
        mask[:, col] = True
        mask[left_row, : col + 1] = True
        mask[right_row, col:] = True
    return mask


def _grow_connected_mask(h: int, w: int) -> np.ndarray:
    mask = _shape_seed(h, w)
    base_area = int(mask.sum())
    target = random.randint(base_area, h * w)
    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in DIRS4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        assert frontier, "connected growth should have a non-empty frontier until full"
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True

    if random.choice((False, True)):
        mask = np.flipud(mask)
    if random.choice((False, True)):
        mask = np.fliplr(mask)
    return mask.copy()


def _dimension_candidates(fingerprint: int, max_h: int, max_w: int, small_limit: int) -> list[tuple[int, int]]:
    candidates = []
    for small in range(2, small_limit + 1):
        long_side = small if fingerprint == 1 else small * fingerprint
        dims = {(small, long_side)} if fingerprint == 1 else {(small, long_side), (long_side, small)}
        for h, w in dims:
            if h <= max_h and w <= max_w:
                candidates.append((h, w))
    return sorted(candidates)


def _make_random_mask(fingerprint: int, max_h: int, max_w: int, small_limit: int) -> np.ndarray | None:
    dims = _dimension_candidates(fingerprint, max_h, max_w, small_limit)
    if not dims:
        return None
    h, w = random.choice(dims)
    return _grow_connected_mask(h, w)


def _mark_unavailable(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int = PLACEMENT_GAP) -> None:
    rr0 = max(0, int(r0 - gap))
    cc0 = max(0, int(c0 - gap))
    rr1 = min(blocked.shape[0], int(r0 + h + gap))
    cc1 = min(blocked.shape[1], int(c0 + w + gap))
    blocked[rr0:rr1, cc0:cc1] = True


def _bbox_fits(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    return not bool(blocked[r0 : r0 + h, c0 : c0 + w].any())


def _choose_position(h: int, w: int, blocked: np.ndarray) -> tuple[int, int] | None:
    options = [
        (r0, c0)
        for r0 in range(blocked.shape[0] - h + 1)
        for c0 in range(blocked.shape[1] - w + 1)
        if _bbox_fits(blocked, r0, c0, h, w)
    ]
    if not options:
        return None
    return random.choice(options)


def _place_left_objects(left_objects: list[dict], panel_h: int, panel_w: int) -> tuple[dict[int, tuple[int, int]], np.ndarray] | tuple[None, None]:
    blocked = np.zeros((panel_h, panel_w), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}
    order = sorted(range(len(left_objects)), key=lambda i: left_objects[i]["mask"].size, reverse=True)
    for idx in order:
        h, w = left_objects[idx]["mask"].shape
        pos = _choose_position(h, w, blocked)
        if pos is None:
            return None, None
        placements[idx] = pos
        _mark_unavailable(blocked, pos[0], pos[1], h, w)
    return placements, blocked


def _place_right_output_objects(
    right_objects: list[dict],
    panel_h: int,
    panel_w: int,
    left_blocked: np.ndarray,
    overlap_idx: int | None,
    overlap_pos: tuple[int, int] | None,
) -> dict[int, tuple[int, int]] | None:
    blocked = np.zeros((panel_h, panel_w), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}

    if overlap_idx is not None and overlap_pos is not None:
        h, w = right_objects[overlap_idx]["mask"].shape
        if not _bbox_fits(blocked, overlap_pos[0], overlap_pos[1], h, w):
            return None
        placements[overlap_idx] = overlap_pos
        _mark_unavailable(blocked, overlap_pos[0], overlap_pos[1], h, w)

    order = sorted(
        (idx for idx in range(len(right_objects)) if idx not in placements),
        key=lambda i: right_objects[i]["mask"].size,
        reverse=True,
    )
    for idx in order:
        h, w = right_objects[idx]["mask"].shape
        combined = blocked | left_blocked
        pos = _choose_position(h, w, combined)
        if pos is None:
            return None
        placements[idx] = pos
        _mark_unavailable(blocked, pos[0], pos[1], h, w)

    return placements


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    canvas[r0 : r0 + h, c0 : c0 + w][mask] = int(color)


def _build_left_object(fingerprint: int, color: int, panel_h: int, panel_w: int, small_limit: int) -> dict | None:
    mask = _make_random_mask(fingerprint, panel_h, panel_w, small_limit)
    if mask is None:
        return None
    return {"fingerprint": int(fingerprint), "color": int(color), "mask": mask}


def _build_right_object(
    fingerprint: int,
    color: int,
    panel_h: int,
    panel_w: int,
    small_limit: int,
    forced_output_mask: np.ndarray | None = None,
) -> dict | None:
    if forced_output_mask is None:
        output_mask = _make_random_mask(fingerprint, panel_h, panel_w, small_limit)
        if output_mask is None:
            return None
    else:
        output_mask = np.array(forced_output_mask, dtype=bool, copy=True)
    return {"fingerprint": int(fingerprint), "color": int(color), "mask": output_mask}


def _attempt_generate() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)
    object_count = random.choice(OBJECT_COUNT_CHOICES)
    small_limit = SMALL_SIDE_LIMIT_BY_COUNT[object_count]

    fingerprints = random.sample(FINGERPRINT_CHOICES, object_count)
    overlap_mode = random.choice((False, True))
    overlap_idx = random.randrange(object_count) if overlap_mode else None

    needed_colors = 2 + 2 * object_count
    palette = random.sample(PALETTE, needed_colors)
    bg = int(palette[0])
    sep = int(palette[1])
    left_colors = [int(c) for c in palette[2 : 2 + object_count]]
    right_colors = [int(c) for c in palette[2 + object_count :]]

    left_objects: list[dict] = []
    for fingerprint, color in zip(fingerprints, left_colors):
        obj = _build_left_object(int(fingerprint), int(color), panel_h, panel_w, small_limit)
        if obj is None:
            return None
        left_objects.append(obj)

    right_objects: list[dict] = []
    for idx, (fingerprint, color) in enumerate(zip(fingerprints, right_colors)):
        forced_mask = left_objects[idx]["mask"] if overlap_idx == idx else None
        obj = _build_right_object(int(fingerprint), int(color), panel_h, panel_w, small_limit, forced_mask)
        if obj is None:
            return None
        right_objects.append(obj)

    left_positions, left_blocked = _place_left_objects(left_objects, panel_h, panel_w)
    if left_positions is None or left_blocked is None:
        return None

    overlap_pos = left_positions[overlap_idx] if overlap_idx is not None else None
    right_out_positions = _place_right_output_objects(
        right_objects, panel_h, panel_w, left_blocked if not overlap_mode else left_blocked, overlap_idx, overlap_pos
    )
    if right_out_positions is None:
        return None

    output = np.full((panel_h, panel_w), bg, dtype=int)
    for idx, obj in enumerate(left_objects):
        r0, c0 = left_positions[idx]
        _paint_mask(output, obj["mask"], obj["color"], r0, c0)
        obj["pos"] = (int(r0), int(c0))

    input_grid = np.full((panel_h, 2 * panel_w + 1), bg, dtype=int)
    input_grid[:, panel_w] = sep
    for idx, obj in enumerate(left_objects):
        r0, c0 = left_positions[idx]
        _paint_mask(input_grid[:, :panel_w], obj["mask"], obj["color"], r0, c0)

    mirrored_overlap = False
    for idx, obj in enumerate(right_objects):
        out_r0, out_c0 = right_out_positions[idx]
        _paint_mask(output, obj["mask"], obj["color"], out_r0, out_c0)

        input_mask = np.fliplr(obj["mask"])
        h, w = input_mask.shape
        in_c0 = int(panel_w - out_c0 - w)
        assert 0 <= in_c0 <= panel_w - w
        _paint_mask(input_grid[:, panel_w + 1 :], input_mask, obj["color"], out_r0, in_c0)

        left_patch = output[out_r0 : out_r0 + h, out_c0 : out_c0 + w]
        if np.any(left_patch == obj["color"]):
            mirrored_overlap = True

    # The overlap-detection rule is generator-internal: compare right-object pixels
    # against the left-only canvas before the right layer is applied.
    left_only = np.full((panel_h, panel_w), bg, dtype=int)
    for idx, obj in enumerate(left_objects):
        r0, c0 = left_positions[idx]
        _paint_mask(left_only, obj["mask"], obj["color"], r0, c0)
    actual_overlap = False
    for idx, obj in enumerate(right_objects):
        r0, c0 = right_out_positions[idx]
        h, w = obj["mask"].shape
        if np.any(obj["mask"] & (left_only[r0 : r0 + h, c0 : c0 + w] != bg)):
            actual_overlap = True
            break
    if bool(actual_overlap) != bool(overlap_mode):
        return None

    try:
        solved = main(input_grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, output):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
