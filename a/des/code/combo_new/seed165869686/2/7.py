import random
from collections import deque

import numpy as np


BG = 0
MASTER_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLOR_POOL = tuple(range(1, 10))

LAYER_PATTERNS = {
    1: ((2,), (3,), (4,)),
    2: ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3)),
    3: ((2, 2, 2), (2, 2, 3), (2, 3, 2), (3, 2, 2)),
}


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _component_coords(grid: np.ndarray, sr: int, sc: int, seen: np.ndarray) -> list[tuple[int, int]]:
    color = int(grid[sr, sc])
    q = deque([(sr, sc)])
    seen[sr, sc] = True
    coords: list[tuple[int, int]] = []
    h, w = grid.shape

    while q:
        r, c = q.popleft()
        coords.append((r, c))
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                seen[nr, nc] = True
                q.append((nr, nc))
    return coords


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    coord_set = set(coords)
    if len(coord_set) != 2 * h + 2 * w - 4:
        return False

    for r, c in coord_set:
        if r not in (r0, r1) and c not in (c0, c1):
            return False

    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return False
    return True


def _find_frames(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    frames = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            coords = _component_coords(grid, r, c, seen)
            assert _is_rect_frame(coords), "every foreground component must be a 1-pixel rectangular frame"
            frames.append({"color": int(grid[r, c]), "bbox": _bbox(coords)})

    assert frames, "expected at least one rectangular frame"
    return frames


def _fill_from_frames(grid: np.ndarray, frames: list[dict[str, int | tuple[int, int, int, int]]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for frame in sorted(frames, key=lambda item: _bbox_area(item["bbox"]), reverse=True):
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(frame["color"])
    out[grid != BG] = grid[grid != BG]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nested rectangular-frame gaps by propagating each border color inward to the next layer.

    Input structure:
    - The background is black.
    - The foreground consists of one or more axis-aligned hollow rectangular frames.
    - Frames are one pixel thick; some are nested into 2-to-4-layer groups with at least one black cell between neighboring borders.
    - Different groups do not overlap, though colors may repeat across groups.

    Rule:
    - Identify every rectangular frame as a separate connected component.
    - Process frames from larger bounding boxes to smaller ones and fill each frame's strict interior with that frame's border color.
    - Later fills from deeper nested frames overwrite the center of earlier larger fills, so each gap keeps the color of the immediately surrounding frame.
    - Restore all original non-black border pixels unchanged.

    Key insight:
    - Each frame is a visible color instruction for everything inside it until a deeper frame overrides that region, so the solution depends on nesting depth rather than absolute position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frames = _find_frames(g)
    return _fill_from_frames(g, frames)


def _choose_pattern(rng: random.Random) -> tuple[int, ...]:
    group_count = rng.choice((1, 2, 3))
    return rng.choice(LAYER_PATTERNS[group_count])


def _sample_group_spec(
    rng: random.Random, layer_count: int, total_groups: int
) -> dict[str, object]:
    if layer_count == 4:
        inner_max = 5 if total_groups == 1 else 4
        inset_choices = (2, 3) if total_groups == 1 else (2,)
    elif layer_count == 3:
        inner_max = 6 if total_groups == 1 else 5 if total_groups == 2 else 4
        inset_choices = (2, 3) if total_groups < 3 else (2,)
    else:
        inner_max = 7 if total_groups == 1 else 5
        inset_choices = (2, 3, 4) if total_groups == 1 else (2, 3)

    inner_h = rng.randint(3, inner_max)
    inner_w = rng.randint(3, inner_max)
    bboxes = [(0, 0, inner_h - 1, inner_w - 1)]

    for _ in range(layer_count - 1):
        prev_r0, prev_c0, prev_r1, prev_c1 = bboxes[-1]
        top = rng.choice(inset_choices)
        bottom = rng.choice(inset_choices)
        left = rng.choice(inset_choices)
        right = rng.choice(inset_choices)
        bboxes.append((prev_r0 - top, prev_c0 - left, prev_r1 + bottom, prev_c1 + right))

    min_r = min(r0 for r0, _, _, _ in bboxes)
    min_c = min(c0 for _, c0, _, _ in bboxes)
    norm = []
    for bbox in bboxes:
        r0, c0, r1, c1 = bbox
        norm.append((r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c))
    norm.sort(key=_bbox_area, reverse=True)

    colors = rng.sample(COLOR_POOL, layer_count)
    frames = [{"bbox": bbox, "color": color} for bbox, color in zip(norm, colors)]
    outer_r0, outer_c0, outer_r1, outer_c1 = norm[0]
    return {
        "frames": frames,
        "height": outer_r1 - outer_r0 + 1,
        "width": outer_c1 - outer_c0 + 1,
    }


def _candidate_positions(
    occupied: np.ndarray, height: int, width: int
) -> list[tuple[int, int]]:
    limit = occupied.shape[0]
    positions = []
    for r0 in range(limit - height + 1):
        for c0 in range(limit - width + 1):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(limit, r0 + height + 1)
            cc1 = min(limit, c0 + width + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _place_groups(
    rng: random.Random, groups: list[dict[str, object]]
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((MASTER_SIZE, MASTER_SIZE), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(groups)
    order = sorted(
        range(len(groups)),
        key=lambda idx: int(groups[idx]["height"]) * int(groups[idx]["width"]),
        reverse=True,
    )

    for idx in order:
        group = groups[idx]
        positions = _candidate_positions(occupied, int(group["height"]), int(group["width"]))
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        placements[idx] = (r0, c0)
        occupied[r0 : r0 + int(group["height"]), c0 : c0 + int(group["width"])] = True

    return [placement for placement in placements if placement is not None]


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _render_input(
    groups: list[dict[str, object]], placements: list[tuple[int, int]]
) -> tuple[np.ndarray, list[dict[str, int | tuple[int, int, int, int]]]]:
    grid = np.zeros((MASTER_SIZE, MASTER_SIZE), dtype=int)
    frames: list[dict[str, int | tuple[int, int, int, int]]] = []

    for group, (base_r, base_c) in zip(groups, placements):
        for frame in group["frames"]:
            r0, c0, r1, c1 = frame["bbox"]
            abs_bbox = (base_r + r0, base_c + c0, base_r + r1, base_c + c1)
            _draw_frame(grid, abs_bbox, int(frame["color"]))
            frames.append({"bbox": abs_bbox, "color": int(frame["color"])})
    return grid, frames


def _crop_to_random_canvas(
    rng: random.Random, inp: np.ndarray, out: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())

    top = rng.randint(0, r0)
    left = rng.randint(0, c0)
    bottom = rng.randint(r1, inp.shape[0] - 1)
    right = rng.randint(c1, inp.shape[1] - 1)
    return inp[top : bottom + 1, left : right + 1], out[top : bottom + 1, left : right + 1]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pattern = _choose_pattern(rng)
    groups = [_sample_group_spec(rng, layer_count, len(pattern)) for layer_count in pattern]
    placements = _place_groups(rng, groups)
    if placements is None:
        return None

    inp, frames = _render_input(groups, placements)
    expected = _fill_from_frames(inp, frames)
    inp, expected = _crop_to_random_canvas(rng, inp, expected)

    if np.array_equal(inp, expected):
        return None

    try:
        got = main(inp)
    except Exception:
        return None

    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
