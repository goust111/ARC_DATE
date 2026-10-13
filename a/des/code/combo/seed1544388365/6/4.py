import random

import numpy as np


RING_SIZE = 5
BG = 0
AZURE = 8
RED = 2
RING_COLORS = (1, 3, 4, 5, 6, 7, 9)
DIRS = ("up", "down", "left", "right")
DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ORTH_TO_DELTA = {
    "up": (0, 1),
    "down": (0, 1),
    "left": (1, 0),
    "right": (1, 0),
}
GAP_POSITIONS = {
    "up": (0, 2),
    "down": (4, 2),
    "left": (2, 0),
    "right": (2, 4),
}
GEN_ATTEMPTS = 400
WORK_CANVAS = 84
RING_COUNT_RANGE = (3, 6)
FORWARD_STEP_RANGE = (8, 14)
ORTH_JITTER_RANGE = (-3, 3)
MARGIN_RANGE = (2, 6)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR_TO_DELTA.values():
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _ring_template(color: int, direction: str, bg: int = BG) -> np.ndarray:
    tile = np.full((RING_SIZE, RING_SIZE), bg, dtype=int)
    tile[0, :] = color
    tile[-1, :] = color
    tile[:, 0] = color
    tile[:, -1] = color
    gr, gc = GAP_POSITIONS[direction]
    tile[gr, gc] = bg
    return tile


def _parse_ring(tile: np.ndarray, bg: int) -> tuple[int, str]:
    sub = np.asarray(tile, dtype=int)
    assert sub.shape == (RING_SIZE, RING_SIZE), "ring bbox must be 5x5"

    colors = set(map(int, np.unique(sub))) - {bg}
    assert len(colors) == 1, "ring must use exactly one foreground color"
    color = int(next(iter(colors)))

    assert bool((sub[1:4, 1:4] == bg).all()), "ring interior must be hollow"

    gaps = []
    for direction, (gr, gc) in GAP_POSITIONS.items():
        if int(sub[gr, gc]) == bg:
            gaps.append(direction)
    assert len(gaps) == 1, "ring must have exactly one gap"
    gap_direction = gaps[0]
    gap_pos = GAP_POSITIONS[gap_direction]

    for r in range(RING_SIZE):
        for c in range(RING_SIZE):
            on_border = r in (0, RING_SIZE - 1) or c in (0, RING_SIZE - 1)
            if not on_border:
                assert int(sub[r, c]) == bg
                continue
            if (r, c) == gap_pos:
                assert int(sub[r, c]) == bg
            else:
                assert int(sub[r, c]) == color

    return color, gap_direction


def _extract_rings_and_markers(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    rings: list[dict] = []
    markers: list[tuple[int, int, int]] = []

    for comp in _color_components(g, bg):
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        if coords.shape[0] == 1:
            markers.append((int(r0), int(c0), int(comp["color"])))
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        color, direction = _parse_ring(sub, bg)
        rings.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "coords": coords,
                "tile": sub.copy(),
                "color": color,
                "direction": direction,
                "center": (int(r0 + 2), int(c0 + 2)),
            }
        )

    assert rings, "expected at least one ring"
    assert markers, "expected marker pixels"
    return rings, markers


def _point_to_object_distance(point: tuple[int, int], coords: np.ndarray) -> int:
    pr, pc = point
    return int(np.min(np.abs(coords[:, 0] - pr) + np.abs(coords[:, 1] - pc)))


def _choose_start_ring(rings: list[dict], start_marker: tuple[int, int]) -> int:
    scored = []
    for idx, ring in enumerate(rings):
        r0, c0, _r1, _c1 = ring["bbox"]
        scored.append((_point_to_object_distance(start_marker, ring["coords"]), r0, c0, idx))
    scored.sort()
    assert len(scored) >= 1
    if len(scored) >= 2:
        assert scored[0][0] != scored[1][0], "ambiguous start ring"
    return int(scored[0][-1])


def _direction_key(direction: str, src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int, int] | None:
    sr, sc = src
    dr = int(dst[0] - sr)
    dc = int(dst[1] - sc)

    if direction == "up":
        if dr >= 0:
            return None
        forward = -dr
        orth = abs(dc)
    elif direction == "down":
        if dr <= 0:
            return None
        forward = dr
        orth = abs(dc)
    elif direction == "left":
        if dc >= 0:
            return None
        forward = -dc
        orth = abs(dr)
    else:
        if dc <= 0:
            return None
        forward = dc
        orth = abs(dr)

    dist2 = dr * dr + dc * dc
    return (dist2, orth, forward)


def _choose_next_ring(current_idx: int, remaining: set[int], rings: list[dict]) -> int:
    current = rings[current_idx]
    direction = current["direction"]
    src = current["center"]
    candidates = []

    for idx in sorted(remaining):
        ring = rings[idx]
        key = _direction_key(direction, src, ring["center"])
        if key is None:
            continue
        r0, c0, _r1, _c1 = ring["bbox"]
        candidates.append((key, r0, c0, idx))

    assert candidates, f"no ring found in direction {direction}"
    candidates.sort()
    if len(candidates) >= 2:
        assert candidates[0][0] != candidates[1][0], "ambiguous next ring"
    return int(candidates[0][-1])


def _trace_order(rings: list[dict], markers: list[tuple[int, int, int]]) -> list[int]:
    start_marker = min(markers, key=lambda item: (item[0], item[1]))
    current = _choose_start_ring(rings, (int(start_marker[0]), int(start_marker[1])))
    remaining = set(range(len(rings)))
    order = []

    while remaining:
        assert current in remaining
        remaining.remove(current)
        order.append(current)
        if not remaining:
            break
        current = _choose_next_ring(current, remaining, rings)

    assert len(order) == len(rings)
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5x5 gap-rings by following the start marker and the chain of gap directions.

    The top-left singleton marker selects the first ring. Each ring's missing border pixel
    points to the next unvisited ring in that cardinal half-plane. The output packs the
    visited rings tightly from left to right while preserving each ring's orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    rings, markers = _extract_rings_and_markers(g)
    order = _trace_order(rings, markers)
    bg = _background_color(g)

    out = np.full((RING_SIZE, RING_SIZE * len(order)), bg, dtype=int)
    for i, idx in enumerate(order):
        out[:, i * RING_SIZE : (i + 1) * RING_SIZE] = rings[idx]["tile"]
    return out


def _bbox_from_center(center: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = center
    return (int(r - 2), int(c - 2), int(r + 2), int(c + 2))


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _render_layout(
    centers: list[tuple[int, int]],
    directions: list[str],
    ring_color: int,
    azure_pos: tuple[int, int],
    red_pos: tuple[int, int],
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    bboxes = [_bbox_from_center(center) for center in centers]
    min_r = min(min(r0 for r0, _c0, _r1, _c1 in bboxes), azure_pos[0], red_pos[0])
    min_c = min(min(c0 for _r0, c0, _r1, _c1 in bboxes), azure_pos[1], red_pos[1])
    shift_r = -min_r
    shift_c = -min_c

    shifted_bboxes = []
    max_r = max(red_pos[0], azure_pos[0])
    max_c = max(red_pos[1], azure_pos[1])
    for r0, c0, r1, c1 in bboxes:
        sr0, sc0, sr1, sc1 = r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c
        shifted_bboxes.append((sr0, sc0, sr1, sc1))
        max_r = max(max_r, sr1)
        max_c = max(max_c, sc1)

    azure = (azure_pos[0] + shift_r, azure_pos[1] + shift_c)
    red = (red_pos[0] + shift_r, red_pos[1] + shift_c)
    max_r = max(max_r, azure[0], red[0])
    max_c = max(max_c, azure[1], red[1])

    grid = np.full((max_r + 1, max_c + 1), BG, dtype=int)
    for bbox, direction in zip(shifted_bboxes, directions):
        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = _ring_template(ring_color, direction, BG)

    grid[azure] = AZURE
    grid[red] = RED
    return grid, shifted_bboxes


def _validate_generated_grid(grid: np.ndarray, expected_bboxes: list[tuple[int, int, int, int]]) -> bool:
    try:
        rings, markers = _extract_rings_and_markers(grid)
        order = _trace_order(rings, markers)
        recovered = [tuple(map(int, rings[idx]["bbox"])) for idx in order]
        if recovered != [tuple(map(int, bbox)) for bbox in expected_bboxes]:
            return False
        out = main(grid)
        if out.shape != (RING_SIZE, RING_SIZE * len(expected_bboxes)):
            return False
        return True
    except AssertionError:
        return False


def _sample_chain() -> tuple[list[tuple[int, int]], list[str]] | None:
    count = random.randint(*RING_COUNT_RANGE)
    centers = [(WORK_CANVAS // 2, WORK_CANVAS // 2)]
    directions: list[str] = []
    used_bboxes = [_bbox_from_center(centers[0])]

    for _ in range(count - 1):
        placed = False
        for _trial in range(80):
            direction = random.choice(DIRS)
            step = random.randint(*FORWARD_STEP_RANGE)
            jitter = random.randint(*ORTH_JITTER_RANGE)
            dr, dc = DIR_TO_DELTA[direction]
            orr, orc = ORTH_TO_DELTA[direction]
            prev_r, prev_c = centers[-1]
            next_center = (
                int(prev_r + dr * step + orr * jitter),
                int(prev_c + dc * step + orc * jitter),
            )
            bbox = _bbox_from_center(next_center)
            if not (
                2 <= bbox[0]
                and 2 <= bbox[1]
                and bbox[2] < WORK_CANVAS - 2
                and bbox[3] < WORK_CANVAS - 2
            ):
                continue
            if any(_bboxes_overlap(bbox, old, pad=1) for old in used_bboxes):
                continue
            centers.append(next_center)
            used_bboxes.append(bbox)
            directions.append(direction)
            placed = True
            break
        if not placed:
            return None

    directions.append(random.choice(DIRS))
    return centers, directions


def _pick_marker_positions(bboxes: list[tuple[int, int, int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    start_bbox = bboxes[0]
    max_r = max(r1 for _r0, _c0, r1, _c1 in bboxes)
    max_c = max(c1 for _r0, _c0, _r1, c1 in bboxes)

    azure_candidates = []
    for dr in range(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1):
        for dc in range(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1):
            pos = (start_bbox[0] - dr, start_bbox[1] - dc)
            if pos[0] < 0 or pos[1] < 0:
                continue
            azure_candidates.append(pos)
    random.shuffle(azure_candidates)

    red_candidates = []
    for dr in range(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1):
        for dc in range(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1):
            red_candidates.append((max_r + dr, max_c + dc))
    random.shuffle(red_candidates)

    for azure in azure_candidates:
        if any(bbox[0] <= azure[0] <= bbox[2] and bbox[1] <= azure[1] <= bbox[3] for bbox in bboxes):
            continue
        for red in red_candidates:
            if red == azure:
                continue
            if any(bbox[0] <= red[0] <= bbox[2] and bbox[1] <= red[1] <= bbox[3] for bbox in bboxes):
                continue
            return azure, red
    return None


def _gen_attempt() -> np.ndarray | None:
    chain = _sample_chain()
    if chain is None:
        return None
    centers, directions = chain
    bboxes = [_bbox_from_center(center) for center in centers]
    marker_positions = _pick_marker_positions(bboxes)
    if marker_positions is None:
        return None
    azure_pos, red_pos = marker_positions

    ring_color = random.choice(RING_COLORS)
    grid, shifted_bboxes = _render_layout(centers, directions, ring_color, azure_pos, red_pos)
    if not _validate_generated_grid(grid, shifted_bboxes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
