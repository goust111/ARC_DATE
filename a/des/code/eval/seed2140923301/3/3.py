from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _rect_border(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _track_color_for_grey(grid: np.ndarray, grey_pos: tuple[int, int]) -> int:
    h, w = grid.shape
    r, c = grey_pos
    counts: Counter[int] = Counter()
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            value = int(grid[rr, cc])
            if value not in (BG, GREY):
                counts[value] += 1
    assert counts, f"grey pixel at {grey_pos} has no colored ring neighbors"
    track_color, freq = counts.most_common(1)[0]
    assert freq >= 2, f"grey pixel at {grey_pos} does not sit on a ring border"
    return int(track_color)


def _track_component(grid: np.ndarray, grey_pos: tuple[int, int], track_color: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    q: deque[tuple[int, int]] = deque([grey_pos])
    seen = {grey_pos}
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if (rr, cc) in seen:
                continue
            value = int(grid[rr, cc])
            if value == track_color or value == GREY:
                seen.add((rr, cc))
                q.append((rr, cc))
    return seen


def _extract_ring(grid: np.ndarray, grey_pos: tuple[int, int]) -> dict[str, object]:
    track_color = _track_color_for_grey(grid, grey_pos)
    component = _track_component(grid, grey_pos, track_color)

    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, f"ring at {grey_pos} is too small"

    border = _rect_border(r0, c0, r1, c1)
    border_set = set(border)
    assert component == border_set, f"ring at {grey_pos} is not a full rectangular border"
    assert sum(int(grid[r, c] == GREY) for r, c in border) == 1, "each ring must contain exactly one grey car"
    for r, c in border:
        value = int(grid[r, c])
        assert value in (track_color, GREY), "ring border contains an unexpected color"

    inside = grid[r0 + 1 : r1, c0 + 1 : c1]
    markers = [int(v) for v in inside.flat if int(v) != BG]
    assert 1 <= len(markers) <= 8, "ring interior must contain 1..8 markers"
    marker_colors = set(markers)
    assert len(marker_colors) == 1, "all markers inside one ring must share a color"
    marker_color = marker_colors.pop()
    assert marker_color not in (track_color, GREY), "marker color must differ from track color and grey"

    return {
        "bbox": (r0, c0, r1, c1),
        "border": border,
        "track_color": int(track_color),
        "marker_color": int(marker_color),
        "steps": len(markers),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each grey car clockwise around its rectangular track by the number of
    interior markers, then recolor it to the marker color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    grey_positions = [tuple(map(int, pos)) for pos in np.argwhere(g == GREY)]
    assert grey_positions, "expected at least one grey car"

    seen_boxes: set[tuple[int, int, int, int]] = set()
    for grey_pos in grey_positions:
        ring = _extract_ring(g, grey_pos)
        bbox = ring["bbox"]
        assert bbox not in seen_boxes, "multiple grey cars detected on the same ring"
        seen_boxes.add(bbox)

        border = ring["border"]
        old_idx = border.index(grey_pos)
        new_pos = border[(old_idx + int(ring["steps"])) % len(border)]

        out[grey_pos] = int(ring["track_color"])
        out[new_pos] = int(ring["marker_color"])

    return out


def _sample_ring_spec(rng: np.random.Generator) -> dict[str, int]:
    steps = int(rng.integers(1, 9))
    for _ in range(100):
        height = int(rng.integers(3, 9))
        width = int(rng.integers(3, 9))
        if (height - 2) * (width - 2) >= steps:
            return {"height": height, "width": width, "steps": steps}
    raise RuntimeError("failed to sample a ring size with enough interior capacity")


def _pick_canvas(specs: list[dict[str, int]], rng: np.random.Generator) -> tuple[int, int]:
    total_area = sum((spec["height"] + 2) * (spec["width"] + 2) for spec in specs)
    base = int(np.ceil(np.sqrt(total_area * 1.7)))
    min_h = max(spec["height"] for spec in specs) + 2
    min_w = max(spec["width"] for spec in specs) + 2
    h_low = max(10, min_h, base - 2)
    w_low = max(10, min_w, base - 2)
    h_high = max(h_low + 1, min(31, h_low + 7))
    w_high = max(w_low + 1, min(31, w_low + 7))
    return int(rng.integers(h_low, h_high)), int(rng.integers(w_low, w_high))


def _place_specs(specs: list[dict[str, int]], h: int, w: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int]] = []
    order = list(range(len(specs)))
    rng.shuffle(order)
    top_lefts: list[tuple[int, int] | None] = [None] * len(specs)

    for idx in order:
        spec = specs[idx]
        rh, rw = spec["height"], spec["width"]
        candidates = [
            (r0, c0)
            for r0 in range(1, h - rh)
            for c0 in range(1, w - rw)
        ]
        rng.shuffle(candidates)
        chosen: tuple[int, int] | None = None
        for r0, c0 in candidates:
            rr0, cc0 = r0 - 1, c0 - 1
            rr1, cc1 = r0 + rh, c0 + rw
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            chosen = (r0, c0)
            break
        if chosen is None:
            return None
        top_lefts[idx] = chosen
        placements.append(chosen)

    return [pos for pos in top_lefts if pos is not None]


def _draw_ring(
    grid: np.ndarray,
    r0: int,
    c0: int,
    spec: dict[str, int],
    track_color: int,
    marker_color: int,
    rng: np.random.Generator,
) -> None:
    h = spec["height"]
    w = spec["width"]
    steps = spec["steps"]
    r1, c1 = r0 + h - 1, c0 + w - 1

    border = _rect_border(r0, c0, r1, c1)
    for r, c in border:
        grid[r, c] = track_color

    inner_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    marker_ids = rng.choice(len(inner_cells), size=steps, replace=False)
    for idx in np.atleast_1d(marker_ids):
        rr, cc = inner_cells[int(idx)]
        grid[rr, cc] = marker_color

    grey_index = int(rng.integers(0, len(border)))
    gr, gc = border[grey_index]
    grid[gr, gc] = GREY


def _validate_generated(grid: np.ndarray, expected_rings: int) -> bool:
    try:
        grey_positions = [tuple(map(int, pos)) for pos in np.argwhere(grid == GREY)]
        if len(grey_positions) != expected_rings:
            return False

        boxes = set()
        for grey_pos in grey_positions:
            ring = _extract_ring(grid, grey_pos)
            boxes.add(ring["bbox"])
        if len(boxes) != expected_rings:
            return False

        _ = main(grid)
        return True
    except AssertionError:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    ring_count = int(rng.integers(1, 5))
    specs = [_sample_ring_spec(rng) for _ in range(ring_count)]
    h, w = _pick_canvas(specs, rng)

    placements = _place_specs(specs, h, w, rng)
    if placements is None:
        return None

    colors = list(rng.permutation(GEN_COLORS))
    grid = np.zeros((h, w), dtype=int)
    for i, (r0, c0) in enumerate(placements):
        track_color = int(colors[2 * i])
        marker_color = int(colors[2 * i + 1])
        _draw_ring(grid, r0, c0, specs[i], track_color, marker_color, rng)

    if not _validate_generated(grid, ring_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
