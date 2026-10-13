from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
FRAGMENT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = ("up", "right", "down", "left")

# --------------- helpers ---------------


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(pts)
    return parts


def _point_direction_and_key(
    r: int, c: int, r0: int, c0: int, r1: int, c1: int
) -> tuple[str | None, tuple[int, int, int, int]]:
    dr2 = 2 * int(r) - (int(r0) + int(r1))
    dc2 = 2 * int(c) - (int(c0) + int(c1))
    dist2 = int(dr2 * dr2 + dc2 * dc2)
    key = (dist2, abs(int(dr2)) + abs(int(dc2)), int(r), int(c))
    if dr2 == 0 or dc2 == 0:
        return None, key
    if dr2 < 0 and dc2 < 0:
        return "up", key
    if dr2 < 0 and dc2 > 0:
        return "right", key
    if dr2 > 0 and dc2 > 0:
        return "down", key
    return "left", key


def _fragment_sequences(
    fragments: list[tuple[int, int, int]], bbox: tuple[int, int, int, int]
) -> dict[str, list[int]]:
    r0, c0, r1, c1 = map(int, bbox)
    groups: dict[str, list[tuple[int, int, int, int, int]]] = {d: [] for d in DIRS}
    for r, c, color in fragments:
        direction, key = _point_direction_and_key(int(r), int(c), r0, c0, r1, c1)
        assert direction is not None, "fragment cannot lie on a center axis"
        groups[direction].append((*key, int(color)))
    return {
        direction: [color for *_prefix, color in sorted(groups[direction])]
        for direction in DIRS
    }


def _paint_tentacles(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    gaps: dict[str, int],
    sequences: dict[str, list[int]],
) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    starts = {
        "up": (r0 - 1, int(gaps["up"])),
        "right": (int(gaps["right"]), c1 + 1),
        "down": (r1 + 1, int(gaps["down"])),
        "left": (int(gaps["left"]), c0 - 1),
    }
    deltas = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
    h, w = map(int, out.shape)

    for direction in DIRS:
        sr, sc = starts[direction]
        dr, dc = deltas[direction]
        for i, color in enumerate(sequences[direction]):
            rr = sr + i * dr
            cc = sc + i * dc
            assert 0 <= rr < h and 0 <= cc < w, "tentacle must stay in bounds"
            assert int(out[rr, cc]) == BG, "tentacle path must be empty in the input"
            out[rr, cc] = int(color)


def _parse_container_bbox(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int
) -> dict[str, object] | None:
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None

    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    if any(int(grid[r, c]) != FRAME for r, c in corners):
        return None

    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == FRAME):
        return None

    top_gap = [c for c in range(c0 + 1, c1) if int(grid[r0, c]) != FRAME]
    bottom_gap = [c for c in range(c0 + 1, c1) if int(grid[r1, c]) != FRAME]
    left_gap = [r for r in range(r0 + 1, r1) if int(grid[r, c0]) != FRAME]
    right_gap = [r for r in range(r0 + 1, r1) if int(grid[r, c1]) != FRAME]
    if not (len(top_gap) == len(bottom_gap) == len(left_gap) == len(right_gap) == 1):
        return None

    tg, bgap = top_gap[0], bottom_gap[0]
    lg, rg = left_gap[0], right_gap[0]
    if int(grid[r0, tg]) != BG or int(grid[r1, bgap]) != BG:
        return None
    if int(grid[lg, c0]) != BG or int(grid[rg, c1]) != BG:
        return None

    for c in range(c0 + 1, c1):
        want_top = BG if c == tg else FRAME
        want_bottom = BG if c == bgap else FRAME
        if int(grid[r0, c]) != want_top or int(grid[r1, c]) != want_bottom:
            return None
    for r in range(r0 + 1, r1):
        want_left = BG if r == lg else FRAME
        want_right = BG if r == rg else FRAME
        if int(grid[r, c0]) != want_left or int(grid[r, c1]) != want_right:
            return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gaps": {"up": tg, "right": rg, "down": bgap, "left": lg},
    }


def _find_containers(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    grey_rows = [r for r in range(h) if np.any(grid[r, :] == FRAME)]
    grey_cols = [c for c in range(w) if np.any(grid[:, c] == FRAME)]

    found: dict[tuple[int, int, int, int], dict[str, object]] = {}
    for i, r0 in enumerate(grey_rows):
        for r1 in grey_rows[i + 1 :]:
            if r1 - r0 < 4:
                continue
            shared_cols = [c for c in grey_cols if int(grid[r0, c]) == FRAME and int(grid[r1, c]) == FRAME]
            for j, c0 in enumerate(shared_cols):
                for c1 in shared_cols[j + 1 :]:
                    if c1 - c0 < 4:
                        continue
                    parsed = _parse_container_bbox(grid, r0, c0, r1, c1)
                    if parsed is not None:
                        found[(r0, c0, r1, c1)] = parsed

    containers = list(found.values())
    containers.sort(key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow outward tentacles from each grey container based on the colors and counts
    of the fragments in its four interior quadrants.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    containers = _find_containers(g)
    assert containers, "expected at least one grey container"

    out = g.copy()
    for container in containers:
        r0, c0, r1, c1 = map(int, container["bbox"])
        fragments: list[tuple[int, int, int]] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                color = int(g[r, c])
                if color not in (BG, FRAME):
                    fragments.append((r, c, color))

        sequences = _fragment_sequences(fragments, (r0, c0, r1, c1))
        _paint_tentacles(out, (r0, c0, r1, c1), container["gaps"], sequences)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
CONTAINER_COUNT_RANGE = (1, 3)
OUTER_H_RANGE = (6, 11)
OUTER_W_RANGE = (6, 11)
MAX_FRAGMENTS_PER_DIRECTION = 3
MAX_TOTAL_FRAGMENTS = len(FRAGMENT_COLORS)
PLACEMENT_TRIES = 200
COUNT_TRIES = 80
CELL_TRIES = 150
GEN_RETRIES = 400
SEPARATION = 1


def _quadrant_candidates(h: int, w: int) -> dict[str, list[tuple[int, int, int]]]:
    bbox = (0, 0, int(h) - 1, int(w) - 1)
    groups: dict[str, list[tuple[int, int, int]]] = {d: [] for d in DIRS}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            direction, key = _point_direction_and_key(r, c, *bbox)
            if direction is None:
                continue
            groups[direction].append((int(r), int(c), int(key[0])))
    return groups


def _sample_count(max_k: int) -> int:
    options = list(range(max_k + 1))
    weights = [3] + [4 if k == 1 else 2 for k in range(1, max_k + 1)]
    return int(random.choices(options, weights=weights, k=1)[0])


def _sample_cells(
    candidates: list[tuple[int, int, int]], count: int
) -> list[tuple[int, int, int]] | None:
    if count == 0:
        return []
    for _ in range(CELL_TRIES):
        chosen = random.sample(candidates, count)
        if len({r for r, _c, _d in chosen}) != count:
            continue
        if len({c for _r, c, _d in chosen}) != count:
            continue
        if len({d for _r, _c, d in chosen}) != count:
            continue
        return chosen
    return None


def _build_container_spec() -> dict[str, object] | None:
    h = random.randint(*OUTER_H_RANGE)
    w = random.randint(*OUTER_W_RANGE)
    candidates = _quadrant_candidates(h, w)

    max_counts = {}
    for direction in DIRS:
        rows = {r for r, _c, _d in candidates[direction]}
        cols = {c for _r, c, _d in candidates[direction]}
        dists = {d for _r, _c, d in candidates[direction]}
        max_counts[direction] = min(
            MAX_FRAGMENTS_PER_DIRECTION,
            len(candidates[direction]),
            len(rows),
            len(cols),
            len(dists),
        )
        if max_counts[direction] <= 0:
            return None

    counts: dict[str, int] | None = None
    for _ in range(COUNT_TRIES):
        proposal = {direction: _sample_count(int(max_counts[direction])) for direction in DIRS}
        total = sum(proposal.values())
        if 1 <= total <= MAX_TOTAL_FRAGMENTS:
            counts = proposal
            break
    if counts is None:
        return None

    chosen_cells: list[tuple[int, int, int]] = []
    for direction in DIRS:
        selected = _sample_cells(candidates[direction], int(counts[direction]))
        if selected is None:
            return None
        chosen_cells.extend(selected)

    colors = random.sample(FRAGMENT_COLORS, len(chosen_cells))
    random.shuffle(colors)
    fragments_local: list[tuple[int, int, int]] = []
    for (r, c, _dist), color in zip(chosen_cells, colors):
        fragments_local.append((int(r), int(c), int(color)))

    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME

    gaps = {
        "up": random.randint(1, w - 2),
        "right": random.randint(1, h - 2),
        "down": random.randint(1, w - 2),
        "left": random.randint(1, h - 2),
    }
    patch[0, int(gaps["up"])] = BG
    patch[int(gaps["right"]), -1] = BG
    patch[-1, int(gaps["down"])] = BG
    patch[int(gaps["left"]), 0] = BG

    for r, c, color in fragments_local:
        patch[r, c] = int(color)

    sequences = _fragment_sequences(fragments_local, (0, 0, h - 1, w - 1))
    return {
        "patch": patch,
        "shape": (h, w),
        "gaps": gaps,
        "tentacles": sequences,
        "reach": {direction: len(sequences[direction]) for direction in DIRS},
    }


def _reserved_rect(
    spec: dict[str, object], top: int, left: int
) -> tuple[int, int, int, int]:
    h, w = map(int, spec["shape"])
    reach = spec["reach"]
    return (
        int(top) - int(reach["up"]) - SEPARATION,
        int(left) - int(reach["left"]) - SEPARATION,
        int(top) + h - 1 + int(reach["down"]) + SEPARATION,
        int(left) + w - 1 + int(reach["right"]) + SEPARATION,
    )


def _rects_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_specs(
    height: int, width: int, specs: list[dict[str, object]]
) -> list[tuple[dict[str, object], int, int]] | None:
    placed: list[tuple[dict[str, object], int, int]] = []
    reserved: list[tuple[int, int, int, int]] = []

    ordered = sorted(
        specs,
        key=lambda spec: (
            (int(spec["shape"][0]) + int(spec["reach"]["up"]) + int(spec["reach"]["down"]))
            * (int(spec["shape"][1]) + int(spec["reach"]["left"]) + int(spec["reach"]["right"]))
        ),
        reverse=True,
    )

    for spec in ordered:
        h, w = map(int, spec["shape"])
        reach = spec["reach"]
        min_r = int(reach["up"]) + SEPARATION
        max_r = int(height) - h - int(reach["down"]) - SEPARATION
        min_c = int(reach["left"]) + SEPARATION
        max_c = int(width) - w - int(reach["right"]) - SEPARATION
        if max_r < min_r or max_c < min_c:
            return None

        success = False
        for _ in range(PLACEMENT_TRIES):
            top = random.randint(min_r, max_r)
            left = random.randint(min_c, max_c)
            rect = _reserved_rect(spec, top, left)
            if any(_rects_overlap(rect, other) for other in reserved):
                continue
            reserved.append(rect)
            placed.append((spec, top, left))
            success = True
            break
        if not success:
            return None

    return placed


def _expected_output(
    grid: np.ndarray, placements: list[tuple[dict[str, object], int, int]]
) -> np.ndarray:
    out = grid.copy()
    for spec, top, left in placements:
        h, w = map(int, spec["shape"])
        bbox = (top, left, top + h - 1, left + w - 1)
        gaps = {
            "up": left + int(spec["gaps"]["up"]),
            "right": top + int(spec["gaps"]["right"]),
            "down": left + int(spec["gaps"]["down"]),
            "left": top + int(spec["gaps"]["left"]),
        }
        _paint_tentacles(out, bbox, gaps, spec["tentacles"])
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*CONTAINER_COUNT_RANGE)
    specs: list[dict[str, object]] = []
    for _ in range(n):
        spec = _build_container_spec()
        if spec is None:
            return None
        specs.append(spec)

    if not any(max(len(spec["tentacles"][direction]) for direction in DIRS) >= 2 for spec in specs):
        return None

    min_h = max(int(spec["shape"][0]) + int(spec["reach"]["up"]) + int(spec["reach"]["down"]) + 2 * SEPARATION for spec in specs)
    min_w = max(int(spec["shape"][1]) + int(spec["reach"]["left"]) + int(spec["reach"]["right"]) + 2 * SEPARATION for spec in specs)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None
    height = random.randint(max(GRID_H_RANGE[0], min_h), GRID_H_RANGE[1])
    width = random.randint(max(GRID_W_RANGE[0], min_w), GRID_W_RANGE[1])

    placements = _place_specs(height, width, specs)
    if placements is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for spec, top, left in placements:
        patch = spec["patch"]
        h, w = map(int, patch.shape)
        if np.any(grid[top : top + h, left : left + w] != BG):
            return None
        grid[top : top + h, left : left + w] = patch

    expected = _expected_output(grid, placements)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
