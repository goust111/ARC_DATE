from __future__ import annotations

from collections import Counter, deque
from typing import Iterable

import numpy as np


BG = 0
RED = 2
GREY = 5
SEED_COLORS = np.array([1, 3, 4, 6, 7, 8, 9], dtype=int)
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]

MIN_SIDE = 16
MAX_SIDE = 30
MIN_ARROWS = 2
MAX_ARROWS = 5
MIN_STEP = 4
MAX_STEP = 7
MAX_RETRIES = 400


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in offsets:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(cells)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _arrow_cells(center: tuple[int, int], direction: tuple[int, int], tip_side: int) -> list[tuple[int, int]]:
    r, c = center
    dr, dc = direction
    if (dr, dc) == (-1, 0):
        shaft = [(r + 1, c), (r, c), (r - 1, c)]
        tip = (r - 1, c + tip_side)
    elif (dr, dc) == (1, 0):
        shaft = [(r - 1, c), (r, c), (r + 1, c)]
        tip = (r + 1, c + tip_side)
    elif (dr, dc) == (0, -1):
        shaft = [(r, c + 1), (r, c), (r, c - 1)]
        tip = (r + tip_side, c - 1)
    elif (dr, dc) == (0, 1):
        shaft = [(r, c - 1), (r, c), (r, c + 1)]
        tip = (r + tip_side, c + 1)
    else:
        raise ValueError(f"invalid direction: {direction}")
    return shaft + [tip]


def _parse_arrow(comp: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(comp) != 4:
        return None
    pts = sorted((int(r), int(c)) for r, c in comp)
    row_counts = Counter(r for r, _ in pts)
    col_counts = Counter(c for _, c in pts)

    shaft_cols = [c for c, cnt in col_counts.items() if cnt == 3]
    if shaft_cols:
        shaft_c = shaft_cols[0]
        shaft_rows = sorted(r for r, c in pts if c == shaft_c)
        if shaft_rows[2] - shaft_rows[0] != 2:
            return None
        extras = [(r, c) for r, c in pts if c != shaft_c]
        if len(extras) != 1:
            return None
        er, ec = extras[0]
        if ec not in (shaft_c - 1, shaft_c + 1):
            return None
        if er == shaft_rows[0]:
            direction = (-1, 0)
        elif er == shaft_rows[2]:
            direction = (1, 0)
        else:
            return None
        return {
            "cells": set(pts),
            "center": (shaft_rows[1], shaft_c),
            "direction": direction,
        }

    shaft_rows = [r for r, cnt in row_counts.items() if cnt == 3]
    if shaft_rows:
        shaft_r = shaft_rows[0]
        shaft_cols2 = sorted(c for r, c in pts if r == shaft_r)
        if shaft_cols2[2] - shaft_cols2[0] != 2:
            return None
        extras = [(r, c) for r, c in pts if r != shaft_r]
        if len(extras) != 1:
            return None
        er, ec = extras[0]
        if er not in (shaft_r - 1, shaft_r + 1):
            return None
        if ec == shaft_cols2[0]:
            direction = (0, -1)
        elif ec == shaft_cols2[2]:
            direction = (0, 1)
        else:
            return None
        return {
            "cells": set(pts),
            "center": (shaft_r, shaft_cols2[1]),
            "direction": direction,
        }

    return None


def _find_seed(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    reds = np.argwhere(grid == RED)
    if reds.size == 0:
        raise ValueError("missing red start marker")

    mask = (grid != BG) & (grid != GREY) & (grid != RED)
    comps = _components(mask, connectivity=8)
    if not comps:
        raise ValueError("missing seed pattern")

    candidate = None
    if len(reds) == 1:
        rr, rc = map(int, reds[0])
        matched = []
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            if (r0, c0) == (rr + 1, rc + 1):
                matched.append((r0, c0, r1, c1, comp))
        if len(matched) == 1:
            _, _, r1, c1, comp = matched[0]
            r0, c0 = rr + 1, rc + 1
            candidate = (r0, c0, r1, c1, comp)

    if candidate is None:
        if len(comps) != 1:
            reds0 = tuple(int(v) for v in reds[0])
            scored = []
            for comp in comps:
                r0, c0, r1, c1 = _bbox(comp)
                center = ((r0 + r1) // 2, (c0 + c1) // 2)
                dist = abs(center[0] - reds0[0]) + abs(center[1] - reds0[1])
                scored.append((dist, r0, c0, r1, c1, comp))
            _, r0, c0, r1, c1, comp = min(scored)
            candidate = (r0, c0, r1, c1, comp)
        else:
            comp = comps[0]
            r0, c0, r1, c1 = _bbox(comp)
            candidate = (r0, c0, r1, c1, comp)

    r0, c0, r1, c1, _ = candidate
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    return patch, center


def _trace_chain(arrows: list[dict[str, object]], seed_center: tuple[int, int]) -> list[int]:
    if not arrows:
        return []

    start_idx = min(
        range(len(arrows)),
        key=lambda i: (
            abs(arrows[i]["center"][0] - seed_center[0]) + abs(arrows[i]["center"][1] - seed_center[1]),
            arrows[i]["center"][0],
            arrows[i]["center"][1],
        ),
    )

    order: list[int] = []
    used: set[int] = set()
    current = start_idx
    while True:
        order.append(current)
        used.add(current)
        cr, cc = arrows[current]["center"]
        dr, dc = arrows[current]["direction"]
        candidates = []
        for i, arrow in enumerate(arrows):
            if i in used:
                continue
            nr, nc = arrow["center"]
            if dr != 0 and nc == cc and (nr - cr) * dr > 0:
                candidates.append((abs(nr - cr), nr, nc, i))
            elif dc != 0 and nr == cr and (nc - cc) * dc > 0:
                candidates.append((abs(nc - cc), nr, nc, i))
        if not candidates:
            break
        current = min(candidates)[-1]
    return order


def _paste_patch(out: np.ndarray, patch: np.ndarray, center: tuple[int, int]) -> None:
    ph, pw = patch.shape
    r0 = center[0] - ph // 2
    c0 = center[1] - pw // 2
    r1 = r0 + ph
    c1 = c0 + pw
    if not (0 <= r0 and 0 <= c0 and r1 <= out.shape[0] and c1 <= out.shape[1]):
        raise ValueError("patch placement exceeds grid bounds")
    out[r0:r1, c0:c1] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy a colored seed sprite along a chain of grey L-arrows, rotating it 90° clockwise at each stop.

    Input structure:
    - The background is black.
    - One multicolor seed pattern of size at most 3x3 appears in the grid, with a single red start pixel placed diagonally above-left of the seed's top-left corner.
    - Several grey L-shaped arrows are scattered in the grid. Each arrow is a 3-pixel shaft plus a 1-pixel side tip attached at the shaft's pointing end.
    - The nearest arrow to the seed begins a unique center-to-center chain, and each arrow points toward the next arrow in that chain.

    Rule:
    - Find the seed pattern near the red start marker.
    - Detect every grey L-shape, infer its shaft direction, and use the shaft midpoint as the arrow center.
    - Start from the arrow whose center has the smallest Manhattan distance to the seed center.
    - At each visited arrow center, draw a copy of the seed pattern centered there, rotating it 90° clockwise relative to the previous copy.
    - Follow the current arrow's pointing direction to the nearest next arrow center on the same row or column, and stop when no further arrow exists. Keep the original seed and red marker unchanged.

    Key insight:
    - The task combines three layers: recognize direction from each L-arrow, order the arrows into a single path chain, then propagate the seed by cumulative quarter-turn rotations along that chain.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    seed_patch, seed_center = _find_seed(g)

    arrow_comps = _components(g == GREY, connectivity=4)
    arrows = []
    for comp in arrow_comps:
        parsed = _parse_arrow(comp)
        if parsed is not None:
            arrows.append(parsed)

    order = _trace_chain(arrows, seed_center)
    for step, idx in enumerate(order, start=1):
        rotated = np.rot90(seed_patch, -step)
        _paste_patch(out, rotated, arrows[idx]["center"])
    return out


def _neighbors8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    ]


def _center_ok(center: tuple[int, int], h: int, w: int) -> bool:
    r, c = center
    return 1 <= r < h - 1 and 1 <= c < w - 1


def _pattern_rotations_distinct(patch: np.ndarray) -> bool:
    sigs = {np.rot90(patch, k).tobytes() for k in range(4)}
    return len(sigs) == 4


def _make_seed_pattern(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        palette = rng.choice(SEED_COLORS, size=int(rng.integers(2, 5)), replace=False)
        patch = rng.choice(palette, size=(3, 3), replace=True).astype(int)
        if len(set(map(int, patch.flatten()))) < 2:
            continue
        if not _pattern_rotations_distinct(patch):
            continue
        return patch
    raise RuntimeError("failed to create a non-symmetric seed pattern")


def _choose_next_dir(rng: np.random.Generator, prev_idx: int | None) -> int:
    if prev_idx is None:
        return int(rng.integers(0, 4))
    options = [prev_idx, (prev_idx - 1) % 4, (prev_idx + 1) % 4]
    return int(options[int(rng.integers(0, len(options)))])


def _dilated(cells: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for cell in cells:
        out.update(_neighbors8(cell))
    return out


def _generate_arrows(rng: np.random.Generator, h: int, w: int, count: int) -> list[dict[str, object]] | None:
    for _ in range(250):
        centers: list[tuple[int, int]] = [
            (int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2)))
        ]
        move_dir_indices: list[int] = []
        prev_idx: int | None = None
        ok = True

        for _step_idx in range(count - 1):
            dir_idx = _choose_next_dir(rng, prev_idx)
            step = int(rng.integers(MIN_STEP, MAX_STEP + 1))
            dr, dc = DIRS4[dir_idx]
            nr = centers[-1][0] + dr * step
            nc = centers[-1][1] + dc * step
            next_center = (nr, nc)
            if not _center_ok(next_center, h, w):
                ok = False
                break
            if any(abs(nr - r) <= 3 and abs(nc - c) <= 3 for r, c in centers):
                ok = False
                break
            centers.append(next_center)
            move_dir_indices.append(dir_idx)
            prev_idx = dir_idx

        if not ok:
            continue

        valid_last = []
        last_r, last_c = centers[-1]
        for idx, (dr, dc) in enumerate(DIRS4):
            blocked = False
            for other_r, other_c in centers[:-1]:
                if dr != 0 and other_c == last_c and (other_r - last_r) * dr > 0:
                    blocked = True
                    break
                if dc != 0 and other_r == last_r and (other_c - last_c) * dc > 0:
                    blocked = True
                    break
            if not blocked:
                valid_last.append(idx)
        if not valid_last:
            continue

        dir_indices = move_dir_indices + [int(valid_last[int(rng.integers(0, len(valid_last)))])]
        arrows: list[dict[str, object]] = []
        occupied: set[tuple[int, int]] = set()
        forbidden: set[tuple[int, int]] = set()
        for center, dir_idx in zip(centers, dir_indices):
            tip_side = -1 if int(rng.integers(0, 2)) == 0 else 1
            cells = _arrow_cells(center, DIRS4[dir_idx], tip_side)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                ok = False
                break
            if any(cell in forbidden for cell in cells):
                ok = False
                break
            occupied.update(cells)
            forbidden.update(_dilated(cells))
            arrows.append(
                {
                    "center": center,
                    "direction": DIRS4[dir_idx],
                    "cells": set(cells),
                }
            )
        if not ok:
            continue

        traced = _trace_chain(arrows, centers[0])
        if traced != list(range(count)):
            continue
        return arrows
    return None


def _seed_candidates(
    h: int,
    w: int,
    arrows: list[dict[str, object]],
) -> list[tuple[int, int]]:
    first_center = arrows[0]["center"]
    other_centers = [arrow["center"] for arrow in arrows[1:]]
    arrow_cells = set().union(*(arrow["cells"] for arrow in arrows))
    arrow_forbidden = _dilated(arrow_cells)
    candidates: list[tuple[int, int]] = []

    for sr in range(2, h - 1):
        for sc in range(2, w - 1):
            seed_cells = {
                (sr + dr, sc + dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
            }
            marker = (sr - 2, sc - 2)
            if marker[0] < 0 or marker[1] < 0:
                continue
            if any(cell in arrow_forbidden for cell in seed_cells | {marker}):
                continue

            d0 = abs(sr - first_center[0]) + abs(sc - first_center[1])
            if not (4 <= d0 <= 8):
                continue
            if any(abs(sr - ar) <= 2 and abs(sc - ac) <= 2 for ar, ac in [arrow["center"] for arrow in arrows]):
                continue
            if any(abs(sr - ar) + abs(sc - ac) <= d0 for ar, ac in other_centers):
                continue
            candidates.append((sr, sc))
    return candidates


def _build_grid(
    h: int,
    w: int,
    seed_center: tuple[int, int],
    seed_patch: np.ndarray,
    arrows: list[dict[str, object]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    sr, sc = seed_center
    grid[sr - 1 : sr + 2, sc - 1 : sc + 2] = seed_patch
    grid[sr - 2, sc - 2] = RED
    for arrow in arrows:
        for r, c in arrow["cells"]:
            grid[r, c] = GREY
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    count = int(rng.integers(MIN_ARROWS, MAX_ARROWS + 1))

    arrows = _generate_arrows(rng, h, w, count)
    if arrows is None:
        return None

    candidates = _seed_candidates(h, w, arrows)
    if not candidates:
        return None

    seed_center = candidates[int(rng.integers(0, len(candidates)))]
    seed_patch = _make_seed_pattern(rng)
    grid = _build_grid(h, w, seed_center, seed_patch, arrows)

    # Validate that the generated instance satisfies the solver's assumptions.
    parsed_arrows = []
    for comp in _components(grid == GREY, connectivity=4):
        parsed = _parse_arrow(comp)
        if parsed is not None:
            parsed_arrows.append(parsed)
    if len(parsed_arrows) != count:
        return None

    _, parsed_seed_center = _find_seed(grid)
    dists = sorted(
        abs(parsed_seed_center[0] - arrow["center"][0]) + abs(parsed_seed_center[1] - arrow["center"][1])
        for arrow in parsed_arrows
    )
    if len(dists) < 2 or dists[0] == dists[1]:
        return None

    order = _trace_chain(parsed_arrows, parsed_seed_center)
    if len(order) != count:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
