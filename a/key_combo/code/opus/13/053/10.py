from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
SOURCE_COLORS = (1, 3, 4, 6, 7, 8, 9)
ODD_SIDES = (13, 15, 17, 19)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _find_cross_axes(grid: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == RED)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == RED)]
    assert len(rows) == 1 and len(cols) == 1, "expected one full red row and one full red column"
    axis_r = int(rows[0])
    axis_c = int(cols[0])
    assert 2 * axis_r + 1 == grid.shape[0], "horizontal frame line must be centered"
    assert 2 * axis_c + 1 == grid.shape[1], "vertical frame line must be centered"
    return axis_r, axis_c


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    H, W = mask.shape
    comps: list[np.ndarray] = []

    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _paint_mask(out: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = np.argwhere(mask)
    out[r0 + coords[:, 0], c0 + coords[:, 1]] = int(color)


def _global_coords(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)]


def _build_output_from_specs(
    shape: tuple[int, int],
    axis_r: int,
    axis_c: int,
    placements: list[dict],
    mode: str = "rule",
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[axis_r, :] = RED
    out[:, axis_c] = RED

    for placement in placements:
        _paint_mask(out, placement["mask"], placement["r0"], placement["c0"], placement["color"])

    for placement in placements:
        coords = _global_coords(placement["mask"], placement["r0"], placement["c0"])
        area = int(placement["area"])
        is_large = area >= 3
        if mode == "all_large":
            is_large = True
        elif mode == "all_small":
            is_large = False

        for r, c in coords:
            out[2 * axis_r - r, c] = GREY
        if is_large:
            for r, c in coords:
                out[r, 2 * axis_c - c] = GREY
                out[2 * axis_r - r, 2 * axis_c - c] = GREY

    return out


def _random_polyomino(area: int, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    frontier.add((r + dr, c + dc))
            frontier -= cells
            cells.add(random.choice(tuple(frontier)))

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0 = min(rows)
        c0 = min(cols)
        norm = {(r - r0, c - c0) for r, c in cells}
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h > max_h or w > max_w:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True

        k = random.randrange(4)
        if k:
            mask = np.rot90(mask, k)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        return mask.copy()

    raise RuntimeError("failed to sample a compact polyomino")


def _sample_component_count(min_quadrant: int) -> int:
    if min_quadrant <= 6:
        return random.choice((3, 4))
    if min_quadrant == 7:
        return random.choice((3, 4, 5))
    return random.choice((4, 5))


def _sample_areas(count: int) -> list[int]:
    areas = [3, 2]
    while len(areas) < count:
        areas.append(random.choice((1, 1, 2, 3, 4, 4, 5)))
    if count >= 4 and not any(area > 3 for area in areas):
        areas[-1] = random.choice((4, 5, 6))
    if count >= 4 and random.random() < 0.35:
        areas[random.randrange(2, count)] = 6
    random.shuffle(areas)
    return areas


def _try_place_components(axis_r: int, axis_c: int, areas: list[int], colors: list[int]) -> list[dict] | None:
    placements: list[dict] = []
    occupied = np.zeros((axis_r, axis_c), dtype=bool)
    max_h = min(4, axis_r - 1)
    max_w = min(4, axis_c - 1)

    specs = []
    for area, color in zip(areas, colors):
        mask = _random_polyomino(int(area), max_h=max_h, max_w=max_w)
        specs.append({"area": int(area), "color": int(color), "mask": mask})

    specs.sort(key=lambda item: (-item["area"], -item["mask"].shape[0] * item["mask"].shape[1]))

    for spec in specs:
        mask = spec["mask"]
        h, w = mask.shape
        max_r0 = axis_r - h - 1
        max_c0 = axis_c - w - 1
        if max_r0 < 0 or max_c0 < 0:
            return None

        blocked = _dilate8(occupied)
        candidates = [(r0, c0) for r0 in range(max_r0 + 1) for c0 in range(max_c0 + 1)]
        random.shuffle(candidates)

        placed = False
        for r0, c0 in candidates:
            if np.any(blocked[r0 : r0 + h, c0 : c0 + w] & mask):
                continue
            occupied[r0 : r0 + h, c0 : c0 + w] |= mask
            placements.append(
                {
                    "area": spec["area"],
                    "color": spec["color"],
                    "mask": mask,
                    "r0": int(r0),
                    "c0": int(c0),
                }
            )
            placed = True
            break

        if not placed:
            return None

    return placements


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    placements: list[dict],
    axis_r: int,
    axis_c: int,
) -> bool:
    got = main(inp)
    if not np.array_equal(got, expected):
        return False

    areas = [int(item["area"]) for item in placements]
    if not any(area < 3 for area in areas) or not any(area >= 3 for area in areas):
        return False
    if 2 not in areas or 3 not in areas:
        return False

    if np.array_equal(expected, _build_output_from_specs(inp.shape, axis_r, axis_c, placements, mode="all_large")):
        return False
    if np.array_equal(expected, _build_output_from_specs(inp.shape, axis_r, axis_c, placements, mode="all_small")):
        return False

    source_mask = (inp[:axis_r, :axis_c] != BLACK) & (inp[:axis_r, :axis_c] != RED)
    if len(_components4(source_mask)) != len(placements):
        return False

    for placement in placements:
        coords = _global_coords(placement["mask"], placement["r0"], placement["c0"])
        mirrored_down = [(2 * axis_r - r, c) for r, c in coords]
        mirrored_right = [(r, 2 * axis_c - c) for r, c in coords]
        mirrored_diag = [(2 * axis_r - r, 2 * axis_c - c) for r, c in coords]

        if not all(expected[r, c] == GREY for r, c in mirrored_down):
            return False
        if placement["area"] >= 3:
            if not all(expected[r, c] == GREY for r, c in mirrored_right + mirrored_diag):
                return False
        else:
            if any(expected[r, c] == GREY for r, c in mirrored_right + mirrored_diag):
                return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror left-top components across a centered red cross, with area deciding whether the mirror is full or downward-only.

    Input structure:
    - The grid has a black background and one full red horizontal line plus one full red vertical line crossing at the center.
    - The red cross divides the canvas into four equal quadrants.
    - Only the top-left quadrant contains colored 4-connected components, separated from each other and from the red lines.

    Rule:
    - Use the centered red cross as the pair of reflection axes and inspect only the non-black, non-red components in the top-left quadrant.
    - Keep the original top-left components unchanged.
    - For each source component with area at least 3, place gray copies reflected to the top-right, bottom-left, and bottom-right quadrants.
    - For each source component with area below 3, place only its gray reflection across the horizontal axis into the bottom-left quadrant.
    - Keep the red cross unchanged and leave all other cells black.

    Key insight:
    - Component area does not choose a color; it chooses which mirror operation branch applies: full two-axis reflection for larger components, but only the downward reflection for smaller ones.
    """
    g = np.asarray(grid, dtype=int)
    axis_r, axis_c = _find_cross_axes(g)

    out = np.zeros_like(g)
    out[axis_r, :] = RED
    out[:, axis_c] = RED
    out[:axis_r, :axis_c] = g[:axis_r, :axis_c]

    source = g[:axis_r, :axis_c]
    comps = _components4((source != BLACK) & (source != RED))
    for comp in comps:
        coords = [(int(r), int(c)) for r, c in comp]
        for r, c in coords:
            out[2 * axis_r - r, c] = GREY
        if len(coords) >= 3:
            for r, c in coords:
                out[r, 2 * axis_c - c] = GREY
                out[2 * axis_r - r, 2 * axis_c - c] = GREY

    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        H = random.choice(ODD_SIDES)
        W = random.choice(ODD_SIDES)
        axis_r = H // 2
        axis_c = W // 2
        count = _sample_component_count(min(axis_r, axis_c))
        areas = _sample_areas(count)
        colors = random.sample(SOURCE_COLORS, count)
        placements = _try_place_components(axis_r, axis_c, areas, colors)
        if placements is None:
            continue

        inp = np.zeros((H, W), dtype=int)
        inp[axis_r, :] = RED
        inp[:, axis_c] = RED
        for placement in placements:
            _paint_mask(inp, placement["mask"], placement["r0"], placement["c0"], placement["color"])

        expected = _build_output_from_specs(inp.shape, axis_r, axis_c, placements)
        if _semantic_checks(inp, expected, placements, axis_r, axis_c):
            return inp

    raise RuntimeError("failed to generate a valid puzzle instance")
