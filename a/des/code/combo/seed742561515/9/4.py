import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 100
PLACEMENT_ATTEMPTS = 160
SEED_PICK_ATTEMPTS = 60


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_palette(row: np.ndarray) -> tuple[list[int], int]:
    row = np.asarray(row, dtype=int)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < row.size:
        color = int(row[i])
        j = i + 1
        while j < row.size and int(row[j]) == color:
            j += 1
        runs.append((color, j - i))
        i = j
    assert len(runs) >= 2, "expected at least two palette segments"
    widths = {width for _, width in runs}
    assert len(widths) == 1, "palette segments must have equal widths"
    colors = [color for color, _ in runs]
    assert len(colors) == len(set(colors)), "palette segment colors must be distinct"
    return colors, int(next(iter(widths)))


def _identify_scene_colors(scene: np.ndarray) -> tuple[int, int]:
    colors = [int(color) for color in np.unique(scene) if int(color) != 0]
    assert len(colors) == 2, "expected exactly one boundary color and one seed color"

    seed_candidates: list[int] = []
    boundary_candidates: list[int] = []
    for color in colors:
        comps = _components(scene == color)
        if comps and all(len(comp) == 1 for comp in comps):
            seed_candidates.append(color)
        else:
            boundary_candidates.append(color)

    assert len(seed_candidates) == 1 and len(boundary_candidates) == 1
    return int(boundary_candidates[0]), int(seed_candidates[0])


def _enclosed_regions(scene: np.ndarray, boundary_color: int) -> list[list[tuple[int, int]]]:
    h, w = scene.shape
    passable = scene != int(boundary_color)
    regions: list[list[tuple[int, int]]] = []
    for comp in _components(passable):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp):
            continue
        regions.append(comp)
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the yellow seed pixels inside each outlined region and use that count
    as a 1-based index into the bottom palette row, then flood-fill the whole
    region interior with the chosen palette color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    palette_colors, _segment_width = _parse_palette(g[-1])
    scene = g[:-1]
    boundary_color, seed_color = _identify_scene_colors(scene)

    out = g.copy()
    regions = _enclosed_regions(scene, boundary_color)
    assert regions, "expected at least one enclosed region"

    for region in regions:
        seed_count = sum(1 for r, c in region if int(scene[r, c]) == seed_color)
        assert 1 <= seed_count <= len(palette_colors), "seed count must index the palette"
        fill_color = int(palette_colors[seed_count - 1])
        for r, c in region:
            out[r, c] = fill_color

    out[-1] = g[-1]
    return out


def _shape_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w or not mask[nr, nc]:
                boundary[rr, cc] = True
                break
    return boundary


def _rect_mask(rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    min_r = min(r0 for r0, _, _, _ in rects)
    min_c = min(c0 for _, _, c0, _ in rects)
    max_r = max(r1 for _, r1, _, _ in rects)
    max_c = max(c1 for _, _, _, c1 in rects)
    mask = np.zeros((max_r - min_r, max_c - min_c), dtype=bool)
    for r0, r1, c0, c1 in rects:
        mask[r0 - min_r : r1 - min_r, c0 - min_c : c1 - min_c] = True
    return mask


def _pick_seed_positions(
    rng: random.Random, interior_mask: np.ndarray, count: int
) -> list[tuple[int, int]] | None:
    cells = [(int(r), int(c)) for r, c in np.argwhere(interior_mask)]
    if len(cells) < count:
        return None
    for _ in range(SEED_PICK_ATTEMPTS):
        order = cells[:]
        rng.shuffle(order)
        chosen: list[tuple[int, int]] = []
        for r, c in order:
            if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    return None


def _make_region_shape(
    rng: random.Random, seed_count: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]] | None:
    kinds = ("rect", "one_arm", "two_arm")
    side_names = ("top", "bottom", "left", "right")

    for _ in range(SHAPE_ATTEMPTS):
        kind = rng.choice(kinds)
        base_h = rng.randint(4 + (seed_count >= 4), 6 + (seed_count >= 6))
        base_w = rng.randint(4 + (seed_count >= 5), 7 + (seed_count >= 6))
        rects = [(0, base_h, 0, base_w)]

        arm_count = {"rect": 0, "one_arm": 1, "two_arm": 2}[kind]
        if arm_count:
            for side in rng.sample(side_names, arm_count):
                if side in ("top", "bottom"):
                    arm_h = rng.randint(2, 3)
                    arm_w = rng.randint(3, base_w)
                    c0 = rng.randint(0, base_w - arm_w)
                    if side == "top":
                        rects.append((-arm_h, 0, c0, c0 + arm_w))
                    else:
                        rects.append((base_h, base_h + arm_h, c0, c0 + arm_w))
                else:
                    arm_w = rng.randint(2, 3)
                    arm_h = rng.randint(3, base_h)
                    r0 = rng.randint(0, base_h - arm_h)
                    if side == "left":
                        rects.append((r0, r0 + arm_h, -arm_w, 0))
                    else:
                        rects.append((r0, r0 + arm_h, base_w, base_w + arm_w))

        shape = _rect_mask(rects)
        boundary = _shape_boundary(shape)
        interior = shape & ~boundary
        if interior.sum() < seed_count:
            continue
        seeds = _pick_seed_positions(rng, interior, seed_count)
        if seeds is not None:
            return shape, boundary, seeds
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        for dr, dc in DIR8:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _validate_attempt(
    grid: np.ndarray,
    region_infos: list[dict],
) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(out[-1], grid[-1]):
        return False

    for info in region_infos:
        fill_color = int(info["fill_color"])
        interior = info["interior_cells"]
        boundary = info["boundary_cells"]
        if not interior or not boundary:
            return False
        for r, c in interior:
            if int(out[r, c]) != fill_color:
                return False
        for r, c in boundary:
            if int(out[r, c]) != int(info["boundary_color"]):
                return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    for _ in range(40):
        segment_count = rng.randint(3, 6)
        segment_width = rng.randint(2, 4)
        width = segment_count * segment_width
        if width >= 12:
            break
    else:
        return None

    max_regions = 2 + int(width >= 16) + int(width >= 20)
    region_count = rng.randint(2, min(4, max_regions))
    height = rng.randint(max(11, 2 * region_count + 8), 19)
    upper_h = height - 1

    colors = list(range(1, 10))
    rng.shuffle(colors)
    boundary_color = int(colors[0])
    seed_color = int(colors[1])
    palette_colors = [int(color) for color in colors[2 : 2 + segment_count]]

    seed_counts = rng.sample(range(1, segment_count + 1), region_count)

    region_specs: list[dict] = []
    for seed_count in seed_counts:
        made = _make_region_shape(rng, seed_count)
        if made is None:
            return None
        shape, boundary_mask, seeds = made
        region_specs.append(
            {
                "seed_count": int(seed_count),
                "shape": shape,
                "boundary_mask": boundary_mask,
                "seed_cells": seeds,
                "interior_mask": shape & ~boundary_mask,
                "fill_color": int(palette_colors[seed_count - 1]),
            }
        )

    region_specs.sort(
        key=lambda spec: int(spec["shape"].shape[0] * spec["shape"].shape[1]),
        reverse=True,
    )

    grid = np.zeros((height, width), dtype=int)
    forbidden = np.zeros((upper_h, width), dtype=bool)
    region_infos: list[dict] = []

    for spec in region_specs:
        shape = np.asarray(spec["shape"], dtype=bool)
        boundary_mask = np.asarray(spec["boundary_mask"], dtype=bool)
        interior_mask = np.asarray(spec["interior_mask"], dtype=bool)
        sh, sw = shape.shape
        if sh > upper_h - 2 or sw > width - 2:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(1, upper_h - sh - 1)
            left = rng.randint(1, width - sw - 1)

            candidate = np.zeros((upper_h, width), dtype=bool)
            candidate[top : top + sh, left : left + sw] = shape
            if np.any(forbidden & candidate):
                continue

            dilated = _dilate8(candidate)
            if np.any(forbidden & dilated):
                continue

            boundary_cells = [
                (top + int(r), left + int(c))
                for r, c in np.argwhere(boundary_mask)
            ]
            interior_cells = [
                (top + int(r), left + int(c))
                for r, c in np.argwhere(interior_mask)
            ]

            for r, c in boundary_cells:
                grid[r, c] = boundary_color
            for r, c in spec["seed_cells"]:
                grid[top + int(r), left + int(c)] = seed_color

            forbidden |= dilated
            region_infos.append(
                {
                    "fill_color": int(spec["fill_color"]),
                    "boundary_cells": boundary_cells,
                    "interior_cells": interior_cells,
                    "boundary_color": boundary_color,
                }
            )
            placed = True
            break

        if not placed:
            return None

    for idx, color in enumerate(palette_colors):
        start = idx * segment_width
        grid[-1, start : start + segment_width] = int(color)

    if not _validate_attempt(grid, region_infos):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid palette-count fill puzzle")
