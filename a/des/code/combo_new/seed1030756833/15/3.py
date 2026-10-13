from __future__ import annotations

import random
from collections import deque

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_PERIOD = 6


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _l_pattern(height: int, width: int, orientation: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if orientation == 0:  # vertical arm points up
        mask[:, 0] = True
        mask[-1, :] = True
    elif orientation == 1:  # horizontal arm points right
        mask[0, :] = True
        mask[:, 0] = True
    elif orientation == 2:  # vertical arm points down
        mask[0, :] = True
        mask[:, -1] = True
    else:  # horizontal arm points left
        mask[:, -1] = True
        mask[-1, :] = True
    return mask


def _classify_l_shape(mask: np.ndarray) -> int | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    norm = mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    h, w = norm.shape
    if h < 2 or w < 2:
        return None
    for orientation in range(4):
        if np.array_equal(norm, _l_pattern(h, w, orientation)):
            return orientation
    return None


def _occluder_candidate(
    g: np.ndarray, color: int, cells: list[tuple[int, int]]
) -> dict[str, int] | None:
    r0, c0, r1, c1 = _bbox(cells)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 4 or width < 4:
        return None

    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    color_mask = patch == int(color)
    if int(color_mask.sum()) != len(cells):
        return None
    if not (
        np.all(color_mask[0, :])
        and np.all(color_mask[-1, :])
        and np.all(color_mask[:, 0])
        and np.all(color_mask[:, -1])
    ):
        return None

    hole = ~color_mask
    if not bool(hole.any()):
        return None
    if bool(hole[0, :].any() or hole[-1, :].any() or hole[:, 0].any() or hole[:, -1].any()):
        return None

    hole_values = set(map(int, patch[hole].ravel()))
    if len(hole_values) != 1:
        return None

    hole_components = _components(hole)
    if len(hole_components) != 1:
        return None

    orientation = _classify_l_shape(hole)
    if orientation is None:
        return None

    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "color": int(color),
        "hole_color": int(next(iter(hole_values))),
        "orientation": int(orientation),
    }


def _detect_occluders(g: np.ndarray) -> list[dict[str, int]]:
    occluders: list[dict[str, int]] = []
    for color in map(int, np.unique(g)):
        comps = _components(g == color)
        infos: list[dict[str, int]] = []
        for comp in comps:
            info = _occluder_candidate(g, color, comp)
            if info is None:
                infos = []
                break
            infos.append(info)
        if infos:
            occluders.extend(infos)

    assert occluders, "expected at least one rectangular occluder with an L-shaped hole"

    occupied = np.zeros(g.shape, dtype=bool)
    for occ in sorted(occluders, key=lambda x: (x["r0"], x["c0"])):
        region = occupied[occ["r0"] : occ["r1"] + 1, occ["c0"] : occ["c1"] + 1]
        assert not bool(region.any()), "occluder rectangles must not overlap"
        region[:, :] = True
    return sorted(occluders, key=lambda x: (x["r0"], x["c0"]))


def _infer_tile(g: np.ndarray, visible: np.ndarray) -> np.ndarray:
    h, w = g.shape
    periods = [
        (ph * pw, ph, pw)
        for ph in range(2, min(MAX_PERIOD, h) + 1)
        for pw in range(2, min(MAX_PERIOD, w) + 1)
    ]
    for _area, ph, pw in sorted(periods):
        tile = np.full((ph, pw), -1, dtype=int)
        seen = np.zeros((ph, pw), dtype=bool)
        ok = True
        for r, c in np.argwhere(visible):
            rr, cc = int(r) % ph, int(c) % pw
            value = int(g[int(r), int(c)])
            if seen[rr, cc] and int(tile[rr, cc]) != value:
                ok = False
                break
            tile[rr, cc] = value
            seen[rr, cc] = True
        if ok and bool(np.all(seen)):
            return tile
    raise AssertionError("uncovered pixels must determine a small periodic tile")


def _periodic_patch(tile: np.ndarray, r0: int, c0: int, height: int, width: int) -> np.ndarray:
    rows = (np.arange(r0, r0 + height) % tile.shape[0]).astype(int)
    cols = (np.arange(c0, c0 + width) % tile.shape[1]).astype(int)
    return tile[rows[:, None], cols[None, :]].astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a hidden periodic tiling and replace each holed occluder with the rotation encoded by its L-shaped hole.

    Input structure:
    - A small multicolor tile, typically 3x3 or 4x4, repeats across the whole grid.
    - Several solid-color rectangular occluders cover parts of that tiling; occluder colors are not used by the tile.
    - Each occluder rectangle contains exactly one connected background-color L-shaped hole, in one of four orientations.

    Rule:
    - Detect every occluder rectangle and treat the entire rectangle, including its L-shaped hole, as hidden.
    - Infer the repeated tile from all non-occluded cells and reconstruct the original tile pattern inside each rectangle.
    - Rotate each reconstructed block around its center by the amount encoded by the hole direction: up=0, right=90, down=180, left=270 degrees.
    - Fill the whole rectangle with that rotated block, so the hole is also covered by the restored pattern.

    Key insight:
    The rectangle marks where the periodic image was hidden, while the internal L-shaped hole is a separate directional instruction for rotating the recovered patch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    occluders = _detect_occluders(g)
    visible = np.ones(g.shape, dtype=bool)
    for occ in occluders:
        visible[occ["r0"] : occ["r1"] + 1, occ["c0"] : occ["c1"] + 1] = False

    tile = _infer_tile(g, visible)
    out = g.copy()
    for occ in occluders:
        height = occ["r1"] - occ["r0"] + 1
        width = occ["c1"] - occ["c0"] + 1
        restored = _periodic_patch(tile, occ["r0"], occ["c0"], height, width)
        rotated = np.rot90(restored, -int(occ["orientation"]))
        assert rotated.shape == (height, width), "90-degree rotations require square occluder rectangles"
        out[occ["r0"] : occ["r1"] + 1, occ["c0"] : occ["c1"] + 1] = rotated
    return out.astype(int)


# ------------------------- generator -------------------------

TILE_SHAPES = ((3, 3), (3, 4), (4, 3), (4, 4))
TILE_COLOR_COUNTS = (3, 4)
MANDATORY_ORIENTATIONS = (0, 1, 2, 3)
GEN_RETRIES = 400
TILE_RETRIES = 500
PLACEMENT_RETRIES = 600
RECT_SIDE_RANGE = (5, 8)
HOLE_SIDE_RANGE = (2, 4)
PADDING = 1


def _make_periodic_grid(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = (np.arange(height) % tile.shape[0]).astype(int)
    cols = (np.arange(width) % tile.shape[1]).astype(int)
    return tile[rows[:, None], cols[None, :]].astype(int)


def _make_tile(th: int, tw: int, colors: list[int]) -> np.ndarray | None:
    bg = int(colors[0])
    payload = [int(c) for c in colors[1:]]
    area = th * tw
    min_payload = max(len(payload), area // 3)
    max_payload = min(area - 2, area // 2)

    for _ in range(TILE_RETRIES):
        tile = np.full((th, tw), bg, dtype=int)
        cells = [(r, c) for r in range(th) for c in range(tw)]
        chosen = random.sample(cells, random.randint(min_payload, max_payload))
        assignments = payload[:]
        assignments.extend(random.choice(payload) for _ in range(len(chosen) - len(assignments)))
        random.shuffle(assignments)
        for (r, c), color in zip(chosen, assignments):
            tile[r, c] = int(color)

        probe = _make_periodic_grid(tile, th * 4, tw * 4)
        try:
            inferred = _infer_tile(probe, np.ones(probe.shape, dtype=bool))
        except AssertionError:
            continue
        if inferred.shape == tile.shape and np.array_equal(inferred, tile):
            return tile
    return None


def _rect_clear(occupied: np.ndarray, r0: int, c0: int, side: int) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - PADDING)
    cc0 = max(0, c0 - PADDING)
    rr1 = min(h, r0 + side + PADDING)
    cc1 = min(w, c0 + side + PADDING)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _rotations_are_visible(block: np.ndarray) -> bool:
    return all(not np.array_equal(block, np.rot90(block, -k)) for k in (1, 2, 3))


def _place_occluders(
    base: np.ndarray, orientations: list[int], colors: list[int]
) -> list[dict[str, int]] | None:
    h, w = base.shape
    occupied = np.zeros(base.shape, dtype=bool)
    placements: list[dict[str, int]] = []

    for orientation, color in zip(orientations, colors):
        placed = False
        max_side = min(RECT_SIDE_RANGE[1], h - 2, w - 2)
        if max_side < RECT_SIDE_RANGE[0]:
            return None
        for _ in range(PLACEMENT_RETRIES):
            side = random.randint(RECT_SIDE_RANGE[0], max_side)
            r0 = random.randint(1, h - side - 1)
            c0 = random.randint(1, w - side - 1)
            if not _rect_clear(occupied, r0, c0, side):
                continue
            block = base[r0 : r0 + side, c0 : c0 + side]
            if not _rotations_are_visible(block):
                continue

            max_hole = min(HOLE_SIDE_RANGE[1], side - 2)
            if max_hole < HOLE_SIDE_RANGE[0]:
                continue
            hole_h = random.randint(HOLE_SIDE_RANGE[0], max_hole)
            hole_w = random.randint(HOLE_SIDE_RANGE[0], max_hole)
            hr = random.randint(1, side - hole_h - 1)
            hc = random.randint(1, side - hole_w - 1)

            occupied[r0 : r0 + side, c0 : c0 + side] = True
            placements.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "side": int(side),
                    "color": int(color),
                    "orientation": int(orientation),
                    "hr": int(hr),
                    "hc": int(hc),
                    "hole_h": int(hole_h),
                    "hole_w": int(hole_w),
                }
            )
            placed = True
            break
        if not placed:
            return None
    return placements


def _render_generated(
    base: np.ndarray, bg: int, placements: list[dict[str, int]]
) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    expected = base.copy()
    for occ in placements:
        r0, c0, side = occ["r0"], occ["c0"], occ["side"]
        inp[r0 : r0 + side, c0 : c0 + side] = int(occ["color"])
        hole = _l_pattern(occ["hole_h"], occ["hole_w"], occ["orientation"])
        for rr, cc in np.argwhere(hole):
            inp[r0 + occ["hr"] + int(rr), c0 + occ["hc"] + int(cc)] = int(bg)

        block = base[r0 : r0 + side, c0 : c0 + side]
        expected[r0 : r0 + side, c0 : c0 + side] = np.rot90(block, -int(occ["orientation"]))
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    th, tw = random.choice(TILE_SHAPES)
    n_occluders = 4 + int(random.random() < 0.5)
    color_count = random.choice(TILE_COLOR_COUNTS)
    colors = random.sample(range(10), color_count + n_occluders)
    tile_colors = colors[:color_count]
    occluder_colors = colors[color_count:]
    bg = int(tile_colors[0])

    tile = _make_tile(th, tw, tile_colors)
    if tile is None:
        return None

    row_reps = random.randint(max(5, (18 + th - 1) // th), 30 // th)
    col_reps = random.randint(max(5, (18 + tw - 1) // tw), 30 // tw)
    height, width = th * row_reps, tw * col_reps
    base = _make_periodic_grid(tile, height, width)

    orientations = list(MANDATORY_ORIENTATIONS)
    if n_occluders > len(orientations):
        orientations.append(random.randrange(4))
    random.shuffle(orientations)

    placements = _place_occluders(base, orientations, occluder_colors)
    if placements is None:
        return None

    inp, expected = _render_generated(base, bg, placements)
    try:
        actual = main(inp)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp.astype(int)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-occluder input")
