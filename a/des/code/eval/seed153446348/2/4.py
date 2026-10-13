from __future__ import annotations

import numpy as np


BG = 0
GRAY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
RNG = np.random.default_rng()
MAX_RETRIES = 400

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ORIENTS = ("up", "down", "left", "right")


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _expected_wall_mask(height: int, width: int, orient: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if orient == "up":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
    elif orient == "down":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
    elif orient == "left":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, -1] = True
    elif orient == "right":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
    else:
        raise ValueError(f"unknown orientation: {orient}")
    return mask


def _lid_cells(bbox: tuple[int, int, int, int], orient: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if orient == "up":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if orient == "down":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if orient == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    if orient == "right":
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise ValueError(f"unknown orientation: {orient}")


def _stripe_positions(size: int, count: int) -> list[int]:
    assert 1 <= count <= size
    raw = np.linspace(0, size - 1, count)
    positions: list[int] = []
    last = -1
    for i, value in enumerate(raw):
        pos = int(round(float(value)))
        lo = last + 1
        hi = size - (count - i)
        pos = max(lo, min(pos, hi))
        positions.append(pos)
        last = pos
    return positions


def _is_full_line(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height != 1 and width != 1:
        return False
    return int(coords.shape[0]) == height * width


def _parse_container(grid: np.ndarray, coords: np.ndarray) -> dict:
    colors = {int(grid[r, c]) for r, c in coords}
    non_gray = colors - {GRAY}
    assert len(non_gray) == 1
    color = next(iter(non_gray))

    color_coords = np.array([(int(r), int(c)) for r, c in coords if int(grid[r, c]) == color], dtype=int)
    r0, c0, r1, c1 = _bbox(color_coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 4 and width >= 4

    rel = np.zeros((height, width), dtype=bool)
    for r, c in color_coords:
        rel[r - r0, c - c0] = True

    orient = None
    for candidate in ORIENTS:
        if np.array_equal(rel, _expected_wall_mask(height, width, candidate)):
            orient = candidate
            break
    assert orient is not None

    gray_coords = [(int(r), int(c)) for r, c in coords if int(grid[r, c]) == GRAY]
    for gr, gc in gray_coords:
        assert r0 < gr < r1 and c0 < gc < c1

    lid_cells = _lid_cells((r0, c0, r1, c1), orient)
    mouth = len(lid_cells)
    assert mouth >= 2

    return {
        "bbox": (r0, c0, r1, c1),
        "color": int(color),
        "orient": orient,
        "lid_cells": lid_cells,
        "mouth": mouth,
    }


def _count_cavities(grid: np.ndarray, bbox: tuple[int, int, int, int], wall_color: int) -> int:
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    empty = (patch != wall_color) & (patch != GRAY)
    return len(_component_coords(empty))


def _draw_stripes(grid: np.ndarray, container: dict, count: int) -> None:
    r0, c0, r1, c1 = container["bbox"]
    color = int(container["color"])
    orient = container["orient"]

    if orient in ("up", "down"):
        rows = _stripe_positions(r1 - r0 - 1, count)
        for rel_r in rows:
            rr = r0 + 1 + rel_r
            for cc in range(c0 + 1, c1):
                if int(grid[rr, cc]) == BG:
                    grid[rr, cc] = color
    else:
        cols = _stripe_positions(c1 - c0 - 1, count)
        for rel_c in cols:
            cc = c0 + 1 + rel_c
            for rr in range(r0 + 1, r1):
                if int(grid[rr, cc]) == BG:
                    grid[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match gray lids to U-shaped containers by mouth width, close each container,
    then draw as many equally spaced stripes as the number of sealed cavities.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _component_coords(g != BG)
    containers: list[dict] = []
    lids: list[dict] = []

    for coords in comps:
        values = {int(g[r, c]) for r, c in coords}
        if values == {GRAY}:
            mask = np.zeros(g.shape, dtype=bool)
            mask[coords[:, 0], coords[:, 1]] = True
            assert _is_full_line(mask)
            lids.append({"coords": coords, "length": int(coords.shape[0])})
            continue
        containers.append(_parse_container(g, coords))

    assert containers and len(containers) == len(lids)

    mouths = [int(container["mouth"]) for container in containers]
    lengths = [int(lid["length"]) for lid in lids]
    assert len(set(mouths)) == len(mouths)
    assert sorted(mouths) == sorted(lengths)

    out = g.copy()
    for lid in lids:
        out[lid["coords"][:, 0], lid["coords"][:, 1]] = BG

    for container in containers:
        for rr, cc in container["lid_cells"]:
            out[rr, cc] = GRAY
        cavity_count = _count_cavities(out, container["bbox"], int(container["color"]))
        assert cavity_count >= 1
        _draw_stripes(out, container, cavity_count)

    return out


def _random_composition(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    if parts == 1:
        return [total]
    cuts = sorted(RNG.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _partition_positions(mouth: int, chambers: int) -> list[int]:
    assert 1 <= chambers <= (mouth + 1) // 2
    widths = _random_composition(mouth - (chambers - 1), chambers)
    positions: list[int] = []
    cursor = widths[0] + 1
    for width in widths[1:]:
        positions.append(cursor)
        cursor += width + 1
    return positions


def _build_container_patch(color: int, orient: str, mouth: int, depth: int, chambers: int) -> np.ndarray:
    assert depth >= chambers
    if orient in ("up", "down"):
        patch = np.zeros((depth + 2, mouth + 2), dtype=int)
    else:
        patch = np.zeros((mouth + 2, depth + 2), dtype=int)

    wall_mask = _expected_wall_mask(*patch.shape, orient)
    patch[wall_mask] = color

    for pos in _partition_positions(mouth, chambers):
        if orient == "up" or orient == "down":
            patch[1:-1, pos] = GRAY
        else:
            patch[pos, 1:-1] = GRAY
    return patch


def _build_lid_patch(length: int, vertical: bool) -> np.ndarray:
    if vertical:
        patch = np.zeros((length, 1), dtype=int)
    else:
        patch = np.zeros((1, length), dtype=int)
    patch[patch == 0] = GRAY
    return patch


def _dilate(mask: np.ndarray) -> np.ndarray:
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


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    mask = patch != BG
    allowed: list[tuple[int, int]] = []
    blocked = _dilate(occupied)
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if not np.any(blocked[r0 : r0 + ph, c0 : c0 + pw] & mask):
                allowed.append((r0, c0))
    if not allowed:
        return False
    r0, c0 = allowed[int(RNG.integers(0, len(allowed)))]
    grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
    occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
    return True


def _generate_attempt() -> np.ndarray | None:
    height = int(RNG.integers(22, 33))
    width = int(RNG.integers(22, 33))
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    n_containers = int(RNG.integers(1, 4))
    mouths = RNG.choice(np.arange(3, 10), size=n_containers, replace=False).tolist()
    colors = RNG.choice(np.array(PALETTE), size=n_containers, replace=False).tolist()

    objects: list[np.ndarray] = []
    for color, mouth in zip(colors, mouths):
        orient = ORIENTS[int(RNG.integers(0, len(ORIENTS)))]
        depth = int(RNG.integers(2, 8))
        max_chambers = min((mouth + 1) // 2, depth, 4)
        chambers = int(RNG.integers(1, max_chambers + 1))
        objects.append(_build_container_patch(int(color), orient, int(mouth), depth, chambers))

        needs_vertical = orient in ("left", "right")
        same_orientation = bool(RNG.integers(0, 2))
        lid_vertical = needs_vertical if same_orientation else (not needs_vertical)
        objects.append(_build_lid_patch(int(mouth), lid_vertical))

    objects.sort(key=lambda patch: int(np.count_nonzero(patch)), reverse=True)
    for patch in objects:
        if not _place_patch(grid, occupied, patch):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        candidate = _generate_attempt()
        if candidate is None:
            continue
        try:
            result = main(candidate)
        except Exception:
            continue
        if np.array_equal(candidate, result):
            continue
        return candidate
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
