from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BACKGROUND = 1
BLACK = 0
GREY = 5
MARKER_COLORS = (2, 3, 4, 6, 7, 8, 9)
SECTORS = ("up", "down", "left", "right")
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sector_of(r: int, c: int, anchor_r: int, anchor_c: int) -> str | None:
    dr = int(r) - int(anchor_r)
    dc = int(c) - int(anchor_c)
    if dr == 0 and dc == 0:
        return None
    adr = abs(dr)
    adc = abs(dc)
    if adr == adc:
        return None
    if adr < adc:
        return "right" if dc > 0 else "left"
    return "down" if dr > 0 else "up"


def _ray_cells(shape: tuple[int, int], anchor_r: int, anchor_c: int) -> set[tuple[int, int]]:
    cells = {(int(anchor_r), int(anchor_c))}
    h, w = shape
    for dr, dc in DIAGONALS:
        rr, cc = anchor_r, anchor_c
        while True:
            rr += dr
            cc += dc
            if not (0 <= rr < h and 0 <= cc < w):
                break
            cells.add((int(rr), int(cc)))
    return cells


def _draw_rays(out: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    out[anchor_r, anchor_c] = BLACK
    for dr, dc in DIAGONALS:
        rr, cc = anchor_r, anchor_c
        while True:
            rr += dr
            cc += dc
            if not _in_bounds(out.shape, rr, cc):
                break
            out[rr, cc] = BLACK


def _object_sector(coords: np.ndarray, anchor_r: int, anchor_c: int) -> str:
    sectors = {
        _sector_of(int(r), int(c), anchor_r, anchor_c)
        for r, c in coords
    }
    assert len(sectors) == 1 and None not in sectors, "grey object must lie in exactly one sector"
    return next(iter(sectors))


def _min_sqdist(coords: np.ndarray, marker: tuple[int, int]) -> int:
    mr, mc = marker
    d2 = (coords[:, 0] - int(mr)) ** 2 + (coords[:, 1] - int(mc)) ** 2
    return int(d2.min())


def _extract_markers(grid: np.ndarray, anchor_r: int, anchor_c: int) -> list[tuple[int, int, int, str]]:
    markers = []
    for r, c in np.argwhere((grid != BACKGROUND) & (grid != BLACK) & (grid != GREY)):
        rr = int(r)
        cc = int(c)
        sector = _sector_of(rr, cc, anchor_r, anchor_c)
        assert sector is not None, "marker may not lie on a diagonal ray"
        markers.append((rr, cc, int(grid[rr, cc]), sector))
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey object by the nearest marker restricted to its diagonal-ray sector.

    Input structure:
    - Blue background with exactly one interior black anchor pixel.
    - Several small 4-connected grey objects and several isolated non-grey marker pixels.
    - The anchor is not on the border; grey objects and markers are off the four diagonals through the anchor.

    Rule:
    - Extend the black anchor into four diagonal rays to the grid edges.
    - Use those rays as sector boundaries: every non-ray cell belongs to exactly one of up, down, left, or right according to the anchor-relative |dr| and |dc| comparison.
    - For each grey object, consider only markers in the same sector; compute the minimum squared Euclidean distance from the object to each candidate marker and recolor the whole object by the uniquely nearest marker's color.
    - Erase all marker pixels to blue, while keeping the black anchor and all four diagonal rays.

    Key insight:
    - The nearest marker is not chosen globally; the diagonal rays act as sector firewalls, so a closer marker across a ray must be ignored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    black_coords = np.argwhere(g == BLACK)
    assert len(black_coords) == 1, "input must contain exactly one black anchor"
    anchor_r, anchor_c = map(int, black_coords[0])

    grey_objects = _components(g == GREY)
    assert grey_objects, "expected grey objects"

    markers = _extract_markers(g, anchor_r, anchor_c)
    markers_by_sector: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for r, c, color, sector in markers:
        markers_by_sector[sector].append((r, c, color))

    out = g.copy()
    for coords in grey_objects:
        sector = _object_sector(coords, anchor_r, anchor_c)
        candidates = markers_by_sector[sector]
        assert candidates, f"sector {sector} has no marker"
        distances = [_min_sqdist(coords, (mr, mc)) for mr, mc, _color in candidates]
        best = min(distances)
        assert distances.count(best) == 1, "nearest same-sector marker must be unique"
        winner = candidates[distances.index(best)][2]
        out[coords[:, 0], coords[:, 1]] = int(winner)

    for r, c, _color, _sector in markers:
        out[r, c] = BACKGROUND
    _draw_rays(out, anchor_r, anchor_c)
    return out


_BASE_SHAPES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)),
)


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in shape)
    c0 = min(c for _, c in shape)
    return tuple(sorted((r - r0, c - c0) for r, c in shape))


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    vars_seen = set()
    base = np.array(shape, dtype=int)
    for flip in (1, -1):
        pts = base.copy()
        pts[:, 1] *= flip
        for _ in range(4):
            norm = _normalize_shape(tuple(map(tuple, pts)))
            vars_seen.add(norm)
            pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
    return list(vars_seen)


SHAPE_VARIANTS = [variant for shape in _BASE_SHAPES for variant in _shape_variants(shape)]


def _boundary_gap(r: int, c: int, anchor_r: int, anchor_c: int) -> int:
    return abs(abs(r - anchor_r) - abs(c - anchor_c))


def _translate_shape(shape: tuple[tuple[int, int], ...], top_r: int, top_c: int) -> tuple[tuple[int, int], ...]:
    return tuple((top_r + r, top_c + c) for r, c in shape)


def _cell_clear(
    cell: tuple[int, int],
    blocked: set[tuple[int, int]],
    spaced: set[tuple[int, int]],
) -> bool:
    r, c = cell
    if cell in blocked:
        return False
    for rr, cc in _neighbors8(r, c):
        if (rr, cc) in spaced:
            return False
    return cell not in spaced


def _all_cells_clear(
    cells: tuple[tuple[int, int], ...],
    blocked: set[tuple[int, int]],
    spaced: set[tuple[int, int]],
) -> bool:
    return all(_cell_clear(cell, blocked, spaced) for cell in cells)


def _sector_cells(shape: tuple[int, int], anchor_r: int, anchor_c: int, sector: str) -> set[tuple[int, int]]:
    cells = set()
    h, w = shape
    for r in range(h):
        for c in range(w):
            if _sector_of(r, c, anchor_r, anchor_c) == sector:
                cells.add((r, c))
    return cells


def _place_object(
    shape: tuple[int, int],
    anchor_r: int,
    anchor_c: int,
    sector: str,
    blocked: set[tuple[int, int]],
    spaced: set[tuple[int, int]],
    require_near_ray: bool,
) -> tuple[tuple[int, int], ...] | None:
    h, w = shape
    candidates: list[tuple[tuple[int, int], ...]] = []
    near_candidates: list[tuple[tuple[int, int], ...]] = []
    for variant in SHAPE_VARIANTS:
        vr = max(r for r, _ in variant) + 1
        vc = max(c for _, c in variant) + 1
        for top_r in range(h - vr + 1):
            for top_c in range(w - vc + 1):
                cells = _translate_shape(variant, top_r, top_c)
                if any(
                    _sector_of(r, c, anchor_r, anchor_c) != sector
                    or r == anchor_r
                    or c == anchor_c
                    or _boundary_gap(r, c, anchor_r, anchor_c) == 0
                    for r, c in cells
                ):
                    continue
                if not _all_cells_clear(cells, blocked, spaced):
                    continue
                if len({r for r, _ in cells}) < 2 or len({c for _, c in cells}) < 2:
                    continue
                candidates.append(cells)
                if min(_boundary_gap(r, c, anchor_r, anchor_c) for r, c in cells) <= 2:
                    near_candidates.append(cells)
    if require_near_ray and near_candidates:
        return random.choice(near_candidates)
    if candidates:
        return random.choice(candidates)
    return None


def _marker_candidates(
    shape: tuple[int, int],
    anchor_r: int,
    anchor_c: int,
    sector: str,
    blocked: set[tuple[int, int]],
    spaced: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for r in range(h):
        for c in range(w):
            if _sector_of(r, c, anchor_r, anchor_c) != sector:
                continue
            if not _cell_clear((r, c), blocked, spaced):
                continue
            out.append((r, c))
    return out


def _pick_marker(
    shape: tuple[int, int],
    anchor_r: int,
    anchor_c: int,
    sector: str,
    blocked: set[tuple[int, int]],
    spaced: set[tuple[int, int]],
    target_object: np.ndarray | None,
    min_d2: int,
    max_d2: int,
) -> tuple[int, int] | None:
    cells = _marker_candidates(shape, anchor_r, anchor_c, sector, blocked, spaced)
    random.shuffle(cells)
    if target_object is None:
        return cells[0] if cells else None
    scored = []
    for cell in cells:
        d2 = _min_sqdist(target_object, cell)
        if min_d2 <= d2 <= max_d2:
            scored.append((d2, cell))
    if scored:
        scored.sort(key=lambda item: (item[0], item[1]))
        if random.random() < 0.5:
            return scored[0][1]
        return random.choice([cell for _d2, cell in scored])
    return None


def _independent_expected(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    black_coords = np.argwhere(g == BLACK)
    assert len(black_coords) == 1
    anchor_r, anchor_c = map(int, black_coords[0])

    grey_objects = _components(g == GREY)
    assert 3 <= len(grey_objects) <= 5
    markers = _extract_markers(g, anchor_r, anchor_c)
    marker_colors = [color for _r, _c, color, _sector in markers]
    assert len(marker_colors) == len(set(marker_colors))

    for r, c, _color, _sector in markers:
        assert _boundary_gap(r, c, anchor_r, anchor_c) > 0
        for rr, cc in _neighbors8(r, c):
            if _in_bounds(g.shape, rr, cc):
                assert not (g[rr, cc] == GREY), "marker may not touch a grey object"
                assert not (g[rr, cc] not in (BACKGROUND, BLACK) and (rr, cc) != (r, c)), "marker must be isolated"

    objects_info = []
    markers_by_sector: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for r, c, color, sector in markers:
        markers_by_sector[sector].append((r, c, color))

    winner_colors = []
    sector_has_choice = False
    firewall_case = False
    for coords in grey_objects:
        sector = _object_sector(coords, anchor_r, anchor_c)
        area = int(coords.shape[0])
        assert 3 <= area <= 6
        candidates = markers_by_sector[sector]
        assert candidates
        if len(candidates) >= 2:
            sector_has_choice = True
        distances = [_min_sqdist(coords, (mr, mc)) for mr, mc, _color in candidates]
        best = min(distances)
        assert distances.count(best) == 1
        winner_idx = distances.index(best)
        winner = candidates[winner_idx]
        winner_colors.append(int(winner[2]))

        all_markers = [(mr, mc, color, sec) for sec, items in markers_by_sector.items() for mr, mc, color in items]
        global_d2 = [_min_sqdist(coords, (mr, mc)) for mr, mc, _color, _sec in all_markers]
        gbest = min(global_d2)
        if global_d2.count(gbest) == 1:
            gidx = global_d2.index(gbest)
            if all_markers[gidx][2] != winner[2] and gbest < best:
                firewall_case = True

        objects_info.append(
            {
                "coords": coords,
                "sector": sector,
                "winner_color": int(winner[2]),
                "winner_d2": int(best),
            }
        )

    assert len(set(winner_colors)) >= 2
    assert sector_has_choice
    assert firewall_case

    expected = np.full_like(g, BACKGROUND)
    expected[:] = g
    for info in objects_info:
        coords = info["coords"]
        expected[coords[:, 0], coords[:, 1]] = info["winner_color"]
    for r, c, _color, _sector in markers:
        expected[r, c] = BACKGROUND
    _draw_rays(expected, anchor_r, anchor_c)

    return expected, {
        "anchor": (anchor_r, anchor_c),
        "objects": objects_info,
        "markers": markers,
    }


def _build_candidate() -> np.ndarray | None:
    h = random.randint(15, 20)
    w = random.randint(15, 20)
    grid = np.full((h, w), BACKGROUND, dtype=int)

    anchor_r = random.randint(3, h - 4)
    anchor_c = random.randint(3, w - 4)
    grid[anchor_r, anchor_c] = BLACK

    object_count = random.randint(3, 5)
    active_count = random.randint(2, min(4, object_count))
    active_sectors = random.sample(SECTORS, active_count)
    focus_sector = random.choice(active_sectors)

    sector_counts = {sector: 1 for sector in active_sectors}
    extras = object_count - active_count
    if extras > 0 and sector_counts[focus_sector] == 1:
        sector_counts[focus_sector] += 1
        extras -= 1
    while extras > 0:
        sector_counts[random.choice(active_sectors)] += 1
        extras -= 1

    blocked = _ray_cells(grid.shape, anchor_r, anchor_c)
    spaced: set[tuple[int, int]] = set()
    objects_by_sector: dict[str, list[np.ndarray]] = defaultdict(list)
    focus_done = False
    for sector in active_sectors:
        for _idx in range(sector_counts[sector]):
            cells = _place_object(
                grid.shape,
                anchor_r,
                anchor_c,
                sector,
                blocked,
                spaced,
                require_near_ray=(sector == focus_sector and not focus_done),
            )
            if cells is None:
                return None
            coords = np.array(cells, dtype=int)
            grid[coords[:, 0], coords[:, 1]] = GREY
            objects_by_sector[sector].append(coords)
            blocked.update(cells)
            spaced.update(cells)
            if sector == focus_sector and not focus_done:
                focus_done = True

    focus_object = objects_by_sector[focus_sector][0]
    base_marker_specs: list[tuple[str, np.ndarray | None, int, int, bool]] = []
    for sector in active_sectors:
        target = focus_object if sector == focus_sector else random.choice(objects_by_sector[sector])
        if sector == focus_sector:
            base_marker_specs.append((sector, target, 12, 200, True))
        else:
            base_marker_specs.append((sector, target, 4, 160, True))
    base_marker_specs.append((focus_sector, focus_object, 20, 260, False))

    unused_colors = list(MARKER_COLORS)
    random.shuffle(unused_colors)
    marker_entries: list[tuple[int, int, int, str]] = []
    for sector, target, min_d2, max_d2, _primary in base_marker_specs:
        cell = _pick_marker(grid.shape, anchor_r, anchor_c, sector, blocked, spaced, target, min_d2, max_d2)
        if cell is None:
            return None
        color = unused_colors.pop()
        rr, cc = cell
        grid[rr, cc] = color
        marker_entries.append((rr, cc, color, sector))
        blocked.add(cell)
        spaced.add(cell)

    decoy_sectors = [sector for sector in SECTORS if sector != focus_sector]
    random.shuffle(decoy_sectors)
    focus_same_sector = [entry for entry in marker_entries if entry[3] == focus_sector]
    focus_winner_d2 = min(_min_sqdist(focus_object, (r, c)) for r, c, _color, _sector in focus_same_sector)
    decoy_cell = None
    decoy_sector = None
    for sector in decoy_sectors:
        cells = _marker_candidates(grid.shape, anchor_r, anchor_c, sector, blocked, spaced)
        random.shuffle(cells)
        better = [
            cell for cell in cells
            if 2 <= _min_sqdist(focus_object, cell) < focus_winner_d2
        ]
        if better:
            decoy_cell = min(better, key=lambda cell: _min_sqdist(focus_object, cell))
            decoy_sector = sector
            break
    if decoy_cell is None or not unused_colors:
        return None

    decoy_color = unused_colors.pop()
    dr, dc = decoy_cell
    grid[dr, dc] = decoy_color
    marker_entries.append((dr, dc, decoy_color, decoy_sector))
    blocked.add(decoy_cell)
    spaced.add(decoy_cell)

    if unused_colors and random.random() < 0.45:
        extra_sector = random.choice(active_sectors)
        extra_target = random.choice(objects_by_sector[extra_sector])
        extra_cell = _pick_marker(grid.shape, anchor_r, anchor_c, extra_sector, blocked, spaced, extra_target, 6, 180)
        if extra_cell is not None:
            extra_color = unused_colors.pop()
            er, ec = extra_cell
            grid[er, ec] = extra_color
            marker_entries.append((er, ec, extra_color, extra_sector))
            blocked.add(extra_cell)
            spaced.add(extra_cell)

    try:
        expected, _info = _independent_expected(grid)
    except AssertionError:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid diagonal-ray partition sample")
