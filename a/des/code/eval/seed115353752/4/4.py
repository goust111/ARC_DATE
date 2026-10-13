from __future__ import annotations

import random

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BRANCHES = ("row", "col", "diag")

GEN_RETRIES = 300
MODULE_RETRIES = 400
SHAPE_RETRIES = 200
MIN_MODULES = 2
MAX_MODULES = 4
PORTAL_SEP = 5
DIAG_SEP = 3


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append({"color": color, "coords": np.array(coords, dtype=int)})
    return comps


def _extract_structure(grid: np.ndarray) -> tuple[list[dict], dict[int, list[tuple[int, int]]], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _find_components(g)

    portal_pairs: dict[int, list[tuple[int, int]]] = {}
    regions: list[dict] = []
    for comp in comps:
        if int(comp["coords"].shape[0]) == 1:
            coord = tuple(map(int, comp["coords"][0]))
            portal_pairs.setdefault(int(comp["color"]), []).append(coord)
        else:
            regions.append(comp)

    assert portal_pairs, "expected at least one portal pair"
    for color, coords in portal_pairs.items():
        assert len(coords) == 2, f"portal color {color} must appear exactly twice"

    portal_colors = set(portal_pairs)
    region_colors = {int(comp["color"]) for comp in regions}
    assert portal_colors.isdisjoint(region_colors), "portal and region colors must be disjoint"

    portal_lookup = {
        (int(r), int(c)): int(color)
        for color, coords in portal_pairs.items()
        for r, c in coords
    }

    parsed_regions: list[dict] = []
    for comp in regions:
        adjacent: set[tuple[int, int]] = set()
        for r, c in comp["coords"]:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in portal_lookup:
                    adjacent.add((nr, nc))
        assert len(adjacent) == 1, "each region must border exactly one portal"
        portal = next(iter(adjacent))
        parsed_regions.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "portal": portal,
                "portal_color": int(portal_lookup[portal]),
            }
        )

    return parsed_regions, portal_pairs, portal_lookup


def _mirror_offsets(offsets: set[tuple[int, int]], mode: str) -> set[tuple[int, int]]:
    if mode == "row":
        return {(dr, -dc) for dr, dc in offsets}
    if mode == "col":
        return {(-dr, dc) for dr, dc in offsets}
    return {(-dr, -dc) for dr, dc in offsets}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport each monochrome region through its adjacent paired portal.

    For a region anchored at one portal, copy every pixel's offset from that
    portal to the paired portal. The copy is mirrored based on how the two
    portals are arranged: same row -> left/right mirror, same column -> up/down
    mirror, otherwise mirror both axes. Original regions are erased and portal
    pixels remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    regions, portal_pairs, portal_lookup = _extract_structure(g)

    out = g.copy()
    for region in regions:
        out[region["coords"][:, 0], region["coords"][:, 1]] = BG

    for region in regions:
        src = region["portal"]
        a, b = portal_pairs[region["portal_color"]]
        dst = b if src == a else a

        if src[0] == dst[0]:
            mode = "row"
        elif src[1] == dst[1]:
            mode = "col"
        else:
            mode = "diag"

        for r, c in region["coords"]:
            dr = int(r) - int(src[0])
            dc = int(c) - int(src[1])
            if mode == "row":
                dc = -dc
            elif mode == "col":
                dr = -dr
            else:
                dr = -dr
                dc = -dc

            rr = int(dst[0]) + int(dr)
            cc = int(dst[1]) + int(dc)
            assert 0 <= rr < h and 0 <= cc < w, "teleported region must stay in bounds"
            assert (rr, cc) not in portal_lookup, "teleported region may not overwrite a portal"
            assert int(out[rr, cc]) in (BG, int(region["color"])), "teleported regions may not conflict"
            out[rr, cc] = int(region["color"])

    return out


def _sample_modes(count: int) -> list[str]:
    assert MIN_MODULES <= count <= MAX_MODULES
    if count == 2:
        return random.sample(list(BRANCHES), 2)
    if count == 3:
        modes = list(BRANCHES)
        random.shuffle(modes)
        return modes
    modes = list(BRANCHES) + [random.choice(BRANCHES)]
    random.shuffle(modes)
    return modes


def _shape_anchor_bounds(offsets: set[tuple[int, int]], h: int, w: int) -> tuple[int, int, int, int]:
    rows = [dr for dr, _dc in offsets]
    cols = [dc for _dr, dc in offsets]
    row_lo = max(1, -min(rows))
    row_hi = min(h - 2, h - 1 - max(rows))
    col_lo = max(1, -min(cols))
    col_hi = min(w - 2, w - 1 - max(cols))
    return row_lo, row_hi, col_lo, col_hi


def _adjacent_count(cells: set[tuple[int, int]], portal: tuple[int, int]) -> int:
    pr, pc = portal
    return sum(abs(r - pr) + abs(c - pc) == 1 for r, c in cells)


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _grow_shape(mode: str) -> set[tuple[int, int]]:
    for _ in range(SHAPE_RETRIES):
        area = random.randint(3, 8)
        max_r = random.randint(2, 4)
        max_c = random.randint(2, 4)
        contact = random.choice(DIRS4)
        banned_neighbors = {step for step in DIRS4 if step != contact}

        cells = {contact}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if (nr, nc) in cells or (nr, nc) == (0, 0):
                        continue
                    if (nr, nc) in banned_neighbors:
                        continue
                    if abs(nr) > max_r or abs(nc) > max_c:
                        continue
                    frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(list(frontier)))

        if len(cells) != area:
            continue
        if _mirror_offsets(cells, mode) == cells:
            continue
        return cells

    raise RuntimeError("failed to generate a non-degenerate region shape")


def _sample_portals(
    mode: str,
    src_offsets: set[tuple[int, int]],
    dst_offsets: set[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    src_r0, src_r1, src_c0, src_c1 = _shape_anchor_bounds(src_offsets, h, w)
    dst_r0, dst_r1, dst_c0, dst_c1 = _shape_anchor_bounds(dst_offsets, h, w)

    if mode == "row":
        rows = list(range(max(src_r0, dst_r0), min(src_r1, dst_r1) + 1))
        pairs = [
            (sc, dc)
            for sc in range(src_c0, src_c1 + 1)
            for dc in range(dst_c0, dst_c1 + 1)
            if abs(sc - dc) >= PORTAL_SEP
        ]
        if not rows or not pairs:
            return None
        row = random.choice(rows)
        src_c, dst_c = random.choice(pairs)
        return (row, src_c), (row, dst_c)

    if mode == "col":
        cols = list(range(max(src_c0, dst_c0), min(src_c1, dst_c1) + 1))
        pairs = [
            (sr, dr)
            for sr in range(src_r0, src_r1 + 1)
            for dr in range(dst_r0, dst_r1 + 1)
            if abs(sr - dr) >= PORTAL_SEP
        ]
        if not cols or not pairs:
            return None
        col = random.choice(cols)
        src_r, dst_r = random.choice(pairs)
        return (src_r, col), (dst_r, col)

    row_pairs = [
        (sr, dr)
        for sr in range(src_r0, src_r1 + 1)
        for dr in range(dst_r0, dst_r1 + 1)
        if abs(sr - dr) >= DIAG_SEP
    ]
    col_pairs = [
        (sc, dc)
        for sc in range(src_c0, src_c1 + 1)
        for dc in range(dst_c0, dst_c1 + 1)
        if abs(sc - dc) >= DIAG_SEP
    ]
    if not row_pairs or not col_pairs:
        return None
    src_r, dst_r = random.choice(row_pairs)
    src_c, dst_c = random.choice(col_pairs)
    return (src_r, src_c), (dst_r, dst_c)


def _place_module(
    inp: np.ndarray,
    out: np.ndarray,
    blocked: set[tuple[int, int]],
    region_color: int,
    portal_color: int,
    mode: str,
) -> bool:
    h, w = inp.shape

    for _ in range(MODULE_RETRIES):
        src_offsets = _grow_shape(mode)
        dst_offsets = _mirror_offsets(src_offsets, mode)
        portals = _sample_portals(mode, src_offsets, dst_offsets, h, w)
        if portals is None:
            continue
        src_portal, dst_portal = portals

        src_cells = {(src_portal[0] + dr, src_portal[1] + dc) for dr, dc in src_offsets}
        dst_cells = {(dst_portal[0] + dr, dst_portal[1] + dc) for dr, dc in dst_offsets}
        module_cells = set(src_cells) | set(dst_cells) | {src_portal, dst_portal}

        if len(src_cells & dst_cells) != 0:
            continue
        if src_portal == dst_portal:
            continue
        if src_portal in src_cells or src_portal in dst_cells or dst_portal in src_cells or dst_portal in dst_cells:
            continue
        if module_cells & blocked:
            continue
        if _adjacent_count(src_cells, src_portal) != 1:
            continue
        if _adjacent_count(dst_cells, dst_portal) != 1:
            continue
        if _adjacent_count(src_cells, dst_portal) != 0:
            continue
        if _adjacent_count(dst_cells, src_portal) != 0:
            continue

        for r, c in src_cells:
            inp[r, c] = region_color
        for r, c in dst_cells:
            out[r, c] = region_color

        sr, sc = src_portal
        dr, dc = dst_portal
        inp[sr, sc] = portal_color
        inp[dr, dc] = portal_color
        out[sr, sc] = portal_color
        out[dr, dc] = portal_color

        blocked.update(_halo(module_cells, h, w))
        return True

    return False


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    module_count = random.randint(MIN_MODULES, MAX_MODULES)
    modes = _sample_modes(module_count)

    base = 12 + 3 * module_count
    h = random.randint(base, min(30, base + 6))
    w = random.randint(base, min(30, base + 6))

    palette = random.sample(list(range(1, 10)), 2 * module_count)
    region_colors = palette[:module_count]
    portal_colors = palette[module_count:]
    random.shuffle(region_colors)
    random.shuffle(portal_colors)

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    blocked: set[tuple[int, int]] = set()

    for region_color, portal_color, mode in zip(region_colors, portal_colors, modes):
        if not _place_module(inp, out, blocked, region_color, portal_color, mode):
            return None

    try:
        _extract_structure(inp)
        solved = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid portal puzzle input")
