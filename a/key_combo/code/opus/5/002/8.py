from __future__ import annotations

import itertools
import random

import numpy as np


BACKGROUND = 0
HUB_COLOR = 5
SEED_COLORS = (1, 2, 3, 4)
SECTORS = ("up", "right", "down", "left")

# Minimal helpers adapted from arc_utils.py so this file stays SourceFileLoader-safe.
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _find_hub(grid: np.ndarray) -> tuple[int, int]:
    candidates = []
    for comp in _components_of_color(grid, HUB_COLOR):
        if len(comp) != 9:
            continue
        r0, c0, r1, c1 = bbox_coords(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == HUB_COLOR):
            candidates.append((r0 + 1, c0 + 1))
    assert len(candidates) == 1, "expected exactly one isolated 3x3 gray hub"
    return candidates[0]


def _is_hub_cell(r: int, c: int, hr: int, hc: int) -> bool:
    return abs(r - hr) <= 1 and abs(c - hc) <= 1


def _sector_of(r: int, c: int, hr: int, hc: int) -> str | None:
    if _is_hub_cell(r, c, hr, hc):
        return None
    dr = r - hr
    dc = c - hc
    if abs(dr) == abs(dc):
        return None
    if r < hr and abs(dc) < hr - r:
        return "up"
    if c > hc and abs(dr) < c - hc:
        return "right"
    if r > hr and abs(dc) < r - hr:
        return "down"
    if c < hc and abs(dr) < hc - c:
        return "left"
    return None


def _sector_margin(r: int, c: int, hr: int, hc: int) -> int:
    sector = _sector_of(r, c, hr, hc)
    if sector == "up":
        return (hr - r) - abs(c - hc)
    if sector == "right":
        return (c - hc) - abs(r - hr)
    if sector == "down":
        return (r - hr) - abs(c - hc)
    if sector == "left":
        return (hc - c) - abs(r - hr)
    return -999


def _draw_rays(out: np.ndarray, hr: int, hc: int) -> None:
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        r, c = hr, hc
        while in_bounds(out.shape, r, c):
            out[r, c] = HUB_COLOR
            r += dr
            c += dc


def _extract_seed_infos(grid: np.ndarray, hr: int, hc: int) -> list[dict]:
    infos: list[dict] = []
    for color in SEED_COLORS:
        comps = _components_of_color(grid, color)
        assert len(comps) == 1, "expected one seed region per seed color"
        coords = comps[0]
        sectors = {_sector_of(r, c, hr, hc) for r, c in coords}
        assert len(sectors) == 1 and None not in sectors, "seed must lie fully inside one sector"
        infos.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": bbox_coords(coords),
                "size": len(coords),
                "sector": next(iter(sectors)),
            }
        )
    return infos


def _hub_winner(filled_counts: dict[int, int], seed_sizes: dict[int, int]) -> int:
    best_fill = max(filled_counts.values())
    tied = [color for color in SEED_COLORS if int(filled_counts[color]) == int(best_fill)]
    if len(tied) == 1:
        return tied[0]
    smallest = min(seed_sizes[color] for color in tied)
    narrowed = [color for color in tied if int(seed_sizes[color]) == int(smallest)]
    assert len(narrowed) == 1, "input should not leave the hub color ambiguous"
    return narrowed[0]


def _solve_with_details(grid: np.ndarray, *, reverse_sector: str | None = None) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    hr, hc = _find_hub(g)
    seed_infos = _extract_seed_infos(g, hr, hc)

    out = g.copy()
    _draw_rays(out, hr, hc)

    sector_sizes = {sector: 0 for sector in SECTORS}
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            sector = _sector_of(r, c, hr, hc)
            if sector is not None:
                sector_sizes[sector] += 1

    by_sector = {sector: [] for sector in SECTORS}
    for info in seed_infos:
        by_sector[info["sector"]].append(info)

    seed_sizes = {info["color"]: info["size"] for info in seed_infos}
    filled_counts = {color: 0 for color in SEED_COLORS}
    fill_by_color: dict[int, set[tuple[int, int]]] = {color: set() for color in SEED_COLORS}

    for sector in SECTORS:
        ordered = sorted(by_sector[sector], key=lambda info: (info["bbox"][0], info["bbox"][1]))
        if reverse_sector == sector:
            ordered = list(reversed(ordered))
        for info in ordered:
            color = info["color"]
            queue = list(info["coords"])
            seen = set(info["coords"])
            while queue:
                r, c = queue.pop()
                for nr, nc in neighbors4(r, c):
                    if (nr, nc) in seen:
                        continue
                    if not in_bounds(out.shape, nr, nc):
                        continue
                    if _sector_of(nr, nc, hr, hc) != sector:
                        continue
                    if int(out[nr, nc]) != BACKGROUND:
                        continue
                    out[nr, nc] = color
                    filled_counts[color] += 1
                    fill_by_color[color].add((nr, nc))
                    seen.add((nr, nc))
                    queue.append((nr, nc))

    winner = _hub_winner(filled_counts, seed_sizes)
    out[hr, hc] = winner
    details = {
        "hub": (hr, hc),
        "seed_infos": seed_infos,
        "sector_sizes": sector_sizes,
        "filled_counts": filled_counts,
        "seed_sizes": seed_sizes,
        "winner": winner,
        "fill_by_color": fill_by_color,
    }
    return out, details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot four diagonal rays from the gray hub, flood each ray-bounded sector from its seeds, and recolor the hub center by the largest flood.

    Input structure:
    - A black background contains one isolated solid 3x3 gray hub square.
    - Outside the hub are four small 4-connected seed regions in colors 1, 2, 3, and 4.
    - Each seed lies strictly inside one of the four sectors that will be formed by the hub-centered diagonal rays.

    Rule:
    - Find the hub center and extend gray rays from it along the four diagonal directions until the grid border.
    - Treat hub cells and ray cells as sector boundaries; every other cell belongs to the up, right, down, or left sector using the puzzle's strict coordinate inequalities.
    - In each sector, flood only black cells from that sector's seeds in top-left bbox order, without crossing hub cells, ray cells, prior fills, or the grid edge.
    - Count only the newly flooded pixels per seed color, then recolor the hub center with the color having the largest flood total; if those totals tie, choose the tied color whose original seed region is smaller.

    Key insight:
    - The hub first sends information outward by carving the grid into four ray-defined sectors, then the flood sizes inside those sectors send information back inward to decide the hub's final center color.
    """
    return _solve_with_details(grid)[0]


SHAPES_BY_SIZE = {
    2: [
        {(0, 0), (0, 1)},
    ],
    3: [
        {(0, 0), (0, 1), (0, 2)},
        {(0, 0), (1, 0), (1, 1)},
    ],
    4: [
        {(0, 0), (0, 1), (0, 2), (0, 3)},
        {(0, 0), (0, 1), (1, 0), (1, 1)},
        {(0, 0), (1, 0), (2, 0), (2, 1)},
        {(0, 1), (1, 0), (1, 1), (1, 2)},
        {(0, 1), (0, 2), (1, 0), (1, 1)},
    ],
    5: [
        {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)},
        {(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)},
        {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
        {(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)},
        {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)},
    ],
}
SECTOR_PERMS = list(itertools.permutations(SECTORS))


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_variants(size: int) -> list[tuple[tuple[int, int], ...]]:
    variants = set()
    for base in SHAPES_BY_SIZE[size]:
        current = set(base)
        for _ in range(4):
            current = {(c, -r) for r, c in current}
            for flipped in (current, {(-r, c) for r, c in current}, {(r, -c) for r, c in current}):
                variants.add(_normalize(flipped))
    return [tuple(shape) for shape in variants]


ALL_SHAPE_VARIANTS = {size: _shape_variants(size) for size in SHAPES_BY_SIZE}
_GEN_COUNTER = 0
_GEOMETRY_CACHE: dict[str, list[tuple[int, int, int, int, dict, dict, dict]]] = {}


def _geometry_ok(mode: str, sector_sizes: dict[str, int]) -> bool:
    largest_sector = max(SECTORS, key=lambda sector: sector_sizes[sector])
    if mode == "size":
        return any(
            sector != largest_sector and sector_sizes[largest_sector] - sector_sizes[sector] <= 2
            for sector in SECTORS
        )
    if mode == "tie":
        sizes = [sector_sizes[sector] for sector in SECTORS]
        return any(1 <= abs(a - b) <= 3 for a, b in itertools.combinations(sizes, 2))
    if mode == "double":
        return True
    return False


def _buffered(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set(coords)
    for r, c in coords:
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _sector_cells(h: int, w: int, hr: int, hc: int) -> dict[str, set[tuple[int, int]]]:
    cells = {sector: set() for sector in SECTORS}
    for r in range(h):
        for c in range(w):
            sector = _sector_of(r, c, hr, hc)
            if sector is not None:
                cells[sector].add((r, c))
    return cells


def _safe_sector_cells(h: int, w: int, hr: int, hc: int) -> dict[str, set[tuple[int, int]]]:
    safe = {sector: set() for sector in SECTORS}
    for r in range(h):
        for c in range(w):
            sector = _sector_of(r, c, hr, hc)
            if sector is None:
                continue
            if _sector_margin(r, c, hr, hc) < 2:
                continue
            if r in (0, h - 1) or c in (0, w - 1):
                continue
            safe[sector].add((r, c))
    return safe


def _pick_hub_and_sizes(mode: str):
    if mode not in _GEOMETRY_CACHE:
        geometries: list[tuple[int, int, int, int, dict, dict, dict]] = []
        for h in [19, 21, 23, 25, 27]:
            for w in [19, 21, 23, 25, 27]:
                base_r = h // 2
                base_c = w // 2
                offsets_r = [d for d in (-4, -3, -2, -1, 1, 2, 3, 4) if 5 <= base_r + d <= h - 6]
                offsets_c = [d for d in (-4, -3, -2, -1, 1, 2, 3, 4) if 5 <= base_c + d <= w - 6]
                for dr in offsets_r:
                    for dc in offsets_c:
                        hr = base_r + dr
                        hc = base_c + dc
                        sector_cells = _sector_cells(h, w, hr, hc)
                        safe_cells = _safe_sector_cells(h, w, hr, hc)
                        sector_sizes = {sector: len(cells) for sector, cells in sector_cells.items()}
                        if min(sector_sizes.values()) < 18 or min(len(cells) for cells in safe_cells.values()) < 8:
                            continue
                        largest_sector = max(SECTORS, key=lambda sector: sector_sizes[sector])
                        if sum(sector_sizes[s] == sector_sizes[largest_sector] for s in SECTORS) != 1 and mode != "double":
                            continue
                        if not _geometry_ok(mode, sector_sizes):
                            continue
                        geometries.append((h, w, hr, hc, sector_cells, safe_cells, sector_sizes))
        _GEOMETRY_CACHE[mode] = geometries

    geometries = _GEOMETRY_CACHE[mode]
    if geometries:
        for _ in range(40):
            h, w, hr, hc, sector_cells, safe_cells, sector_sizes = random.choice(geometries)
            plan = _choose_plan(mode, sector_sizes)
            if plan is not None:
                return h, w, hr, hc, sector_cells, safe_cells, sector_sizes, plan
    raise RuntimeError(f"failed to sample a valid hub for mode {mode}")


def _choose_plan(mode: str, sector_sizes: dict[str, int]) -> dict | None:
    colors = list(SEED_COLORS)
    if mode == "size":
        perms = SECTOR_PERMS[:]
        random.shuffle(perms)
        for perm in perms:
            color_sector = {color: perm[i] for i, color in enumerate(colors)}
            for sizes_tuple in itertools.product(range(2, 6), repeat=4):
                seed_sizes = {color: sizes_tuple[i] for i, color in enumerate(colors)}
                fills = {color: sector_sizes[color_sector[color]] - seed_sizes[color] for color in colors}
                best = max(fills.values())
                if sum(v == best for v in fills.values()) != 1:
                    continue
                winner = max(colors, key=lambda color: fills[color])
                if color_sector[winner] == max(SECTORS, key=lambda sector: sector_sizes[sector]):
                    continue
                return {"color_sector": color_sector, "seed_sizes": seed_sizes}
        return None

    if mode == "tie":
        perms = SECTOR_PERMS[:]
        random.shuffle(perms)
        for perm in perms:
            color_sector = {color: perm[i] for i, color in enumerate(colors)}
            for sizes_tuple in itertools.product(range(2, 6), repeat=4):
                seed_sizes = {color: sizes_tuple[i] for i, color in enumerate(colors)}
                fills = {color: sector_sizes[color_sector[color]] - seed_sizes[color] for color in colors}
                best = max(fills.values())
                tied = [color for color in colors if fills[color] == best]
                if len(tied) != 2:
                    continue
                if len({seed_sizes[color] for color in tied}) != 2:
                    continue
                if any(fills[color] >= best for color in colors if color not in tied):
                    continue
                return {"color_sector": color_sector, "seed_sizes": seed_sizes}
        return None

    if mode == "double":
        sectors = list(SECTORS)
        random.shuffle(sectors)
        sector_order = sorted(sectors, key=lambda sector: sector_sizes[sector], reverse=True)
        for double_sector in sector_order:
            for empty_sector in sectors:
                if empty_sector == double_sector:
                    continue
                singles = [sector for sector in sectors if sector not in {double_sector, empty_sector}]
                random.shuffle(colors)
                double_colors = colors[:2]
                single_colors = colors[2:]
                color_sector = {
                    double_colors[0]: double_sector,
                    double_colors[1]: double_sector,
                    single_colors[0]: singles[0],
                    single_colors[1]: singles[1],
                }
                for sizes_tuple in itertools.product(range(2, 6), repeat=4):
                    seed_sizes = {color: sizes_tuple[i] for i, color in enumerate(colors)}
                    double_fill = sector_sizes[double_sector] - seed_sizes[double_colors[0]] - seed_sizes[double_colors[1]]
                    single_fills = [sector_sizes[color_sector[color]] - seed_sizes[color] for color in single_colors]
                    if double_fill <= max(single_fills):
                        continue
                    return {
                        "color_sector": color_sector,
                        "seed_sizes": seed_sizes,
                        "double_sector": double_sector,
                        "double_colors": tuple(double_colors),
                    }
        return None

    raise ValueError(f"unknown mode {mode}")


def _place_shape(
    h: int,
    w: int,
    allowed: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    size: int,
) -> set[tuple[int, int]] | None:
    variants = ALL_SHAPE_VARIANTS[size][:]
    random.shuffle(variants)
    for shape in variants:
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        candidates = []
        for r0 in range(h - sh + 1):
            for c0 in range(w - sw + 1):
                coords = {(r0 + r, c0 + c) for r, c in shape}
                if not coords.issubset(allowed):
                    continue
                if not coords.isdisjoint(forbidden):
                    continue
                candidates.append(coords)
        if candidates:
            return random.choice(candidates)
    return None


def _build_input(mode: str) -> np.ndarray:
    h, w, hr, hc, _sector_cells_map, safe_cells, _sector_sizes, plan = _pick_hub_and_sizes(mode)
    grid = np.zeros((h, w), dtype=int)
    grid[hr - 1 : hr + 2, hc - 1 : hc + 2] = HUB_COLOR

    forbidden = set()
    for r in range(hr - 2, hr + 3):
        for c in range(hc - 2, hc + 3):
            if in_bounds(grid.shape, r, c):
                forbidden.add((r, c))

    placements: dict[int, set[tuple[int, int]]] = {}
    colors = list(SEED_COLORS)
    colors.sort(key=lambda color: plan["seed_sizes"][color], reverse=True)
    for color in colors:
        sector = plan["color_sector"][color]
        coords = _place_shape(h, w, safe_cells[sector], forbidden, plan["seed_sizes"][color])
        if coords is None:
            raise RuntimeError("placement failed")
        placements[color] = coords
        forbidden |= _buffered(coords, grid.shape)

    for color, coords in placements.items():
        for r, c in coords:
            grid[r, c] = color
    return grid


def _validate_instance(inp: np.ndarray, mode: str) -> bool:
    out, details = _solve_with_details(inp)
    hr, hc = details["hub"]
    if (hr, hc) == (inp.shape[0] // 2, inp.shape[1] // 2):
        return False
    if np.array_equal(inp, out):
        return False
    if int(out[hr, hc]) == HUB_COLOR:
        return False

    sector_members = {sector: [] for sector in SECTORS}
    for info in details["seed_infos"]:
        sector_members[info["sector"]].append(info)
    counts = sorted(len(v) for v in sector_members.values())

    if mode == "size":
        if counts != [1, 1, 1, 1]:
            return False
        raw_largest_sector = max(SECTORS, key=lambda sector: details["sector_sizes"][sector])
        winner_sector = next(info["sector"] for info in details["seed_infos"] if info["color"] == details["winner"])
        if winner_sector == raw_largest_sector:
            return False
        return True

    if mode == "double":
        if counts != [0, 1, 1, 2]:
            return False
        double_sector = next(sector for sector, infos in sector_members.items() if len(infos) == 2)
        reversed_out, reversed_details = _solve_with_details(inp, reverse_sector=double_sector)
        if int(reversed_out[hr, hc]) == int(out[hr, hc]):
            return False
        if reversed_details["winner"] == details["winner"]:
            return False
        return True

    if mode == "tie":
        if counts != [1, 1, 1, 1]:
            return False
        best = max(details["filled_counts"].values())
        tied = [color for color in SEED_COLORS if details["filled_counts"][color] == best]
        if len(tied) != 2:
            return False
        tied_sizes = {color: details["seed_sizes"][color] for color in tied}
        if len(set(tied_sizes.values())) != 2:
            return False
        smaller = min(tied, key=lambda color: tied_sizes[color])
        if details["winner"] != smaller:
            return False
        return True

    return False


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    modes = ("size", "double", "tie")
    mode = modes[_GEN_COUNTER % len(modes)]
    for _ in range(400):
        try:
            grid = _build_input(mode)
        except RuntimeError:
            continue
        if _validate_instance(grid, mode):
            _GEN_COUNTER += 1
            return grid
    raise RuntimeError(f"failed to generate a valid input for mode {mode}")
