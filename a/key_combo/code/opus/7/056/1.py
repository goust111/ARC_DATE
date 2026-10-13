from __future__ import annotations

import numpy as np


BLACK = 0
WALL = 5
SOURCE_COLORS = (6, 7)
SEED_COLORS = (1, 2, 3, 4, 8, 9)
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

MAX_GENERATION_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (16, 23)
GRID_WIDTH_RANGE = (16, 23)
INTERIOR_HEIGHT_RANGE = (4, 7)
INTERIOR_WIDTH_RANGE = (4, 9)
BORDER_MARGIN = 2
CAVITY_GAP = 2
SOURCE_CLEARANCE = 2


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _draw_wall(grid: np.ndarray, wall_box: tuple[int, int, int, int]) -> None:
    wr0, wc0, wr1, wc1 = wall_box
    grid[wr0, wc0 : wc1 + 1] = WALL
    grid[wr1, wc0 : wc1 + 1] = WALL
    grid[wr0 : wr1 + 1, wc0] = WALL
    grid[wr0 : wr1 + 1, wc1] = WALL


def _find_cavities(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    non_wall = grid != WALL
    comps = _components(non_wall)
    outside_mask = np.zeros_like(non_wall, dtype=bool)
    cavities: list[dict] = []

    for comp in comps:
        if _touches_border(comp, grid.shape):
            for r, c in comp:
                outside_mask[r, c] = True
            continue

        r0, c0, r1, c1 = _bbox(comp)
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            raise AssertionError("cavity interior must be a filled rectangle")

        colors = {int(grid[r, c]) for r, c in comp} - {BLACK}
        if len(colors) != 1:
            raise AssertionError("each cavity must contain exactly one seed color")
        seed_color = next(iter(colors))
        seed_coords = [(r, c) for r, c in comp if int(grid[r, c]) == seed_color]
        sr0, sc0, sr1, sc1 = _bbox(seed_coords)
        tile = grid[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()

        cavities.append(
            {
                "interior": (r0, c0, r1, c1),
                "wall": (r0 - 1, c0 - 1, r1 + 1, c1 + 1),
                "seed_color": seed_color,
                "seed_top_left": (sr0, sc0),
                "tile": tile,
            }
        )

    if not cavities:
        raise AssertionError("expected at least one cavity")
    return cavities, outside_mask


def _entry_points_for_source(
    shape: tuple[int, int], source: tuple[int, int], cavities: list[dict]
) -> dict[int, list[tuple[int, int]]]:
    h, w = shape
    entries: dict[int, list[tuple[int, int]]] = {}
    for dr, dc in DIAGONALS:
        active: set[int] = set()
        r, c = source
        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break
            for idx, cavity in enumerate(cavities):
                r0, c0, r1, c1 = cavity["interior"]
                inside = r0 <= r <= r1 and c0 <= c <= c1
                if inside and idx not in active:
                    entries.setdefault(idx, []).append((r, c))
                    active.add(idx)
                elif not inside and idx in active:
                    active.remove(idx)
    return entries


def _collect_anchor_entries(
    shape: tuple[int, int], sources: list[tuple[int, int]], cavities: list[dict]
) -> dict[int, list[tuple[int, int]]]:
    collected: dict[int, list[tuple[int, int]]] = {idx: [] for idx in range(len(cavities))}
    for source in sources:
        src_entries = _entry_points_for_source(shape, source, cavities)
        for idx, pts in src_entries.items():
            collected[idx].extend(pts)
    return collected


def _tile_cavity(
    out: np.ndarray,
    cavity: dict,
    anchor: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = cavity["interior"]
    tile = cavity["tile"]
    tile_h, tile_w = tile.shape
    ar, ac = anchor
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            out[r, c] = int(tile[(r - ar) % tile_h, (c - ac) % tile_w])


def main(grid: np.ndarray) -> np.ndarray:
    """Tile each grey-walled cavity using a phase anchored by the earliest diagonal ray entry.

    Input structure:
    - The grid has a black background, several grey rectangular walls, and 2-3 enclosed rectangular cavities.
    - Each cavity interior is black except for one small single-color seed tile.
    - Outside the walls, 1-2 isolated colored pixels sit in the background and imply diagonal rays.

    Rule:
    - Treat every outside isolated pixel as emitting rays in all four diagonal directions, continuing through walls to the grid edge.
    - For each cavity, find every ray entry into the cavity interior and choose the lexicographically smallest first interior entry cell as the anchor.
    - Use that anchor as the tile's top-left phase origin, periodically repeat the cavity's seed tile across the full cavity, crop at the cavity edges, keep walls unchanged, and clear everything outside the cavities back to black.

    Key insight:
    - The visible seed says what pattern repeats, but the ray entry point says how that pattern is phase-aligned inside each cavity.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    cavities, outside_mask = _find_cavities(g)
    source_coords = [
        (int(r), int(c))
        for r, c in np.argwhere((g != BLACK) & (g != WALL) & outside_mask)
    ]
    if not source_coords:
        raise AssertionError("expected at least one outside source pixel")

    entries = _collect_anchor_entries(g.shape, source_coords, cavities)

    out = np.full_like(g, BLACK)
    out[g == WALL] = WALL
    for idx, cavity in enumerate(cavities):
        if not entries[idx]:
            raise AssertionError("every cavity must be hit by at least one ray")
        anchor = min(entries[idx])
        _tile_cavity(out, cavity, anchor)
    return out


def _boxes_far_apart(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _random_tile(rng: np.random.Generator, color: int) -> np.ndarray:
    tile_h, tile_w = ((2, 2) if int(rng.integers(0, 2)) == 0 else (2, 3))
    while True:
        density = rng.uniform(0.45, 0.85)
        mask = rng.random((tile_h, tile_w)) < density
        if not mask.any() or mask.all():
            continue
        if not mask[0].any() or not mask[-1].any():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        tile = np.full((tile_h, tile_w), BLACK, dtype=int)
        tile[mask] = color
        return tile


def _place_cavities(rng: np.random.Generator, h: int, w: int, count: int) -> list[dict] | None:
    cavities: list[dict] = []
    dims: list[tuple[int, int]] = []
    h_hi = INTERIOR_HEIGHT_RANGE[1]
    w_hi = INTERIOR_WIDTH_RANGE[1]
    if count == 3:
        h_hi -= 1
        w_hi -= 1
    for _ in range(count):
        dims.append(
            (
                int(rng.integers(INTERIOR_HEIGHT_RANGE[0], h_hi)),
                int(rng.integers(INTERIOR_WIDTH_RANGE[0], w_hi)),
            )
        )
    dims.sort(key=lambda x: x[0] * x[1], reverse=True)

    for interior_h, interior_w in dims:
        candidates: list[tuple[int, int]] = []
        r_min = BORDER_MARGIN + 1
        c_min = BORDER_MARGIN + 1
        r_max = h - interior_h - BORDER_MARGIN - 1
        c_max = w - interior_w - BORDER_MARGIN - 1
        if r_min > r_max or c_min > c_max:
            return None
        for r0 in range(r_min, r_max + 1):
            for c0 in range(c_min, c_max + 1):
                wall_box = (r0 - 1, c0 - 1, r0 + interior_h, c0 + interior_w)
                if all(_boxes_far_apart(wall_box, cav["wall"], CAVITY_GAP) for cav in cavities):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = candidates[int(rng.integers(0, len(candidates)))]
        cavities.append(
            {
                "interior": (r0, c0, r0 + interior_h - 1, c0 + interior_w - 1),
                "wall": (r0 - 1, c0 - 1, r0 + interior_h, c0 + interior_w),
            }
        )
    return cavities


def _add_seeds(rng: np.random.Generator, cavities: list[dict]) -> None:
    colors = list(rng.choice(np.array(SEED_COLORS, dtype=int), size=len(cavities), replace=False))
    for cavity, color in zip(cavities, colors):
        tile = _random_tile(rng, int(color))
        tile_h, tile_w = tile.shape
        r0, c0, r1, c1 = cavity["interior"]
        sr0 = int(rng.integers(r0, r1 - tile_h + 2))
        sc0 = int(rng.integers(c0, c1 - tile_w + 2))
        cavity["tile"] = tile
        cavity["seed_top_left"] = (sr0, sc0)
        cavity["seed_color"] = int(color)


def _render_input_grid(
    shape: tuple[int, int], cavities: list[dict], sources: list[tuple[tuple[int, int], int]]
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for cavity in cavities:
        _draw_wall(grid, cavity["wall"])
        sr0, sc0 = cavity["seed_top_left"]
        tile = cavity["tile"]
        th, tw = tile.shape
        grid[sr0 : sr0 + th, sc0 : sc0 + tw] = tile
    for (r, c), color in sources:
        grid[r, c] = int(color)
    return grid


def _render_expected_output(
    shape: tuple[int, int], cavities: list[dict], source_coords: list[tuple[int, int]]
) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    out = np.full(shape, BLACK, dtype=int)
    for cavity in cavities:
        _draw_wall(out, cavity["wall"])
    entries = _collect_anchor_entries(shape, source_coords, cavities)
    for idx, cavity in enumerate(cavities):
        if not entries[idx]:
            raise AssertionError("cavity missing ray coverage")
        _tile_cavity(out, cavity, min(entries[idx]))
    return out, entries


def _source_allowed(
    coord: tuple[int, int], cavities: list[dict], chosen: list[tuple[int, int]]
) -> bool:
    r, c = coord
    for cavity in cavities:
        wr0, wc0, wr1, wc1 = cavity["wall"]
        if wr0 - SOURCE_CLEARANCE <= r <= wr1 + SOURCE_CLEARANCE and wc0 - SOURCE_CLEARANCE <= c <= wc1 + SOURCE_CLEARANCE:
            return False
    for rr, cc in chosen:
        if max(abs(r - rr), abs(c - cc)) <= SOURCE_CLEARANCE:
            return False
    return True


def _candidate_sources(
    grid: np.ndarray, cavities: list[dict]
) -> list[dict]:
    infos: list[dict] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK:
                continue
            coord = (r, c)
            if not _source_allowed(coord, cavities, []):
                continue
            entries = _entry_points_for_source(grid.shape, coord, cavities)
            covered = {idx for idx, pts in entries.items() if pts}
            if covered:
                infos.append({"coord": coord, "entries": entries, "covered": covered})
    return infos


def _merge_entries(a: dict[int, list[tuple[int, int]]], b: dict[int, list[tuple[int, int]]], n: int) -> dict[int, list[tuple[int, int]]]:
    merged = {idx: list(a.get(idx, [])) for idx in range(n)}
    for idx in range(n):
        merged[idx].extend(b.get(idx, []))
    return merged


def _choose_sources(
    rng: np.random.Generator, shape: tuple[int, int], cavities: list[dict], infos: list[dict]
) -> tuple[list[tuple[tuple[int, int], int]], dict[int, list[tuple[int, int]]]] | None:
    target = set(range(len(cavities)))

    singles = []
    for info in infos:
        if info["covered"] == target:
            multi = any(len(info["entries"].get(idx, [])) >= 2 for idx in target)
            singles.append((info, multi))

    pairs = []
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            a = infos[i]
            b = infos[j]
            if not _source_allowed(a["coord"], cavities, [b["coord"]]):
                continue
            if not _source_allowed(b["coord"], cavities, [a["coord"]]):
                continue
            covered = a["covered"] | b["covered"]
            if covered != target:
                continue
            entries = _merge_entries(a["entries"], b["entries"], len(cavities))
            multi = any(len(entries[idx]) >= 2 for idx in target)
            pairs.append(((a, b), entries, multi))

    use_pair = bool(pairs) and (not singles or rng.random() < 0.65)
    if use_pair:
        preferred = [item for item in pairs if item[2]]
        pool = preferred if preferred else pairs
        (a, b), entries, _ = pool[int(rng.integers(0, len(pool)))]
        colors = list(SOURCE_COLORS)
        if int(rng.integers(0, 2)) == 1:
            colors.reverse()
        return [((a["coord"][0], a["coord"][1]), colors[0]), ((b["coord"][0], b["coord"][1]), colors[1])], entries

    if singles:
        preferred = [item for item in singles if item[1]]
        pool = preferred if preferred else singles
        info, _ = pool[int(rng.integers(0, len(pool)))]
        return [((info["coord"][0], info["coord"][1]), SOURCE_COLORS[0])], {
            idx: list(info["entries"].get(idx, [])) for idx in range(len(cavities))
        }

    if pairs:
        (a, b), entries, _ = pairs[int(rng.integers(0, len(pairs)))]
        return [((a["coord"][0], a["coord"][1]), SOURCE_COLORS[0]), ((b["coord"][0], b["coord"][1]), SOURCE_COLORS[1])], entries

    return None


def _has_phase_shift(cavities: list[dict], entries: dict[int, list[tuple[int, int]]]) -> bool:
    for idx, cavity in enumerate(cavities):
        anchor = min(entries[idx])
        sr0, sc0 = cavity["seed_top_left"]
        tile_h, tile_w = cavity["tile"].shape
        if ((anchor[0] - sr0) % tile_h, (anchor[1] - sc0) % tile_w) != (0, 0):
            return True
    return False


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    cavity_count = 3 if rng.random() < 0.4 else 2
    if cavity_count == 3:
        h = int(rng.integers(max(GRID_HEIGHT_RANGE[0], 18), GRID_HEIGHT_RANGE[1]))
        w = int(rng.integers(max(GRID_WIDTH_RANGE[0], 18), GRID_WIDTH_RANGE[1]))
    else:
        h = int(rng.integers(*GRID_HEIGHT_RANGE))
        w = int(rng.integers(*GRID_WIDTH_RANGE))

    cavities = _place_cavities(rng, h, w, cavity_count)
    if cavities is None:
        return None
    _add_seeds(rng, cavities)

    empty_grid = _render_input_grid((h, w), cavities, [])
    infos = _candidate_sources(empty_grid, cavities)
    if not infos:
        return None

    chosen = _choose_sources(rng, (h, w), cavities, infos)
    if chosen is None:
        return None
    sources, entries = chosen

    if not _has_phase_shift(cavities, entries):
        return None

    inp = _render_input_grid((h, w), cavities, sources)
    expected, _ = _render_expected_output((h, w), cavities, [coord for coord, _ in sources])

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
