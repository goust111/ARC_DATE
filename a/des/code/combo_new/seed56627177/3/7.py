from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
GREY = 5
EMITTER_COLORS = (7, 8, 9)
PORTAL_COLORS = (1, 2, 3, 4, 6)
DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _center(cells: tuple[tuple[int, int], ...]) -> tuple[float, float]:
    return (
        sum(r for r, _ in cells) / len(cells),
        sum(c for _, c in cells) / len(cells),
    )


def _line_to_edge(start: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = start
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _portal_orientation(cells: tuple[tuple[int, int], ...]) -> str:
    rows = {r for r, _ in cells}
    return "H" if len(rows) == 1 else "V"


def _portal_exit_direction(
    src_cells: tuple[tuple[int, int], ...],
    dst_cells: tuple[tuple[int, int], ...],
    orient: str,
    h: int,
    w: int,
    *,
    strict: bool,
) -> tuple[int, int] | None:
    src_r, src_c = _center(src_cells)
    dst_r, dst_c = _center(dst_cells)
    if orient == "H":
        if dst_c > src_c:
            return (0, 1)
        if dst_c < src_c:
            return (0, -1)
        if strict:
            return None
        left_space = min(c for _, c in dst_cells)
        right_space = (w - 1) - max(c for _, c in dst_cells)
        return (0, 1) if right_space >= left_space else (0, -1)

    if dst_r > src_r:
        return (1, 0)
    if dst_r < src_r:
        return (-1, 0)
    if strict:
        return None
    up_space = min(r for r, _ in dst_cells)
    down_space = (h - 1) - max(r for r, _ in dst_cells)
    return (1, 0) if down_space >= up_space else (-1, 0)


def _portal_exit_cell(
    cells: tuple[tuple[int, int], ...], orient: str, direction: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(cells)
    if orient == "H":
        row = r0
        return (row, c1 + 1) if direction == (0, 1) else (row, c0 - 1)
    col = c0
    return (r1 + 1, col) if direction == (1, 0) else (r0 - 1, col)


def _portal_variants_for_hit(
    hit: tuple[int, int], direction: tuple[int, int], h: int, w: int
) -> list[dict]:
    r, c = hit
    dr, dc = direction
    variants: list[dict] = []

    def add(orient: str, cells: tuple[tuple[int, int], ...]) -> None:
        if all(0 <= rr < h and 0 <= cc < w for rr, cc in cells):
            variants.append({"orient": orient, "cells": cells})

    if dr == 0:
        if dc == 1:
            add("H", ((r, c), (r, c + 1)))
        else:
            add("H", ((r, c - 1), (r, c)))
        add("V", ((r, c), (r + 1, c)))
        add("V", ((r - 1, c), (r, c)))
    else:
        if dr == 1:
            add("V", ((r, c), (r + 1, c)))
        else:
            add("V", ((r - 1, c), (r, c)))
        add("H", ((r, c), (r, c + 1)))
        add("H", ((r, c - 1), (r, c)))

    uniq: dict[tuple[tuple[int, int], ...], dict] = {}
    for item in variants:
        key = tuple(sorted(item["cells"]))
        uniq[key] = {"orient": item["orient"], "cells": key}
    return list(uniq.values())


def _object_placeable(
    cells: tuple[tuple[int, int], ...],
    obj_cells: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    own = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in obj_cells or (r, c) in path_cells:
            return False
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in obj_cells and (nr, nc) not in own:
                return False
    return True


def _segment_is_free(
    cells: list[tuple[int, int]], obj_cells: set[tuple[int, int]], path_cells: set[tuple[int, int]]
) -> bool:
    return all(cell not in obj_cells and cell not in path_cells for cell in cells)


def _random_portal_candidates(
    obj_cells: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    h: int,
    w: int,
    rng: random.Random,
    *,
    want: int = 40,
    tries: int = 200,
) -> list[dict]:
    candidates: list[dict] = []
    seen: set[tuple[str, tuple[tuple[int, int], ...]]] = set()
    for _ in range(tries):
        orient = rng.choice(("H", "V"))
        if orient == "H":
            r = rng.randrange(h)
            c = rng.randrange(w - 1)
            cells = ((r, c), (r, c + 1))
        else:
            r = rng.randrange(h - 1)
            c = rng.randrange(w)
            cells = ((r, c), (r + 1, c))
        key = (orient, cells)
        if key in seen:
            continue
        seen.add(key)
        if _object_placeable(cells, obj_cells, path_cells, h, w):
            candidates.append({"orient": orient, "cells": cells})
            if len(candidates) >= want:
                break
    rng.shuffle(candidates)
    return candidates


def _emitter_direction(emitter: dict, h: int, w: int) -> tuple[int, int]:
    cells = emitter["cells"]
    r0, c0, r1, c1 = _bbox(list(cells))
    if r0 == r1:
        left_space = c0
        right_space = (w - 1) - c1
        return (0, 1) if right_space >= left_space else (0, -1)
    up_space = r0
    down_space = (h - 1) - r1
    return (1, 0) if down_space >= up_space else (-1, 0)


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], dict[int, int], dict[tuple[int, int], int]]:
    comps = _components(grid != BG)
    emitters: list[dict] = []
    portals: list[dict] = []
    endpoint_colors: set[int] = set()

    for comp in comps:
        cells = tuple(sorted(comp))
        colors = [int(grid[r, c]) for r, c in cells]
        counts = Counter(colors)
        r0, c0, r1, c1 = _bbox(comp)
        if len(cells) == 3 and (r0 == r1 or c0 == c1):
            if sorted(counts.values()) != [1, 2]:
                raise AssertionError("invalid emitter color pattern")
            repeated = max(counts, key=lambda color: counts[color])
            center_color = min(counts, key=lambda color: counts[color])
            ordered = sorted(cells, key=(lambda cell: cell[1]) if r0 == r1 else (lambda cell: cell[0]))
            center = ordered[1]
            if int(grid[center]) != int(center_color):
                raise AssertionError("emitter center must be the colored middle pixel")
            if int(grid[ordered[0]]) != int(repeated) or int(grid[ordered[2]]) != int(repeated):
                raise AssertionError("emitter endpoints must share the grey color")
            endpoint_colors.add(int(repeated))
            emitters.append(
                {
                    "cells": cells,
                    "center": center,
                    "center_color": int(center_color),
                    "endpoint_color": int(repeated),
                }
            )
        elif len(cells) == 2 and (r0 == r1 or c0 == c1) and len(counts) == 1:
            portals.append({"cells": cells, "color": int(colors[0]), "orient": _portal_orientation(cells)})
        else:
            raise AssertionError("unexpected foreground object")

    assert emitters, "expected at least one emitter"
    assert len(endpoint_colors) == 1, "all emitters must share the grey endpoint color"

    by_color: dict[int, list[int]] = defaultdict(list)
    for idx, portal in enumerate(portals):
        by_color[int(portal["color"])].append(idx)

    mate: dict[int, int] = {}
    for idxs in by_color.values():
        assert len(idxs) == 2, "each portal color must define exactly one pair"
        a, b = idxs
        mate[a] = b
        mate[b] = a

    portal_at: dict[tuple[int, int], int] = {}
    for idx, portal in enumerate(portals):
        for cell in portal["cells"]:
            portal_at[cell] = idx

    emitters.sort(key=lambda item: (item["center"][0], item["center"][1]))
    return emitters, portals, mate, portal_at


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored rays from gray-capped emitters through paired portals, recoloring each teleported segment by portal color.

    Input structure:
    - The background is black.
    - Each emitter is a 3-cell horizontal or vertical line with two grey end cells and one colored center cell.
    - Each portal is a monochrome 1x2 or 2x1 domino, and every portal color appears in exactly two separate dominoes.

    Rule:
    - For each emitter, infer its launch direction along the bar toward the side with more open grid space and begin the ray just beyond the outer grey cap.
    - Paint the ray across background cells in the emitter-center color until the ray reaches a portal cell or the grid edge.
    - When a ray reaches a portal, stop before the portal, jump to the other portal of the same color, rotate to that portal's own orientation, and launch away from the incident portal.
    - Continue tracing after each teleport, replacing the ray color with the portal-pair color, while keeping all emitter and portal cells unchanged.

    Key insight:
    - The task is a cascade of straight ray segments: emitters choose the initial direction from geometry, while each portal pair both relocates the ray and replaces its direction and color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    emitters, portals, mate, portal_at = _parse_scene(g)
    out = g.copy()

    for emitter in emitters:
        direction = _emitter_direction(emitter, h, w)
        dr, dc = direction
        sr, sc = emitter["center"]
        r = sr + 2 * dr
        c = sc + 2 * dc
        color = int(emitter["center_color"])
        seen: set[tuple[int, int, int, int, int]] = set()

        while 0 <= r < h and 0 <= c < w:
            state = (r, c, dr, dc, color)
            if state in seen:
                break
            seen.add(state)

            if int(g[r, c]) == BG:
                if int(out[r, c]) == BG:
                    out[r, c] = color
                r += dr
                c += dc
                continue

            portal_idx = portal_at.get((r, c))
            if portal_idx is None:
                break

            dst_idx = mate[portal_idx]
            src_portal = portals[portal_idx]
            dst_portal = portals[dst_idx]
            direction2 = _portal_exit_direction(
                src_portal["cells"], dst_portal["cells"], dst_portal["orient"], h, w, strict=False
            )
            assert direction2 is not None
            dr, dc = direction2
            color = int(dst_portal["color"])
            r, c = _portal_exit_cell(dst_portal["cells"], dst_portal["orient"], direction2)

    return out


def _random_emitter_candidates(
    obj_cells: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    h: int,
    w: int,
    teleports: int,
    rng: random.Random,
) -> list[dict]:
    need = 2 if teleports == 0 else 3
    candidates: list[dict] = []

    for _ in range(120):
        direction = rng.choice(DIRS)
        if direction == (0, 1):
            if w < 7:
                continue
            r = rng.randrange(h)
            c = rng.choice(list(range(1, w // 2)))
            cells = ((r, c - 1), (r, c), (r, c + 1))
        elif direction == (0, -1):
            if w < 7:
                continue
            r = rng.randrange(h)
            c = rng.choice(list(range((w + 1) // 2, w - 1)))
            cells = ((r, c - 1), (r, c), (r, c + 1))
        elif direction == (1, 0):
            if h < 7:
                continue
            r = rng.choice(list(range(1, h // 2)))
            c = rng.randrange(w)
            cells = ((r - 1, c), (r, c), (r + 1, c))
        else:
            if h < 7:
                continue
            r = rng.choice(list(range((h + 1) // 2, h - 1)))
            c = rng.randrange(w)
            cells = ((r - 1, c), (r, c), (r + 1, c))

        cells = tuple(sorted(cells))
        if not _object_placeable(cells, obj_cells, path_cells, h, w):
            continue
        center = cells[1]
        start = (center[0] + 2 * direction[0], center[1] + 2 * direction[1])
        if not (0 <= start[0] < h and 0 <= start[1] < w):
            continue
        if len(_line_to_edge(start, direction, h, w)) < need:
            continue
        candidates.append({"cells": cells, "center": center, "direction": direction, "start": start})

    rng.shuffle(candidates)
    return candidates


def _build_chain_from(
    start: tuple[int, int],
    direction: tuple[int, int],
    teleports: int,
    obj_cells: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[set[tuple[int, int]], list[tuple[dict, dict]]] | None:
    line = _line_to_edge(start, direction, h, w)
    need = 2 if teleports == 0 else 3
    if len(line) < need:
        return None

    if teleports == 0:
        if _segment_is_free(line, obj_cells, path_cells):
            return set(line), []
        return None

    hit_indices = list(range(2, len(line)))
    rng.shuffle(hit_indices)
    hit_indices = hit_indices[: min(len(hit_indices), 10)]

    for hit_idx in hit_indices:
        path = line[:hit_idx]
        hit = line[hit_idx]
        if not _segment_is_free(path, obj_cells, path_cells):
            continue

        src_variants = _portal_variants_for_hit(hit, direction, h, w)
        rng.shuffle(src_variants)
        for src in src_variants:
            src_cells = src["cells"]
            if not _object_placeable(src_cells, obj_cells, path_cells, h, w):
                continue

            obj2 = obj_cells | set(src_cells)
            path2 = path_cells | set(path)
            dst_candidates = _random_portal_candidates(obj2, path2, h, w, rng)
            for dst in dst_candidates:
                next_dir = _portal_exit_direction(src_cells, dst["cells"], dst["orient"], h, w, strict=True)
                if next_dir is None:
                    continue
                next_start = _portal_exit_cell(dst["cells"], dst["orient"], next_dir)
                next_need = 2 if teleports - 1 == 0 else 3
                if not (0 <= next_start[0] < h and 0 <= next_start[1] < w):
                    continue
                if next_start in obj2 or next_start in path2:
                    continue
                if len(_line_to_edge(next_start, next_dir, h, w)) < next_need:
                    continue

                obj3 = obj2 | set(dst["cells"])
                result = _build_chain_from(next_start, next_dir, teleports - 1, obj3, path2, h, w, rng)
                if result is None:
                    continue
                sub_paths, sub_pairs = result
                return set(path) | sub_paths, [(src, dst)] + sub_pairs

    return None


def _sample_chain_lengths(rng: random.Random) -> list[int]:
    if rng.random() < 0.5:
        lengths = [rng.choice((2, 3)), rng.choice((0, 1))]
    else:
        lengths = [rng.choice((2, 3)), rng.choice((0, 1)), rng.choice((0, 1))]
    rng.shuffle(lengths)
    return lengths


def _render_scene(h: int, w: int, chains: list[dict], rng: random.Random) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    emitter_colors = list(EMITTER_COLORS)
    portal_colors = list(PORTAL_COLORS)
    rng.shuffle(emitter_colors)
    rng.shuffle(portal_colors)

    portal_color_idx = 0
    for chain_idx, chain in enumerate(chains):
        emitter_color = emitter_colors[chain_idx]
        for cell in chain["emitter"]["cells"]:
            grid[cell] = GREY
        grid[chain["emitter"]["center"]] = emitter_color

        for src, dst in chain["pairs"]:
            portal_color = portal_colors[portal_color_idx]
            portal_color_idx += 1
            for cell in src["cells"]:
                grid[cell] = portal_color
            for cell in dst["cells"]:
                grid[cell] = portal_color

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    chain_lengths = _sample_chain_lengths(rng)
    total_pairs = sum(chain_lengths)
    base = 16 + 2 * total_pairs
    h = rng.randint(max(18, base - 2), min(30, base + 4))
    w = rng.randint(max(18, base - 2), min(30, base + 4))

    obj_cells: set[tuple[int, int]] = set()
    path_cells: set[tuple[int, int]] = set()
    chains: list[dict] = []

    for teleports in sorted(chain_lengths, reverse=True):
        placed = None
        emitter_candidates = _random_emitter_candidates(obj_cells, path_cells, h, w, teleports, rng)
        for emitter in emitter_candidates:
            obj2 = obj_cells | set(emitter["cells"])
            result = _build_chain_from(emitter["start"], emitter["direction"], teleports, obj2, path_cells, h, w, rng)
            if result is None:
                continue
            seg_cells, pairs = result
            chain_obj_cells = set(emitter["cells"])
            for src, dst in pairs:
                chain_obj_cells.update(src["cells"])
                chain_obj_cells.update(dst["cells"])
            placed = {"emitter": emitter, "pairs": pairs, "paths": seg_cells, "obj_cells": chain_obj_cells}
            break

        if placed is None:
            return None

        chains.append(placed)
        obj_cells.update(placed["obj_cells"])
        path_cells.update(placed["paths"])

    if sum(len(chain["pairs"]) for chain in chains) != total_pairs:
        return None
    if total_pairs > len(PORTAL_COLORS) or len(chains) > len(EMITTER_COLORS):
        return None

    grid = _render_scene(h, w, chains, rng)
    if not np.any(grid == BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape or np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid portal-ray ARC input")
