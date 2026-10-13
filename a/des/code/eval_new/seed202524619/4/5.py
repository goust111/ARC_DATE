from __future__ import annotations

from collections import Counter
import random
from collections.abc import Iterable

import numpy as np


_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as arrays of (r, c) coords."""
    H, W = mask.shape
    vis = np.zeros((H, W), dtype=bool)
    comps: list[np.ndarray] = []
    for sr in range(H):
        for sc in range(W):
            if not bool(mask[sr, sc]) or bool(vis[sr, sc]):
                continue
            stack = [(int(sr), int(sc))]
            vis[sr, sc] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in _DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and not bool(vis[nr, nc]):
                        vis[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _neighbors(r: int, c: int, H: int, W: int) -> Iterable[tuple[int, int]]:
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            yield nr, nc


def _is_two_cell_segment(comp: np.ndarray) -> bool:
    if len(comp) != 2:
        return False
    (r0, c0), (r1, c1) = map(tuple, comp)
    dr, dc = abs(int(r0 - r1)), abs(int(c0 - c1))
    return (dr, dc) in {(1, 0), (0, 1)}


def _is_straight_line(comp: np.ndarray) -> bool:
    if len(comp) < 2:
        return False
    r0, r1 = comp[:, 0].min(), comp[:, 0].max()
    c0, c1 = comp[:, 1].min(), comp[:, 1].max()
    cells = {tuple(cell) for cell in comp}

    if r0 == r1:
        if len(comp) != int(c1 - c0 + 1):
            return False
        rs = int(r0)
        for c in range(int(c0), int(c1) + 1):
            if (rs, c) not in cells:
                return False
        return True
    if c0 == c1:
        if len(comp) != int(r1 - r0 + 1):
            return False
        for r in range(int(r0), int(r1) + 1):
            if (r, int(c0)) not in cells:
                return False
        return True
    return False


def _segment_endpoints(comp: np.ndarray) -> list[tuple[int, int]]:
    comp_set = {tuple(cell) for cell in comp}
    endpoints: list[tuple[int, int]] = []
    H = int(comp[:, 0].max()) + 1
    W = int(comp[:, 1].max()) + 1
    for r, c in comp:
        deg = 0
        for nr, nc in _neighbors(int(r), int(c), H, W):
            if (nr, nc) in comp_set:
                deg += 1
        if deg <= 1:
            endpoints.append((int(r), int(c)))
    return endpoints


def _path_color_candidates(g: np.ndarray, bg: int, comps_by_color: dict[int, list[np.ndarray]]) -> list[int]:
    candidates: list[tuple[int, int]] = []
    for color, comps in comps_by_color.items():
        if color == bg or not comps:
            continue
        if all(_is_straight_line(comp) for comp in comps):
            total = int(sum(len(comp) for comp in comps))
            candidates.append((total, color))
    candidates.sort(reverse=True)
    if not candidates:
        return []
    top = candidates[0][0]
    return [color for total, color in candidates if total == top]


def main(grid: np.ndarray) -> np.ndarray:
    """Follow green path segments through same-color portal pairs and extend from the paired exit.

    Input structure:
    - Background is black; green path pixels form one-cell-wide straight segments.
    - There are several two-pixel same-color portal pairs (1×2 or 2×1), one pixel of each pair touching exactly one green segment endpoint.
    - Each pair has one portal touching green (入口) and one partner portal (出口). For the entry portal, the other portal in the pair is the paired exit portal.

    Rule:
    - Identify portal pairs by color. For each pair, find the cell touching green as entry and the other cell in that portal as direction marker.
    - For the corresponding green segment endpoint touching that entry cell, clear that entire segment and redraw the same number of green pixels from the exit portal's direction-pixel.
    - The drawn segment goes straight in the entry-induced direction for exactly the original segment length.
    - If the traced segment reaches another portal entry, teleport through that pair and continue with the same stepping process (cascade).
    - Portal pixels remain unchanged.

    Key insight:
    The only stable mapping key is color pairing plus the entry-side orientation: this fixes the out direction, and all movement is iterative length-preserving tracing.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    counts = Counter(map(int, g.flatten()))
    if not counts:
        return g.copy()

    bg = int(counts.most_common(1)[0][0])
    colors = sorted(int(c) for c in counts.keys() if int(c) != bg)
    if not colors:
        return g.copy()

    comps_by_color: dict[int, list[np.ndarray]] = {
        color: _components(g == color)
        for color in colors
    }

    # Step 1: identify portal-pair colors.
    portal_colors: set[int] = set()
    for color, comps in comps_by_color.items():
        if len(comps) == 2 and all(_is_two_cell_segment(c) for c in comps):
            portal_colors.add(color)

    # Step 2: guess the main path color and infer entry->direction vectors.
    path_candidates = _path_color_candidates(g, bg, comps_by_color)
    if not path_candidates:
        return g.copy()
    path_color = int(path_candidates[0])
    path_mask = g == path_color

    portal_info: dict[int, dict[str, object]] = {}
    coord_to_pair_color: dict[tuple[int, int], int] = {}

    for color in portal_colors:
        comps = comps_by_color[color]
        entry_comp_idx = None
        entry_cell: tuple[int, int] | None = None

        # The entry portal component touches green once.
        for idx, comp in enumerate(comps):
            for r, c in map(tuple, comp):
                for nr, nc in _neighbors(int(r), int(c), H, W):
                    if int(g[nr, nc]) == int(path_color):
                        if entry_cell is not None:
                            # ambiguous / invalid mapping, skip this color.
                            entry_comp_idx = None
                            break
                        entry_comp_idx = idx
                        entry_cell = (int(r), int(c))
                        break
                if entry_comp_idx is None:
                    continue
                if entry_comp_idx is not None and entry_cell is not None:
                    break
            if entry_comp_idx is None:
                continue
            break

        if entry_comp_idx is None or entry_cell is None:
            continue

        entry_comp = comps[int(entry_comp_idx)]
        exit_comp = comps[1 - int(entry_comp_idx)]

        # entry component has two pixels; one is entry pixel, the other is direction pixel.
        other_cells = [tuple(c) for c in entry_comp if tuple(c) != entry_cell]
        if not other_cells:
            continue
        dir_pixel = other_cells[0]
        out_dir = (int(dir_pixel[0] - entry_cell[0]), int(dir_pixel[1] - entry_cell[1]))

        exit_cells = [tuple(c) for c in exit_comp]
        exit_a, exit_b = exit_cells
        # Choose the exit direction-pixel according to the same direction as the entry portal.
        if (exit_b[0] - exit_a[0], exit_b[1] - exit_a[1]) == out_dir:
            exit_start = exit_b
        elif (exit_a[0] - exit_b[0], exit_a[1] - exit_b[1]) == out_dir:
            exit_start = exit_a
        else:
            exit_start = exit_a

        exit_partner = exit_b if exit_start == exit_a else exit_a

        pinfo = {
            "out_dir": out_dir,
            "entry_pixel": entry_cell,
            "entry_pixels": {entry_cell},
            "exit_pixels": {exit_a, exit_b},
            "exit_start": exit_start,
            "exit_partner": {exit_start: exit_partner, exit_partner: exit_start},
        }
        portal_info[color] = pinfo

        for pt in [entry_cell, dir_pixel, exit_a, exit_b]:
            coord_to_pair_color[tuple(pt)] = color

    if not portal_info:
        return g.copy()

    # Step 3: find move specs from each entry-side green segment.
    entry_pixels = {
        tuple(v["entry_pixel"]) for v in portal_info.values() if isinstance(v, dict)
    }
    move_specs: list[tuple[np.ndarray, int, int]] = []
    path_components = _components(path_mask)

    for comp in path_components:
        if len(comp) < 2 or not _is_straight_line(comp):
            continue
        endpoints = _segment_endpoints(comp)
        if len(endpoints) != 2:
            continue

        endpoint = None
        link_color = None
        for ep in endpoints:
            if ep in entry_pixels:
                endpoint = ep
                link_color = coord_to_pair_color.get(ep)
                break
        if endpoint is None or link_color is None or link_color not in portal_info:
            continue

        length = int(len(comp))
        if length <= 0:
            continue
        move_specs.append((comp, length, int(link_color)))

    if not move_specs:
        return g.copy()

    out = g.copy()
    # Remove originals before rewriting to keep length-preserving teleport behavior.
    for comp, _length, _color in move_specs:
        out[comp[:, 0], comp[:, 1]] = int(bg)

    # Step 4: emit each moved segment with cascading teleport.
    for _comp, length, color in move_specs:
        info = portal_info[color]
        r, c = info["exit_start"]  # type: ignore[assignment]
        dr, dc = info["out_dir"]    # type: ignore[assignment]
        remain = int(length)
        seen: set[tuple[int, int, int, int, int]] = set()

        while remain > 0:
            state = (int(r), int(c), int(dr), int(dc), int(remain))
            if state in seen:
                break
            seen.add(state)

            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            target = int(out[nr, nc])
            key = (nr, nc)

            if key in coord_to_pair_color:
                hit_color = int(coord_to_pair_color[key])
                hit_info = portal_info.get(hit_color)
                if hit_info is not None:
                    if key in hit_info["entry_pixels"]:  # type: ignore[index]
                        r, c = int(hit_info["exit_start"][0]), int(hit_info["exit_start"][1])  # type: ignore[index]
                        dr, dc = int(hit_info["out_dir"][0]), int(hit_info["out_dir"][1])  # type: ignore[index]
                        continue
                    if key in hit_info["exit_pixels"]:  # type: ignore[index]
                        partner = hit_info["exit_partner"][key]  # type: ignore[index]
                        partner = tuple(partner)
                        dr, dc = int(partner[0] - nr), int(partner[1] - nc)
                        r, c = int(partner[0]), int(partner[1])
                        continue

            if target != int(bg):
                break

            out[nr, nc] = int(path_color)
            r, c = nr, nc
            remain -= 1

    return out


# ------------------------------- generator -------------------------------

_BG = 0
_PATH_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _in_bounds(pt: tuple[int, int], H: int, W: int) -> bool:
    r, c = pt
    return 0 <= r < H and 0 <= c < W


def _mark(occupied: np.ndarray, cells: Iterable[tuple[int, int]]) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _try_place_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    path_color: int,
    portal_color: int,
    length: int,
    entry: tuple[int, int],
    seg_dir: tuple[int, int],
    out_dir: tuple[int, int],
) -> dict[str, object] | None:
    H, W = grid.shape
    if length < 2:
        return None

    er, ec = entry
    path_cells: list[tuple[int, int]] = []
    for k in range(length):
        r, c = er + seg_dir[0] * k, ec + seg_dir[1] * k
        if not _in_bounds((r, c), H, W):
            return None
        path_cells.append((r, c))

    for rc in path_cells:
        if occupied[rc]:
            return None

    pr, pc = er + out_dir[0], ec + out_dir[1]
    if not _in_bounds((pr, pc), H, W) or occupied[pr, pc]:
        return None
    if (pr, pc) in path_cells:
        return None

    # Exit portal orientation follows the out-direction axis.
    if out_dir[0] == 0:
        axis_choices = [(0, 1), (0, -1)]
    else:
        axis_choices = [(1, 0), (-1, 0)]

    for _ in range(200):
        dr, dc = random.choice(axis_choices)
        if _in_bounds((er, ec), H, W):
            # randomize anchor, then place [anchor, anchor + axis]
            if dr == 0:
                rr = random.randint(0, H - 1)
                cc = random.randint(0, W - 2)
                p0, p1 = (rr, cc), (rr, cc + 1)
            else:
                rr = random.randint(0, H - 2)
                cc = random.randint(0, W - 1)
                p0, p1 = (rr, cc), (rr + 1, cc)
            if dr == 0 and dc < 0:
                p0, p1 = p1, p0
            if dr < 0 and dc == 0:
                p0, p1 = p1, p0
            if random.random() < 0.5:
                p0, p1 = p1, p0

            if (p0[0] + dr, p0[1] + dc) != p1:
                continue
            if occupied[p0] or occupied[p1]:
                continue

            # Avoid accidental overlap with this pair's own path+entry portal.
            blocked = set(path_cells) | {entry, (pr, pc)}
            if (p0 in blocked) or (p1 in blocked):
                continue

            _mark(occupied, path_cells)
            _mark(occupied, {entry, (pr, pc), p0, p1})

            for rr2, cc2 in path_cells:
                grid[rr2, cc2] = path_color
            grid[er, ec] = portal_color
            grid[pr, pc] = portal_color
            grid[p0[0], p0[1]] = portal_color
            grid[p1[0], p1[1]] = portal_color

            exit_comp = [p0, p1]
            if (exit_comp[1][0] - exit_comp[0][0], exit_comp[1][1] - exit_comp[0][1]) == out_dir:
                exit_start = exit_comp[1]
                exit_partner = exit_comp[0]
            elif (exit_comp[0][0] - exit_comp[1][0], exit_comp[0][1] - exit_comp[1][1]) == out_dir:
                exit_start = exit_comp[0]
                exit_partner = exit_comp[1]
            else:
                exit_start = exit_comp[0]
                exit_partner = exit_comp[1]

            return {
                "color": int(portal_color),
                "path_length": int(length),
                "entry_pixel": (int(er), int(ec)),
                "entry_partner": (int(pr), int(pc)),
                "out_dir": (int(out_dir[0]), int(out_dir[1])),
                "exit_start": (int(exit_start[0]), int(exit_start[1])),
                "exit_partner": (int(exit_partner[0]), int(exit_partner[1])),
                "exit_pixels": {tuple(exit_comp[0]), tuple(exit_comp[1])},
            }

    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(10, 28)
    W = random.randint(10, 28)
    n_pairs = random.randint(1, 4)

    path_color = random.choice(_PATH_POOL)
    available_colors = [c for c in _PATH_POOL if c != path_color]
    if len(available_colors) < n_pairs:
        return None
    portal_colors = random.sample(available_colors, n_pairs)

    grid = np.full((H, W), int(_BG), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    pair_specs: list[dict[str, object]] = []
    rays: list[list[tuple[int, int]]] = []

    cascade_prob = 0.6 if n_pairs > 1 else 0.0
    for i, color in enumerate(portal_colors):
        linked = bool(cascade_prob and i > 0 and rays and random.random() < 0.7)

        length = random.randint(2, min(8, max(2, min(H, W) - 2)))
        placed = None

        for attempt in range(220):
            if linked:
                ray = random.choice(rays)
                cand_entries = [p for p in ray if not occupied[p]]
                if not cand_entries:
                    linked = False
                else:
                    entry = random.choice(cand_entries)
                    seg_dir = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                    out_dir = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                    if out_dir == (0, 0):
                        out_dir = (0, 1)
                    placed = _try_place_pair(
                        grid,
                        occupied,
                        path_color,
                        color,
                        length,
                        entry,
                        seg_dir,
                        out_dir,
                    )
                if placed is not None:
                    break

            # Independent placement.
            entry = (random.randint(0, H - 1), random.randint(0, W - 1))
            if occupied[entry]:
                continue
            seg_dir = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            out_dir = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            placed = _try_place_pair(
                grid,
                occupied,
                path_color,
                color,
                length,
                entry,
                seg_dir,
                out_dir,
            )
            if placed is not None:
                break

        if placed is None:
            return None

        pair_specs.append(placed)
        es = placed["exit_start"]
        od = placed["out_dir"]
        esr, esc = int(es[0]), int(es[1])

        ray: list[tuple[int, int]] = []
        for k in range(1, int(placed["path_length"]) + 1):
            rr, cc = esr + od[0] * k, esc + od[1] * k
            if not (0 <= rr < H and 0 <= cc < W):
                break
            ray.append((rr, cc))
        rays.append(ray)

    # Quick local oracle check: ensure solver runs and produces a grid.
    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None

    # Keep generated inputs diverse and non-degenerate.
    colors = Counter(map(int, grid.flatten()))
    if int(colors.get(path_color, 0)) < 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid portal-chain input")
