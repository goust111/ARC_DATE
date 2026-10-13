import random
from collections import Counter, deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_BG = 0
_MAROON = 9

# Tooth directions in (top, bottom, left, right)
_TOP = 0
_BOTTOM = 1
_LEFT = 2
_RIGHT = 3

# Generator palette and bounds.
_GENERATOR_PALETTE = (1, 2, 3, 4, 5, 6, 7, 8)
_MIN_GRID = 14
_MAX_GRID = 30
_MAX_GENERATION_ATTEMPTS = 400

# Generator layout knobs.


def _neighbors4(r: int, c: int) -> Iterable[Tuple[int, int]]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _inside(H: int, W: int, pos: Tuple[int, int]) -> bool:
    r, c = pos
    return 0 <= r < H and 0 <= c < W


def _find_components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    """Find 4-connected components of non-background pixels."""
    H, W = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) == int(bg) or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[Tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and int(grid[nr, nc]) != int(bg):
                        visited[nr, nc] = True
                        q.append((nr, nc))

            comps.append(np.array(coords, dtype=int))

    return comps


def _tooth_count(cells: set[Tuple[int, int]], start_r: int, start_c: int, dr: int, dc: int) -> int:
    """Count outward teeth up to length 2 from a given start in one direction."""
    k = 0
    for step in range(1, 3):
        rr = start_r + dr * step
        cc = start_c + dc * step
        if (int(rr), int(cc)) not in cells:
            break
        k += 1
    return k


def _extract_gear(grid: np.ndarray, comp: np.ndarray, bg: int) -> Optional[dict]:
    """Extract one gear from one connected non-bg component.

    A valid gear has exactly one full 3x3 center block and any of the four side teeth
    of length 0~2, plus optional single maroon marker inside the center block.
    """
    H, W = grid.shape
    cells = set((int(r), int(c)) for r, c in comp)
    if len(cells) < 9:
        return None

    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)

    candidates = []

    for core_r in range(min_r, max_r - 1):
        for core_c in range(min_c, max_c - 1):
            core: list[Tuple[int, int]] = []
            valid = True
            for dr in range(3):
                for dc in range(3):
                    pos = (core_r + dr, core_c + dc)
                    if pos not in cells:
                        valid = False
                        break
                    core.append(pos)
                if not valid:
                    break
            if not valid:
                continue

            core_vals = [int(grid[r, c]) for r, c in core]
            non_bg = [v for v in core_vals if v != bg]
            if not non_bg:
                continue

            cnt = Counter(non_bg)
            marker_count = cnt.get(_MAROON, 0)
            if marker_count > 1:
                continue

            body_candidates = [k for k in cnt if k != _MAROON]
            if len(body_candidates) != 1:
                continue
            body_color = int(body_candidates[0])
            body_count = cnt[body_color]
            if body_count + marker_count != 9 or body_count < 8:
                continue

            top = _tooth_count(cells, core_r, core_c + 1, -1, 0)
            bottom = _tooth_count(cells, core_r + 2, core_c + 1, 1, 0)
            left = _tooth_count(cells, core_r + 1, core_c, 0, -1)
            right = _tooth_count(cells, core_r + 1, core_c + 2, 0, 1)
            counts = (top, bottom, left, right)

            expected: set[Tuple[int, int]] = set(core)
            for k in range(1, top + 1):
                expected.add((core_r - k, core_c + 1))
            for k in range(1, bottom + 1):
                expected.add((core_r + 2 + k, core_c + 1))
            for k in range(1, left + 1):
                expected.add((core_r + 1, core_c - k))
            for k in range(1, right + 1):
                expected.add((core_r + 1, core_c + 2 + k))

            if expected != cells:
                continue

            core_set = set(core)
            teeth = expected - core_set

            driver = any(int(grid[r, c]) == _MAROON for r, c in core)
            marker_positions = [(int(r), int(c)) for r, c in core if int(grid[r, c]) == _MAROON]

            candidates.append(
                {
                    "center": (core_r + 1, core_c + 1),
                    "core_top_left": (core_r, core_c),
                    "counts": counts,
                    "coords": np.array(sorted(expected), dtype=int),
                    "core_cells": core_set,
                    "teeth_cells": teeth,
                    "driver": bool(driver),
                    "marker_positions": marker_positions,
                }
            )

    if len(candidates) != 1:
        return None
    return candidates[0]


def _rotate_point(r: int, c: int, center_r: int, center_c: int, clockwise: bool) -> Tuple[int, int]:
    dr = int(r - center_r)
    dc = int(c - center_c)
    if clockwise:
        rr = center_r + dc
        cc = center_c - dr
    else:
        rr = center_r - dc
        cc = center_c + dr
    return int(rr), int(cc)


def _build_gear_graph(gears: list[dict], H: int, W: int) -> list[list[int]]:
    n = len(gears)
    adjacency = [set() for _ in range(n)]
    cell_to_gear: dict[Tuple[int, int], int] = {}
    for i, g in enumerate(gears):
        for r, c in g["coords"]:
            pos = (int(r), int(c))
            assert pos not in cell_to_gear, f"overlapping components at {pos}"
            cell_to_gear[pos] = i

    for i, gi in enumerate(gears):
        for r, c in gi["teeth_cells"]:
            for nr, nc in _neighbors4(r, c):
                j = cell_to_gear.get((nr, nc))
                if j is None or j == i:
                    continue
                if (nr, nc) in gears[j]["teeth_cells"]:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

    return [sorted(list(s)) for s in adjacency]


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate 90° gear rotations through a contact chain, starting from the maroon marker gear.

    Input structure:
    - Black background, and several disjoint gears. Each gear is one connected non-background
      component shaped as a 3×3 core plus optional side teeth (length 0~2) protruding from
      side midpoints. The whole gear is one color, except the driven gear has one maroon pixel
      inside its 3×3 core.

    Rule:
    - Detect the single maroon-marked gear as the driver and rotate it 90° clockwise.
    - Any gear whose teeth touch an already-rotated gear rotates 90° counterclockwise,
      and direction flips at each propagation step along the contact graph.
    - Gears not connected to the driver by tooth contact remain unchanged.
    - Rotate each affected gear around its core-center while preserving cell colors (including maroon).

    Key insight:
    - The non-symmetric tooth counts around a 3×3 core identify each gear's orientation and teeth,
      and meshing is exactly determined by 4-neighbor contacts between teeth cells.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = _BG

    comps = _find_components(g, bg)
    assert len(comps), "expected at least one non-background component"

    gears: list[dict] = []
    for comp in comps:
        gear = _extract_gear(g, comp, bg)
        assert gear is not None, "a component does not match the expected gear template"
        gears.append(gear)

    driver = [i for i, gspec in enumerate(gears) if gspec["driver"]]
    assert len(driver) == 1, "expected exactly one driver gear marked by maroon"
    driver_idx = int(driver[0])

    graph = _build_gear_graph(gears, H, W)

    # 1 clockwise, -1 counterclockwise, 0 unchanged.
    rotate_dir = [0] * len(gears)
    rotate_dir[driver_idx] = 1
    q = deque([driver_idx])
    while q:
        u = q.popleft()
        for v in graph[u]:
            need = -rotate_dir[u]
            if rotate_dir[v] == 0:
                rotate_dir[v] = need
                q.append(v)
            else:
                assert rotate_dir[v] == need, "gear contact graph has contradictory rotation parity"

    rotating = [i for i, d in enumerate(rotate_dir) if d != 0]

    out = g.copy()
    for i in rotating:
        for r, c in gears[i]["coords"]:
            out[r, c] = int(bg)

    for i in rotating:
        clockwise = rotate_dir[i] == 1
        cr, cc = gears[i]["center"]
        for r, c in gears[i]["coords"]:
            color = int(g[int(r), int(c)])
            nr, nc = _rotate_point(int(r), int(c), cr, cc, clockwise)
            assert _inside(H, W, (nr, nc)), "rotation moved a gear out of bounds"
            assert out[nr, nc] == int(bg), "rotation would overwrite a non-rotated object"
            out[nr, nc] = color

    return out


# --------------- generator ----------------

def _gear_coords(top: int, left: int, counts: Tuple[int, int, int, int]) -> Tuple[set[Tuple[int, int]], set[Tuple[int, int]], set[Tuple[int, int]]]:
    top_cnt, bottom_cnt, left_cnt, right_cnt = counts
    coords: set[Tuple[int, int]] = set()
    for dr in range(3):
        for dc in range(3):
            coords.add((top + dr, left + dc))
    for k in range(1, top_cnt + 1):
        coords.add((top - k, left + 1))
    for k in range(1, bottom_cnt + 1):
        coords.add((top + 2 + k, left + 1))
    for k in range(1, left_cnt + 1):
        coords.add((top + 1, left - k))
    for k in range(1, right_cnt + 1):
        coords.add((top + 1, left + 2 + k))

    core: set[Tuple[int, int]] = {(top + r, left + c) for r in range(3) for c in range(3)}
    return coords, core, coords - core


def _random_counts(required: Optional[int] = None) -> Tuple[int, int, int, int]:
    while True:
        cnt = [random.randint(0, 2) for _ in range(4)]
        if required is not None:
            cnt[required] = random.randint(1, 2)

        if any(v > 0 for v in cnt) and len(set(cnt)) > 1:
            return (cnt[_TOP], cnt[_BOTTOM], cnt[_LEFT], cnt[_RIGHT])


def _place_single_gear(H: int, W: int) -> Optional[dict]:
    for _ in range(100):
        counts = _random_counts()
        top = random.randint(2, H - 5)
        left = random.randint(2, W - 5)
        coords, core, teeth = _gear_coords(top, left, counts)
        if all(0 <= r < H and 0 <= c < W for r, c in coords):
            return {
                "top": top,
                "left": left,
                "counts": counts,
                "coords": coords,
                "core": core,
                "teeth": teeth,
                "cell_set": set(coords),
            }
    return None


def _tooth_contacts(candidate: dict, existing: list[dict]) -> Tuple[set[int], bool]:
    adj: set[int] = set()
    for e_idx, e in enumerate(existing):
        et = e["teeth"]
        for r, c in candidate["teeth"]:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in e["cell_set"]:
                    if (nr, nc) not in et:
                        return set(), False
                    adj.add(e_idx)

        for r, c in candidate["cell_set"]:
            if (r, c) in candidate["teeth"]:
                continue
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in e["cell_set"]:
                    return set(), False

    return adj, True


def _build_chain(H: int, W: int, length: int) -> Optional[List[dict]]:
    assert length >= 2

    for _ in range(300):
        first_counts = _random_counts()
        first_top = random.randint(2, H - 5)
        first_left = random.randint(2, W - 5)
        coords, core, teeth = _gear_coords(first_top, first_left, first_counts)

        chain = [
            {
                "top": first_top,
                "left": first_left,
                "counts": first_counts,
                "coords": coords,
                "core": core,
                "teeth": teeth,
                "cell_set": set(coords),
            }
        ]

        prev_dir: Optional[str] = None
        ok = True

        for idx in range(1, length):
            prev = chain[idx - 1]
            prev_counts = prev["counts"]
            prev_r = prev["top"]
            prev_c = prev["left"]

            direction_order = ["R", "D", "L", "U"]
            random.shuffle(direction_order)
            placed = False

            for direction in direction_order:
                if prev_dir == "R" and direction == "L":
                    continue
                if prev_dir == "L" and direction == "R":
                    continue
                if prev_dir == "U" and direction == "D":
                    continue
                if prev_dir == "D" and direction == "U":
                    continue

                if direction == "R":
                    required_prev = _RIGHT
                    other = _LEFT
                    dtop, dleft = 0, 3 + prev_counts[required_prev]
                elif direction == "L":
                    required_prev = _LEFT
                    other = _RIGHT
                    dtop, dleft = 0, -(3 + prev_counts[required_prev])
                elif direction == "D":
                    required_prev = _BOTTOM
                    other = _TOP
                    dtop, dleft = 3 + prev_counts[required_prev], 0
                else:  # "U"
                    required_prev = _TOP
                    other = _BOTTOM
                    dtop, dleft = -(3 + prev_counts[required_prev]), 0

                if prev_counts[required_prev] == 0:
                    continue

                next_counts = _random_counts(required=other)
                ntop = prev_r + dtop
                nleft = prev_c + dleft

                ncoords, ncore, nteeth = _gear_coords(ntop, nleft, next_counts)
                if not all(0 <= r < H and 0 <= c < W for r, c in ncoords):
                    continue

                overlaps = any((r, c) in chain[i]["cell_set"] for i in range(len(chain)) for (r, c) in ncoords)
                if overlaps:
                    continue

                adj, valid = _tooth_contacts({"coords": ncoords, "core": ncore, "teeth": nteeth, "cell_set": set(ncoords)}, chain)
                if not valid or adj != {idx - 1}:
                    continue

                chain.append(
                    {
                        "top": ntop,
                        "left": nleft,
                        "counts": next_counts,
                        "coords": ncoords,
                        "core": ncore,
                        "teeth": nteeth,
                        "cell_set": set(ncoords),
                    }
                )
                prev_dir = direction
                placed = True
                break

            if not placed:
                ok = False
                break

        if ok and len(chain) == length:
            return chain

    return None


def _add_isolated(H: int, W: int, specs: list[dict], needed: int) -> bool:
    for _ in range(300):
        if needed == 0:
            return True
        placed = True

        top = random.randint(2, H - 5)
        left = random.randint(2, W - 5)
        counts = _random_counts()
        coords, core, teeth = _gear_coords(top, left, counts)
        if not all(0 <= r < H and 0 <= c < W for r, c in coords):
            continue

        overlaps = any((r, c) in s["cell_set"] for s in specs for (r, c) in coords)
        if overlaps:
            continue

        # Must stay isolated from all existing gears.
        adj, valid = _tooth_contacts({"coords": coords, "core": core, "teeth": teeth, "cell_set": set(coords)}, specs)
        if not valid or adj:
            continue

        specs.append(
            {
                "top": top,
                "left": left,
                "counts": counts,
                "coords": coords,
                "core": core,
                "teeth": teeth,
                "cell_set": set(coords),
            }
        )
        needed -= 1

        if not _add_isolated(H, W, specs, needed):
            specs.pop()
            return False
        return True

    return False


def generate_input() -> np.ndarray:
    """Randomly synthesize a valid gear input that exercises isolated, chain, and branching behavior."""
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        H = random.randint(_MIN_GRID, _MAX_GRID)
        W = random.randint(_MIN_GRID, _MAX_GRID)

        chain_roll = random.random()
        if chain_roll < 0.33:
            chain_len = 1
            isolated_extra = random.randint(1, 3)
        elif chain_roll < 0.66:
            chain_len = random.randint(2, 3)
            isolated_extra = random.randint(0, 2)
        else:
            chain_len = random.randint(2, 5)
            isolated_extra = random.randint(0, 1)

        gears: list[dict] = []
        if chain_len == 1:
            single = _place_single_gear(H, W)
            if single is None:
                continue
            gears.append(single)
        else:
            chain = _build_chain(H, W, chain_len)
            if chain is None:
                continue
            gears.extend(chain)

        # Add isolated gears if requested.
        if len(gears) + isolated_extra > len(_GENERATOR_PALETTE):
            continue
        if not _add_isolated(H, W, gears, isolated_extra):
            continue

        # Keep at least one driver and unique colors, with enough color diversity.
        body_colors = random.sample(_GENERATOR_PALETTE, len(gears))
        driver_idx = random.randrange(len(gears))

        grid = np.full((H, W), _BG, dtype=int)
        for idx, spec in enumerate(gears):
            color = body_colors[idx]
            for r, c in spec["coords"]:
                grid[r, c] = int(color)

            if idx == driver_idx:
                mr, mc = random.choice(tuple(spec["core"]))
                grid[mr, mc] = _MAROON

        # Self-check that every generated sample is valid for the current main rule.
        try:
            _ = main(grid)
        except Exception:
            continue

        # Ensure the generator never creates degenerate maroon-only ambiguity.
        if np.sum(grid == _MAROON) != 1:
            continue
        return grid

    raise RuntimeError("failed to generate valid gear input")
