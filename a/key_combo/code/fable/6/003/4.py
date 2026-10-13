from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np


UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

try:
    from arc_utils import components_mask
except Exception:
    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        seen = np.zeros((h, w), dtype=bool)
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if int(connectivity) == 8:
            dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        out: list[np.ndarray] = []
        for r0 in range(h):
            for c0 in range(w):
                if seen[r0, c0] or not mask[r0, c0]:
                    continue
                q = deque([(r0, c0)])
                seen[r0, c0] = True
                coords: list[tuple[int, int]] = []
                while q:
                    r, c = q.popleft()
                    coords.append((r, c))
                    for dr, dc in dirs:
                        rr = r + dr
                        cc = c + dc
                        if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            q.append((rr, cc))
                out.append(np.array(coords, dtype=int))
        return out


BG_COLOR = 0
CLOUD_COLOR = 2
TERRAIN_COLOR = 6
WATER_COLOR = 8

MAX_ATTEMPTS = 400
FORCED_BRANCHES = ("rim", "breach", "tunnel")
_GEN_COUNTER = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay water from the lone cloud through only the basins reachable via the lowest competing outlet on each step.

    Input structure:
    - The lower part is a solid terrain block whose column-wise upper contour
      alternates between ridge segments and flat basin floors.
    - Some one-column ridges contain a single-cell breach, and some wider ridges
      contain a one-cell-wide background tunnel joining the two neighboring basins.
    - The top row contains exactly one cloud-source pixel above one basin.

    Rule:
    - Water falls vertically from the cloud into its basin.
    - In every reached basin, compare the two rim overflows with any breach or
      tunnel mouth on the basin walls; the lowest candidate outlet wins.
    - Fill that basin up to one row below the winning outlet, color the actual
      exit path, and continue into the adjacent downstream basin; repeat until
      the water leaves the grid edge.

    Key insight:
    - The decisive outlet is not always a skyline spill point: internal wall
      openings compete with rim heights, and a tunnel can redirect the relay to
      a different downstream chain while untouched basins stay dry.
    """
    g = np.asarray(grid, dtype=int)
    world = _parse_world(g)
    visited, chosen = _trace_flow(world)

    out = g.copy()
    source = world["basins"][world["source_basin"]]
    out[1 : source["floor"], world["cloud_col"]] = WATER_COLOR

    for basin_idx, cand in zip(visited, chosen, strict=False):
        basin = world["basins"][basin_idx]
        out[cand["row"] + 1 : basin["floor"], basin["c0"] : basin["c1"] + 1] = WATER_COLOR
        _paint_transition(out, world, cand)

    return out


def _parse_world(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    top_counts = Counter(map(int, g[0].tolist()))
    assert len(top_counts) == 2, "top row must contain only background and one cloud source"
    cloud_color = int(min(top_counts, key=lambda c: top_counts[int(c)]))
    bg = int(max(top_counts, key=lambda c: top_counts[int(c)]))
    cloud_cols = np.flatnonzero(g[0] == cloud_color)
    assert cloud_cols.size == 1, "cloud source must be unique"
    cloud_col = int(cloud_cols[0])

    other_colors = [int(c) for c in np.unique(g) if int(c) not in (bg, cloud_color)]
    assert len(other_colors) == 1, "expected exactly one terrain color"
    terrain = int(other_colors[0])

    surface: list[int] = []
    for c in range(w):
        terrain_rows = np.flatnonzero(g[:, c] == terrain)
        assert terrain_rows.size > 0, "every column must contain terrain"
        surface.append(int(terrain_rows[0]))

    segments: list[dict] = []
    c = 0
    while c < w:
        row = int(surface[c])
        c0 = c
        while c + 1 < w and int(surface[c + 1]) == row:
            c += 1
        segments.append({"c0": int(c0), "c1": int(c), "row": int(row)})
        c += 1

    assert len(segments) >= 7 and len(segments) % 2 == 1, "terrain must alternate barrier and basin segments"

    basins: list[dict] = []
    col_to_basin: dict[int, int] = {}
    for basin_idx in range((len(segments) - 1) // 2):
        seg_idx = 2 * basin_idx + 1
        seg = segments[seg_idx]
        left = segments[seg_idx - 1]
        right = segments[seg_idx + 1]
        width = int(seg["c1"] - seg["c0"] + 1)
        assert width >= 2, "basin floors must span at least two columns"
        assert int(seg["row"]) > int(left["row"]) and int(seg["row"]) > int(right["row"]), "invalid basin profile"
        basin = {
            "idx": int(basin_idx),
            "seg_idx": int(seg_idx),
            "c0": int(seg["c0"]),
            "c1": int(seg["c1"]),
            "floor": int(seg["row"]),
            "left_barrier": int(seg_idx - 1),
            "right_barrier": int(seg_idx + 1),
            "candidates": [],
        }
        basins.append(basin)
        for col in range(basin["c0"], basin["c1"] + 1):
            col_to_basin[int(col)] = int(basin_idx)

    source_basin = None
    for basin in basins:
        if basin["c0"] <= cloud_col <= basin["c1"]:
            source_basin = int(basin["idx"])
            break
    assert source_basin is not None, "cloud must sit above a basin"

    for basin in basins:
        left_seg = segments[basin["left_barrier"]]
        right_seg = segments[basin["right_barrier"]]
        basin["candidates"].append(
            {
                "kind": "rim",
                "side": "left",
                "row": int(left_seg["row"]) - 1,
                "dest": int(basin["idx"]) - 1 if int(basin["idx"]) > 0 else None,
                "cells": [],
                "barrier": int(basin["left_barrier"]),
            }
        )
        basin["candidates"].append(
            {
                "kind": "rim",
                "side": "right",
                "row": int(right_seg["row"]) - 1,
                "dest": int(basin["idx"]) + 1 if int(basin["idx"]) + 1 < len(basins) else None,
                "cells": [],
                "barrier": int(basin["right_barrier"]),
            }
        )

    void_mask = np.zeros((h, w), dtype=bool)
    for c, top in enumerate(surface):
        void_mask[top:, c] = g[top:, c] == bg

    void_components: list[dict] = []
    for coords in components_mask(void_mask, connectivity=4):
        cells = sorted((int(r), int(c)) for r, c in coords)
        touch: dict[int, dict] = {}
        for r, c in cells:
            if c - 1 in col_to_basin and r < basins[col_to_basin[c - 1]]["floor"]:
                idx = int(col_to_basin[c - 1])
                rec = touch.setdefault(idx, {"side": "right", "rows": []})
                rec["rows"].append(int(r))
            if c + 1 in col_to_basin and r < basins[col_to_basin[c + 1]]["floor"]:
                idx = int(col_to_basin[c + 1])
                rec = touch.setdefault(idx, {"side": "left", "rows": []})
                rec["rows"].append(int(r))

        touched = sorted(touch)
        if not touched:
            continue
        assert len(touched) <= 2, "openings may touch at most two basins"
        comp_kind = "breach" if len(cells) == 1 else "tunnel"
        comp = {"kind": comp_kind, "cells": cells, "touch": touch, "touched": touched}
        void_components.append(comp)

        for basin_idx in touched:
            others = [x for x in touched if int(x) != int(basin_idx)]
            basins[basin_idx]["candidates"].append(
                {
                    "kind": comp_kind,
                    "side": str(touch[basin_idx]["side"]),
                    "row": int(max(touch[basin_idx]["rows"])),
                    "dest": int(others[0]) if others else None,
                    "cells": list(cells),
                    "barrier": None,
                }
            )

    return {
        "grid": g,
        "shape": (int(h), int(w)),
        "bg": int(bg),
        "terrain": int(terrain),
        "cloud_color": int(cloud_color),
        "cloud_col": int(cloud_col),
        "surface": surface,
        "segments": segments,
        "basins": basins,
        "source_basin": int(source_basin),
        "void_components": void_components,
    }


def _trace_flow(world: dict) -> tuple[list[int], list[dict]]:
    basins = world["basins"]
    current = int(world["source_basin"])
    visited: list[int] = []
    chosen: list[dict] = []

    while current is not None:
        assert current not in visited, "flow graph must not loop"
        basin = basins[current]
        rows = [int(c["row"]) for c in basin["candidates"]]
        best = max(rows)
        winners = [c for c in basin["candidates"] if int(c["row"]) == best]
        assert len(winners) == 1, "lowest outlet must be unique"
        pick = winners[0]
        visited.append(int(current))
        chosen.append(pick)
        nxt = pick["dest"]
        if nxt is None:
            break
        current = int(nxt)

    return visited, chosen


def _paint_transition(out: np.ndarray, world: dict, cand: dict) -> None:
    basins = world["basins"]
    if cand["kind"] == "rim":
        seg = world["segments"][cand["barrier"]]
        path_row = int(seg["row"]) - 1
        out[path_row, seg["c0"] : seg["c1"] + 1] = WATER_COLOR
        if cand["dest"] is None:
            return
        dest = basins[int(cand["dest"])]
        drop_col = int(dest["c1"]) if cand["side"] == "left" else int(dest["c0"])
        out[path_row : dest["floor"], drop_col] = WATER_COLOR
        return

    for r, c in cand["cells"]:
        out[int(r), int(c)] = WATER_COLOR
    if cand["dest"] is None:
        return
    dest = basins[int(cand["dest"])]
    drop_col = int(dest["c1"]) if cand["side"] == "left" else int(dest["c0"])
    out[int(cand["row"]) : dest["floor"], drop_col] = WATER_COLOR


def _path_barrier_index(start: int, direction: int, basin_idx: int) -> int:
    return int(basin_idx + 1) if int(direction) > 0 else int(basin_idx)


def _choose_dry_side_floors(rng: random.Random, start_floor: int, count: int) -> list[int]:
    vals: list[int] = []
    current = int(start_floor) - rng.randint(0, 2)
    for _ in range(count):
        current = max(7, current - rng.randint(0, 2))
        vals.append(int(current))
    return vals


def _gen_attempt(forced_branch: str) -> np.ndarray | None:
    rng = random.Random()
    n_basins = rng.randint(3, 4)
    direction = rng.choice([-1, 1])
    start = 1 if n_basins == 3 else rng.choice([1, 2])
    if start <= 0 or start >= n_basins - 1:
        return None

    path = list(range(start, n_basins)) if direction > 0 else list(range(start, -1, -1))
    path_len = len(path)
    start_floor_max = 19 - 3 * (path_len - 1)
    if start_floor_max < 9:
        return None
    start_floor = rng.randint(9, start_floor_max)

    floors = [0] * n_basins
    floors[start] = int(start_floor)
    if direction > 0:
        for i in range(start + 1, n_basins):
            floors[i] = floors[i - 1] + 3
        left_vals = _choose_dry_side_floors(rng, start_floor, start)
        for offset, i in enumerate(range(start - 1, -1, -1)):
            floors[i] = left_vals[offset]
    else:
        for i in range(start - 1, -1, -1):
            floors[i] = floors[i + 1] + 3
        right_vals = _choose_dry_side_floors(rng, start_floor, n_basins - start - 1)
        for offset, i in enumerate(range(start + 1, n_basins)):
            floors[i] = right_vals[offset]

    basin_widths = [rng.randint(2, 4) for _ in range(n_basins)]
    barrier_widths = [rng.randint(1, 3) for _ in range(n_basins + 1)]
    barrier_tops = [0] * (n_basins + 1)
    barrier_kind: list[str | None] = [None] * (n_basins + 1)
    barrier_rows: list[int | None] = [None] * (n_basins + 1)

    outgoing_type: dict[int, str] = {}
    for pos, basin_idx in enumerate(path[:-1]):
        if pos == 0:
            outgoing_type[int(basin_idx)] = forced_branch
        else:
            outgoing_type[int(basin_idx)] = rng.choices(
                ["rim", "breach", "tunnel"],
                weights=[4, 3, 3],
                k=1,
            )[0]

    def barrier_cap(j: int) -> int:
        if j == 0:
            return int(floors[0]) - 2
        if j == n_basins:
            return int(floors[-1]) - 2
        return min(int(floors[j - 1]), int(floors[j])) - 2

    source_outgoing_barrier = _path_barrier_index(start, direction, start)
    source_spill = int(floors[start]) - 3
    opposite_barrier = int(start) if direction > 0 else int(start + 1)
    exit_barrier = int(n_basins) if direction > 0 else 0

    for j in range(n_basins + 1):
        cap = barrier_cap(j)
        if j == exit_barrier:
            barrier_tops[j] = int(floors[path[-1]]) - 2
            continue

        owner = None
        for basin_idx in path[:-1]:
            if _path_barrier_index(start, direction, basin_idx) == j:
                owner = int(basin_idx)
                break

        if owner is not None:
            spill = int(floors[owner]) - 3
            kind = outgoing_type[owner]
            if kind == "rim":
                barrier_widths[j] = rng.randint(1, 3)
                barrier_tops[j] = spill + 1
            elif kind == "breach":
                barrier_widths[j] = 1
                barrier_tops[j] = max(2, spill - rng.randint(2, 4))
                barrier_kind[j] = "breach"
                barrier_rows[j] = spill
            else:
                barrier_widths[j] = rng.randint(2, 4)
                barrier_tops[j] = max(2, spill - rng.randint(2, 4))
                barrier_kind[j] = "tunnel"
                barrier_rows[j] = spill
            if barrier_tops[j] >= cap + 1:
                barrier_tops[j] = max(2, cap - 1)
            continue

        if j == opposite_barrier and forced_branch == "rim":
            desired = min(cap, source_spill - rng.randint(1, 2))
            if desired < 4:
                return None
            extra_kind = rng.choice(["breach", "tunnel"])
            if extra_kind == "breach":
                barrier_widths[j] = 1
            else:
                barrier_widths[j] = rng.randint(2, 4)
            barrier_tops[j] = max(2, desired - rng.randint(2, 3))
            barrier_kind[j] = extra_kind
            barrier_rows[j] = int(desired)
            continue

        barrier_tops[j] = max(2, cap - rng.randint(3, 5))
        if j not in (0, n_basins) and rng.random() < 0.35 and cap >= barrier_tops[j] + 2:
            if barrier_widths[j] == 1:
                barrier_kind[j] = "breach"
                barrier_rows[j] = rng.randint(barrier_tops[j] + 1, cap)
            else:
                barrier_kind[j] = "tunnel"
                barrier_rows[j] = rng.randint(barrier_tops[j] + 1, cap)

    total_width = sum(basin_widths) + sum(barrier_widths)
    if total_width > 30:
        return None

    barrier_spans: list[tuple[int, int]] = []
    basin_spans: list[tuple[int, int]] = []
    cursor = 0
    for basin_idx in range(n_basins):
        b0 = cursor
        b1 = cursor + barrier_widths[basin_idx] - 1
        barrier_spans.append((int(b0), int(b1)))
        cursor = b1 + 1

        s0 = cursor
        s1 = cursor + basin_widths[basin_idx] - 1
        basin_spans.append((int(s0), int(s1)))
        cursor = s1 + 1
    barrier_spans.append((int(cursor), int(cursor + barrier_widths[-1] - 1)))
    width = int(cursor + barrier_widths[-1])

    surface = np.zeros(width, dtype=int)
    for j, (c0, c1) in enumerate(barrier_spans):
        surface[c0 : c1 + 1] = int(barrier_tops[j])
    for i, (c0, c1) in enumerate(basin_spans):
        surface[c0 : c1 + 1] = int(floors[i])

    height = int(surface.max()) + rng.randint(3, 5)
    if height > 30:
        return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    for c, top in enumerate(surface.tolist()):
        grid[int(top) :, int(c)] = TERRAIN_COLOR

    for j, kind in enumerate(barrier_kind):
        if kind is None:
            continue
        row = barrier_rows[j]
        assert row is not None
        c0, c1 = barrier_spans[j]
        if kind == "breach":
            if c0 != c1:
                return None
            grid[int(row), int(c0)] = BG_COLOR
        else:
            grid[int(row), int(c0) : int(c1) + 1] = BG_COLOR

    source_c0, source_c1 = basin_spans[start]
    cloud_col = rng.randint(source_c0, source_c1)
    grid[0, int(cloud_col)] = CLOUD_COLOR

    if not _semantic_checks(grid, forced_branch, direction, start):
        return None
    return grid


def _semantic_checks(grid: np.ndarray, forced_branch: str, direction: int, source_idx: int) -> bool:
    try:
        world = _parse_world(grid)
        visited, chosen = _trace_flow(world)
        out = main(grid)
    except Exception:
        return False

    basins = world["basins"]
    if int(world["source_basin"]) != int(source_idx):
        return False
    if len(visited) < 2 or len(visited) >= len(basins):
        return False
    if chosen[0]["kind"] != forced_branch:
        return False

    source_types = {str(c["kind"]) for c in basins[source_idx]["candidates"]}
    if "rim" not in source_types:
        return False
    if forced_branch == "rim" and not ({"breach", "tunnel"} & source_types):
        return False
    if forced_branch in {"breach", "tunnel"} and forced_branch not in source_types:
        return False

    diffs = [visited[i + 1] - visited[i] for i in range(len(visited) - 1)]
    if diffs and any(int(d) != int(direction) for d in diffs):
        return False

    wet_mask = out == WATER_COLOR
    if int(wet_mask.sum()) <= 0:
        return False

    touched = set(map(int, visited))
    for basin in basins:
        cavity = wet_mask[0 : basin["floor"], basin["c0"] : basin["c1"] + 1]
        wet_cells = int(cavity.sum())
        if basin["idx"] in touched and wet_cells == 0:
            return False
        if basin["idx"] not in touched and wet_cells != 0:
            return False

    for basin in basins:
        rows = [int(c["row"]) for c in basin["candidates"]]
        if rows.count(max(rows)) != 1:
            return False

    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    forced_branch = FORCED_BRANCHES[_GEN_COUNTER % len(FORCED_BRANCHES)]
    _GEN_COUNTER += 1
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(forced_branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid cloud-relay terrain input")
