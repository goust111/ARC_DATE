from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


Coord = tuple[int, int]

BG = 0
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_minimap(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    h, w = grid.shape
    assert int(grid[0, 0]) != bg, "minimap must touch the top-left corner"

    row_gap = None
    for r in range(1, h):
        if np.all(grid[r, :] == bg):
            row_gap = r
            break
    col_gap = None
    for c in range(1, w):
        if np.all(grid[:, c] == bg):
            col_gap = c
            break
    assert row_gap is not None and col_gap is not None, "minimap/background separator not found"

    mini = grid[:row_gap, :col_gap].copy()
    assert np.all(np.any(mini != bg, axis=1)), "every minimap row must contain a non-bg pixel"
    assert np.all(np.any(mini != bg, axis=0)), "every minimap col must contain a non-bg pixel"
    return (0, 0, row_gap - 1, col_gap - 1), mini


def _find_objects(
    grid: np.ndarray, bg: int, ignore_bbox: tuple[int, int, int, int] | None = None
) -> tuple[list[dict], list[dict]]:
    anchors: list[dict] = []
    singles: list[dict] = []
    ignore = None
    if ignore_bbox is not None:
        ir0, ic0, ir1, ic1 = ignore_bbox
        ignore = (int(ir0), int(ic0), int(ir1), int(ic1))
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        for coords in _components_mask(grid == color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            if ignore is not None and ignore[0] <= r0 <= r1 <= ignore[2] and ignore[1] <= c0 <= c1 <= ignore[3]:
                continue
            area = int(coords.shape[0])
            if area == 9 and (r1 - r0 + 1, c1 - c0 + 1) == (3, 3):
                if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                    anchors.append(
                        {
                            "color": color,
                            "bbox": (r0, c0, r1, c1),
                            "center": (r0 + 1, c0 + 1),
                        }
                    )
                    continue
            if area == 1:
                rr, cc = map(int, coords[0])
                singles.append({"color": color, "pos": (rr, cc)})
    return anchors, singles


def _minimap_pixels(minimap: np.ndarray) -> list[tuple[int, int, int]]:
    pts: list[tuple[int, int, int]] = []
    for r in range(minimap.shape[0]):
        for c in range(minimap.shape[1]):
            color = int(minimap[r, c])
            if color != BG:
                pts.append((r, c, color))
    assert pts, "minimap must contain non-bg pixels"
    return pts


def _mapping_candidates(
    minimap: np.ndarray, existing_anchors: list[dict], shape: tuple[int, int]
) -> list[tuple[int, int, int]]:
    h, w = shape
    pts = _minimap_pixels(minimap)
    existing = [(a["center"][0], a["center"][1], int(a["color"])) for a in existing_anchors]
    assert len(existing) >= 2, "need at least two existing anchors"

    raw: set[tuple[int, int, int]] = set()
    max_s = max(h, w)
    for mr, mc, color in pts:
        for ar, ac, acol in existing:
            if acol != color:
                continue
            for s in range(1, max_s + 1):
                raw.add((s, ar - mr * s, ac - mc * s))

    valid: list[tuple[int, int, int]] = []
    for s, dr, dc in raw:
        if dr < minimap.shape[0] + 2 or dc < minimap.shape[1] + 2:
            continue
        centers: dict[Coord, int] = {}
        ok = True
        for mr, mc, color in pts:
            rr = dr + mr * s
            cc = dc + mc * s
            if not (1 <= rr < h - 1 and 1 <= cc < w - 1):
                ok = False
                break
            key = (int(rr), int(cc))
            if key in centers:
                ok = False
                break
            centers[key] = color
        if not ok:
            continue
        for ar, ac, color in existing:
            if centers.get((ar, ac)) != color:
                ok = False
                break
        if ok:
            valid.append((int(s), int(dr), int(dc)))
    return sorted(set(valid))


def _full_anchors(minimap: np.ndarray, s: int, dr: int, dc: int) -> list[dict]:
    anchors: list[dict] = []
    for mr, mc, color in _minimap_pixels(minimap):
        rr = dr + mr * s
        cc = dc + mc * s
        anchors.append(
            {
                "mini": (mr, mc),
                "color": color,
                "center": (int(rr), int(cc)),
                "bbox": (int(rr - 1), int(cc - 1), int(rr + 1), int(cc + 1)),
            }
        )
    return anchors


def _anchor_lookup(anchors: list[dict]) -> dict[Coord, dict]:
    return {tuple(a["center"]): a for a in anchors}


def _assign_scatter(single: dict, anchors: list[dict]) -> tuple[dict, str]:
    pr, pc = single["pos"]
    color = int(single["color"])
    same = [a for a in anchors if int(a["color"]) == color]
    candidates: list[tuple[dict, str, int]] = []

    for anchor in same:
        ar, ac = anchor["center"]
        if pr == ar and pc < ac:
            if not any(a2["center"][0] == ar and pc < a2["center"][1] < ac for a2 in same):
                candidates.append((anchor, "W", ac - pc))
        if pr == ar and pc > ac:
            if not any(a2["center"][0] == ar and ac < a2["center"][1] < pc for a2 in same):
                candidates.append((anchor, "E", pc - ac))
        if pc == ac and pr < ar:
            if not any(a2["center"][1] == ac and pr < a2["center"][0] < ar for a2 in same):
                candidates.append((anchor, "N", ar - pr))
        if pc == ac and pr > ar:
            if not any(a2["center"][1] == ac and ar < a2["center"][0] < pr for a2 in same):
                candidates.append((anchor, "S", pr - ar))

    assert len(candidates) == 1, "each scatter pixel must belong to exactly one same-color anchor ray"
    anchor, direction, _dist = candidates[0]
    return anchor, direction


def _compress_groups(scatter: list[dict], anchors: list[dict]) -> dict[tuple[Coord, str], list[dict]]:
    groups: dict[tuple[Coord, str], list[dict]] = defaultdict(list)
    for px in scatter:
        anchor, direction = _assign_scatter(px, anchors)
        ar, ac = anchor["center"]
        pr, pc = px["pos"]
        dist = abs(pr - ar) + abs(pc - ac)
        groups[(anchor["center"], direction)].append({"pixel": px, "dist": dist, "anchor": anchor})
    for items in groups.values():
        items.sort(key=lambda item: int(item["dist"]))
    return groups


def _destination(center: Coord, direction: str, index: int) -> Coord:
    r, c = center
    if direction == "N":
        return (r - 2 - index, c)
    if direction == "S":
        return (r + 2 + index, c)
    if direction == "W":
        return (r, c - 2 - index)
    if direction == "E":
        return (r, c + 2 + index)
    raise ValueError(direction)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode anchor targets from the top-left minimap, complete the missing 3x3 anchors, then compact same-color ray pixels against those anchors.

    Input structure:
    - Black is the background.
    - A small top-left minimap contains colored and black cells, and is separated from the main area by at least one full background row and column.
    - The main area contains some existing 3x3 solid monochrome anchor blocks plus isolated single pixels of matching colors placed on horizontal or vertical rays through anchor centers.

    Rule:
    - Read the minimap as anchor-center coordinates after an integer scale-and-offset mapping from minimap cells to main-grid anchor centers.
    - Infer that mapping from the already present 3x3 anchors, then place every missing 3x3 anchor whose minimap cell is colored.
    - For each isolated pixel, find the unique same-color anchor whose center shares the pixel's row or column and has no other same-color anchor center between them.
    - Remove the minimap and original isolated pixels, and repack each ray's pixels tightly against the corresponding anchor edge in near-to-far order.

    Key insight:
    The minimap does not tell you how to recolor anything directly; it first decodes the missing anchor positions, and only those completed anchors make the later ray-compaction step unambiguous.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    assert bg == BG, "task uses black background"

    mini_bbox, minimap = _find_minimap(g, bg)
    existing_anchors, singles = _find_objects(g, bg, ignore_bbox=mini_bbox)
    candidates = _mapping_candidates(minimap, existing_anchors, g.shape)
    assert len(candidates) == 1, "minimap-to-anchor mapping must be unique"
    s, dr, dc = candidates[0]

    anchors = _full_anchors(minimap, s, dr, dc)
    anchor_by_center = _anchor_lookup(anchors)
    existing_centers = {tuple(a["center"]) for a in existing_anchors}
    for center in existing_centers:
        assert center in anchor_by_center, "existing anchors must agree with the decoded minimap"

    out = g.copy()
    r0, c0, r1, c1 = mini_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = bg
    for px in singles:
        pr, pc = px["pos"]
        out[pr, pc] = bg

    for anchor in anchors:
        ar, ac = anchor["center"]
        out[ar - 1 : ar + 2, ac - 1 : ac + 2] = int(anchor["color"])

    groups = _compress_groups(singles, anchors)
    for (center, direction), items in groups.items():
        color = int(items[0]["pixel"]["color"])
        for idx, _item in enumerate(items):
            rr, cc = _destination(center, direction, idx)
            assert _in_bounds(out.shape, rr, cc)
            assert int(out[rr, cc]) == bg
            out[rr, cc] = color

    return out


def _choose_positions(rows: int, cols: int, count: int) -> list[Coord] | None:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(cells)
    chosen: list[Coord] = []
    for r, c in cells:
        if any(rr == r or cc == c for rr, cc in chosen):
            continue
        chosen.append((r, c))
        if len(chosen) == count:
            break
    if len(chosen) == count:
        return chosen
    return None


def _sample_minimap() -> tuple[np.ndarray, list[dict]] | None:
    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    count = random.randint(max(4, rows), min(rows * cols - 1, 7))
    for _ in range(80):
        coords = random.sample([(r, c) for r in range(rows) for c in range(cols)], count)
        row_hits = {r for r, _c in coords}
        col_hits = {c for _r, c in coords}
        if len(row_hits) != rows or len(col_hits) != cols:
            continue

        color_count = random.randint(2, min(4, count))
        colors = random.sample(PALETTE, color_count)
        random.shuffle(coords)
        assigned: list[dict] = []
        by_color: dict[int, list[Coord]] = {color: [] for color in colors}
        ok = True

        for idx, (r, c) in enumerate(coords):
            if idx < color_count:
                color = colors[idx]
            else:
                options = [
                    color
                    for color in colors
                    if all(rr != r and cc != c for rr, cc in by_color[color])
                ]
                if not options:
                    ok = False
                    break
                color = random.choice(options)
            by_color[color].append((r, c))
            assigned.append({"mini": (r, c), "color": color})

        if not ok:
            continue
        if count > color_count and not any(len(by_color[color]) >= 2 for color in colors):
            continue

        mini = np.zeros((rows, cols), dtype=int)
        for item in assigned:
            mr, mc = item["mini"]
            mini[mr, mc] = int(item["color"])
        return mini, assigned
    return None


def _anchor_cells(center: Coord) -> set[Coord]:
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _ray_cells(
    center: Coord, direction: str, shape: tuple[int, int], blocked: set[Coord]
) -> list[Coord]:
    r, c = center
    if direction == "N":
        dr, dc = -1, 0
        rr, cc = r - 2, c
    elif direction == "S":
        dr, dc = 1, 0
        rr, cc = r + 2, c
    elif direction == "W":
        dr, dc = 0, -1
        rr, cc = r, c - 2
    else:
        dr, dc = 0, 1
        rr, cc = r, c + 2

    cells: list[Coord] = []
    while _in_bounds(shape, rr, cc) and (rr, cc) not in blocked:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _direction_capacity(
    anchor: dict, direction: str, shape: tuple[int, int], blocked: set[Coord]
) -> tuple[int, list[Coord]]:
    cells = _ray_cells(anchor["center"], direction, shape, blocked)
    if len(cells) >= 5:
        return 2, cells
    if len(cells) >= 2:
        return 1, cells
    return 0, cells


def _pick_existing(anchors: list[dict]) -> set[Coord] | None:
    total = len(anchors)
    indices = list(range(total))
    random.shuffle(indices)
    existing_n = random.randint(2, total - 2)
    chosen = indices[:existing_n]
    centers = {tuple(anchors[i]["center"]) for i in chosen}
    rows = {anchors[i]["mini"][0] for i in chosen}
    cols = {anchors[i]["mini"][1] for i in chosen}
    if len(rows) + len(cols) < 3:
        return None
    return centers


def _build_assignment(
    anchors: list[dict], existing_centers: set[Coord], shape: tuple[int, int]
) -> dict[tuple[int, str], int] | None:
    blocked = set()
    for anchor in anchors:
        blocked |= _anchor_cells(anchor["center"])

    per_color_dirs: dict[int, list[dict]] = defaultdict(list)
    for idx, anchor in enumerate(anchors):
        for direction in ("N", "S", "W", "E"):
            cap, cells = _direction_capacity(anchor, direction, shape, blocked)
            if cap > 0:
                per_color_dirs[int(anchor["color"])].append(
                    {
                        "idx": idx,
                        "direction": direction,
                        "cap": cap,
                        "cells": cells,
                        "missing": tuple(anchor["center"]) not in existing_centers,
                    }
                )

    targets = {color: random.randint(3, 5) for color in per_color_dirs}
    if set(targets) != {int(anchor["color"]) for anchor in anchors}:
        return None

    assignment: dict[tuple[int, str], int] = {}
    for color, infos in per_color_dirs.items():
        if sum(int(info["cap"]) for info in infos) < targets[color]:
            return None

    missing_infos = [info for infos in per_color_dirs.values() for info in infos if info["missing"]]
    if not missing_infos:
        return None
    reserve = random.choice(missing_infos)
    reserve_key = (int(reserve["idx"]), str(reserve["direction"]))
    assignment[reserve_key] = 1
    targets[int(anchors[reserve["idx"]]["color"])] -= 1

    for color, infos in per_color_dirs.items():
        order = infos[:]
        random.shuffle(order)
        for info in order:
            if targets[color] <= 0:
                break
            key = (int(info["idx"]), str(info["direction"]))
            if assignment.get(key, 0) == 0:
                assignment[key] = 1
                targets[color] -= 1
        for info in order:
            if targets[color] <= 0:
                break
            key = (int(info["idx"]), str(info["direction"]))
            if assignment.get(key, 0) < int(info["cap"]):
                assignment[key] = assignment.get(key, 0) + 1
                targets[color] -= 1
        if targets[color] > 0:
            return None
    return assignment


def _place_scatter(
    grid: np.ndarray,
    minimap: np.ndarray,
    anchors: list[dict],
    existing_centers: set[Coord],
    assignment: dict[tuple[int, str], int],
) -> list[dict] | None:
    full_occ = np.zeros_like(grid, dtype=bool)
    full_occ[: minimap.shape[0], : minimap.shape[1]] = minimap != BG
    for anchor in anchors:
        for rr, cc in _anchor_cells(anchor["center"]):
            full_occ[rr, cc] = True

    placed = np.zeros_like(grid, dtype=bool)
    scatter: list[dict] = []

    for (idx, direction), count in assignment.items():
        anchor = anchors[idx]
        cells = _ray_cells(anchor["center"], direction, grid.shape, {
            (r, c) for r, c in zip(*np.where(full_occ), strict=False)
        })
        if count == 1:
            needed = [1]
        else:
            needed = [2, 4]
        if max(needed) >= len(cells):
            return None
        picks = [cells[i] for i in needed[:count]]
        for rr, cc in picks:
            if full_occ[rr, cc] or placed[rr, cc]:
                return None
            for nr, nc in _neighbors4(rr, cc):
                if _in_bounds(grid.shape, nr, nc) and (full_occ[nr, nc] or placed[nr, nc]):
                    return None
            placed[rr, cc] = True
            scatter.append(
                {
                    "color": int(anchor["color"]),
                    "pos": (int(rr), int(cc)),
                    "anchor": tuple(anchor["center"]),
                    "direction": direction,
                    "anchor_missing": tuple(anchor["center"]) not in existing_centers,
                }
            )
    return scatter


def _render_input(
    shape: tuple[int, int],
    minimap: np.ndarray,
    anchors: list[dict],
    existing_centers: set[Coord],
    scatter: list[dict],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[: minimap.shape[0], : minimap.shape[1]] = minimap
    for anchor in anchors:
        if tuple(anchor["center"]) not in existing_centers:
            continue
        ar, ac = anchor["center"]
        grid[ar - 1 : ar + 2, ac - 1 : ac + 2] = int(anchor["color"])
    for px in scatter:
        rr, cc = px["pos"]
        grid[rr, cc] = int(px["color"])
    return grid


def _semantic_checks(
    inp: np.ndarray,
    minimap: np.ndarray,
    anchors: list[dict],
    existing_centers: set[Coord],
    scatter: list[dict],
) -> bool:
    try:
        mini_bbox, found_mini = _find_minimap(inp, BG)
        if mini_bbox != (0, 0, minimap.shape[0] - 1, minimap.shape[1] - 1):
            return False
        if not np.array_equal(found_mini, minimap):
            return False

        existing, singles = _find_objects(inp, BG, ignore_bbox=mini_bbox)
        if len(existing) < 2 or len(anchors) - len(existing) < 2:
            return False
        candidates = _mapping_candidates(minimap, existing, inp.shape)
        if len(candidates) != 1:
            return False

        full = _full_anchors(minimap, *candidates[0])
        groups = _compress_groups(singles, full)
        if len(singles) != len(scatter):
            return False
        if not any(px["anchor_missing"] for px in scatter):
            return False
        counts = Counter(int(px["color"]) for px in scatter)
        for color in {int(a["color"]) for a in anchors}:
            if not (3 <= counts[color] <= 5):
                return False

        out = main(inp)
        if np.array_equal(out, inp):
            return False
        if np.any(out[: minimap.shape[0], : minimap.shape[1]] != BG):
            return False

        for anchor in full:
            ar, ac = anchor["center"]
            patch = out[ar - 1 : ar + 2, ac - 1 : ac + 2]
            if patch.shape != (3, 3) or not np.all(patch == int(anchor["color"])):
                return False

        for (center, direction), items in groups.items():
            color = int(items[0]["pixel"]["color"])
            for idx, _item in enumerate(items):
                rr, cc = _destination(center, direction, idx)
                if int(out[rr, cc]) != color:
                    return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_minimap()
    if sampled is None:
        return None
    minimap, anchor_specs = sampled

    s = random.choice([5, 6])
    dr = minimap.shape[0] + random.randint(2, 5)
    dc = minimap.shape[1] + random.randint(2, 5)
    anchors = _full_anchors(minimap, s, dr, dc)

    max_r = max(anchor["center"][0] for anchor in anchors)
    max_c = max(anchor["center"][1] for anchor in anchors)
    h_min = max_r + 5
    w_min = max_c + 5
    if h_min > 30 or w_min > 30:
        return None
    h = random.randint(h_min, min(30, h_min + 3))
    w = random.randint(w_min, min(30, w_min + 3))

    existing_centers = _pick_existing(anchors)
    if existing_centers is None:
        return None

    assignment = _build_assignment(anchors, existing_centers, (h, w))
    if assignment is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    scatter = _place_scatter(grid, minimap, anchors, existing_centers, assignment)
    if scatter is None:
        return None

    inp = _render_input((h, w), minimap, anchors, existing_centers, scatter)
    if not _semantic_checks(inp, minimap, anchors, existing_centers, scatter):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input for puzzle 8")
