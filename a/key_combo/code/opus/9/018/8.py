from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable

import numpy as np


BG = 0
PLUS_COLOR = 5
MARKER_COLORS = (1, 2, 6, 7)
GUIDE_COLORS = (3, 4, 8)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_TO_NAME = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    nxt = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            nxt += 1
            stack = [(r, c)]
            labels[r, c] = nxt
            while stack:
                rr, cc = stack.pop()
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = nxt
                        stack.append((nr, nc))
    return labels, nxt


def components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    labels, n = label_mask(mask)
    return [np.argwhere(labels == k) for k in range(1, n + 1)]


def is_thin_plus(coords: np.ndarray) -> tuple[bool, tuple[int, int] | None, int | None]:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 5 or h % 2 == 0:
        return False, None, None
    patch = np.zeros((h, w), dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    patch[local[:, 0], local[:, 1]] = True
    mid = h // 2
    expected = np.zeros_like(patch)
    expected[mid, :] = True
    expected[:, mid] = True
    if not np.array_equal(patch, expected):
        return False, None, None
    return True, (r0 + mid, c0 + mid), mid


def detect_pluses(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = get_bg(grid)
    best_color = None
    best = None
    for color in sorted(set(map(int, grid.flatten())) - {bg}):
        comps = []
        for coords in components_of_mask(grid == color):
            ok, center, arm = is_thin_plus(coords)
            if ok:
                comps.append(
                    {
                        "coords": coords,
                        "bbox": bbox_coords(coords),
                        "center": center,
                        "arm": arm,
                    }
                )
        if not comps:
            continue
        key = (len(comps), sum(len(comp["coords"]) for comp in comps))
        if best is None or key > best:
            best = key
            best_color = int(color)
            best = comps
    if best_color is None or best is None:
        raise AssertionError("expected plus components")
    best.sort(key=lambda item: item["center"])
    for idx, comp in enumerate(best):
        comp["id"] = idx
    return best_color, best


def extend_repeating_tile(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    protect: set[int],
) -> None:
    h, w = tile.shape
    dr, dc = direction
    grid_h, grid_w = out.shape

    def put(rr: int, cc: int, value: int) -> None:
        if value == bg:
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(value)

    if (dr, dc) == (0, 1):
        for k, tc in enumerate(range(c0 + w, grid_w)):
            col_idx = k % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (0, -1):
        for k, tc in enumerate(range(c0 - 1, -1, -1), start=1):
            col_idx = (-k) % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (1, 0):
        for k, tr in enumerate(range(r0 + h, grid_h)):
            row_idx = k % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    elif (dr, dc) == (-1, 0):
        for k, tr in enumerate(range(r0 - 1, -1, -1), start=1):
            row_idx = (-k) % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    else:
        raise ValueError("bad direction")


def analyze_grid(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    plus_color, pluses = detect_pluses(g)
    plus_id_map = np.full((h, w), -1, dtype=int)
    for plus in pluses:
        for r, c in plus["coords"]:
            plus_id_map[int(r), int(c)] = int(plus["id"])

    guides = []
    for pos in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        color = int(g[pos])
        if color != bg and color != plus_color:
            guides.append({"pos": pos, "color": color})

    marker_mask = (g != bg) & (g != plus_color)
    for guide in guides:
        rr, cc = guide["pos"]
        marker_mask[rr, cc] = False

    markers = []
    for coords in components_of_mask(marker_mask):
        colors = {int(g[r, c]) for r, c in coords}
        if len(colors) != 1:
            raise AssertionError("marker component must be monochrome")
        if len(coords) not in (1, 2):
            raise AssertionError("marker component area must be 1 or 2")
        side_hits = set()
        plus_hits = set()
        for r, c in coords:
            r = int(r)
            c = int(c)
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and plus_id_map[nr, nc] >= 0:
                    plus_hits.add(int(plus_id_map[nr, nc]))
                    side_hits.add((r - nr, c - nc))
        if len(plus_hits) != 1 or len(side_hits) != 1:
            raise AssertionError("marker must attach to exactly one plus on one side")
        direction = next(iter(side_hits))
        r0, c0, r1, c1 = bbox_coords(coords)
        if direction[0] != 0:
            if c0 != c1 or (r1 - r0 + 1) != len(coords):
                raise AssertionError("vertical marker shape mismatch")
        else:
            if r0 != r1 or (c1 - c0 + 1) != len(coords):
                raise AssertionError("horizontal marker shape mismatch")
        markers.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "plus_id": next(iter(plus_hits)),
                "direction": direction,
                "color": next(iter(colors)),
            }
        )

    assignments = {}
    for guide in guides:
        r, c = guide["pos"]
        dists = [abs(r - pr) + abs(c - pc) for pr, pc in (plus["center"] for plus in pluses)]
        best = min(dists)
        if dists.count(best) != 1:
            raise AssertionError("guide must have a unique nearest plus")
        assignments[guide["pos"]] = {
            "plus_id": int(dists.index(best)),
            "distance": int(best),
            "color": int(guide["color"]),
        }

    target_color = {}
    for plus in pluses:
        chosen = [
            info for info in assignments.values() if int(info["plus_id"]) == int(plus["id"])
        ]
        if not chosen:
            continue
        chosen.sort(key=lambda item: item["distance"])
        target_color[int(plus["id"])] = int(chosen[0]["color"])

    return {
        "bg": bg,
        "plus_color": plus_color,
        "pluses": pluses,
        "plus_id_map": plus_id_map,
        "guides": guides,
        "markers": markers,
        "assignments": assignments,
        "target_color": target_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each plus's attached marker blocks from corner guides, then extend them as repeating rays.

    Input structure:
    - The background is black.
    - Several large monochrome thin '+' shapes share one main color and do not touch.
    - Small 1- or 2-pixel monochrome marker blocks sit immediately above, below, left, or right of a plus arm.
    - Two or three single guide pixels occupy grid corners and are not the plus color.

    Rule:
    - Find every guide pixel and assign it to the unique nearest plus center by Manhattan distance.
    - Recolor all marker blocks attached to a targeted plus with that guide color; if multiple guides target the same plus, use the closer guide.
    - Remove the guide pixels to background.
    - Marker blocks on untargeted pluses keep their original color.
    - For every marker block, determine which side of its plus it touches and repeat the whole marker tile periodically away from that plus until the grid edge, never overwriting plus pixels.

    Key insight:
    - The corner guides are only remote color sources: they first choose a plus by nearest-center distance, and that transferred marker color then controls the outward ray color.
    """
    info = analyze_grid(np.array(grid, dtype=int))
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(info["bg"])
    plus_color = int(info["plus_color"])

    for guide in info["guides"]:
        rr, cc = guide["pos"]
        out[rr, cc] = bg

    for marker in info["markers"]:
        color = int(info["target_color"].get(marker["plus_id"], marker["color"]))
        for r, c in marker["coords"]:
            out[int(r), int(c)] = color

    for marker in info["markers"]:
        r0, c0, r1, c1 = marker["bbox"]
        tile = crop_bbox(out, (r0, c0, r1, c1))
        extend_repeating_tile(
            out,
            tile,
            r0,
            c0,
            marker["direction"],
            bg=bg,
            protect={plus_color},
        )
    return out


def plus_cells(center: tuple[int, int], arm: int) -> set[tuple[int, int]]:
    r, c = center
    cells = {(r, cc) for cc in range(c - arm, c + arm + 1)}
    cells.update({(rr, c) for rr in range(r - arm, r + arm + 1)})
    return cells


def marker_seed_cells(center: tuple[int, int], arm: int, direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = center
    if direction == (-1, 0):
        return [(r - arm - offset, c) for offset in range(length, 0, -1)]
    if direction == (1, 0):
        return [(r + arm + offset, c) for offset in range(1, length + 1)]
    if direction == (0, -1):
        return [(r, c - arm - offset) for offset in range(length, 0, -1)]
    if direction == (0, 1):
        return [(r, c + arm + offset) for offset in range(1, length + 1)]
    raise ValueError("bad direction")


def projected_ray_cells(center: tuple[int, int], arm: int, direction: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    h, w = shape
    if direction == (-1, 0):
        return {(rr, c) for rr in range(0, r - arm)}
    if direction == (1, 0):
        return {(rr, c) for rr in range(r + arm + 1, h)}
    if direction == (0, -1):
        return {(r, cc) for cc in range(0, c - arm)}
    if direction == (0, 1):
        return {(r, cc) for cc in range(c + arm + 1, w)}
    raise ValueError("bad direction")


def choose_centers(rng: np.random.Generator, h: int, w: int, n: int) -> list[dict] | None:
    pluses: list[dict] = []
    used_rows: list[int] = []
    used_cols: list[int] = []
    for _ in range(n):
        placed = False
        for _try in range(300):
            arm = int(rng.choice([2, 2, 3]))
            r = int(rng.integers(arm + 3, h - arm - 3))
            c = int(rng.integers(arm + 3, w - arm - 3))
            if any(abs(r - rr) < 4 for rr in used_rows):
                continue
            if any(abs(c - cc) < 4 for cc in used_cols):
                continue
            cells = plus_cells((r, c), arm)
            ok = True
            for other in pluses:
                if cells & other["cells"]:
                    ok = False
                    break
                if any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in other["cells"]):
                    ok = False
                    break
            if not ok:
                continue
            pluses.append({"center": (r, c), "arm": arm, "cells": cells})
            used_rows.append(r)
            used_cols.append(c)
            placed = True
            break
        if not placed:
            return None
    return pluses


def choose_markers(
    rng: np.random.Generator,
    h: int,
    w: int,
    pluses: list[dict],
) -> list[dict] | None:
    n = len(pluses)
    counts = [1] * n
    if n == 3 and rng.random() < 0.8:
        counts[int(rng.integers(0, n))] = 2
    total = sum(counts)
    colors = list(rng.permutation(np.array(MARKER_COLORS, dtype=int)))[:total]
    lengths = [int(rng.choice([1, 2])) for _ in range(total)]
    if total >= 2:
        lengths[0] = 1
        lengths[1] = 2
    plus_cells_all = set().union(*(plus["cells"] for plus in pluses))
    ray_cells_all: set[tuple[int, int]] = set()
    markers: list[dict] = []
    color_idx = 0

    order = list(range(n))
    rng.shuffle(order)
    plus_to_dirs: dict[int, list[tuple[int, int]]] = {}

    for plus_idx in order:
        plus = pluses[plus_idx]
        taken_dirs: list[tuple[int, int]] = []
        for _ in range(counts[plus_idx]):
            need_len = lengths[color_idx]
            candidates = DIRS[:]
            rng.shuffle(candidates)
            placed = None
            for direction in candidates:
                if direction in taken_dirs:
                    continue
                seed = marker_seed_cells(plus["center"], plus["arm"], direction, need_len)
                if not all(0 <= r < h and 0 <= c < w for r, c in seed):
                    continue
                if any((r, c) in plus_cells_all for r, c in seed):
                    continue
                ray = projected_ray_cells(plus["center"], plus["arm"], direction, (h, w))
                if ray & (plus_cells_all - plus["cells"]):
                    continue
                if ray & ray_cells_all:
                    continue
                if any(cell in ray_cells_all for cell in seed):
                    continue
                placed = {
                    "plus_index": plus_idx,
                    "direction": direction,
                    "length": need_len,
                    "seed": seed,
                    "ray": ray,
                    "color": int(colors[color_idx]),
                }
                break
            if placed is None:
                return None
            markers.append(placed)
            taken_dirs.append(placed["direction"])
            plus_to_dirs.setdefault(plus_idx, []).append(placed["direction"])
            ray_cells_all |= placed["ray"]
            color_idx += 1
    return markers


def choose_guides(
    rng: np.random.Generator,
    h: int,
    w: int,
    pluses: list[dict],
) -> list[dict] | None:
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    infos = []
    for pos in corners:
        r, c = pos
        dists = [abs(r - pr) + abs(c - pc) for pr, pc in (plus["center"] for plus in pluses)]
        best = min(dists)
        if dists.count(best) != 1:
            continue
        infos.append(
            {
                "pos": pos,
                "plus_index": int(dists.index(best)),
                "distance": int(best),
            }
        )
    combos = []
    for k in (2, 3):
        if len(infos) < k:
            continue
        for combo in combinations(infos, k):
            targeted = {item["plus_index"] for item in combo}
            if len(targeted) >= len(pluses):
                continue
            has_duplicate = len(targeted) < len(combo)
            combos.append((has_duplicate, combo))
    if not combos:
        return None
    prefer_duplicate = rng.random() < 0.55
    filtered = [combo for dup, combo in combos if dup == prefer_duplicate]
    if not filtered:
        filtered = [combo for _, combo in combos]
    chosen = list(filtered[int(rng.integers(0, len(filtered)))])

    if rng.random() < 0.7:
        guide_colors = list(rng.permutation(np.array(GUIDE_COLORS, dtype=int)))[: len(chosen)]
    else:
        guide_colors = [int(rng.choice(GUIDE_COLORS)) for _ in chosen]

    out = []
    for info, color in zip(chosen, guide_colors):
        out.append(
            {
                "pos": info["pos"],
                "plus_index": info["plus_index"],
                "distance": info["distance"],
                "color": int(color),
            }
        )
    return out


def build_input(
    h: int,
    w: int,
    pluses: list[dict],
    markers: list[dict],
    guides: list[dict],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for plus in pluses:
        for r, c in plus["cells"]:
            grid[r, c] = PLUS_COLOR
    for marker in markers:
        for r, c in marker["seed"]:
            grid[r, c] = int(marker["color"])
    for guide in guides:
        r, c = guide["pos"]
        grid[r, c] = int(guide["color"])
    return grid


def validate_instance(
    inp: np.ndarray,
    pluses: list[dict],
    markers: list[dict],
    guides: list[dict],
) -> bool:
    try:
        info = analyze_grid(inp)
    except Exception:
        return False

    if len(info["pluses"]) != len(pluses):
        return False
    if len(info["markers"]) != len(markers):
        return False
    if len(info["guides"]) != len(guides):
        return False
    if len(set(marker["color"] for marker in markers)) != len(markers):
        return False
    if not info["target_color"]:
        return False
    if len(info["target_color"]) >= len(info["pluses"]):
        return False
    if not any(len([g for g in guides if g["plus_index"] == idx]) >= 2 for idx in range(len(pluses))):
        if len(guides) == len(info["target_color"]):
            return False
    out = main(inp)
    if np.array_equal(inp, out):
        return False
    changed_guides = all(out[r, c] == 0 for r, c in (guide["pos"] for guide in guides))
    if not changed_guides:
        return False
    for marker in info["markers"]:
        plus = info["pluses"][marker["plus_id"]]
        center_r, center_c = plus["center"]
        mr0, mc0, mr1, mc1 = marker["bbox"]
        if marker["direction"] == (-1, 0):
            edge_cells = {(rr, center_c) for rr in range(0, center_r - plus["arm"])}
        elif marker["direction"] == (1, 0):
            edge_cells = {(rr, center_c) for rr in range(center_r + plus["arm"] + 1, out.shape[0])}
        elif marker["direction"] == (0, -1):
            edge_cells = {(center_r, cc) for cc in range(0, center_c - plus["arm"])}
        else:
            edge_cells = {(center_r, cc) for cc in range(center_c + plus["arm"] + 1, out.shape[1])}
        color = int(out[mr0, mc0])
        if not edge_cells:
            return False
        if any(int(out[r, c]) not in (0, color) and int(out[r, c]) != PLUS_COLOR for r, c in edge_cells):
            return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        h = int(rng.integers(16, 23))
        w = int(rng.integers(16, 23))
        plus_count = int(rng.choice([3, 3, 4]))
        pluses = choose_centers(rng, h, w, plus_count)
        if pluses is None:
            continue
        markers = choose_markers(rng, h, w, pluses)
        if markers is None:
            continue
        guides = choose_guides(rng, h, w, pluses)
        if guides is None:
            continue
        inp = build_input(h, w, pluses, markers, guides)
        if validate_instance(inp, pluses, markers, guides):
            return inp
    raise RuntimeError("generate_input: failed after retries")
