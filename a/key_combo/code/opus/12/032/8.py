from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
CTRL = 2
GREY = 5
COLOR_POOL = (1, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
QUADRANTS = ("TL", "TR", "BL", "BR")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_legends(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    legends: list[dict] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if r0 not in (0, h - 3) and c0 not in (0, w - 3):
                continue
            box = g[r0 : r0 + 3, c0 : c0 + 3]
            if np.any(box == BG):
                continue
            grey_locs = np.argwhere(box == GREY)
            if len(grey_locs) != 2:
                continue
            if not any(int(r) == 1 and int(c) == 1 for r, c in grey_locs):
                continue
            others = sorted(set(map(int, box.ravel())) - {GREY})
            if len(others) != 1:
                continue
            fill = others[0]
            other_grey = [(int(r), int(c)) for r, c in grey_locs if (int(r), int(c)) != (1, 1)]
            if len(other_grey) != 1:
                continue
            ir, ic = other_grey[0]
            dr, dc = ir - 1, ic - 1
            if abs(dr) + abs(dc) != 1:
                continue
            legends.append(
                {
                    "bbox": (r0, c0, r0 + 2, c0 + 2),
                    "fill": fill,
                    "dir": (dr, dc),
                }
            )
    legends.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
    return legends


def _control_pixels(g: np.ndarray, legend_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = g.shape
    pts = []
    for r, c in np.argwhere(g == CTRL):
        r = int(r)
        c = int(c)
        if not (r == 0 or c == 0 or r == h - 1 or c == w - 1):
            continue
        if legend_mask[r, c]:
            continue
        pts.append((r, c))
    return sorted(pts)


def _object_mask(g: np.ndarray, legend_mask: np.ndarray, controls: list[tuple[int, int]]) -> np.ndarray:
    mask = g != BG
    mask[legend_mask] = False
    for r, c in controls:
        mask[r, c] = False
    comps = [comp for comp in _components(mask) if all(0 < r < g.shape[0] - 1 and 0 < c < g.shape[1] - 1 for r, c in comp)]
    assert comps
    best = max(comps, key=len)
    out = np.zeros_like(mask, dtype=bool)
    for r, c in best:
        out[r, c] = True
    return out


def _split_lines(controls: list[tuple[int, int]], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r_split = None
    c_split = None
    for r, c in controls:
        if r == 0 or r == h - 1:
            c_split = c
        if c == 0 or c == w - 1:
            r_split = r
    assert r_split is not None and c_split is not None
    return int(r_split), int(c_split)


def _fragment_components(g: np.ndarray, object_mask: np.ndarray, r_split: int, c_split: int) -> list[list[tuple[int, int]]]:
    frag_mask = object_mask.copy()
    frag_mask[r_split, :] = False
    frag_mask[:, c_split] = False
    comps = _components(frag_mask)
    assert len(comps) == 3
    return comps


def _dominant_color(g: np.ndarray, comp: list[tuple[int, int]]) -> int:
    counts: Counter[int] = Counter(int(g[r, c]) for r, c in comp if int(g[r, c]) != BG)
    assert counts
    best = max(counts.values())
    tied = {color for color, count in counts.items() if count == best}
    if len(tied) == 1:
        return next(iter(tied))
    r0, c0, r1, c1 = _bbox(comp)
    comp_set = set(comp)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in comp_set and int(g[r, c]) in tied:
                return int(g[r, c])
    raise AssertionError("unreachable tie-break")


def _slide(comp: list[tuple[int, int]], colors: dict[tuple[int, int], int], move: tuple[int, int], shape: tuple[int, int], occupied: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    h, w = shape
    dr, dc = move
    cur = list(comp)
    while True:
        nxt = [(r + dr, c + dc) for r, c in cur]
        ok = True
        for nr, nc in nxt:
            if not (0 <= nr < h and 0 <= nc < w):
                ok = False
                break
            if (nr, nc) in occupied:
                ok = False
                break
        if not ok:
            break
        cur = nxt
    moved: dict[tuple[int, int], int] = {}
    for (old_r, old_c), (nr, nc) in zip(comp, cur):
        moved[(nr, nc)] = int(colors[(old_r, old_c)])
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    """Cut a multicolor object by border rays, decode directions from same-color legends, and keep the unmatched fragment fixed.

    Input structure:
    - A black background contains one interior multicolor 4-connected object plus exactly two red border control pixels.
    - Two 3x3 border legends sit on edges not used by the control pixels; each legend is filled with one object color and contains a grey center plus one grey cardinal indicator.
    - The control row and control column conceptually cut the object into exactly three fragments.

    Rule:
    - Use the two red edge controls to identify one horizontal cut row and one vertical cut column.
    - Split the interior object into fragments by those ray positions, then compute each fragment's dominant non-black color; ties break by first appearance in the fragment bbox under row-major scan.
    - If a fragment's dominant color matches a legend fill color, slide that whole fragment in the legend's indicated cardinal direction until the next step would leave the grid or hit an already fixed fragment.
    - The unique fragment whose dominant color has no legend is the outlier and stays in place.
    - Process the outlier first, then matched fragments in border-legend order from top to bottom and left to right. Remove the red controls/rays and remove the legends from the output.

    Key insight:
    - The cut geometry determines fragment membership, fragment membership determines the color majority, and that majority selects the legend that decides motion; the unmatched-majority fragment is the only one that does not move.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    legends = _extract_legends(g)
    assert len(legends) == 2
    legend_mask = np.zeros((h, w), dtype=bool)
    for info in legends:
        r0, c0, r1, c1 = info["bbox"]
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    controls = _control_pixels(g, legend_mask)
    assert len(controls) == 2
    r_split, c_split = _split_lines(controls, g.shape)

    obj_mask = _object_mask(g, legend_mask, controls)
    frags = _fragment_components(g, obj_mask, r_split, c_split)

    legend_dirs = {info["fill"]: info["dir"] for info in legends}
    frag_infos = []
    for comp in frags:
        colors = {(r, c): int(g[r, c]) for r, c in comp}
        dom = _dominant_color(g, comp)
        frag_infos.append({"coords": comp, "colors": colors, "dom": dom})

    unmatched = [f for f in frag_infos if f["dom"] not in legend_dirs]
    matched = [f for f in frag_infos if f["dom"] in legend_dirs]
    assert len(unmatched) == 1 and len(matched) == 2

    out = np.zeros_like(g)
    occupied: set[tuple[int, int]] = set()

    for pos, color in unmatched[0]["colors"].items():
        out[pos] = color
        occupied.add(pos)

    by_color = {f["dom"]: f for f in matched}
    for info in legends:
        frag = by_color[info["fill"]]
        moved = _slide(frag["coords"], frag["colors"], info["dir"], g.shape, occupied)
        for pos, color in moved.items():
            out[pos] = color
            occupied.add(pos)

    return out


def _grow_fragment(allowed: set[tuple[int, int]], anchor: tuple[int, int], size: int) -> set[tuple[int, int]] | None:
    if anchor not in allowed:
        return None
    comp = {anchor}
    frontier = {
        n
        for n in _neighbors4(*anchor)
        if n in allowed and n not in comp
    }
    while len(comp) < size and frontier:
        nxt = random.choice(sorted(frontier))
        frontier.remove(nxt)
        comp.add(nxt)
        for cand in _neighbors4(*nxt):
            if cand in allowed and cand not in comp:
                frontier.add(cand)
    return comp if len(comp) == size else None


def _legend_patch(fill: int, move: tuple[int, int]) -> np.ndarray:
    patch = np.full((3, 3), fill, dtype=int)
    patch[1, 1] = GREY
    dr, dc = move
    patch[1 + dr, 1 + dc] = GREY
    return patch


def _legend_origin(side: str, h: int, w: int) -> tuple[int, int]:
    if side == "top":
        return 0, random.randint(4, w - 7)
    if side == "bottom":
        return h - 3, random.randint(4, w - 7)
    if side == "left":
        return random.randint(4, h - 7), 0
    return random.randint(4, h - 7), w - 3


def _build_expected(shape: tuple[int, int], fragments: list[dict], legends: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    legend_dirs = {info["fill"]: info["dir"] for info in legends}
    by_dom = {frag["dom"]: frag for frag in fragments}
    unmatched = [frag for frag in fragments if frag["dom"] not in legend_dirs]
    assert len(unmatched) == 1
    occupied: set[tuple[int, int]] = set()
    for pos, color in unmatched[0]["colors"].items():
        out[pos] = color
        occupied.add(pos)
    for info in sorted(legends, key=lambda d: (d["bbox"][0], d["bbox"][1])):
        frag = by_dom[info["fill"]]
        moved = _slide(frag["coords"], frag["colors"], info["dir"], shape, occupied)
        for pos, color in moved.items():
            out[pos] = color
            occupied.add(pos)
    return out


def _paint_fragment(grid: np.ndarray, coords: set[tuple[int, int]], dom: int, accent: int, force_tie: bool) -> dict[tuple[int, int], int] | None:
    pts = sorted(coords)
    n = len(pts)
    if n < 6:
        return None
    if force_tie and n % 2 == 0:
        dom_count = n // 2
    else:
        lo = max(n // 2 + 1, int(np.ceil(0.55 * n)))
        hi = max(lo, n - 2)
        dom_count = random.randint(lo, hi)
    colors: dict[tuple[int, int], int] = {}
    for i, pos in enumerate(pts):
        colors[pos] = dom if i < dom_count else accent
    seen = set(colors.values())
    if len(seen) < 2:
        return None
    for pos, color in colors.items():
        grid[pos] = color
    return colors


def _quadrant_specs(r_split: int, c_split: int, top_h: int, bottom_h: int, left_w: int, right_w: int):
    return {
        "TL": {
            "cells": {(r, c) for r in range(r_split - top_h, r_split) for c in range(c_split - left_w, c_split)},
            "anchor": (r_split - 1, c_split - 1),
        },
        "TR": {
            "cells": {(r, c) for r in range(r_split - top_h, r_split) for c in range(c_split + 1, c_split + 1 + right_w)},
            "anchor": (r_split - 1, c_split + 1),
        },
        "BL": {
            "cells": {(r, c) for r in range(r_split + 1, r_split + 1 + bottom_h) for c in range(c_split - left_w, c_split)},
            "anchor": (r_split + 1, c_split - 1),
        },
        "BR": {
            "cells": {(r, c) for r in range(r_split + 1, r_split + 1 + bottom_h) for c in range(c_split + 1, c_split + 1 + right_w)},
            "anchor": (r_split + 1, c_split + 1),
        },
    }


def _make_input() -> np.ndarray | None:
    h = random.randint(16, 20)
    w = random.randint(16, 20)

    top_h = random.randint(3, 4)
    bottom_h = random.randint(3, 4)
    left_w = random.randint(4, 5)
    right_w = random.randint(4, 5)

    min_r = top_h + 2
    max_r = h - bottom_h - 3
    min_c = left_w + 2
    max_c = w - right_w - 4
    if min_r > max_r or min_c > max_c:
        return None

    r_split = random.randint(min_r, max_r)
    c_split = random.randint(min_c, max_c)
    grid = np.zeros((h, w), dtype=int)

    missing = random.choice(QUADRANTS)
    quadrant_data = _quadrant_specs(r_split, c_split, top_h, bottom_h, left_w, right_w)
    occupied_quads = [q for q in QUADRANTS if q != missing]

    dom_colors = random.sample(COLOR_POOL, 4)
    frag_dom_colors = dom_colors[:3]
    accent = dom_colors[3]
    matched_ids = sorted(random.sample(range(3), 2))
    tie_id = random.randrange(3) if random.random() < 0.35 else None

    fragments: list[dict] = []
    for idx, quad in enumerate(occupied_quads):
        allowed = quadrant_data[quad]["cells"]
        anchor = quadrant_data[quad]["anchor"]
        area = len(allowed)
        target = random.randint(max(6, area // 2), area)
        coords = _grow_fragment(allowed, anchor, target)
        if coords is None:
            return None
        colors = _paint_fragment(grid, coords, frag_dom_colors[idx], accent, idx == tie_id)
        if colors is None:
            return None
        fragments.append(
            {
                "coords": sorted(coords),
                "colors": colors,
                "dom": frag_dom_colors[idx],
                "quad": quad,
            }
        )

    seam = {(r_split, c_split)}
    if any(q in occupied_quads for q in ("TL", "TR")):
        seam.add((r_split - 1, c_split))
    if any(q in occupied_quads for q in ("BL", "BR")):
        seam.add((r_split + 1, c_split))
    if any(q in occupied_quads for q in ("TL", "BL")):
        seam.add((r_split, c_split - 1))
    if any(q in occupied_quads for q in ("TR", "BR")):
        seam.add((r_split, c_split + 1))

    for r, c in seam:
        if not (0 <= r < h and 0 <= c < w):
            return None
        if grid[r, c] != BG:
            return None
        grid[r, c] = random.choice((accent, *frag_dom_colors))

    ctrl_vertical = random.choice(("top", "bottom"))
    ctrl_horizontal = random.choice(("left", "right"))
    if ctrl_vertical == "top":
        grid[0, c_split] = CTRL
    else:
        grid[h - 1, c_split] = CTRL
    if ctrl_horizontal == "left":
        grid[r_split, 0] = CTRL
    else:
        grid[r_split, w - 1] = CTRL

    remaining_sides = [s for s in ("top", "bottom", "left", "right") if s not in {ctrl_vertical, ctrl_horizontal}]
    matched_frags = [fragments[i] for i in matched_ids]
    matched_colors = [frag["dom"] for frag in matched_frags]
    random.shuffle(matched_colors)

    directions = random.sample(DIRS4, 2)
    legends: list[dict] = []
    legend_mask = np.zeros((h, w), dtype=bool)
    for side, fill, move in zip(remaining_sides, matched_colors, directions):
        r0, c0 = _legend_origin(side, h, w)
        r1, c1 = r0 + 2, c0 + 2
        patch = _legend_patch(fill, move)
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != BG):
            return None
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        legends.append({"bbox": (r0, c0, r1, c1), "fill": fill, "dir": move})

    expected = _build_expected(grid.shape, fragments, legends)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    split_mask = grid != BG
    split_mask[legend_mask] = False
    split_mask[0, c_split] = False
    split_mask[r_split, 0 if ctrl_horizontal == "left" else w - 1] = False
    comps = _fragment_components(grid, split_mask, r_split, c_split)
    if sorted(len(comp) for comp in comps) != sorted(len(frag["coords"]) for frag in fragments):
        return None

    doms = [_dominant_color(grid, frag["coords"]) for frag in fragments]
    if len(set(doms)) != 3:
        return None
    if sum(dom in {legend["fill"] for legend in legends} for dom in doms) != 2:
        return None

    moved = 0
    for frag in matched_frags:
        original = set(frag["coords"])
        moved_coords = set(pos for pos, color in np.ndenumerate(actual) if color != BG and color in set(frag["colors"].values()))
        if not original & moved_coords:
            moved += 1
    if np.array_equal(actual, np.zeros_like(actual)):
        return None
    if np.array_equal(actual, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _make_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
