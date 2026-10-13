from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


BG = 0
GREY = 5
COLOR_ORDER = (1, 2, 3, 4, 6, 7, 8, 9)
ORDER_INDEX = {color: i for i, color in enumerate(COLOR_ORDER)}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in NEIGHBORS4]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), m.tobytes()


def _dominant_color(colors: list[int]) -> int:
    counts = Counter(map(int, colors))
    return min(counts, key=lambda color: (-counts[color], ORDER_INDEX[color]))


def _find_exact_matches(pattern_mask: np.ndarray, template_mask: np.ndarray) -> list[tuple[int, int]]:
    pattern_mask = np.asarray(pattern_mask, dtype=bool)
    template_mask = np.asarray(template_mask, dtype=bool)
    h, w = template_mask.shape
    H, W = pattern_mask.shape
    matches: list[tuple[int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if np.array_equal(pattern_mask[r0 : r0 + h, c0 : c0 + w], template_mask):
                matches.append((int(r0), int(c0)))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grey mini-sprites are search templates: find the unique equal-shape foreground patch inside the
    large multicolor component, repaint that patch to its dominant color, and erase the grey sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components(g != BG)
    assert comps, "expected at least one foreground component"
    pattern_coords = max(comps, key=len)

    pattern_mask = np.zeros_like(g, dtype=bool)
    for r, c in pattern_coords:
        pattern_mask[r, c] = True

    templates = _components(g == GREY)
    assert templates, "expected grey template components"
    templates.sort(key=lambda comp: _bbox(comp))

    out = g.copy()
    used_cells: set[tuple[int, int]] = set()
    for coords in templates:
        template_mask, _ = _tight_mask(coords)
        matches = _find_exact_matches(pattern_mask, template_mask)
        assert len(matches) == 1, "each grey template must match exactly one pattern patch"
        r0, c0 = matches[0]

        region_cells: list[tuple[int, int]] = []
        region_colors: list[int] = []
        for dr, dc in np.argwhere(template_mask):
            rr = int(r0 + dr)
            cc = int(c0 + dc)
            region_cells.append((rr, cc))
            region_colors.append(int(g[rr, cc]))

        assert not (set(region_cells) & used_cells), "template matches must be non-overlapping"
        used_cells.update(region_cells)

        fill_color = _dominant_color(region_colors)
        for rr, cc in region_cells:
            out[rr, cc] = fill_color
        for rr, cc in coords:
            out[rr, cc] = BG

    return out


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _build_shape_library() -> list[np.ndarray]:
    shapes: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for h in range(2, 4):
        for w in range(2, 4):
            total = h * w
            for area in range(3, min(6, total - 1) + 1):
                for cells in combinations(range(total), area):
                    mask = np.zeros((h, w), dtype=bool)
                    for idx in cells:
                        mask[idx // w, idx % w] = True
                    if not _is_connected(mask):
                        continue
                    cropped = _crop_bool(mask)
                    ch, cw = map(int, cropped.shape)
                    if ch == 1 or cw == 1:
                        continue
                    if int(cropped.sum()) == ch * cw:
                        continue
                    shapes.setdefault(_mask_key(cropped), cropped.copy())
    library = list(shapes.values())
    library.sort(key=lambda m: (int(m.sum()), int(m.shape[0]), int(m.shape[1]), m.tobytes()))
    return library


SHAPE_LIBRARY = _build_shape_library()
CORNER_PAIRS = (("TL", "BR"), ("TR", "BL"))


def _corner_origin(corner: str, ph: int, pw: int, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, pw - w
    if corner == "BL":
        return ph - h, 0
    if corner == "BR":
        return ph - h, pw - w
    raise AssertionError(corner)


def _shape_cells(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)]


def _adjacent_free_cells(
    cells: list[tuple[int, int]], blocked: np.ndarray, occupied: np.ndarray
) -> set[tuple[int, int]]:
    h, w = blocked.shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not occupied[nr, nc]:
                out.add((int(nr), int(nc)))
    return out


def _path_from_connected(
    connected: np.ndarray, target_cells: list[tuple[int, int]], blocked: np.ndarray
) -> list[tuple[int, int]] | None:
    starts = _adjacent_free_cells(list(map(tuple, np.argwhere(connected))), blocked, connected)
    targets = _adjacent_free_cells(target_cells, blocked, connected)
    if not starts or not targets:
        return None
    if starts & targets:
        return []

    h, w = connected.shape
    free = (~blocked) & (~connected)
    q = deque(starts)
    prev: dict[tuple[int, int], tuple[int, int] | None] = {cell: None for cell in starts}
    while q:
        r, c = q.popleft()
        if (r, c) in targets:
            path: list[tuple[int, int]] = []
            cur: tuple[int, int] | None = (r, c)
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path
        for nr, nc in _neighbors4(r, c):
            nxt = (int(nr), int(nc))
            if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and nxt not in prev:
                prev[nxt] = (r, c)
                q.append(nxt)
    return None


def _template_modes(k: int) -> list[str]:
    modes = [random.choice(("majority", "tie")) for _ in range(k)]
    if k >= 2 and len(set(modes)) == 1:
        modes[0] = "majority"
        modes[1] = "tie"
        random.shuffle(modes)
    return modes


def _assign_region_colors(area: int, palette: list[int], mode: str) -> tuple[list[int], int]:
    if mode == "tie":
        if area == 3:
            tied_colors = random.sample(palette, 3)
            colors = tied_colors[:]
        elif area == 4:
            tied_colors = random.sample(palette, 2)
            colors = [tied_colors[0], tied_colors[0], tied_colors[1], tied_colors[1]]
        elif area == 5:
            tied_colors = random.sample(palette, 2)
            extra_pool = [color for color in palette if color not in tied_colors]
            extra = random.choice(extra_pool)
            colors = [tied_colors[0], tied_colors[0], tied_colors[1], tied_colors[1], extra]
        else:
            if len(palette) >= 3 and random.random() < 0.5:
                tied_colors = random.sample(palette, 3)
                colors = [tied_colors[0], tied_colors[0], tied_colors[1], tied_colors[1], tied_colors[2], tied_colors[2]]
            else:
                tied_colors = random.sample(palette, 2)
                colors = [tied_colors[0]] * 3 + [tied_colors[1]] * 3
        random.shuffle(colors)
        return colors, min(tied_colors, key=lambda color: ORDER_INDEX[color])

    dom = random.choice(palette)
    dom_count = random.randint(area // 2 + 1, area - 1)
    others = [random.choice(palette) for _ in range(area - dom_count)]
    colors = [dom] * dom_count + others
    random.shuffle(colors)
    return colors, dom


def _dilate(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _build_pattern() -> tuple[np.ndarray, list[dict[str, object]], dict[tuple[tuple[int, int], bytes], tuple[int, int]]] | None:
    ph = random.randint(8, 12)
    pw = random.randint(8, 12)
    k = random.randint(2, 4)

    if k == 2:
        corners = list(random.choice(CORNER_PAIRS))
    elif k == 3:
        missing = random.choice(("TL", "TR", "BL", "BR"))
        corners = [corner for corner in ("TL", "TR", "BL", "BR") if corner != missing]
    else:
        corners = ["TL", "TR", "BL", "BR"]
    random.shuffle(corners)

    shapes = random.sample(SHAPE_LIBRARY, k)
    placements: list[dict[str, object]] = []
    blocked = np.zeros((ph, pw), dtype=bool)
    for corner, mask in zip(corners, shapes):
        h, w = map(int, mask.shape)
        r0, c0 = _corner_origin(corner, ph, pw, h, w)
        blocked[r0 : r0 + h, c0 : c0 + w] = True
        placements.append(
            {
                "corner": corner,
                "mask": mask.copy(),
                "origin": (int(r0), int(c0)),
                "cells": _shape_cells(mask, r0, c0),
            }
        )

    center_candidates = [
        (r, c)
        for r in range(max(0, ph // 2 - 1), min(ph, ph // 2 + 2))
        for c in range(max(0, pw // 2 - 1), min(pw, pw // 2 + 2))
        if not blocked[r, c]
    ]
    if not center_candidates:
        return None

    hub = random.choice(center_candidates)
    connected = np.zeros((ph, pw), dtype=bool)
    connected[hub] = True

    random.shuffle(placements)
    for info in placements:
        path = _path_from_connected(connected, info["cells"], blocked)
        if path is None:
            return None
        for r, c in path:
            connected[r, c] = True
        for r, c in info["cells"]:
            connected[r, c] = True

    if not _is_connected(connected):
        return None

    match_lookup: dict[tuple[tuple[int, int], bytes], tuple[int, int]] = {}
    for info in placements:
        mask = np.asarray(info["mask"], dtype=bool)
        origin = tuple(info["origin"])
        matches = _find_exact_matches(connected, mask)
        if matches != [origin]:
            return None
        match_lookup[_mask_key(mask)] = origin

    placements.sort(key=lambda info: _bbox(list(info["cells"])))
    return connected, placements, match_lookup


def _paint_pattern(
    pattern_mask: np.ndarray, placements: list[dict[str, object]]
) -> tuple[np.ndarray, list[tuple[list[tuple[int, int]], int]]]:
    palette = random.sample(list(COLOR_ORDER), random.randint(3, 5))
    colored = np.zeros(pattern_mask.shape, dtype=int)
    pattern_cells = [tuple(map(int, cell)) for cell in np.argwhere(pattern_mask)]
    for r, c in pattern_cells:
        colored[r, c] = random.choice(palette)

    modes = _template_modes(len(placements))
    expected_regions: list[tuple[list[tuple[int, int]], int]] = []
    template_cells: set[tuple[int, int]] = set()
    for info, mode in zip(placements, modes):
        cells = list(info["cells"])
        colors, dominant = _assign_region_colors(len(cells), palette, mode)
        random.shuffle(cells)
        for (r, c), color in zip(cells, colors):
            colored[r, c] = int(color)
        expected_regions.append((cells[:], int(dominant)))
        template_cells.update(cells)

    used = set(map(int, colored[pattern_mask]))
    missing = [color for color in palette if color not in used]
    non_template = [cell for cell in pattern_cells if cell not in template_cells]
    if len(non_template) < len(missing):
        raise RuntimeError("not enough non-template cells to force palette coverage")
    random.shuffle(non_template)
    for color, (r, c) in zip(missing, non_template):
        colored[r, c] = int(color)

    assert 3 <= len(set(map(int, colored[pattern_mask]))) <= 5
    return colored, expected_regions


def _place_templates(
    grid: np.ndarray, placements: list[dict[str, object]], forbidden: np.ndarray
) -> list[list[tuple[int, int]]] | None:
    h, w = grid.shape
    occupied = forbidden.copy()
    out_templates: list[list[tuple[int, int]]] = []
    zone_c0 = min(w - 2, max(0, np.max(np.argwhere(forbidden)[:, 1]) + 3))
    for info in placements:
        mask = np.asarray(info["mask"], dtype=bool)
        th, tw = map(int, mask.shape)
        options: list[tuple[int, int]] = []
        for r0 in range(h - th + 1):
            for c0 in range(zone_c0, w - tw + 1):
                if _dilate(occupied)[r0 : r0 + th, c0 : c0 + tw].any():
                    continue
                options.append((int(r0), int(c0)))
        if not options:
            return None
        r0, c0 = random.choice(options)
        cells = _shape_cells(mask, r0, c0)
        for rr, cc in cells:
            grid[rr, cc] = GREY
            occupied[rr, cc] = True
        out_templates.append(cells)
    return out_templates


def _attempt() -> np.ndarray | None:
    built = _build_pattern()
    if built is None:
        return None
    pattern_mask, placements, _ = built
    pattern_colored, expected_regions = _paint_pattern(pattern_mask, placements)

    ph, pw = pattern_mask.shape
    H = random.randint(max(18, ph + 4), 24)
    W = random.randint(max(22, pw + 8), 30)
    max_left = W - pw - 6
    if max_left < 1:
        return None
    pr0 = random.randint(1, min(4, H - ph - 1))
    pc0 = random.randint(1, max_left)

    inp = np.zeros((H, W), dtype=int)
    pattern_global = np.zeros((H, W), dtype=bool)
    for r, c in np.argwhere(pattern_mask):
        rr = int(pr0 + r)
        cc = int(pc0 + c)
        inp[rr, cc] = int(pattern_colored[r, c])
        pattern_global[rr, cc] = True

    template_cells = _place_templates(inp, placements, pattern_global)
    if template_cells is None:
        return None

    expected = inp.copy()
    for cells in template_cells:
        for r, c in cells:
            expected[r, c] = BG
    for info, (_, dominant) in zip(placements, expected_regions):
        r0, c0 = info["origin"]
        mask = np.asarray(info["mask"], dtype=bool)
        for dr, dc in np.argwhere(mask):
            expected[pr0 + int(r0) + int(dr), pc0 + int(c0) + int(dc)] = int(dominant)

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None

    grey_comps = _components(inp == GREY)
    if len(grey_comps) != len(placements):
        return None
    big = max(_components(inp != BG), key=len)
    big_mask = np.zeros_like(inp, dtype=bool)
    for r, c in big:
        big_mask[r, c] = True
    if not np.array_equal(big_mask, pattern_global):
        return None

    for info in placements:
        mask = np.asarray(info["mask"], dtype=bool)
        target = (pr0 + int(info["origin"][0]), pc0 + int(info["origin"][1]))
        if _find_exact_matches(pattern_global, mask) != [target]:
            return None

    used_colors = set(map(int, inp[pattern_global]))
    if not (3 <= len(used_colors) <= 5):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
