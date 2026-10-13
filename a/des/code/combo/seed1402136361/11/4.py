from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _leaf_neighbors(coord_set: set[tuple[int, int]], r: int, c: int) -> list[tuple[int, int]]:
    neighbors = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nb = (int(r + dr), int(c + dc))
        if nb in coord_set:
            neighbors.append(nb)
    return neighbors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count edge leaf pixels on each side of every object and extend a centered bar outward by that count.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    out = g.copy()

    for comp in _components(g, bg):
        color = int(comp["color"])
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        coord_set = set(coords)

        core: list[tuple[int, int]] = []
        leaves: list[tuple[int, int, tuple[int, int]]] = []
        for r, c in coords:
            neighbors = _leaf_neighbors(coord_set, r, c)
            if len(neighbors) == 1:
                leaves.append((int(r), int(c), neighbors[0]))
            else:
                core.append((int(r), int(c)))

        if not core:
            continue

        rows = [r for r, _ in core]
        cols = [c for _, c in core]
        r0 = int(min(rows))
        r1 = int(max(rows))
        c0 = int(min(cols))
        c1 = int(max(cols))
        mid_r = (r0 + r1) // 2
        mid_c = (c0 + c1) // 2

        counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        for r, c, (nr, nc) in leaves:
            if r == nr - 1 and c == nc:
                counts["top"] += 1
            elif r == nr + 1 and c == nc:
                counts["bottom"] += 1
            elif c == nc - 1 and r == nr:
                counts["left"] += 1
            elif c == nc + 1 and r == nr:
                counts["right"] += 1

        for k in range(int(counts["top"])):
            rr = int(r0 - 1 - k)
            if 0 <= rr < h:
                out[rr, mid_c] = color
        for k in range(int(counts["bottom"])):
            rr = int(r1 + 1 + k)
            if 0 <= rr < h:
                out[rr, mid_c] = color
        for k in range(int(counts["left"])):
            cc = int(c0 - 1 - k)
            if 0 <= cc < w:
                out[mid_r, cc] = color
        for k in range(int(counts["right"])):
            cc = int(c1 + 1 + k)
            if 0 <= cc < w:
                out[mid_r, cc] = color

    return out


def _sample_non_adjacent(length: int, count: int) -> list[int]:
    if count == 0:
        return []

    choices: list[list[int]] = []

    def rec(start: int, remaining: int, acc: list[int]) -> None:
        if remaining == 0:
            choices.append(acc.copy())
            return
        last = length - (2 * remaining - 1)
        for idx in range(start, last + 1):
            acc.append(int(idx))
            rec(int(idx + 2), int(remaining - 1), acc)
            acc.pop()

    rec(0, count, [])
    return random.choice(choices)


def _make_object_spec(color: int) -> dict:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    side_lengths = {"top": w, "bottom": w, "left": h, "right": h}

    while True:
        sides: dict[str, list[int]] = {}
        active_sides = 0
        for side, span in side_lengths.items():
            max_count = (span + 1) // 2
            if random.random() < 0.5:
                count = 0
            else:
                count = random.randint(1, max_count)
            sides[side] = _sample_non_adjacent(span, count)
            if count > 0:
                active_sides += 1
        if active_sides > 0:
            break

    top = len(sides["top"])
    bottom = len(sides["bottom"])
    left = len(sides["left"])
    right = len(sides["right"])
    return {
        "color": int(color),
        "h": int(h),
        "w": int(w),
        "sides": sides,
        "reserve_bbox_area": int((h + top + bottom) * (w + left + right)),
        "top_extent": int(top),
        "bottom_extent": int(bottom),
        "left_extent": int(left),
        "right_extent": int(right),
    }


def _bboxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    if ar1 + gap < br0 or br1 + gap < ar0:
        return False
    if ac1 + gap < bc0 or bc1 + gap < ac0:
        return False
    return True


def _place_specs(height: int, width: int, specs: list[dict]) -> list[tuple[dict, int, int]] | None:
    placed: list[tuple[dict, int, int]] = []
    reserved: list[tuple[int, int, int, int]] = []

    for spec in sorted(specs, key=lambda item: -int(item["reserve_bbox_area"])):
        h = int(spec["h"])
        w = int(spec["w"])
        top = int(spec["top_extent"])
        bottom = int(spec["bottom_extent"])
        left = int(spec["left_extent"])
        right = int(spec["right_extent"])

        min_r = top
        max_r = height - h - bottom
        min_c = left
        max_c = width - w - right
        if min_r > max_r or min_c > max_c:
            return None

        found = False
        for _ in range(250):
            r0 = random.randint(min_r, max_r)
            c0 = random.randint(min_c, max_c)
            bbox = (r0 - top, c0 - left, r0 + h - 1 + bottom, c0 + w - 1 + right)
            if any(_bboxes_too_close(bbox, other, gap=1) for other in reserved):
                continue
            reserved.append(bbox)
            placed.append((spec, int(r0), int(c0)))
            found = True
            break
        if not found:
            return None

    return placed


def _paint_object(grid: np.ndarray, spec: dict, r0: int, c0: int) -> None:
    color = int(spec["color"])
    h = int(spec["h"])
    w = int(spec["w"])
    grid[r0 : r0 + h, c0 : c0 + w] = color

    for offset in spec["sides"]["top"]:
        grid[r0 - 1, c0 + int(offset)] = color
    for offset in spec["sides"]["bottom"]:
        grid[r0 + h, c0 + int(offset)] = color
    for offset in spec["sides"]["left"]:
        grid[r0 + int(offset), c0 - 1] = color
    for offset in spec["sides"]["right"]:
        grid[r0 + int(offset), c0 + w] = color


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 26)
    width = random.randint(16, 26)
    bg = random.randint(0, 9)
    n_objects = random.randint(1, 4)

    palette = [color for color in range(10) if color != bg]
    colors = random.sample(palette, n_objects)
    specs = [_make_object_spec(color) for color in colors]
    placements = _place_specs(height, width, specs)
    if placements is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    for spec, r0, c0 in placements:
        _paint_object(grid, spec, r0, c0)

    if int(np.sum(grid == bg)) <= int(np.sum(grid != bg)):
        return None
    if len(_components(grid, bg)) != n_objects:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
