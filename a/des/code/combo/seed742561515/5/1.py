from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
MAROON = 9
LEGEND_COLORS = (1, 2, 3, 4, 6, 8)

MAX_GENERATION_ATTEMPTS = 400
MAX_OBJECT_ATTEMPTS = 250
MAX_SHAPE_ATTEMPTS = 250


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            comp: list[tuple[int, int]] = []
            stack = [(int(r), int(c))]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _boundary(mask: np.ndarray) -> np.ndarray:
    h, w = map(int, mask.shape)
    out = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc, h, w):
            if not bool(mask[nr, nc]):
                out[rr, cc] = True
                break
        else:
            if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                out[rr, cc] = True
    return out


def _parse_legend(row: np.ndarray) -> list[int]:
    width = int(row.shape[0])
    assert width % 2 == 0 and width >= 2
    colors: list[int] = []
    for c in range(0, width, 2):
        color = int(row[c])
        assert int(row[c + 1]) == color
        colors.append(color)
    assert len(set(colors)) == len(colors)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count orange markers inside each maroon closed outline, look up the
    corresponding color in the bottom 2-cell legend strip, and fill the whole
    enclosed interior with that color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert h >= 4 and w >= 8 and w % 2 == 0

    legend = _parse_legend(g[-1])
    assert bool(np.all(g[-2, :] == g[-2, 0]))

    upper = g[:-2, :]
    uh, uw = map(int, upper.shape)
    passable = (upper == BLACK) | (upper == ORANGE)

    out = g.copy()
    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == uh - 1 or c == uw - 1 for r, c in comp):
            continue
        marker_count = sum(int(upper[r, c] == ORANGE) for r, c in comp)
        if 1 <= marker_count <= len(legend):
            fill = legend[marker_count - 1]
            for r, c in comp:
                out[r, c] = fill
    return out


def _column_blob(height: int, width: int) -> np.ndarray:
    top = [random.randint(0, max(0, height // 4))]
    first_bottom_lo = max(top[0] + 3, height - 1 - max(1, height // 4))
    bottom = [random.randint(first_bottom_lo, height - 1)]

    for _c in range(1, width):
        t = top[-1] + random.choice((-1, 0, 1))
        t = max(0, min(t, height - 4))
        b = bottom[-1] + random.choice((-1, 0, 1))
        b = max(t + 3, min(b, height - 1))
        top.append(int(t))
        bottom.append(int(b))

    mask = np.zeros((height, width), dtype=bool)
    for c in range(width):
        mask[top[c] : bottom[c] + 1, c] = True
    return mask


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[int(rows[0]) : int(rows[-1]) + 1, int(cols[0]) : int(cols[-1]) + 1]


def _pick_markers(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    pool = cells[:]
    random.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for r, c in pool:
        if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
            chosen.append((int(r), int(c)))
            if len(chosen) == count:
                return chosen
    remaining = [cell for cell in pool if cell not in chosen]
    chosen.extend(remaining[: count - len(chosen)])
    return chosen


def _generate_object(required_markers: int, max_h: int, max_w: int) -> dict | None:
    min_side = 5 if required_markers <= 4 else 6
    if max_h < min_side or max_w < min_side:
        return None

    for _ in range(MAX_SHAPE_ATTEMPTS):
        height = random.randint(min_side, max_h)
        width = random.randint(min_side, max_w)

        if random.random() < 0.5:
            mask = _column_blob(height, width)
        else:
            mask = _column_blob(width, height).T

        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)

        mask = _trim(mask)
        border = _boundary(mask)
        interior = mask & (~border)
        comps = _components(interior)
        if len(comps) != 1:
            continue
        interior_cells = comps[0]
        if len(interior_cells) < required_markers:
            continue

        marker_cells = _pick_markers(interior_cells, required_markers)
        sprite = np.zeros(mask.shape, dtype=int)
        sprite[border] = MAROON
        for r, c in marker_cells:
            sprite[r, c] = ORANGE

        return {
            "sprite": sprite,
            "interior": [(int(r), int(c)) for r, c in interior_cells],
            "count": int(required_markers),
            "area": int(mask.sum()),
        }
    return None


def _place_objects(scene_h: int, scene_w: int, objects: list[dict]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((scene_h, scene_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(objects)
    order = sorted(range(len(objects)), key=lambda i: (-int(objects[i]["area"]), random.random()))

    for idx in order:
        sprite = objects[idx]["sprite"]
        sh, sw = map(int, sprite.shape)
        if scene_h - sh <= 1 or scene_w - sw <= 1:
            return None

        positions = [
            (r0, c0)
            for r0 in range(1, scene_h - sh)
            for c0 in range(1, scene_w - sw)
        ]
        random.shuffle(positions)

        placed = False
        for r0, c0 in positions:
            rr0 = r0 - 1
            rr1 = r0 + sh + 1
            cc0 = c0 - 1
            cc1 = c0 + sw + 1
            if bool(occupied[rr0:rr1, cc0:cc1].any()):
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            placements[idx] = (int(r0), int(c0))
            placed = True
            break

        if not placed:
            return None

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _expected_output(inp: np.ndarray, legend: list[int], objects: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    for obj, (r0, c0) in zip(objects, placements):
        fill = int(legend[int(obj["count"]) - 1])
        for dr, dc in obj["interior"]:
            out[r0 + dr, c0 + dc] = fill
    return out


def _generate_case() -> np.ndarray | None:
    legend_len = random.randint(4, len(LEGEND_COLORS))
    width = 2 * legend_len
    object_count = random.randint(1, 3)
    scene_h = random.randint(max(8, 6 + 2 * object_count), 15)
    total_h = scene_h + 2

    legend = random.sample(LEGEND_COLORS, legend_len)
    if object_count <= legend_len and random.random() < 0.7:
        marker_counts = random.sample(list(range(1, legend_len + 1)), object_count)
    else:
        marker_counts = [random.randint(1, legend_len) for _ in range(object_count)]
    if object_count > 1 and len(set(marker_counts)) == 1:
        marker_counts[-1] = 1 + (marker_counts[-1] % legend_len)

    max_obj_h = min(10, scene_h - 2)
    max_obj_w = min(9, width - 2)

    objects: list[dict] = []
    for count in marker_counts:
        obj = _generate_object(int(count), max_obj_h, max_obj_w)
        if obj is None:
            return None
        objects.append(obj)

    placements = _place_objects(scene_h, width, objects)
    if placements is None or len(placements) != len(objects):
        return None

    inp = np.full((total_h, width), BLACK, dtype=int)
    inp[scene_h, :] = GREY
    for i, color in enumerate(legend):
        inp[scene_h + 1, 2 * i : 2 * i + 2] = int(color)

    for obj, (r0, c0) in zip(objects, placements):
        sprite = obj["sprite"]
        sh, sw = map(int, sprite.shape)
        patch = inp[r0 : r0 + sh, c0 : c0 + sw]
        mask = sprite != BLACK
        patch[mask] = sprite[mask]

    expected = _expected_output(inp, legend, objects, placements)
    actual = main(inp)
    if np.array_equal(actual, inp):
        return None
    if not np.array_equal(actual, expected):
        return None

    passable = ((inp[:-2] == BLACK) | (inp[:-2] == ORANGE))
    enclosed = [
        comp
        for comp in _components(passable)
        if not any(
            r == 0 or c == 0 or r == scene_h - 1 or c == width - 1
            for r, c in comp
        )
    ]
    if len(enclosed) != len(objects):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
