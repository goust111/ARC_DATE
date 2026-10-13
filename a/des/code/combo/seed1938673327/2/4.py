from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FRAME_COLOR = 1
PALETTE = tuple(range(2, 10))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400
_GEN_COUNTER = 0


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(np.array(coords, dtype=int))
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return np.array(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _paste(dst: np.ndarray, sprite: np.ndarray, r0: int, c0: int, *, transparent: int) -> None:
    h, w = sprite.shape
    sub = dst[r0 : r0 + h, c0 : c0 + w]
    keep = sprite != int(transparent)
    sub[keep] = sprite[keep]


def _marker_candidates(line: list[tuple[int, int]], frame_color: int) -> list[tuple[int, int]]:
    return [(int(idx), int(color)) for idx, color in line if int(color) != int(frame_color)]


def _extract_objects(
    grid: np.ndarray,
    *,
    inner_bbox: tuple[int, int, int, int],
    bg: int,
) -> list[dict]:
    ir0, ic0, ir1, ic1 = inner_bbox
    inside = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]
    objects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(inside) if int(v) != int(bg)):
        for comp in _components(inside == int(color)):
            rows = comp[:, 0]
            cols = comp[:, 1]
            r0 = int(rows.min())
            c0 = int(cols.min())
            r1 = int(rows.max())
            c1 = int(cols.max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[rows - r0, cols - c0] = True
            sprite = np.full(mask.shape, int(bg), dtype=int)
            sprite[mask] = int(color)
            objects.append(
                {
                    "color": int(color),
                    "area": int(mask.sum()),
                    "sprite": sprite,
                    "r0": int(ir0 + r0),
                    "c0": int(ic0 + c0),
                }
            )
    return objects


def _sort_objects(objects: list[dict], ascending: bool) -> list[dict]:
    if ascending:
        key = lambda obj: (int(obj["area"]), int(obj["color"]), int(obj["r0"]), int(obj["c0"]))
    else:
        key = lambda obj: (-int(obj["area"]), int(obj["color"]), int(obj["r0"]), int(obj["c0"]))
    return sorted(objects, key=key)


def _arrangement_positions(
    objects: list[dict],
    *,
    inner_h: int,
    inner_w: int,
    horizontal: bool,
) -> list[tuple[int, int]]:
    assert objects
    dims = [tuple(map(int, obj["sprite"].shape)) for obj in objects]
    positions: list[tuple[int, int]] = []

    if horizontal:
        total_w = sum(w for _h, w in dims) + len(dims) - 1
        band_h = max(h for h, _w in dims)
        assert total_w <= int(inner_w) and band_h <= int(inner_h)
        base_r = (int(inner_h) - band_h) // 2
        cur_c = (int(inner_w) - total_w) // 2
        for obj in objects:
            h, w = map(int, obj["sprite"].shape)
            positions.append((int(base_r + (band_h - h) // 2), int(cur_c)))
            cur_c += int(w + 1)
    else:
        total_h = sum(h for h, _w in dims) + len(dims) - 1
        band_w = max(w for _h, w in dims)
        assert total_h <= int(inner_h) and band_w <= int(inner_w)
        cur_r = (int(inner_h) - total_h) // 2
        base_c = (int(inner_w) - band_w) // 2
        for obj in objects:
            h, w = map(int, obj["sprite"].shape)
            positions.append((int(cur_r), int(base_c + (band_w - w) // 2)))
            cur_r += int(h + 1)

    return positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the objects inside a frame according to whether opposite-edge markers
    match any object color, then arrange the sorted objects in a centered row or
    column and remove the marker pixels.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(_dominant_color(g))

    fr0, fc0, fr1, fc1 = _bbox(g != int(bg))
    assert fr1 - fr0 >= 2 and fc1 - fc0 >= 2

    perimeter = [int(g[fr0, c]) for c in range(fc0, fc1 + 1)]
    perimeter += [int(g[fr1, c]) for c in range(fc0, fc1 + 1)]
    perimeter += [int(g[r, fc0]) for r in range(fr0 + 1, fr1)]
    perimeter += [int(g[r, fc1]) for r in range(fr0 + 1, fr1)]
    frame_color = int(Counter(perimeter).most_common(1)[0][0])

    top_marks = _marker_candidates([(c, int(g[fr0, c])) for c in range(fc0 + 1, fc1)], frame_color)
    bottom_marks = _marker_candidates([(c, int(g[fr1, c])) for c in range(fc0 + 1, fc1)], frame_color)
    left_marks = _marker_candidates([(r, int(g[r, fc0])) for r in range(fr0 + 1, fr1)], frame_color)
    right_marks = _marker_candidates([(r, int(g[r, fc1])) for r in range(fr0 + 1, fr1)], frame_color)

    assert len(top_marks) == len(bottom_marks) == len(left_marks) == len(right_marks) == 1

    tb_color = int(top_marks[0][1])
    lr_color = int(left_marks[0][1])
    assert int(bottom_marks[0][1]) == int(tb_color)
    assert int(right_marks[0][1]) == int(lr_color)
    assert int(tb_color) != int(lr_color)

    inner_bbox = (int(fr0 + 1), int(fc0 + 1), int(fr1 - 1), int(fc1 - 1))
    ir0, ic0, ir1, ic1 = inner_bbox
    objects = _extract_objects(g, inner_bbox=inner_bbox, bg=bg)
    assert objects

    object_colors = {int(obj["color"]) for obj in objects}
    ascending = bool(int(tb_color) in object_colors)
    horizontal = bool(int(lr_color) in object_colors)
    ordered = _sort_objects(objects, ascending=ascending)

    out = g.copy()
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = int(bg)
    out[fr0, fc0 : fc1 + 1] = int(frame_color)
    out[fr1, fc0 : fc1 + 1] = int(frame_color)
    out[fr0 : fr1 + 1, fc0] = int(frame_color)
    out[fr0 : fr1 + 1, fc1] = int(frame_color)

    positions = _arrangement_positions(
        ordered,
        inner_h=int(ir1 - ir0 + 1),
        inner_w=int(ic1 - ic0 + 1),
        horizontal=horizontal,
    )
    for obj, (rr, cc) in zip(ordered, positions):
        _paste(out, obj["sprite"], int(ir0 + rr), int(ic0 + cc), transparent=bg)

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(int(r - min_r), int(c - min_c)) for r, c in cells}


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    norm = _normalize_cells(cells)
    h = max(r for r, _c in norm) + 1
    w = max(c for _r, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[int(r), int(c)] = True
    return mask


def _random_mask(area: int) -> np.ndarray | None:
    for _ in range(250):
        cells = {(0, 0)}
        while len(cells) < int(area):
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if (nr, nc) in cells:
                        continue
                    trial = cells | {(nr, nc)}
                    rows = [p[0] for p in trial]
                    cols = [p[1] for p in trial]
                    if max(rows) - min(rows) + 1 <= 4 and max(cols) - min(cols) + 1 <= 4:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != int(area):
            continue

        mask = _mask_from_cells(cells)
        mask = np.rot90(mask, random.randrange(4))
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        mask = _crop_mask(mask)

        if area >= 4 and (mask.shape[0] == 1 or mask.shape[1] == 1) and random.random() < 0.7:
            continue
        if area >= 5 and bool(mask.all()) and random.random() < 0.65:
            continue
        return np.array(mask, dtype=bool)
    return None


def _make_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.full(mask.shape, BG, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _build_objects(colors: list[int], areas: list[int]) -> list[dict] | None:
    objects: list[dict] = []
    for color, area in zip(colors, areas):
        mask = _random_mask(int(area))
        if mask is None:
            return None
        objects.append(
            {
                "color": int(color),
                "area": int(area),
                "sprite": _make_sprite(mask, int(color)),
                "r0": -1,
                "c0": -1,
            }
        )
    return objects


def _scatter_positions(objects: list[dict], inner_h: int, inner_w: int) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int] | None] = [None] * len(objects)
    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    order = sorted(
        range(len(objects)),
        key=lambda idx: int(objects[idx]["sprite"].shape[0] * objects[idx]["sprite"].shape[1]),
        reverse=True,
    )

    for idx in order:
        h, w = map(int, objects[idx]["sprite"].shape)
        min_r = 1
        max_r = int(inner_h - h - 1)
        min_c = 1
        max_c = int(inner_w - w - 1)
        if max_r < min_r or max_c < min_c:
            return None
        candidates = [(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)]
        random.shuffle(candidates)
        placed = False
        for r, c in candidates:
            if not bool(occupied[r - 1 : r + h + 1, c - 1 : c + w + 1].any()):
                occupied[r - 1 : r + h + 1, c - 1 : c + w + 1] = True
                positions[idx] = (int(r), int(c))
                placed = True
                break
        if not placed:
            return None

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _choose_marker_colors(
    object_colors: list[int],
    *,
    ascending: bool,
    horizontal: bool,
) -> tuple[int, int] | None:
    unused = [color for color in PALETTE if int(color) not in set(object_colors)]
    if ascending:
        tb_color = int(random.choice(object_colors))
    else:
        if not unused:
            return None
        tb_color = int(random.choice(unused))

    if horizontal:
        candidates = [color for color in object_colors if int(color) != int(tb_color)]
    else:
        candidates = [color for color in unused if int(color) != int(tb_color)]
    if not candidates:
        return None
    lr_color = int(random.choice(candidates))
    return int(tb_color), int(lr_color)


def _render_expected(
    shape: tuple[int, int],
    *,
    frame_bbox: tuple[int, int, int, int],
    objects: list[dict],
    ascending: bool,
    horizontal: bool,
) -> np.ndarray:
    fr0, fc0, fr1, fc1 = frame_bbox
    ir0, ic0, ir1, ic1 = int(fr0 + 1), int(fc0 + 1), int(fr1 - 1), int(fc1 - 1)
    out = np.full(shape, BG, dtype=int)
    out[fr0, fc0 : fc1 + 1] = FRAME_COLOR
    out[fr1, fc0 : fc1 + 1] = FRAME_COLOR
    out[fr0 : fr1 + 1, fc0] = FRAME_COLOR
    out[fr0 : fr1 + 1, fc1] = FRAME_COLOR
    ordered = _sort_objects(objects, ascending=ascending)
    positions = _arrangement_positions(
        ordered,
        inner_h=int(ir1 - ir0 + 1),
        inner_w=int(ic1 - ic0 + 1),
        horizontal=horizontal,
    )
    for obj, (rr, cc) in zip(ordered, positions):
        _paste(out, obj["sprite"], int(ir0 + rr), int(ic0 + cc), transparent=BG)
    return out


def _gen_attempt(ascending: bool, horizontal: bool) -> np.ndarray | None:
    n_objects = random.randint(3, 5)
    colors = random.sample(list(PALETTE), n_objects)
    areas = random.sample(list(range(3, 9)), n_objects)
    objects = _build_objects(colors, areas)
    if objects is None:
        return None

    dims = [tuple(map(int, obj["sprite"].shape)) for obj in objects]
    total_w = sum(w for _h, w in dims) + len(dims) - 1
    total_h = sum(h for h, _w in dims) + len(dims) - 1
    max_h = max(h for h, _w in dims)
    max_w = max(w for _h, w in dims)

    if horizontal:
        inner_w = int(total_w + random.randint(3, 7))
        inner_h = int(max(max_h + random.randint(6, 10), 10))
    else:
        inner_h = int(total_h + random.randint(3, 7))
        inner_w = int(max(max_w + random.randint(6, 10), 10))

    positions = _scatter_positions(objects, inner_h=inner_h, inner_w=inner_w)
    if positions is None:
        return None

    marker_colors = _choose_marker_colors(colors, ascending=ascending, horizontal=horizontal)
    if marker_colors is None:
        return None
    tb_color, lr_color = marker_colors

    top_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)

    height = int(top_margin + inner_h + 2 + bottom_margin)
    width = int(left_margin + inner_w + 2 + right_margin)
    fr0 = int(top_margin)
    fc0 = int(left_margin)
    fr1 = int(fr0 + inner_h + 1)
    fc1 = int(fc0 + inner_w + 1)

    inp = np.full((height, width), BG, dtype=int)
    inp[fr0, fc0 : fc1 + 1] = FRAME_COLOR
    inp[fr1, fc0 : fc1 + 1] = FRAME_COLOR
    inp[fr0 : fr1 + 1, fc0] = FRAME_COLOR
    inp[fr0 : fr1 + 1, fc1] = FRAME_COLOR

    for obj, (r, c) in zip(objects, positions):
        obj["r0"] = int(fr0 + 1 + r)
        obj["c0"] = int(fc0 + 1 + c)
        _paste(inp, obj["sprite"], int(obj["r0"]), int(obj["c0"]), transparent=BG)

    inp[fr0, random.randint(fc0 + 1, fc1 - 1)] = int(tb_color)
    inp[fr1, random.randint(fc0 + 1, fc1 - 1)] = int(tb_color)
    inp[random.randint(fr0 + 1, fr1 - 1), fc0] = int(lr_color)
    inp[random.randint(fr0 + 1, fr1 - 1), fc1] = int(lr_color)

    expected = _render_expected(
        inp.shape,
        frame_bbox=(fr0, fc0, fr1, fc1),
        objects=objects,
        ascending=ascending,
        horizontal=horizontal,
    )
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected) and not np.array_equal(inp, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    combo = int(_GEN_COUNTER % 4)
    _GEN_COUNTER += 1
    ascending = bool(combo & 1)
    horizontal = bool(combo & 2)

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(ascending=ascending, horizontal=horizontal)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
