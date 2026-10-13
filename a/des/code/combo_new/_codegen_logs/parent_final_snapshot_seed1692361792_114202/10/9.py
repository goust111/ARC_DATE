import random
from typing import Iterable

import numpy as np


BG = 0
CANVAS_SIZE = 30
MAX_GENERATION_ATTEMPTS = 300
ALL_KINDS = [(1, "H"), (1, "V"), (2, "H"), (2, "V"), (3, "H"), (3, "V")]
NON_BG_COLORS = tuple(range(1, 10))
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _kind_sort_key(kind: tuple[int, str]) -> tuple[int, int]:
    period, orient = kind
    return period, 0 if orient == "H" else 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _run_lengths(seq: Iterable[int]) -> tuple[list[int], list[int]]:
    values = list(map(int, seq))
    assert values, "stripe sequence must be non-empty"
    colors = [values[0]]
    runs = [1]
    for value in values[1:]:
        if value == colors[-1]:
            runs[-1] += 1
        else:
            colors.append(value)
            runs.append(1)
    return colors, runs


def _classify_object(sub: np.ndarray) -> tuple[tuple[int, str], int]:
    height, width = map(int, sub.shape)
    assert height >= 4 and width >= 4, "objects must have a visible border and interior"
    border_values = np.concatenate(
        [sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]]
    )
    border_colors = set(map(int, border_values))
    assert len(border_colors) == 1, "object border must be monochrome"
    border_color = next(iter(border_colors))

    interior = sub[1:-1, 1:-1]
    assert interior.size > 0 and np.all(interior != BG), "object interior must be non-background"

    rows_uniform = bool(np.all(interior == interior[:, :1]))
    cols_uniform = bool(np.all(interior == interior[:1, :]))
    assert rows_uniform ^ cols_uniform, "interior texture must be horizontal or vertical stripes"

    if rows_uniform:
        orient = "H"
        seq = interior[:, 0]
    else:
        orient = "V"
        seq = interior[0, :]

    colors, runs = _run_lengths(seq.tolist())
    assert len(colors) >= 2, "stripe texture must alternate at least once"
    assert len(set(colors)) == 2, "stripe texture must use exactly two colors"
    assert all(colors[i] != colors[i - 1] for i in range(1, len(colors)))
    assert len(set(runs)) == 1, "every stripe band must have the same thickness"

    period = int(runs[0])
    assert 1 <= period <= 3, "stripe period must stay within 1..3"
    return (period, orient), border_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each striped rectangle border by the marker assigned to its stripe class.

    Input structure:
    - The background is black.
    - Several filled rectangular objects appear on the background; each has a
      one-cell-thick monochrome border and a non-black striped interior.
    - Every interior is made from two alternating colors, either in horizontal
      row bands or vertical column bands, with band thickness 1 to 3.
    - Separate 2x2 monochrome marker squares appear elsewhere in one left-to-right
      sequence, with one marker per distinct stripe class.

    Rule:
    - Detect each rectangular object and read its interior stripe orientation and
      stripe thickness.
    - Group objects by the pair (period, orientation).
    - Sort the groups by increasing period, breaking ties with horizontal before
      vertical, and match them to the markers from left to right.
    - Repaint each object's border with its matched marker color while keeping
      every interior pattern and every marker square unchanged.

    Key insight:
    - The grouping key is the visible stripe geometry inside each rectangle, not
      the rectangle's size, position, border color, or the stripe colors themselves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    markers: list[dict[str, object]] = []
    objects: list[dict[str, object]] = []

    for coords in _components(g != BG):
        r0, c0, r1, c1 = _bbox(coords)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        height, width = map(int, sub.shape)
        assert coords.shape[0] == height * width, "components must fill their bounding box"

        if height == 2 and width == 2 and len(set(map(int, sub.reshape(-1)))) == 1:
            markers.append({"bbox": (r0, c0, r1, c1), "color": int(sub[0, 0])})
            continue

        kind, _border_color = _classify_object(sub)
        objects.append({"bbox": (r0, c0, r1, c1), "kind": kind})

    assert markers and objects, "expected both markers and striped rectangles"
    markers.sort(key=lambda marker: (int(marker["bbox"][1]), int(marker["bbox"][0])))

    kinds = sorted({obj["kind"] for obj in objects}, key=_kind_sort_key)
    assert len(kinds) == len(markers), "marker count must equal the number of stripe classes"
    assert len({int(marker["color"]) for marker in markers}) == len(markers), "markers must have distinct colors"

    color_by_kind = {kind: int(marker["color"]) for kind, marker in zip(kinds, markers)}
    out = g.copy()
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        color = color_by_kind[obj["kind"]]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


def _build_object(kind: tuple[int, str], border_color: int) -> np.ndarray:
    period, orient = kind
    run_count = random.choice([2, 3, 4]) if period == 1 else random.choice([2, 3])
    cross_size = random.randint(3, 7)
    inner_h = period * run_count if orient == "H" else cross_size
    inner_w = cross_size if orient == "H" else period * run_count

    stripe_colors = random.sample([c for c in NON_BG_COLORS if c != border_color], 2)
    interior = np.zeros((inner_h, inner_w), dtype=int)
    if orient == "H":
        for idx in range(run_count):
            color = stripe_colors[idx % 2]
            interior[idx * period : (idx + 1) * period, :] = color
    else:
        for idx in range(run_count):
            color = stripe_colors[idx % 2]
            interior[:, idx * period : (idx + 1) * period] = color

    obj = np.full((inner_h + 2, inner_w + 2), border_color, dtype=int)
    obj[1:-1, 1:-1] = interior
    return obj


def _reserve(occ: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> None:
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(occ.shape[0], r + h + pad)
    c1 = min(occ.shape[1], c + w + pad)
    occ[r0:r1, c0:c1] = True


def _can_place(occ: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(occ.shape[0], r + h + pad)
    c1 = min(occ.shape[1], c + w + pad)
    return not bool(occ[r0:r1, c0:c1].any())


def _gen_attempt() -> np.ndarray | None:
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    occ = np.zeros_like(canvas, dtype=bool)

    type_count = random.randint(2, 4)
    kinds = sorted(random.sample(ALL_KINDS, type_count), key=_kind_sort_key)
    object_count = min(type_count + random.randint(0, 2), 5)

    chosen_colors = random.sample(list(NON_BG_COLORS), object_count + type_count)
    border_colors = chosen_colors[:object_count]
    marker_colors = chosen_colors[object_count:]

    object_kinds = list(kinds)
    while len(object_kinds) < object_count:
        object_kinds.append(random.choice(kinds))
    random.shuffle(object_kinds)

    marker_gap = random.randint(1, 3)
    marker_span = 2 * type_count + marker_gap * (type_count - 1)
    marker_row = random.choice([1, CANVAS_SIZE - 3])
    marker_col = random.randint(1, CANVAS_SIZE - marker_span - 1)
    for idx, color in enumerate(marker_colors):
        c0 = marker_col + idx * (2 + marker_gap)
        canvas[marker_row : marker_row + 2, c0 : c0 + 2] = color
        _reserve(occ, marker_row, c0, 2, 2, pad=1)

    specs = []
    for border_color, kind in zip(border_colors, object_kinds):
        arr = _build_object(kind, border_color)
        specs.append((int(arr.shape[0] * arr.shape[1]), kind, arr))
    specs.sort(key=lambda item: item[0], reverse=True)

    for _area, kind, arr in specs:
        h, w = map(int, arr.shape)
        placed = False
        for _ in range(600):
            r = random.randint(0, CANVAS_SIZE - h)
            c = random.randint(0, CANVAS_SIZE - w)
            if _can_place(occ, r, c, h, w, pad=1):
                canvas[r : r + h, c : c + w] = arr
                _reserve(occ, r, c, h, w, pad=1)
                placed = True
                break
        if not placed:
            return None

    used = np.argwhere(canvas != BG)
    if used.size == 0:
        return None
    r0, c0 = used.min(axis=0)
    r1, c1 = used.max(axis=0)
    top_pad = random.randint(0, min(2, int(r0)))
    left_pad = random.randint(0, min(2, int(c0)))
    bottom_pad = random.randint(0, min(2, int(CANVAS_SIZE - 1 - r1)))
    right_pad = random.randint(0, min(2, int(CANVAS_SIZE - 1 - c1)))
    grid = canvas[r0 - top_pad : r1 + 1 + bottom_pad, c0 - left_pad : c1 + 1 + right_pad]

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid striped-rectangle task")
