import random
from collections import Counter, defaultdict, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SHAPE_POOL = [
    "vert",
    "horiz",
    "L_tl",
    "L_tr",
    "L_bl",
    "L_br",
    "L_tl",
    "L_tr",
    "L_bl",
    "L_br",
    "T_top",
    "T_bottom",
    "T_left",
    "T_right",
    "T_top",
    "T_bottom",
    "T_left",
    "T_right",
    "plus",
]

GEN_ATTEMPTS = 400
COMPONENT_ATTEMPTS = 60
PAIR_ATTEMPTS = 80


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _active_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    mask = grid != bg
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a colored interior area"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    inner = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(inner != bg), "interior area must be fully colored"
    return int(r0), int(c0), int(r1), int(c1)


def _collect_markers(
    grid: np.ndarray, bg: int, r0: int, c0: int, r1: int, c1: int
) -> dict[int, dict[str, list[int]]]:
    h, w = grid.shape
    ih = r1 - r0 + 1
    iw = c1 - c0 + 1
    markers: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: {"top": [], "bottom": [], "left": [], "right": []}
    )

    for c in range(1, w - 1):
        v = int(grid[0, c])
        if v != bg:
            x = c - c0
            assert 1 <= x <= iw - 1
            markers[v]["top"].append(x)
        v = int(grid[h - 1, c])
        if v != bg:
            x = c - c0
            assert 1 <= x <= iw - 1
            markers[v]["bottom"].append(x)

    for r in range(1, h - 1):
        v = int(grid[r, 0])
        if v != bg:
            y = r - r0
            assert 1 <= y <= ih - 1
            markers[v]["left"].append(y)
        v = int(grid[r, w - 1])
        if v != bg:
            y = r - r0
            assert 1 <= y <= ih - 1
            markers[v]["right"].append(y)

    assert markers, "expected edge markers"
    return markers


def _add_vertical(vert: np.ndarray, x: int, y0: int, y1: int) -> None:
    for r in range(y0, y1):
        vert[r, x - 1] = True


def _add_horizontal(horiz: np.ndarray, y: int, x0: int, x1: int) -> None:
    for c in range(x0, x1):
        horiz[y - 1, c] = True


def _reconstruct_blocks(
    grid: np.ndarray, bg: int
) -> tuple[int, int, int, int, np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = _active_bbox(grid, bg)
    ih = r1 - r0 + 1
    iw = c1 - c0 + 1
    vert = np.zeros((ih, iw - 1), dtype=bool)
    horiz = np.zeros((ih - 1, iw), dtype=bool)

    markers = _collect_markers(grid, bg, r0, c0, r1, c1)
    for exits in markers.values():
        for side in ("top", "bottom", "left", "right"):
            assert len(exits[side]) <= 1, "each marker color may exit each side at most once"

        top = exits["top"][0] if exits["top"] else None
        bottom = exits["bottom"][0] if exits["bottom"] else None
        left = exits["left"][0] if exits["left"] else None
        right = exits["right"][0] if exits["right"] else None

        has_top = top is not None
        has_bottom = bottom is not None
        has_left = left is not None
        has_right = right is not None

        if has_top and has_bottom and not has_left and not has_right:
            assert top == bottom
            _add_vertical(vert, top, 0, ih)
        elif has_left and has_right and not has_top and not has_bottom:
            assert left == right
            _add_horizontal(horiz, left, 0, iw)
        elif has_top and has_left and not has_bottom and not has_right:
            _add_vertical(vert, top, 0, left)
            _add_horizontal(horiz, left, 0, top)
        elif has_top and has_right and not has_bottom and not has_left:
            _add_vertical(vert, top, 0, right)
            _add_horizontal(horiz, right, top, iw)
        elif has_bottom and has_left and not has_top and not has_right:
            _add_vertical(vert, bottom, left, ih)
            _add_horizontal(horiz, left, 0, bottom)
        elif has_bottom and has_right and not has_top and not has_left:
            _add_vertical(vert, bottom, right, ih)
            _add_horizontal(horiz, right, bottom, iw)
        elif has_top and has_left and has_right and not has_bottom:
            assert left == right
            _add_vertical(vert, top, 0, left)
            _add_horizontal(horiz, left, 0, iw)
        elif has_bottom and has_left and has_right and not has_top:
            assert left == right
            _add_vertical(vert, bottom, left, ih)
            _add_horizontal(horiz, left, 0, iw)
        elif has_left and has_top and has_bottom and not has_right:
            assert top == bottom
            _add_vertical(vert, top, 0, ih)
            _add_horizontal(horiz, left, 0, top)
        elif has_right and has_top and has_bottom and not has_left:
            assert top == bottom
            _add_vertical(vert, top, 0, ih)
            _add_horizontal(horiz, right, top, iw)
        elif has_top and has_bottom and has_left and has_right:
            assert top == bottom and left == right
            _add_vertical(vert, top, 0, ih)
            _add_horizontal(horiz, left, 0, iw)
        else:
            raise AssertionError("unrecognized marker topology")

    return r0, c0, r1, c1, vert, horiz


def _regions_from_blocks(
    ih: int, iw: int, vert: np.ndarray, horiz: np.ndarray
) -> list[list[tuple[int, int]]]:
    seen = np.zeros((ih, iw), dtype=bool)
    regions: list[list[tuple[int, int]]] = []

    for sr in range(ih):
        for sc in range(iw):
            if seen[sr, sc]:
                continue
            seen[sr, sc] = True
            q = deque([(sr, sc)])
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                if r > 0 and not horiz[r - 1, c] and not seen[r - 1, c]:
                    seen[r - 1, c] = True
                    q.append((r - 1, c))
                if r + 1 < ih and not horiz[r, c] and not seen[r + 1, c]:
                    seen[r + 1, c] = True
                    q.append((r + 1, c))
                if c > 0 and not vert[r, c - 1] and not seen[r, c - 1]:
                    seen[r, c - 1] = True
                    q.append((r, c - 1))
                if c + 1 < iw and not vert[r, c] and not seen[r, c + 1]:
                    seen[r, c + 1] = True
                    q.append((r, c + 1))
            regions.append(cells)
    return regions


def _region_descriptions(
    inner: np.ndarray, regions: list[list[tuple[int, int]]]
) -> list[dict[str, object]]:
    descs: list[dict[str, object]] = []
    for cells in regions:
        counts = Counter(int(inner[r, c]) for r, c in cells)
        base = counts.most_common(1)[0][0]
        cell_set = set(cells)
        off = [(r, c) for r, c in cells if int(inner[r, c]) != base]
        anomaly = None
        if off:
            off_counts = Counter(int(inner[r, c]) for r, c in off)
            if len(off_counts) == 1:
                anomaly_color = next(iter(off_counts))
                rs = [r for r, _ in off]
                cs = [c for _, c in off]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)
                rect = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
                if len(rect) == len(off) and all((r, c) in cell_set for r, c in rect):
                    anomaly = {
                        "color": anomaly_color,
                        "shape": (r1 - r0 + 1, c1 - c0 + 1),
                        "bbox": (r0, c0, r1, c1),
                    }
        descs.append({"cells": cells, "base": base, "anomaly": anomaly})
    return descs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct hidden orthogonal separators from edge markers, then undo each
    region's swapped rectangle so every inferred region becomes monochrome.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    r0, c0, r1, c1, vert, horiz = _reconstruct_blocks(g, bg)
    inner = g[r0 : r1 + 1, c0 : c1 + 1]
    regions = _regions_from_blocks(inner.shape[0], inner.shape[1], vert, horiz)
    info = _region_descriptions(inner, regions)

    out = np.full_like(g, bg)
    for region in info:
        color = int(region["base"])
        for r, c in region["cells"]:
            out[r0 + r, c0 + c] = color
    return out


def _sample_coord(lo: int, hi: int, rng: random.Random) -> int:
    if lo > hi:
        lo, hi = 1, max(1, hi)
    return rng.randint(lo, hi)


def _sample_component(kind: str, ih: int, iw: int, rng: random.Random) -> dict[str, int | str]:
    x = _sample_coord(2, iw - 2, rng)
    y = _sample_coord(2, ih - 2, rng)
    spec: dict[str, int | str] = {"kind": kind}
    if kind == "vert":
        spec["x"] = x
    elif kind == "horiz":
        spec["y"] = y
    else:
        spec["x"] = x
        spec["y"] = y
    return spec


def _component_segments(spec: dict[str, int | str], ih: int, iw: int) -> list[tuple[str, int, int, int]]:
    kind = str(spec["kind"])
    x = int(spec.get("x", -1))
    y = int(spec.get("y", -1))
    if kind == "vert":
        return [("v", x, 0, ih)]
    if kind == "horiz":
        return [("h", y, 0, iw)]
    if kind == "L_tl":
        return [("v", x, 0, y), ("h", y, 0, x)]
    if kind == "L_tr":
        return [("v", x, 0, y), ("h", y, x, iw)]
    if kind == "L_bl":
        return [("v", x, y, ih), ("h", y, 0, x)]
    if kind == "L_br":
        return [("v", x, y, ih), ("h", y, x, iw)]
    if kind == "T_top":
        return [("v", x, 0, y), ("h", y, 0, iw)]
    if kind == "T_bottom":
        return [("v", x, y, ih), ("h", y, 0, iw)]
    if kind == "T_left":
        return [("v", x, 0, ih), ("h", y, 0, x)]
    if kind == "T_right":
        return [("v", x, 0, ih), ("h", y, x, iw)]
    if kind == "plus":
        return [("v", x, 0, ih), ("h", y, 0, iw)]
    raise ValueError(kind)


def _component_nodes(spec: dict[str, int | str], ih: int, iw: int) -> set[tuple[int, int]]:
    nodes: set[tuple[int, int]] = set()
    for axis, a, b0, b1 in _component_segments(spec, ih, iw):
        if axis == "v":
            for y in range(b0, b1 + 1):
                nodes.add((y, a))
        else:
            for x in range(b0, b1 + 1):
                nodes.add((a, x))
    return nodes


def _draw_component(spec: dict[str, int | str], vert: np.ndarray, horiz: np.ndarray) -> None:
    for axis, a, b0, b1 in _component_segments(spec, vert.shape[0], horiz.shape[1]):
        if axis == "v":
            _add_vertical(vert, a, b0, b1)
        else:
            _add_horizontal(horiz, a, b0, b1)


def _component_exits(spec: dict[str, int | str]) -> dict[str, int]:
    kind = str(spec["kind"])
    x = int(spec.get("x", -1))
    y = int(spec.get("y", -1))
    if kind == "vert":
        return {"top": x, "bottom": x}
    if kind == "horiz":
        return {"left": y, "right": y}
    if kind == "L_tl":
        return {"top": x, "left": y}
    if kind == "L_tr":
        return {"top": x, "right": y}
    if kind == "L_bl":
        return {"bottom": x, "left": y}
    if kind == "L_br":
        return {"bottom": x, "right": y}
    if kind == "T_top":
        return {"top": x, "left": y, "right": y}
    if kind == "T_bottom":
        return {"bottom": x, "left": y, "right": y}
    if kind == "T_left":
        return {"left": y, "top": x, "bottom": x}
    if kind == "T_right":
        return {"right": y, "top": x, "bottom": x}
    if kind == "plus":
        return {"top": x, "bottom": x, "left": y, "right": y}
    raise ValueError(kind)


def _region_rectangles(
    cells: list[tuple[int, int]], ih: int, iw: int
) -> dict[tuple[int, int], list[tuple[int, int, int, int]]]:
    mask = np.zeros((ih, iw), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    size_cap = max(2, len(cells) // 4)
    rects: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    for r0 in range(min(rows), max(rows) + 1):
        for c0 in range(min(cols), max(cols) + 1):
            if not mask[r0, c0]:
                continue
            for h in range(1, min(4, ih - r0) + 1):
                for w in range(1, min(4, iw - c0) + 1):
                    area = h * w
                    if area < 2 or area > size_cap:
                        continue
                    if np.all(mask[r0 : r0 + h, c0 : c0 + w]):
                        rects[(h, w)].append((r0, c0, h, w))
    return rects


def _choose_pairs(
    rect_maps: list[dict[tuple[int, int], list[tuple[int, int, int, int]]]],
    rng: random.Random,
) -> list[tuple[int, int, tuple[int, int, int, int], tuple[int, int, int, int]]]:
    n = len(rect_maps)
    if n < 2:
        return []
    desired = 2 if n >= 4 and rng.random() < 0.5 else 1

    for target in range(desired, 0, -1):
        for _ in range(PAIR_ATTEMPTS):
            used: set[int] = set()
            pairs: list[
                tuple[int, int, tuple[int, int, int, int], tuple[int, int, int, int]]
            ] = []
            while len(pairs) < target:
                free = [idx for idx in range(n) if idx not in used]
                options = []
                for i in range(len(free)):
                    for j in range(i + 1, len(free)):
                        a = free[i]
                        b = free[j]
                        common = list(set(rect_maps[a]) & set(rect_maps[b]))
                        if common:
                            options.append((a, b, common))
                if not options:
                    break
                a, b, common = rng.choice(options)
                size = rng.choice(common)
                ra = rng.choice(rect_maps[a][size])
                rb = rng.choice(rect_maps[b][size])
                pairs.append((a, b, ra, rb))
                used.add(a)
                used.add(b)
            if len(pairs) == target:
                return pairs
    return []


def _paint_markers(
    grid: np.ndarray,
    r0: int,
    c0: int,
    spec: dict[str, int | str],
    color: int,
) -> None:
    exits = _component_exits(spec)
    if "top" in exits:
        grid[0, c0 + exits["top"]] = color
    if "bottom" in exits:
        grid[-1, c0 + exits["bottom"]] = color
    if "left" in exits:
        grid[r0 + exits["left"], 0] = color
    if "right" in exits:
        grid[r0 + exits["right"], -1] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 20)
    w = rng.randint(14, 20)
    top_margin = rng.randint(2, 3)
    bottom_margin = rng.randint(2, 3)
    left_margin = rng.randint(2, 3)
    right_margin = rng.randint(2, 3)
    ih = h - top_margin - bottom_margin
    iw = w - left_margin - right_margin
    if ih < 8 or iw < 8:
        return None

    n_components = 2 if min(ih, iw) >= 10 and rng.random() < 0.5 else 1
    vert = np.zeros((ih, iw - 1), dtype=bool)
    horiz = np.zeros((ih - 1, iw), dtype=bool)
    specs: list[dict[str, int | str]] = []
    used_nodes: set[tuple[int, int]] = set()

    for _ in range(n_components):
        placed = False
        for _ in range(COMPONENT_ATTEMPTS):
            kind = rng.choice(SHAPE_POOL)
            spec = _sample_component(kind, ih, iw, rng)
            nodes = _component_nodes(spec, ih, iw)
            if nodes & used_nodes:
                continue
            trial_vert = vert.copy()
            trial_horiz = horiz.copy()
            _draw_component(spec, trial_vert, trial_horiz)
            specs.append(spec)
            vert = trial_vert
            horiz = trial_horiz
            used_nodes |= nodes
            placed = True
            break
        if not placed:
            return None

    regions = _regions_from_blocks(ih, iw, vert, horiz)
    if not (2 <= len(regions) <= 7):
        return None
    if min(len(region) for region in regions) < 8:
        return None
    if len(regions) + n_components > 9:
        return None

    bg_slots = h * w - ih * iw - sum(len(_component_exits(spec)) for spec in specs)
    if bg_slots <= max(len(region) for region in regions):
        return None

    bg = rng.randint(0, 9)
    others = [color for color in range(10) if color != bg]
    rng.shuffle(others)
    marker_colors = others[:n_components]
    region_colors = others[n_components : n_components + len(regions)]

    base_inner = np.zeros((ih, iw), dtype=int)
    rect_maps = []
    for idx, region in enumerate(regions):
        for r, c in region:
            base_inner[r, c] = region_colors[idx]
        rect_maps.append(_region_rectangles(region, ih, iw))

    pairs = _choose_pairs(rect_maps, rng)
    if not pairs:
        return None

    inp_inner = base_inner.copy()
    for a, b, ra, rb in pairs:
        ca = int(region_colors[a])
        cb = int(region_colors[b])
        ar0, ac0, ah, aw = ra
        br0, bc0, bh, bw = rb
        assert (ah, aw) == (bh, bw)
        inp_inner[ar0 : ar0 + ah, ac0 : ac0 + aw] = cb
        inp_inner[br0 : br0 + bh, bc0 : bc0 + bw] = ca

    grid = np.full((h, w), bg, dtype=int)
    grid[
        top_margin : top_margin + ih,
        left_margin : left_margin + iw,
    ] = inp_inner

    for spec, color in zip(specs, marker_colors):
        _paint_markers(grid, top_margin, left_margin, spec, int(color))

    counts = Counter(int(v) for v in grid.ravel())
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None

    expected = np.full((h, w), bg, dtype=int)
    expected[
        top_margin : top_margin + ih,
        left_margin : left_margin + iw,
    ] = base_inner
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-separator swap puzzle")
