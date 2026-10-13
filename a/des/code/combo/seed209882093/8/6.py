from __future__ import annotations

import numpy as np


COLOR_PRIORITY = [2, 7, 4, 3, 1, 8, 6, 9]  # Red, Orange, Yellow, Green, Blue, Azure, Magenta, Maroon
COLOR_RANK = {color: rank for rank, color in enumerate(COLOR_PRIORITY)}
BG_COLOR = 0
GRID_MIN = 20
GRID_MAX = 30
PAIR_COUNT_OPTIONS = [2, 3, 4]
PAIR_COUNT_WEIGHTS = np.array([0.4, 0.4, 0.2], dtype=float)
GAP_MIN = 2
GAP_MAX = 8
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_TRIES = 300
SHAPE_TRIES = 200
_RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _iter_neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _iter_neighbors4(rr, cc, height, width):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _is_connected(mask: np.ndarray) -> bool:
    return len(_connected_components(mask)) == 1


def _count_holes(mask: np.ndarray) -> int:
    holes = 0
    height, width = mask.shape
    for comp in _connected_components(~mask):
        if all(0 < r < height - 1 and 0 < c < width - 1 for r, c in comp):
            holes += 1
    return holes


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _extract_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []

    for r in range(height):
        for c in range(width):
            if bool(seen[r, c]) or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _iter_neighbors4(rr, cc, height, width):
                    if not bool(seen[nr, nc]) and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "holes": _count_holes(mask),
                }
            )
    return components


def _bridge_color(color_a: int, color_b: int) -> int:
    assert color_a in COLOR_RANK and color_b in COLOR_RANK
    return color_a if COLOR_RANK[color_a] <= COLOR_RANK[color_b] else color_b


def _draw_bridge_between_components(out: np.ndarray, comp_a: dict, comp_b: dict) -> None:
    (r0a, c0a, r1a, c1a) = comp_a["bbox"]
    (r0b, c0b, r1b, c1b) = comp_b["bbox"]
    bridge_width = max(int(comp_a["holes"]), int(comp_b["holes"])) + 1
    bridge_color = _bridge_color(int(comp_a["color"]), int(comp_b["color"]))

    row_overlap = min(r1a, r1b) - max(r0a, r0b) + 1
    col_overlap = min(c1a, c1b) - max(c0a, c0b) + 1

    horizontal = c1a < c0b or c1b < c0a
    vertical = r1a < r0b or r1b < r0a
    assert horizontal != vertical

    if horizontal:
        left, right = (comp_a, comp_b) if c0a < c0b else (comp_b, comp_a)
        overlap_start = max(int(left["bbox"][0]), int(right["bbox"][0]))
        assert row_overlap >= bridge_width
        row0 = overlap_start + (row_overlap - bridge_width) // 2
        col0 = int(left["bbox"][3]) + 1
        col1 = int(right["bbox"][1])
        assert col0 < col1
        out[row0 : row0 + bridge_width, col0:col1] = bridge_color
    else:
        top, bottom = (comp_a, comp_b) if r0a < r0b else (comp_b, comp_a)
        overlap_start = max(int(top["bbox"][1]), int(bottom["bbox"][1]))
        assert col_overlap >= bridge_width
        col0 = overlap_start + (col_overlap - bridge_width) // 2
        row0 = int(top["bbox"][2]) + 1
        row1 = int(bottom["bbox"][0])
        assert row0 < row1
        out[row0:row1, col0 : col0 + bridge_width] = bridge_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair exact same-shape objects, then bridge each pair.

    Bridge color = the earlier color in the explicit palette order
    (red, orange, yellow, green, blue, azure, magenta, maroon).
    Bridge width = max hole count in the pair + 1.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    components = _extract_components(g, bg)
    assert components

    by_shape: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for comp in components:
        by_shape.setdefault(_shape_key(comp["mask"]), []).append(comp)

    out = g.copy()
    for group in by_shape.values():
        assert len(group) == 2
        comp_a, comp_b = group
        assert int(comp_a["color"]) != int(comp_b["color"])
        _draw_bridge_between_components(out, comp_a, comp_b)
    return out


def _random_notch(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    mode = str(_RNG.choice(["corner", "edge"]))
    if mode == "corner":
        dh = int(_RNG.integers(1, h))
        dw = int(_RNG.integers(1, w))
        corner = str(_RNG.choice(["tl", "tr", "bl", "br"]))
        if corner == "tl":
            out[:dh, :dw] = False
        elif corner == "tr":
            out[:dh, w - dw :] = False
        elif corner == "bl":
            out[h - dh :, :dw] = False
        else:
            out[h - dh :, w - dw :] = False
        return out

    side = str(_RNG.choice(["top", "bottom", "left", "right"]))
    if side in ("top", "bottom"):
        notch_width = int(_RNG.integers(1, w - 1))
        start = int(_RNG.integers(1, w - notch_width))
        depth = int(_RNG.integers(1, h - 1))
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        out[rows, start : start + notch_width] = False
    else:
        notch_height = int(_RNG.integers(1, h - 1))
        start = int(_RNG.integers(1, h - notch_height))
        depth = int(_RNG.integers(1, w - 1))
        cols = slice(0, depth) if side == "left" else slice(w - depth, w)
        out[start : start + notch_height, cols] = False
    return out


def _all_hole_rectangles(h: int, w: int) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for r0 in range(1, h - 1):
        for r1 in range(r0, h - 2 + 1):
            for c0 in range(1, w - 1):
                for c1 in range(c0, w - 2 + 1):
                    rects.append((r0, r1, c0, c1))
    return rects


def _rects_overlap_or_touch(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]) -> bool:
    ar0, ar1, ac0, ac1 = rect_a
    br0, br1, bc0, bc1 = rect_b
    rows_overlap = max(ar0, br0) <= min(ar1, br1)
    cols_overlap = max(ac0, bc0) <= min(ac1, bc1)
    if rows_overlap and cols_overlap:
        return True

    row_gap = 0
    if ar1 < br0:
        row_gap = br0 - ar1 - 1
    elif br1 < ar0:
        row_gap = ar0 - br1 - 1

    col_gap = 0
    if ac1 < bc0:
        col_gap = bc0 - ac1 - 1
    elif bc1 < ac0:
        col_gap = ac0 - bc1 - 1

    return row_gap == 0 and col_gap == 0


def _generate_shape(target_holes: int) -> np.ndarray:
    for _ in range(SHAPE_TRIES):
        if target_holes == 2:
            dims = [
                (h, w)
                for h in range(3, 7)
                for w in range(3, 7)
                if ((h - 2) >= 2 and (w - 2) >= 1) or ((h - 2) >= 1 and (w - 2) >= 2)
            ]
            h, w = dims[int(_RNG.integers(len(dims)))]
        else:
            h = int(_RNG.integers(3, 7))
            w = int(_RNG.integers(3, 7))

        mask = np.ones((h, w), dtype=bool)
        if target_holes == 0:
            notch_count = int(_RNG.choice([0, 1, 2], p=[0.25, 0.55, 0.20]))
            for _ in range(notch_count):
                mask = _random_notch(mask)
            if not mask.any():
                continue
        else:
            rects = _all_hole_rectangles(h, w)
            if target_holes == 1:
                r0, r1, c0, c1 = rects[int(_RNG.integers(len(rects)))]
                mask[r0 : r1 + 1, c0 : c1 + 1] = False
            else:
                _RNG.shuffle(rects)
                picked: list[tuple[int, int, int, int]] = []
                for rect in rects:
                    if all(not _rects_overlap_or_touch(rect, prev) for prev in picked):
                        picked.append(rect)
                    if len(picked) == 2:
                        break
                if len(picked) != 2:
                    continue
                for r0, r1, c0, c1 in picked:
                    mask[r0 : r1 + 1, c0 : c1 + 1] = False

        if not _is_connected(mask):
            continue
        if _count_holes(mask) != target_holes:
            continue
        return mask
    raise RuntimeError(f"failed to generate a {target_holes}-hole shape")


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = color


def _pair_footprint(mask: np.ndarray, orientation: str, gap: int) -> tuple[int, int]:
    h, w = map(int, mask.shape)
    if orientation == "h":
        return h, 2 * w + gap
    return 2 * h + gap, w


def _grid_size(pair_specs: list[dict]) -> tuple[int, int] | None:
    max_h = max(spec["footprint_h"] for spec in pair_specs)
    max_w = max(spec["footprint_w"] for spec in pair_specs)
    total_area = sum(spec["footprint_h"] * spec["footprint_w"] for spec in pair_specs)
    base = int(np.ceil(np.sqrt(total_area * 1.8)))
    h_low = max(GRID_MIN, max_h + 2, base - 2)
    w_low = max(GRID_MIN, max_w + 2, base - 2)
    if h_low > GRID_MAX or w_low > GRID_MAX:
        return None
    height = int(_RNG.integers(h_low, GRID_MAX + 1))
    width = int(_RNG.integers(w_low, GRID_MAX + 1))
    return height, width


def _rect_is_free(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int = 1) -> bool:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(occupied.shape[0] - 1, r1 + margin)
    cc1 = min(occupied.shape[1] - 1, c1 + margin)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _place_pairs(pair_specs: list[dict], height: int, width: int) -> bool:
    occupied = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(pair_specs)), key=lambda idx: pair_specs[idx]["footprint_h"] * pair_specs[idx]["footprint_w"], reverse=True)
    for idx in order:
        spec = pair_specs[idx]
        fh = int(spec["footprint_h"])
        fw = int(spec["footprint_w"])
        if height - fh < 2 or width - fw < 2:
            return False
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = int(_RNG.integers(1, height - fh))
            c0 = int(_RNG.integers(1, width - fw))
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            if not _rect_is_free(occupied, r0, c0, r1, c1):
                continue
            spec["anchor"] = (r0, c0)
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _render_input(pair_specs: list[dict], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    for spec in pair_specs:
        mask = spec["mask"]
        gap = int(spec["gap"])
        color_a, color_b = spec["colors"]
        r0, c0 = spec["anchor"]
        h, w = map(int, mask.shape)
        if spec["orientation"] == "h":
            _paint_mask(grid, mask, r0, c0, color_a)
            _paint_mask(grid, mask, r0, c0 + w + gap, color_b)
        else:
            _paint_mask(grid, mask, r0, c0, color_a)
            _paint_mask(grid, mask, r0 + h + gap, c0, color_b)
    return grid


def _render_expected(pair_specs: list[dict], inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    for spec in pair_specs:
        mask = spec["mask"]
        h, w = map(int, mask.shape)
        gap = int(spec["gap"])
        holes = int(spec["holes"])
        bridge_width = holes + 1
        bridge_color = _bridge_color(int(spec["colors"][0]), int(spec["colors"][1]))
        r0, c0 = spec["anchor"]
        if spec["orientation"] == "h":
            row0 = r0 + (h - bridge_width) // 2
            col0 = c0 + w
            out[row0 : row0 + bridge_width, col0 : col0 + gap] = bridge_color
        else:
            col0 = c0 + (w - bridge_width) // 2
            row0 = r0 + h
            out[row0 : row0 + gap, col0 : col0 + bridge_width] = bridge_color
    return out


def _build_pair_specs() -> list[dict]:
    pair_count = int(_RNG.choice(PAIR_COUNT_OPTIONS, p=PAIR_COUNT_WEIGHTS))
    colors = list(map(int, _RNG.choice(COLOR_PRIORITY, size=2 * pair_count, replace=False)))

    hole_targets = list(map(int, _RNG.choice([0, 1, 2], size=pair_count, p=[0.45, 0.35, 0.20])))
    if pair_count >= 2 and len(set(hole_targets)) == 1 and _RNG.random() < 0.7:
        idx = int(_RNG.integers(pair_count))
        hole_targets[idx] = int((hole_targets[idx] + int(_RNG.integers(1, 3))) % 3)

    orientations = [str(_RNG.choice(["h", "v"])) for _ in range(pair_count)]
    if pair_count >= 2 and len(set(orientations)) == 1 and _RNG.random() < 0.7:
        orientations[int(_RNG.integers(pair_count))] = "v" if orientations[0] == "h" else "h"

    used_shapes: set[tuple[tuple[int, int], bytes]] = set()
    specs: list[dict] = []
    for idx in range(pair_count):
        for _ in range(SHAPE_TRIES):
            mask = _generate_shape(hole_targets[idx])
            key = _shape_key(mask)
            if key in used_shapes:
                continue
            used_shapes.add(key)
            gap = int(_RNG.integers(GAP_MIN, GAP_MAX + 1))
            footprint_h, footprint_w = _pair_footprint(mask, orientations[idx], gap)
            specs.append(
                {
                    "mask": mask,
                    "holes": hole_targets[idx],
                    "colors": (colors[2 * idx], colors[2 * idx + 1]),
                    "orientation": orientations[idx],
                    "gap": gap,
                    "footprint_h": footprint_h,
                    "footprint_w": footprint_w,
                }
            )
            break
        else:
            raise RuntimeError("failed to build unique pair shape")
    return specs


def _attempt_generate() -> np.ndarray | None:
    pair_specs = _build_pair_specs()
    size = _grid_size(pair_specs)
    if size is None:
        return None
    height, width = size
    if not _place_pairs(pair_specs, height, width):
        return None

    inp = _render_input(pair_specs, height, width)
    expected = _render_expected(pair_specs, inp)

    try:
        actual = main(inp)
    except AssertionError:
        return None

    components = _extract_components(inp, BG_COLOR)
    if len(components) != 2 * len(pair_specs):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _attempt_generate()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
