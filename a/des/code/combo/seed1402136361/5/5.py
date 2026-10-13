from __future__ import annotations

import math
from collections import Counter, deque

import numpy as np


_RNG = np.random.default_rng()
_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_COLORS = tuple(range(10))
_MAX_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(max(counts.items(), key=lambda item: item[1])[0])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                comp.append((rr, cc))
                for dr, dc in _DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for comp in _components4(grid != bg):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each foreground object must be monochrome"
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(comp)
        objects.append(
            {
                "color": color,
                "coords": comp,
                "bbox": (r0, c0, r1, c1),
                "area": len(comp),
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read active colors from the bottom palette row, keep only matching objects,
    and replace each kept object with a centered solid square whose side is the
    square root of the object's area. The palette row is removed from the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    bg = _most_common_color(g)
    palette_colors = [int(v) for v in g[-1] if int(v) != int(bg)]
    active_colors = set(palette_colors)

    src = g[:-1]
    out = np.full(src.shape, int(bg), dtype=int)

    for obj in _extract_objects(src, int(bg)):
        color = int(obj["color"])
        if color not in active_colors:
            continue

        area = int(obj["area"])
        side = int(math.isqrt(area))
        assert side * side == area, "object area must be a perfect square"

        r0, c0, r1, c1 = obj["bbox"]  # type: ignore[assignment]
        nr0 = (int(r0) + int(r1) - side + 1) // 2
        nc0 = (int(c0) + int(c1) - side + 1) // 2
        nr1 = nr0 + side
        nc1 = nc0 + side
        assert 0 <= nr0 <= nr1 <= out.shape[0]
        assert 0 <= nc0 <= nc1 <= out.shape[1]
        out[nr0:nr1, nc0:nc1] = color

    return out


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            nxt = (r + dr, c + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(coords)


def _spans_bbox(coords: set[tuple[int, int]], h: int, w: int) -> bool:
    return (
        any(r == 0 for r, _ in coords)
        and any(r == h - 1 for r, _ in coords)
        and any(c == 0 for _, c in coords)
        and any(c == w - 1 for _, c in coords)
    )


def _shape_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _make_shape(side: int) -> tuple[list[tuple[int, int]], int, int]:
    area = side * side
    dims_options = [(side, side + 2), (side + 2, side), (side, side)]

    for _ in range(200):
        h, w = dims_options[int(_RNG.integers(len(dims_options)))]
        coords = {(r, c) for r in range(h) for c in range(w)}

        while len(coords) > area:
            removable: list[tuple[int, int]] = []
            degrees: list[int] = []
            for cell in coords:
                degree = sum((nbr in coords) for nbr in _shape_neighbors(cell))
                if degree == 4:
                    continue
                candidate = set(coords)
                candidate.remove(cell)
                if _spans_bbox(candidate, h, w) and _is_connected(candidate):
                    removable.append(cell)
                    degrees.append(degree)
            if not removable:
                break

            if _RNG.random() < 0.7:
                min_degree = min(degrees)
                pool = [cell for cell, degree in zip(removable, degrees) if degree == min_degree]
            else:
                pool = removable
            coords.remove(pool[int(_RNG.integers(len(pool)))])

        if len(coords) != area:
            continue
        if (h, w) == (side, side) and _RNG.random() < 0.8:
            continue

        if _RNG.random() < 0.5:
            coords = {(r, w - 1 - c) for r, c in coords}
        if _RNG.random() < 0.5:
            coords = {(h - 1 - r, c) for r, c in coords}
        return sorted(coords), h, w

    raise RuntimeError(f"failed to create shape for side={side}")


def _boxes_separated(
    boxes: list[tuple[int, int, int, int]],
    candidate: tuple[int, int, int, int],
    pad: int = 1,
) -> bool:
    r0, c0, r1, c1 = candidate
    for ar0, ac0, ar1, ac1 in boxes:
        if not (r1 + pad < ar0 or ar1 + pad < r0 or c1 + pad < ac0 or ac1 + pad < c0):
            return False
    return True


def _paint_shape(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    top: int,
    left: int,
    color: int,
) -> None:
    for r, c in coords:
        grid[top + r, left + c] = color


def _build_expected(
    top_h: int,
    w: int,
    bg: int,
    specs: list[dict[str, object]],
    active_colors: set[int],
) -> np.ndarray:
    out = np.full((top_h, w), int(bg), dtype=int)
    for spec in specs:
        color = int(spec["color"])
        if color not in active_colors:
            continue
        r0, c0, r1, c1 = spec["out_box"]  # type: ignore[assignment]
        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    top_h = int(_RNG.integers(14, 21))
    w = int(_RNG.integers(14, 23))
    object_count = int(_RNG.integers(3, 6))

    palette = list(_COLORS)
    _RNG.shuffle(palette)
    bg = int(palette[0])
    colors = [int(v) for v in palette[1 : object_count + 1]]

    side_choices = [3, 4]
    if top_h >= 18 and w >= 18:
        side_choices.append(5)

    raw_specs: list[dict[str, object]] = []
    for color in colors:
        side = int(side_choices[int(_RNG.integers(len(side_choices)))])
        try:
            shape, h, shape_w = _make_shape(side)
        except RuntimeError:
            return None
        raw_specs.append(
            {
                "color": color,
                "side": side,
                "shape": shape,
                "h": h,
                "w": shape_w,
            }
        )

    order = sorted(
        range(len(raw_specs)),
        key=lambda idx: (int(raw_specs[idx]["h"]) * int(raw_specs[idx]["w"]), int(raw_specs[idx]["side"])),
        reverse=True,
    )
    input_boxes: list[tuple[int, int, int, int]] = []
    output_boxes: list[tuple[int, int, int, int]] = []
    placed_specs: list[dict[str, object]] = []

    for idx in order:
        spec = raw_specs[idx]
        h = int(spec["h"])
        shape_w = int(spec["w"])
        side = int(spec["side"])
        placed = False
        for _ in range(300):
            top = int(_RNG.integers(0, top_h - h + 1))
            left = int(_RNG.integers(0, w - shape_w + 1))

            in_box = (top, left, top + h - 1, left + shape_w - 1)
            out_top = top + (h - side) // 2
            out_left = left + (shape_w - side) // 2
            out_box = (out_top, out_left, out_top + side - 1, out_left + side - 1)

            if not _boxes_separated(input_boxes, in_box, pad=1):
                continue
            if not _boxes_separated(output_boxes, out_box, pad=1):
                continue

            spec = dict(spec)
            spec["top"] = top
            spec["left"] = left
            spec["in_box"] = in_box
            spec["out_box"] = out_box
            placed_specs.append(spec)
            input_boxes.append(in_box)
            output_boxes.append(out_box)
            placed = True
            break
        if not placed:
            return None

    active_count = int(_RNG.integers(1, object_count))
    active_colors = set(int(v) for v in _RNG.choice(colors, size=active_count, replace=False).tolist())
    palette_colors = list(active_colors)
    _RNG.shuffle(palette_colors)

    if len(palette_colors) > w:
        return None
    palette_left = int(_RNG.integers(0, w - len(palette_colors) + 1))

    grid = np.full((top_h + 1, w), int(bg), dtype=int)
    for spec in placed_specs:
        _paint_shape(
            grid,
            spec["shape"],  # type: ignore[arg-type]
            int(spec["top"]),
            int(spec["left"]),
            int(spec["color"]),
        )
    for i, color in enumerate(palette_colors):
        grid[top_h, palette_left + i] = int(color)

    if _most_common_color(grid) != bg:
        return None

    objects = _extract_objects(grid[:-1], bg)
    if len(objects) != object_count:
        return None

    expected = _build_expected(top_h, w, bg, placed_specs, active_colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
