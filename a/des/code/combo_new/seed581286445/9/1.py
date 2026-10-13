from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
FILL_COLORS = (1, 2, 3, 4, 6, 7, 8)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _rot90(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h, _ = _shape_dims(shape)
    return _normalize([(c, h - 1 - r) for r, c in shape])


def _flip_h(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _, w = _shape_dims(shape)
    return _normalize([(r, w - 1 - c) for r, c in shape])


def _orientations(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    cur = shape
    for _ in range(4):
        cur = _normalize(cur)
        if cur not in seen:
            seen.add(cur)
            out.append(cur)
        cur = _rot90(cur)
    cur = _flip_h(shape)
    for _ in range(4):
        cur = _normalize(cur)
        if cur not in seen:
            seen.add(cur)
            out.append(cur)
        cur = _rot90(cur)
    return out


def _draw_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], r0: int, c0: int, color: int) -> None:
    for dr, dc in shape:
        grid[r0 + dr, c0 + dc] = color


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _frame_components(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        frame = grid[r0 : r1 + 1, c0 : c1 + 1] == GREY
        if not np.all(frame[0, :]) or not np.all(frame[-1, :]) or not np.all(frame[:, 0]) or not np.all(frame[:, -1]):
            continue
        if np.any(frame[1:-1, 1:-1]):
            continue
        frames.append((r0, c0, r1, c1))
    return sorted(frames)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each maroon seed shape to a legend template and fill its enclosing grey frame with the legend color.

    Input structure:
    - Black is the background.
    - A legend band appears above the main area and contains several separated monochrome template sprites in different colors.
    - Lower in the grid, one or more one-pixel-thick Grey rectangular frames enclose Black interiors.
    - Each frame interior contains exactly one Maroon seed sprite whose pixel shape matches one legend sprite exactly.

    Rule:
    - Read every colored legend sprite above the frames and index it by its normalized pixel shape.
    - For each Grey frame, extract the normalized shape of the Maroon seed inside its interior.
    - Find the legend sprite with the same shape and take that legend color.
    - Fill the entire interior of the frame with that color, replacing both the Maroon seed and the remaining Black cells, while leaving the Grey walls and the legend unchanged.

    Key insight:
    - The Maroon seed is not the output color itself; it is a shape key that selects the correct color from the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    frames = _frame_components(g)
    assert frames, "expected at least one grey frame"

    legend_bottom = min(r0 for r0, _, _, _ in frames)
    legend_zone = g[:legend_bottom, :]
    legend_mask = (legend_zone != BLACK) & (legend_zone != GREY) & (legend_zone != MAROON)
    legend_components = _components(legend_mask)
    assert legend_components, "expected at least one legend template"

    shape_to_color: dict[tuple[tuple[int, int], ...], int] = {}
    for comp in legend_components:
        colors = {int(legend_zone[r, c]) for r, c in comp}
        assert len(colors) == 1, "each legend component must be monochrome"
        shape = _normalize(comp)
        assert shape not in shape_to_color, "legend shapes must be unique"
        shape_to_color[shape] = colors.pop()

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_components = _components(interior == MAROON)
        assert len(seed_components) == 1, "each frame must contain exactly one maroon seed"
        shape = _normalize(seed_components[0])
        assert shape in shape_to_color, "seed shape must exist in the legend"
        fill_color = shape_to_color[shape]
        allowed = {BLACK, MAROON}
        assert set(map(int, np.unique(interior))).issubset(allowed), "frame interiors should contain only black and maroon"
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color

    return out


BASE_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
]


def _shape_pool() -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for base in BASE_SHAPES:
        for shape in _orientations(base):
            if shape not in seen:
                seen.add(shape)
                out.append(shape)
    return out


SHAPE_POOL = _shape_pool()
GEN_RETRIES = 400


def _legend_width(shapes: list[tuple[tuple[int, int], ...]], gaps: list[int]) -> int:
    widths = [_shape_dims(shape)[1] for shape in shapes]
    return 2 + sum(widths) + sum(gaps)


def _choose_shapes() -> list[tuple[tuple[tuple[int, int], ...], int]]:
    for _ in range(100):
        count = random.randint(2, 4)
        shapes = random.sample(SHAPE_POOL, count)
        gaps = [random.randint(2, 4) for _ in range(count - 1)]
        if _legend_width(shapes, gaps) <= 28:
            colors = random.sample(FILL_COLORS, count)
            order = list(range(count))
            random.shuffle(order)
            return [(shapes[i], colors[i]) for i in order]
    raise AssertionError("failed to choose legend shapes")


def _frame_specs(legend: list[tuple[tuple[tuple[int, int], ...], int]]) -> list[dict[str, int | tuple[tuple[int, int], ...]]]:
    base_indices = list(range(len(legend)))
    extras = random.randint(0, 2)
    for _ in range(extras):
        base_indices.append(random.randrange(len(legend)))
    random.shuffle(base_indices)

    specs: list[dict[str, int | tuple[tuple[int, int], ...]]] = []
    for idx in base_indices:
        shape, color = legend[idx]
        sh, sw = _shape_dims(shape)
        interior_h = random.randint(sh + 2, min(9, sh + 5))
        interior_w = random.randint(sw + 2, min(10, sw + 6))
        seed_r = random.randint(1, interior_h - sh - 1)
        seed_c = random.randint(1, interior_w - sw - 1)
        specs.append(
            {
                "shape": shape,
                "color": color,
                "interior_h": interior_h,
                "interior_w": interior_w,
                "seed_r": seed_r,
                "seed_c": seed_c,
            }
        )
    return specs


def _place_frames(grid_h: int, grid_w: int, main_top: int, specs: list[dict[str, int | tuple[tuple[int, int], ...]]]) -> list[tuple[int, int]] | None:
    used = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(
        range(len(specs)),
        key=lambda i: -int((int(specs[i]["interior_h"]) + 4) * (int(specs[i]["interior_w"]) + 4)),
    )

    for idx in order:
        box_h = int(specs[idx]["interior_h"]) + 2
        box_w = int(specs[idx]["interior_w"]) + 2
        candidates: list[tuple[int, int]] = []
        for r0 in range(main_top, grid_h - box_h + 1):
            for c0 in range(0, grid_w - box_w + 1):
                er0 = max(main_top, r0 - 1)
                ec0 = max(0, c0 - 1)
                er1 = min(grid_h - 1, r0 + box_h)
                ec1 = min(grid_w - 1, c0 + box_w)
                if not used[er0 : er1 + 1, ec0 : ec1 + 1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        box_h = int(specs[idx]["interior_h"]) + 2
        box_w = int(specs[idx]["interior_w"]) + 2
        er0 = max(main_top, r0 - 1)
        ec0 = max(0, c0 - 1)
        er1 = min(grid_h - 1, r0 + box_h)
        ec1 = min(grid_w - 1, c0 + box_w)
        used[er0 : er1 + 1, ec0 : ec1 + 1] = True

    return [(r0, c0) for r0, c0 in placements if r0 is not None and c0 is not None]


def _build_pair() -> tuple[np.ndarray, np.ndarray]:
    legend = _choose_shapes()
    specs = _frame_specs(legend)

    legend_shapes = [shape for shape, _ in legend]
    gaps = [random.randint(2, 4) for _ in range(len(legend) - 1)]
    legend_band_h = max(_shape_dims(shape)[0] for shape in legend_shapes) + random.randint(1, 3)
    main_gap = random.randint(1, 2)
    main_top = legend_band_h + main_gap
    min_w = max(_legend_width(legend_shapes, gaps) + 1, max(int(spec["interior_w"]) + 2 for spec in specs) + 1, 16)
    min_h = max(main_top + max(int(spec["interior_h"]) + 2 for spec in specs) + 1, 16)
    if min_w > 30 or min_h > 30:
        raise AssertionError("layout bounds exceeded")

    total_box_area = sum((int(spec["interior_h"]) + 3) * (int(spec["interior_w"]) + 3) for spec in specs)
    area_floor = min(30, max(16, int(np.ceil(total_box_area ** 0.5)) + 8))
    grid_w = random.randint(max(min_w, area_floor - 2), 30)
    grid_h = random.randint(max(min_h, area_floor), 30)

    placements = _place_frames(grid_h, grid_w, main_top, specs)
    if placements is None:
        raise AssertionError("failed to place frames")

    inp = np.full((grid_h, grid_w), BLACK, dtype=int)
    out = inp.copy()

    legend_span = _legend_width(legend_shapes, gaps)
    left_margin = random.randint(1, grid_w - legend_span + 1)
    c = left_margin
    for idx, (shape, color) in enumerate(legend):
        sh, sw = _shape_dims(shape)
        r0 = random.randint(0, legend_band_h - sh)
        _draw_shape(inp, shape, r0, c, color)
        _draw_shape(out, shape, r0, c, color)
        c += sw
        if idx < len(gaps):
            c += gaps[idx]

    for spec, (r0, c0) in zip(specs, placements, strict=False):
        interior_h = int(spec["interior_h"])
        interior_w = int(spec["interior_w"])
        seed_r = int(spec["seed_r"])
        seed_c = int(spec["seed_c"])
        shape = spec["shape"]
        color = int(spec["color"])

        _draw_frame(inp, r0, c0, interior_h + 2, interior_w + 2)
        _draw_frame(out, r0, c0, interior_h + 2, interior_w + 2)
        _draw_shape(inp, shape, r0 + 1 + seed_r, c0 + 1 + seed_c, MAROON)
        out[r0 + 1 : r0 + 1 + interior_h, c0 + 1 : c0 + 1 + interior_w] = color

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, out = _build_pair()
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
