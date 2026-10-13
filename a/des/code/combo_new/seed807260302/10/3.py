from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = np.arange(1, 10)
GENERATION_RETRIES = 500
PATTERN_RETRIES = 100


def _looks_like_square_layer(patch: np.ndarray) -> bool:
    side = int(patch.shape[0])
    if side != int(patch.shape[1]):
        return False
    if bool(patch.all()):
        return True

    rows, cols = np.indices((side, side))
    dist = np.minimum.reduce((rows, cols, side - 1 - rows, side - 1 - cols))
    for thickness in range(1, side // 2 + 1):
        if bool(np.array_equal(patch, dist < thickness)):
            return True
    return False


def _monochrome_square_layers(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    height, width = map(int, grid.shape)
    seen = np.zeros((height, width), dtype=bool)
    layers: list[dict[str, int | tuple[int, int, int, int]]] = []

    for start_r in range(height):
        for start_c in range(width):
            color = int(grid[start_r, start_c])
            if color == BLACK or bool(seen[start_r, start_c]):
                continue

            queue: deque[tuple[int, int]] = deque([(start_r, start_c)])
            seen[start_r, start_c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                r, c = queue.popleft()
                coords.append((int(r), int(c)))
                for dr, dc in DIR4:
                    rr, cc = int(r + dr), int(c + dc)
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if bool(seen[rr, cc]) or int(grid[rr, cc]) != color:
                        continue
                    seen[rr, cc] = True
                    queue.append((rr, cc))

            pts = np.array(coords, dtype=int)
            r0, c0 = map(int, pts.min(axis=0))
            r1, c1 = map(int, pts.max(axis=0))
            side_h = int(r1 - r0 + 1)
            side_w = int(c1 - c0 + 1)
            assert side_h == side_w, "each colored layer must have a square bbox"

            patch = grid[r0 : r1 + 1, c0 : c1 + 1] == color
            assert _looks_like_square_layer(patch), "each component must be a square frame or solid square"

            layers.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "side": int(side_h),
                    "center_r2": int(r0 + r1),
                    "center_c2": int(c0 + c1),
                }
            )

    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flatten concentric square-ring patterns into left-to-right color columns.

    Input structure:
    - The background is black.
    - The grid contains several non-overlapping concentric square patterns.
    - Each pattern has 2 to 4 differently colored square-frame layers, with the
      innermost layer optionally rendered as a solid square.

    Rule:
    - Group colored square layers by their common center.
    - Within each group, read colors from the largest outer square to the
      smallest inner square.
    - Sort the groups by the column of their outer layer's top-left corner.
    - Write one output column per group; shorter color sequences are padded
      with black at the bottom.

    Key insight: The nested two-dimensional rings are just ordered color
    sequences once the common center and outer-to-inner layer order are found.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    by_center: dict[tuple[int, int], list[dict[str, int | tuple[int, int, int, int]]]] = {}
    for layer in _monochrome_square_layers(g):
        key = (int(layer["center_r2"]), int(layer["center_c2"]))
        by_center.setdefault(key, []).append(layer)

    patterns: list[dict[str, int | list[int]]] = []
    for layers in by_center.values():
        layers.sort(key=lambda item: (-int(item["side"]), int(item["r0"]), int(item["c0"])))
        assert 2 <= len(layers) <= 4, "each pattern must have 2 to 4 layers"

        colors = [int(item["color"]) for item in layers]
        assert len(colors) == len(set(colors)), "layers within one pattern must have distinct colors"

        for outer, inner in zip(layers, layers[1:]):
            assert int(outer["side"]) > int(inner["side"]), "nested layers must shrink inward"
            assert int(outer["r0"]) <= int(inner["r0"]) <= int(inner["r1"]) <= int(outer["r1"])
            assert int(outer["c0"]) <= int(inner["c0"]) <= int(inner["c1"]) <= int(outer["c1"])

        outer = layers[0]
        patterns.append(
            {
                "r0": int(outer["r0"]),
                "c0": int(outer["c0"]),
                "colors": colors,
            }
        )

    assert patterns, "expected at least one concentric pattern"
    patterns.sort(key=lambda item: (int(item["c0"]), int(item["r0"])))

    max_layers = max(len(item["colors"]) for item in patterns)
    out = np.full((int(max_layers), len(patterns)), BLACK, dtype=int)
    for col, pattern in enumerate(patterns):
        for row, color in enumerate(pattern["colors"]):
            out[int(row), int(col)] = int(color)
    return out


def _paint_frame(grid: np.ndarray, r0: int, c0: int, side: int, thickness: int, color: int) -> None:
    r1 = int(r0 + side)
    c1 = int(c0 + side)
    t = int(thickness)
    grid[r0 : r0 + t, c0:c1] = int(color)
    grid[r1 - t : r1, c0:c1] = int(color)
    grid[r0:r1, c0 : c0 + t] = int(color)
    grid[r0:r1, c1 - t : c1] = int(color)


def _sample_pattern(rng: np.random.Generator) -> dict[str, int | bool | list[int]] | None:
    for _ in range(PATTERN_RETRIES):
        layer_count = int(rng.integers(2, 5))
        solid_inner = bool(rng.integers(0, 2))
        colors = rng.choice(COLOR_POOL, size=layer_count, replace=False).astype(int).tolist()

        if solid_inner:
            thicknesses = [int(rng.choice([1, 1, 1, 2])) for _ in range(layer_count - 1)]
            inner_side = int(rng.integers(1, 5))
            side = int(inner_side + 2 * sum(thicknesses))
        else:
            thicknesses = [int(rng.choice([1, 1, 1, 2])) for _ in range(layer_count)]
            hole_side = int(rng.integers(1, 4))
            side = int(hole_side + 2 * sum(thicknesses))

        if 3 <= side <= 10:
            return {
                "colors": colors,
                "thicknesses": thicknesses,
                "solid_inner": solid_inner,
                "side": int(side),
            }
    return None


def _render_pattern(grid: np.ndarray, r0: int, c0: int, spec: dict[str, int | bool | list[int]]) -> None:
    colors = list(spec["colors"])
    thicknesses = list(spec["thicknesses"])
    side = int(spec["side"])
    r, c = int(r0), int(c0)

    if bool(spec["solid_inner"]):
        for color, thickness in zip(colors[:-1], thicknesses):
            _paint_frame(grid, r, c, side, int(thickness), int(color))
            r += int(thickness)
            c += int(thickness)
            side -= 2 * int(thickness)
        grid[r : r + side, c : c + side] = int(colors[-1])
    else:
        for color, thickness in zip(colors, thicknesses):
            _paint_frame(grid, r, c, side, int(thickness), int(color))
            r += int(thickness)
            c += int(thickness)
            side -= 2 * int(thickness)


def _expected_from_placements(placements: list[dict[str, int | list[int]]]) -> np.ndarray:
    ordered = sorted(placements, key=lambda item: (int(item["c0"]), int(item["r0"])))
    max_layers = max(len(item["colors"]) for item in ordered)
    out = np.full((int(max_layers), len(ordered)), BLACK, dtype=int)
    for col, item in enumerate(ordered):
        for row, color in enumerate(item["colors"]):
            out[int(row), int(col)] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    pattern_count = int(rng.integers(2, 5))
    specs = [_sample_pattern(rng) for _ in range(pattern_count)]
    if any(spec is None for spec in specs):
        return None
    specs = [spec for spec in specs if spec is not None]

    sides = [int(spec["side"]) for spec in specs]
    gaps = [int(rng.integers(0, 4))]
    gaps.extend(int(rng.integers(1, 4)) for _ in range(pattern_count - 1))
    gaps.append(int(rng.integers(0, 4)))

    base_width = int(sum(sides) + sum(gaps))
    if base_width > 30:
        return None

    slack = int(rng.integers(0, 31 - base_width))
    for _ in range(slack):
        gaps[int(rng.integers(0, len(gaps)))] += 1
    width = int(sum(sides) + sum(gaps))

    min_height = int(max(sides))
    height = int(rng.integers(min_height, 31))
    grid = np.zeros((height, width), dtype=int)

    placements: list[dict[str, int | list[int]]] = []
    col = int(gaps[0])
    for idx, spec in enumerate(specs):
        side = int(spec["side"])
        row = int(rng.integers(0, height - side + 1))
        _render_pattern(grid, row, col, spec)
        placements.append({"r0": int(row), "c0": int(col), "colors": list(spec["colors"])})
        col += int(side + gaps[idx + 1])

    expected = _expected_from_placements(placements)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not bool(np.array_equal(actual, expected)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_RETRIES} attempts")
