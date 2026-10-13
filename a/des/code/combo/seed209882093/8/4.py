import random
from typing import Optional

import numpy as np


BG = 0
WARM_COLORS = (2, 4, 7, 9)
COOL_COLORS = (1, 3, 5, 8)
ALL_FRAME_COLORS = WARM_COLORS + COOL_COLORS

GRID_MIN = 17
GRID_MAX = 31
LAYER_RANGE = (3, 5)
GAP_CHOICES = (2, 2, 3, 4)
GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    if r0 > r1 or c0 > c1:
        return mask

    left = max(c0, 0)
    right = min(c1, w - 1)
    top = max(r0, 0)
    bottom = min(r1, h - 1)
    if left > right or top > bottom:
        return mask

    if 0 <= r0 < h:
        mask[r0, left : right + 1] = True
    if 0 <= r1 < h:
        mask[r1, left : right + 1] = True
    if 0 <= c0 < w:
        mask[top : bottom + 1, c0] = True
    if 0 <= c1 < w:
        mask[top : bottom + 1, c1] = True
    return mask


def _draw_frame(grid: np.ndarray, color: int, box: tuple[int, int, int, int]) -> None:
    grid[_frame_mask(grid.shape, box)] = int(color)


def _camp(color: int) -> str:
    if color in WARM_COLORS:
        return "warm"
    if color in COOL_COLORS:
        return "cool"
    raise AssertionError(f"unexpected frame color: {color}")


def _extract_layers(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    assert 3 <= len(colors) <= 5, "expected 3-5 nested frames"

    layers: list[dict[str, object]] = []
    center_sig: Optional[tuple[int, int]] = None
    for color in colors:
        mask = g == color
        box = _bbox(mask)
        assert box is not None, "each frame color must appear"
        expected = _frame_mask(g.shape, box)
        assert np.array_equal(mask, expected), "each color must form a complete 1-pixel rectangular frame"

        r0, c0, r1, c1 = box
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3
        sig = (r0 + r1, c0 + c1)
        if center_sig is None:
            center_sig = sig
        else:
            assert sig == center_sig, "all frames must share one center"

        layers.append({"color": color, "box": box, "center_sig": sig})

    layers.sort(
        key=lambda layer: (
            -(
                (layer["box"][2] - layer["box"][0] + 1)
                * (layer["box"][3] - layer["box"][1] + 1)
            ),
            layer["box"][0],
            layer["box"][1],
        )
    )

    for outer, inner in zip(layers, layers[1:]):
        or0, oc0, or1, oc1 = outer["box"]
        ir0, ic0, ir1, ic1 = inner["box"]
        assert ir0 - or0 >= 2 and ic0 - oc0 >= 2
        assert or1 - ir1 >= 2 and oc1 - ic1 >= 2

    return layers


def _transform_layer(layer: dict[str, object]) -> tuple[str, tuple[int, ...]]:
    r0, c0, r1, c1 = layer["box"]
    color = int(layer["color"])
    if _camp(color) == "warm":
        return "frame", (r0 - 1, c0 - 1, r1 + 1, c1 + 1)

    nr0, nc0, nr1, nc1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    if nr1 - nr0 + 1 < 3 or nc1 - nc0 + 1 < 3:
        return "point", ((r0 + r1) // 2, (c0 + c1) // 2)
    return "frame", (nr0, nc0, nr1, nc1)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    layers = _extract_layers(g)

    out = np.full(g.shape, BG, dtype=int)
    for layer in reversed(layers):
        color = int(layer["color"])
        kind, geom = _transform_layer(layer)
        if kind == "point":
            r, c = geom
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                out[r, c] = color
        else:
            _draw_frame(out, color, geom)  # inner first, outer later => outer takes priority
    return out


def _random_odd(low: int, high: int) -> int:
    values = [v for v in range(low, high + 1) if v % 2 == 1]
    return random.choice(values)


def _build_extents(
    layer_count: int,
    max_extent: int,
    force_collapse: bool,
    force_overlap: bool,
) -> Optional[list[int]]:
    gaps = [random.choice(GAP_CHOICES) for _ in range(layer_count - 1)]
    if force_overlap and gaps:
        gaps[random.randrange(len(gaps))] = 2

    total_gap = sum(gaps)
    if max_extent - total_gap < 1:
        return None

    if force_collapse:
        inner_extent = 1
    else:
        upper = max_extent - total_gap
        if upper <= 0:
            return None
        inner_extent = random.randint(1, min(4, upper)) if upper >= 4 else random.randint(1, upper)

    extents = [inner_extent]
    for gap in gaps:
        extents.append(extents[-1] + gap)
    if extents[-1] > max_extent:
        return None
    return list(reversed(extents))


def _choose_camps(
    layer_count: int,
    hrs: list[int],
    hcs: list[int],
    force_collapse: bool,
    force_overlap: bool,
) -> Optional[list[bool]]:
    forced: dict[int, bool] = {}

    if force_collapse and (hrs[-1] == 1 or hcs[-1] == 1):
        forced[layer_count - 1] = False

    overlap_candidates = []
    for i in range(layer_count - 1):
        if hrs[i] - hrs[i + 1] == 2 or hcs[i] - hcs[i + 1] == 2:
            overlap_candidates.append(i)
    if force_overlap and overlap_candidates:
        idx = random.choice(overlap_candidates)
        forced[idx] = False
        forced[idx + 1] = True

    forced_warm = sum(1 for v in forced.values() if v)
    forced_cool = sum(1 for v in forced.values() if not v)
    warm_min = max(1, layer_count - len(COOL_COLORS), forced_warm)
    warm_max = min(len(WARM_COLORS), layer_count - 1, layer_count - forced_cool)
    if warm_min > warm_max:
        return None

    warm_total = random.randint(warm_min, warm_max)
    remaining = [i for i in range(layer_count) if i not in forced]
    extra_warm = warm_total - forced_warm
    if not 0 <= extra_warm <= len(remaining):
        return None

    warm_positions = {i for i, v in forced.items() if v}
    warm_positions.update(random.sample(remaining, extra_warm))
    return [i in warm_positions for i in range(layer_count)]


def _choose_colors(is_warm: list[bool]) -> list[int]:
    warm_positions = [i for i, flag in enumerate(is_warm) if flag]
    cool_positions = [i for i, flag in enumerate(is_warm) if not flag]
    warm_colors = random.sample(list(WARM_COLORS), len(warm_positions))
    cool_colors = random.sample(list(COOL_COLORS), len(cool_positions))
    random.shuffle(warm_colors)
    random.shuffle(cool_colors)

    colors = [0] * len(is_warm)
    for idx in warm_positions:
        colors[idx] = warm_colors.pop()
    for idx in cool_positions:
        colors[idx] = cool_colors.pop()
    return colors


def _gen_attempt() -> Optional[np.ndarray]:
    layer_count = random.randint(*LAYER_RANGE)
    force_collapse = random.random() < 0.4
    force_overlap = random.random() < 0.5

    min_extent = 1 + 2 * (layer_count - 1)
    min_h = max(GRID_MIN, 2 * (min_extent + 1) + 1)
    min_w = max(GRID_MIN, 2 * (min_extent + 1) + 1)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = _random_odd(min_h, GRID_MAX)
    w = _random_odd(min_w, GRID_MAX)
    center_r = h // 2
    center_c = w // 2
    max_hr = center_r - 1
    max_hc = center_c - 1

    collapse_on_rows = force_collapse and random.random() < 0.5
    collapse_on_cols = force_collapse and not collapse_on_rows

    hrs = _build_extents(layer_count, max_hr, collapse_on_rows, force_overlap)
    hcs = _build_extents(layer_count, max_hc, collapse_on_cols, force_overlap)
    if hrs is None or hcs is None:
        return None

    is_warm = _choose_camps(layer_count, hrs, hcs, force_collapse, force_overlap)
    if is_warm is None:
        return None
    colors = _choose_colors(is_warm)

    grid = np.full((h, w), BG, dtype=int)
    for hr, hc, color in zip(hrs, hcs, colors):
        box = (center_r - hr, center_c - hc, center_r + hr, center_c + hc)
        _draw_frame(grid, color, box)

    # Validate the generated structure against the solver assumptions.
    layers = _extract_layers(grid)
    assert len(layers) == layer_count
    assert set(colors) == {int(layer["color"]) for layer in layers}
    assert all(int(c) in ALL_FRAME_COLORS for c in colors)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
