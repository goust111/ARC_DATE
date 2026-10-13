from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
MAROON = 9

CONTROL_COLORS = (BLUE, RED, GREEN)
GEN_ATTEMPTS = 500


_TEMPLATE_2 = np.array(
    [
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ],
    dtype=bool,
)

_TEMPLATE_3 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=bool,
)

_LAYER_BAG: list[int] = []
_MODE_BAG: list[str] = []
_QSIZE_BAG: list[int] = []


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    target = {tuple(map(int, p)) for p in coords}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in target and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(target)


def _peel_layers(mask: np.ndarray) -> list[np.ndarray]:
    remaining = mask.astype(bool).copy()
    h, w = remaining.shape
    layers: list[np.ndarray] = []
    while np.any(remaining):
        layer = np.zeros_like(remaining, dtype=bool)
        for r, c in np.argwhere(remaining):
            rr = int(r)
            cc = int(c)
            for nr, nc in _neighbors4(rr, cc):
                if nr < 0 or nr >= h or nc < 0 or nc >= w or not remaining[nr, nc]:
                    layer[rr, cc] = True
                    break
        assert np.any(layer), "peeling must remove at least one pixel"
        layers.append(layer)
        remaining[layer] = False
    return layers


def _transform(mask: np.ndarray, quadrant: str) -> np.ndarray:
    out = mask
    if quadrant in {"bl", "br"}:
        out = np.flipud(out)
    if quadrant in {"tr", "br"}:
        out = np.fliplr(out)
    return out


def _quadrant_slices(mid_r: int, mid_c: int) -> dict[str, tuple[slice, slice]]:
    return {
        "tl": (slice(0, mid_r), slice(0, mid_c)),
        "tr": (slice(0, mid_r), slice(mid_c + 1, None)),
        "bl": (slice(mid_r + 1, None), slice(0, mid_c)),
        "br": (slice(mid_r + 1, None), slice(mid_c + 1, None)),
    }


def _segment_counts(grid: np.ndarray, mid_r: int, mid_c: int) -> dict[str, int]:
    return {
        "top": int(np.count_nonzero(grid[:mid_r, mid_c] == MAROON)),
        "bottom": int(np.count_nonzero(grid[mid_r + 1 :, mid_c] == MAROON)),
        "left": int(np.count_nonzero(grid[mid_r, :mid_c] == MAROON)),
        "right": int(np.count_nonzero(grid[mid_r, mid_c + 1 :] == MAROON)),
    }


def _effective_depths(counts: dict[str, int], layer_count: int) -> dict[str, int]:
    return {
        "tl": min(layer_count, counts["top"] + counts["left"]),
        "tr": min(layer_count, counts["top"] + counts["right"]),
        "bl": min(layer_count, counts["left"] + counts["bottom"]),
        "br": min(layer_count, counts["right"] + counts["bottom"]),
    }


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1, "expected a central divider row and column"

    mid_r = h // 2
    mid_c = w // 2
    qh = mid_r
    qw = mid_c
    assert qh == h - mid_r - 1 and qw == w - mid_c - 1, "quadrants must be equal"
    assert qh == qw, "quadrants must be square"

    divider_colors = set(map(int, np.unique(np.concatenate([g[mid_r, :], g[:, mid_c]]))))
    assert divider_colors.issubset({GREY, MAROON}), "divider must be grey with maroon counters"

    slices = _quadrant_slices(mid_r, mid_c)
    tl = g[slices["tl"]]
    template_mask = tl == ORANGE
    assert np.any(template_mask), "top-left quadrant must contain the orange template"
    assert _is_connected(template_mask), "template must be 4-connected"

    base_layers = _peel_layers(template_mask)
    counts = _segment_counts(g, mid_r, mid_c)
    depths = _effective_depths(counts, len(base_layers))

    controls: dict[str, dict] = {}
    for quadrant, (rs, cs) in slices.items():
        block = g[rs, cs]
        non_default = sorted(set(map(int, np.unique(block))) - {BG, ORANGE})
        if quadrant == "tl":
            assert not non_default, "top-left quadrant should only contain orange template pixels"
            controls[quadrant] = {"color": ORANGE, "mask": np.zeros_like(block, dtype=bool)}
            continue

        assert len(non_default) == 1, "each non-template quadrant must use one control color"
        control_color = int(non_default[0])
        control_mask = block == control_color
        assert 1 <= int(control_mask.sum()) <= 2, "each non-template quadrant needs 1-2 control pixels"
        transformed_template = _transform(template_mask, quadrant)
        assert np.all(~control_mask | transformed_template), "controls must sit on transformed template cells"
        controls[quadrant] = {"color": control_color, "mask": control_mask}

    return {
        "grid": g,
        "mid_r": mid_r,
        "mid_c": mid_c,
        "q": qh,
        "slices": slices,
        "template_mask": template_mask,
        "base_layers": base_layers,
        "segment_counts": counts,
        "depths": depths,
        "controls": controls,
    }


def _render_from_parsed(parsed: dict, *, force_unflipped: bool = False) -> np.ndarray:
    g = parsed["grid"]
    out = np.full_like(g, BG)
    out[parsed["mid_r"], :] = GREY
    out[:, parsed["mid_c"]] = GREY

    for quadrant, (rs, cs) in parsed["slices"].items():
        depth = int(parsed["depths"][quadrant])
        if depth <= 0:
            continue

        transformed_layers = []
        for layer in parsed["base_layers"][:depth]:
            transformed_layers.append(layer if force_unflipped else _transform(layer, quadrant))
        active = np.any(np.stack(transformed_layers, axis=0), axis=0)

        block = g[rs, cs]
        target = out[rs, cs]
        default_color = int(parsed["controls"][quadrant]["color"])
        target[active] = default_color

        control_mask = parsed["controls"][quadrant]["mask"]
        if np.any(control_mask):
            target[active & control_mask] = block[active & control_mask]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the top-left template into all quadrants, then reveal only the counter-selected outer layers in each quadrant.

    Input structure:
    - A black grid is split into four equal square quadrants by a one-cell grey cross divider.
    - The top-left quadrant contains a complete orange template shape on black.
    - The other quadrants contain only 1-2 same-colored control pixels, and the divider segments may contain maroon counter pixels.

    Rule:
    - Peel the orange template into 4-neighbor layers from outside to inside.
    - Reflect that template into the other quadrants: right-top uses a left-right mirror, left-bottom uses an up-down mirror, and right-bottom uses both.
    - For each quadrant, count maroon pixels on its two adjacent divider half-segments and reveal only that many outer layers; revealed pixels take the quadrant's control color, while the top-left keeps orange.

    Key insight:
    - The final pattern is determined jointly by three signals: the template fixes the shape, the divider counters fix the visible depth, and the quadrant control pixels fix the visible color.
    """
    parsed = _parse_input(np.asarray(grid, dtype=int))
    return _render_from_parsed(parsed)


def _bag_draw(pool: list[int] | list[str], values: list[int] | list[str]) -> int | str:
    if not pool:
        pool.extend(values)
        random.shuffle(pool)
    return pool.pop()


def _choose_template() -> tuple[np.ndarray, int]:
    layer_target = int(_bag_draw(_LAYER_BAG, [2, 3, 2, 3, 2, 2]))
    base = _TEMPLATE_2 if layer_target == 2 else _TEMPLATE_3
    mask = base.copy()
    mask = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    assert len(_peel_layers(mask)) == layer_target
    assert not np.array_equal(mask, np.fliplr(mask))
    assert not np.array_equal(mask, np.flipud(mask))
    return mask, layer_target


def _choose_qsize(mask: np.ndarray) -> int:
    h, w = mask.shape
    options = [q for q in (7, 8) if q - h >= 2 and q - w >= 2]
    preferred = int(_bag_draw(_QSIZE_BAG, [7, 8, 7, 8, 7]))
    if preferred in options:
        return preferred
    return int(random.choice(options))


def _place_template_in_quadrant(mask: np.ndarray, q: int) -> np.ndarray:
    h, w = mask.shape
    r0 = random.randint(1, q - h - 1)
    c0 = random.randint(1, q - w - 1)
    out = np.zeros((q, q), dtype=bool)
    out[r0 : r0 + h, c0 : c0 + w] = mask
    return out


def _choose_spaced_indices(length: int, count: int) -> list[int]:
    if count == 0:
        return []
    choices = list(range(length))
    for _ in range(80):
        picked = sorted(random.sample(choices, count))
        if all(b - a >= 2 for a, b in zip(picked, picked[1:])):
            return picked
    return sorted(random.sample(choices, count))


def _segment_cells(q: int, mid: int, name: str) -> list[tuple[int, int]]:
    if name == "top":
        return [(r, mid) for r in range(q)]
    if name == "bottom":
        return [(mid + 1 + r, mid) for r in range(q)]
    if name == "left":
        return [(mid, c) for c in range(q)]
    return [(mid, mid + 1 + c) for c in range(q)]


def _pivot_exists(layer_count: int, counts: tuple[int, int, int, int]) -> bool:
    t, l, r, b = counts
    base = _effective_depths({"top": t, "left": l, "right": r, "bottom": b}, layer_count)
    for idx, value in enumerate(counts):
        if value == 0:
            continue
        reduced = list(counts)
        reduced[idx] -= 1
        probe = _effective_depths(
            {"top": reduced[0], "left": reduced[1], "right": reduced[2], "bottom": reduced[3]},
            layer_count,
        )
        if probe != base:
            return True
    return False


def _choose_counter_pattern(layer_count: int) -> tuple[int, int, int, int]:
    values = list(range(min(layer_count, 3) + 1))
    mode = str(_bag_draw(_MODE_BAG, ["with_zero", "full_no_zero", "mixed", "with_zero"]))
    candidates = []
    for counts in itertools.product(values, repeat=4):
        t, l, r, b = map(int, counts)
        depths = _effective_depths({"top": t, "left": l, "right": r, "bottom": b}, layer_count)
        depth_values = list(depths.values())
        if len(set(depth_values)) < 2:
            continue
        if not any(0 < d < layer_count for d in depth_values):
            continue
        if sum(depths[q] > 0 for q in ("tr", "bl", "br")) < 2:
            continue
        if not _pivot_exists(layer_count, counts):
            continue
        if mode == "with_zero" and 0 not in depth_values:
            continue
        if mode == "full_no_zero" and (0 in depth_values or layer_count not in depth_values):
            continue
        if mode == "mixed" and not (0 in depth_values and layer_count in depth_values):
            continue
        candidates.append((t, l, r, b))
    if not candidates:
        for counts in itertools.product(values, repeat=4):
            t, l, r, b = map(int, counts)
            depths = _effective_depths({"top": t, "left": l, "right": r, "bottom": b}, layer_count)
            depth_values = list(depths.values())
            if len(set(depth_values)) < 2:
                continue
            if not any(0 < d < layer_count for d in depth_values):
                continue
            if sum(depths[q] > 0 for q in ("tr", "bl", "br")) < 2:
                continue
            if _pivot_exists(layer_count, counts):
                candidates.append((t, l, r, b))
    assert candidates, "expected at least one non-degenerate counter pattern"
    return tuple(map(int, random.choice(candidates)))


def _choose_control_colors(depths: dict[str, int]) -> dict[str, int]:
    while True:
        colors = {q: int(random.choice(CONTROL_COLORS)) for q in ("tr", "bl", "br")}
        positive = [colors[q] for q in ("tr", "bl", "br") if depths[q] > 0]
        if len(positive) >= 2 and len(set(positive)) >= 2:
            return colors


def _sample_control_positions(mask: np.ndarray, layer_count: int) -> list[tuple[int, int]]:
    layers = _peel_layers(mask)
    n_controls = 1 if random.random() < 0.45 else 2
    chosen: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()

    if n_controls == 2 and layer_count >= 2:
        wanted_layers = [0, layer_count - 1]
        if layer_count == 3 and random.random() < 0.4:
            wanted_layers = sorted(random.sample(range(layer_count), 2))
        for idx in wanted_layers:
            coords = [tuple(map(int, p)) for p in np.argwhere(layers[idx])]
            random.shuffle(coords)
            for coord in coords:
                if coord not in used:
                    chosen.append(coord)
                    used.add(coord)
                    break

    all_coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    random.shuffle(all_coords)
    while len(chosen) < n_controls:
        coord = all_coords.pop()
        if coord not in used:
            chosen.append(coord)
            used.add(coord)
    return chosen


def _build_input() -> np.ndarray:
    template_core, layer_count = _choose_template()
    q = _choose_qsize(template_core)
    template = _place_template_in_quadrant(template_core, q)
    size = 2 * q + 1
    mid = q

    grid = np.zeros((size, size), dtype=int)
    grid[mid, :] = GREY
    grid[:, mid] = GREY

    grid[:q, :q][template] = ORANGE

    top_count, left_count, right_count, bottom_count = _choose_counter_pattern(layer_count)
    for name, count in {
        "top": top_count,
        "left": left_count,
        "right": right_count,
        "bottom": bottom_count,
    }.items():
        cells = _segment_cells(q, mid, name)
        for idx in _choose_spaced_indices(q, count):
            r, c = cells[idx]
            grid[r, c] = MAROON

    depths = _effective_depths(
        {"top": top_count, "left": left_count, "right": right_count, "bottom": bottom_count},
        layer_count,
    )
    colors = _choose_control_colors(depths)
    slices = _quadrant_slices(mid, mid)
    for quadrant in ("tr", "bl", "br"):
        block = grid[slices[quadrant]]
        transformed = _transform(template, quadrant)
        for r, c in _sample_control_positions(transformed, layer_count):
            block[r, c] = colors[quadrant]

    return grid


def _mutate_counter(grid: np.ndarray, segment: str) -> np.ndarray | None:
    g = grid.copy()
    mid = g.shape[0] // 2
    q = mid
    for r, c in _segment_cells(q, mid, segment):
        if g[r, c] == MAROON:
            g[r, c] = GREY
            return g
    return None


def _validate_case(grid: np.ndarray) -> bool:
    try:
        parsed = _parse_input(grid)
    except Exception:
        return False

    out = _render_from_parsed(parsed)
    layer_count = len(parsed["base_layers"])
    if layer_count not in {2, 3}:
        return False

    template = parsed["template_mask"]
    if np.array_equal(template, np.fliplr(template)) or np.array_equal(template, np.flipud(template)):
        return False

    depth_values = list(parsed["depths"].values())
    if len(set(depth_values)) < 2 or not any(0 < d < layer_count for d in depth_values):
        return False

    if sum(parsed["depths"][q] > 0 for q in ("tr", "bl", "br")) < 2:
        return False

    changed_counter = False
    for segment, count in parsed["segment_counts"].items():
        if count == 0:
            continue
        probe = _mutate_counter(grid, segment)
        if probe is not None and not np.array_equal(main(probe), out):
            changed_counter = True
            break
    if not changed_counter:
        return False

    recolor_changed = False
    for quadrant in ("tr", "bl", "br"):
        if parsed["depths"][quadrant] <= 0:
            continue
        block = grid[parsed["slices"][quadrant]]
        old = parsed["controls"][quadrant]["color"]
        new = next(color for color in CONTROL_COLORS if color != old)
        probe = grid.copy()
        probe_block = probe[parsed["slices"][quadrant]]
        probe_block[block == old] = new
        if not np.array_equal(main(probe), out):
            recolor_changed = True
            break
    if not recolor_changed:
        return False

    if np.array_equal(_render_from_parsed(parsed, force_unflipped=True), out):
        return False

    if not np.all(out[parsed["mid_r"], :] == GREY) or not np.all(out[:, parsed["mid_c"]] == GREY):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    grid = _build_input()
    try:
        solved = main(grid)
    except Exception:
        return None
    if max(grid.shape) > 30 or max(solved.shape) > 30:
        return None
    if not _validate_case(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-layer template input")
