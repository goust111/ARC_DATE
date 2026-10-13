from __future__ import annotations

import hashlib
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

RING_COLORS = (BLUE, RED, GREEN, YELLOW, GREY)
SIDES = ("top", "bottom", "left", "right")
SIDE_TO_STEP = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_RETRY_LIMIT = 600
_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_border(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    if r1 > r0:
        coords.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    if c1 > c0:
        coords.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return coords


def _indicator_side(box: tuple[int, int, int, int], coord: tuple[int, int]) -> str:
    r0, c0, r1, c1 = box
    r, c = coord
    sides = []
    if r == r0 and c0 < c < c1:
        sides.append("top")
    if r == r1 and c0 < c < c1:
        sides.append("bottom")
    if c == c0 and r0 < r < r1:
        sides.append("left")
    if c == c1 and r0 < r < r1:
        sides.append("right")
    assert len(sides) == 1, "indicator must lie on exactly one non-corner ring side"
    return sides[0]


def _draw_ray(out: np.ndarray, start: tuple[int, int], side: str, color: int) -> None:
    dr, dc = SIDE_TO_STEP[side]
    h, w = out.shape
    r, c = start
    while 0 <= r < h and 0 <= c < w:
        out[r, c] = int(color)
        r += dr
        c += dc


def _extract_rings(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, ORANGE, *RING_COLORS}, "unexpected colors in grid"

    comps = _component_coords(g != BLACK)
    assert 3 <= len(comps) <= 5, "expected 3 to 5 separate ring components"

    rings: list[dict[str, object]] = []
    for coords in comps:
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "each ring bbox must enclose a non-empty interior"

        coord_set = {tuple(map(int, rc)) for rc in coords}
        border = set(_rect_border(box))
        assert coord_set == border, "each component must be exactly a 1-pixel rectangular ring"

        values = {int(g[r, c]) for r, c in coord_set}
        assert ORANGE in values, "each ring must contain one orange indicator"
        ring_values = values - {ORANGE}
        assert len(ring_values) == 1, "each ring must have one underlying ring color"
        ring_color = ring_values.pop()

        indicator_cells = [(r, c) for r, c in coord_set if int(g[r, c]) == ORANGE]
        assert len(indicator_cells) == 1, "each ring must have exactly one indicator"
        indicator = indicator_cells[0]
        side = _indicator_side(box, indicator)

        rings.append(
            {
                "coords": coords,
                "box": box,
                "color": int(ring_color),
                "indicator": indicator,
                "side": side,
            }
        )

    rings.sort(key=lambda item: (item["box"][0], item["box"][1]))  # type: ignore[index]
    assert len({int(ring["color"]) for ring in rings}) == len(rings), "ring colors must be distinct"
    for outer, inner in zip(rings, rings[1:]):
        or0, oc0, or1, oc1 = outer["box"]  # type: ignore[index]
        ir0, ic0, ir1, ic1 = inner["box"]  # type: ignore[index]
        assert or0 < ir0 < ir1 < or1 and oc0 < ic0 < ic1 < oc1, "ring bboxes must be strictly nested"
    return rings


def _center_start_from_rings(grid: np.ndarray, rings: list[dict[str, object]]) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = rings[-1]["box"]  # type: ignore[index]
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    assert inner.shape in {(1, 1), (2, 2)}, "innermost ring must surround a 1x1 or 2x2 center hole"
    assert np.all(inner == BLACK), "center hole must be black"
    return r0 + 1, c0 + 1


def _oracle_output_from_specs(
    shape: tuple[int, int],
    start: tuple[int, int],
    layer_specs: list[dict[str, object]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in layer_specs:
        _draw_ray(out, start, str(spec["side"]), int(spec["color"]))
    out[start] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read compass directions from nested ring indicators and emit center-origin rays colored by ring layer.

    Input structure:
    - A black background contains 3-5 separate 1-pixel-thick concentric rectangular rings.
    - Every ring has a unique non-orange color and exactly one orange indicator pixel replacing one border cell.
    - The innermost ring surrounds a black 1x1 or 2x2 center area.

    Rule:
    - Order the rings from outermost to innermost by nesting.
    - The side containing a ring's orange indicator encodes a compass direction: top/up, bottom/down, left/left, right/right.
    - Use the top-left cell of the innermost black center area as the common ray start, draw rays to the grid edge from outer to inner in the rings' own colors, then erase all rings and keep only the rays with the start recolored orange.

    Key insight:
    - The indicator does not say where to grow from its own position; it only encodes direction, while the nesting order controls both reading order and which ray color remains visible when later inner layers overwrite earlier outer ones.
    """
    g = np.asarray(grid, dtype=int)
    rings = _extract_rings(g)
    start = _center_start_from_rings(g, rings)
    out = np.zeros_like(g)
    for ring in rings:
        _draw_ray(out, start, str(ring["side"]), int(ring["color"]))
    out[start] = ORANGE
    return out


def _choose_sides(rng: np.random.Generator, n_layers: int) -> list[str]:
    if n_layers == 3:
        sides = rng.choice(np.array(SIDES), size=3, replace=False).tolist()
    elif n_layers == 4:
        if float(rng.random()) < 0.5:
            sides = rng.choice(np.array(SIDES), size=4, replace=False).tolist()
        else:
            repeated = str(rng.choice(np.array(SIDES)))
            others = rng.choice(np.array([s for s in SIDES if s != repeated]), size=2, replace=False).tolist()
            sides = [repeated, repeated, *others]
            rng.shuffle(sides)
    else:
        repeated = str(rng.choice(np.array(SIDES)))
        others = [s for s in SIDES if s != repeated]
        sides = [repeated, repeated, *others]
        rng.shuffle(sides)
    return [str(side) for side in sides]


def _indicator_cell(
    rng: np.random.Generator,
    box: tuple[int, int, int, int],
    side: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "top":
        return r0, int(rng.integers(c0 + 1, c1))
    if side == "bottom":
        return r1, int(rng.integers(c0 + 1, c1))
    if side == "left":
        return int(rng.integers(r0 + 1, r1)), c0
    return int(rng.integers(r0 + 1, r1)), c1


def _build_input_from_specs(
    rng: np.random.Generator,
    shape: tuple[int, int],
    outer_box: tuple[int, int, int, int],
    colors: list[int],
    sides: list[str],
) -> tuple[np.ndarray, list[dict[str, object]], tuple[int, int]]:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    n_layers = len(colors)
    layer_specs: list[dict[str, object]] = []

    for idx in range(n_layers):
        box = (
            outer_box[0] + 2 * idx,
            outer_box[1] + 2 * idx,
            outer_box[2] - 2 * idx,
            outer_box[3] - 2 * idx,
        )
        color = int(colors[idx])
        side = str(sides[idx])
        for r, c in _rect_border(box):
            grid[r, c] = color
        layer_specs.append({"box": box, "color": color, "side": side})

    start = (layer_specs[-1]["box"][0] + 1, layer_specs[-1]["box"][1] + 1)
    for spec in layer_specs:
        marker_r, marker_c = _indicator_cell(rng, spec["box"], str(spec["side"]))
        grid[marker_r, marker_c] = ORANGE
        spec["indicator"] = (marker_r, marker_c)
    return grid, layer_specs, start


def _move_indicator(grid: np.ndarray, spec: dict[str, object], new_side: str) -> np.ndarray:
    out = grid.copy()
    old_r, old_c = spec["indicator"]  # type: ignore[index]
    out[old_r, old_c] = int(spec["color"])
    r0, c0, r1, c1 = spec["box"]  # type: ignore[index]
    if new_side == "top":
        new_coord = (r0, c0 + 1)
    elif new_side == "bottom":
        new_coord = (r1, c0 + 1)
    elif new_side == "left":
        new_coord = (r0 + 1, c0)
    else:
        new_coord = (r0 + 1, c1)
    out[new_coord] = ORANGE
    return out


def _hash_grid(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _validate_scene(inp: np.ndarray, layer_specs: list[dict[str, object]], start: tuple[int, int]) -> bool:
    expected = _oracle_output_from_specs(inp.shape, start, layer_specs)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    rings = _extract_rings(inp)
    if len(rings) != len(layer_specs):
        return False
    visible_by_side: dict[str, int] = {}
    for spec in layer_specs:
        visible_by_side[str(spec["side"])] = int(spec["color"])
    if len(visible_by_side) < 3:
        return False

    sr, sc = start
    if actual[sr, sc] != ORANGE:
        return False

    for side, color in visible_by_side.items():
        dr, dc = SIDE_TO_STEP[side]
        rr, cc = sr + dr, sc + dc
        if not (0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]):
            return False
        while 0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]:
            if int(actual[rr, cc]) != color:
                return False
            rr += dr
            cc += dc

    for idx in range(len(layer_specs) - 1, -1, -1):
        side = str(layer_specs[idx]["side"])
        alternatives = [candidate for candidate in SIDES if candidate != side]
        for candidate in alternatives:
            mutated = _move_indicator(inp, layer_specs[idx], candidate)
            if np.array_equal(main(mutated), actual):
                continue
            break
        else:
            return False
        break

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_layers = int(rng.integers(3, 6))
    center_size = int(rng.integers(1, 3))
    outer_size = center_size + 4 * n_layers - 2

    top = int(rng.integers(1, 6))
    bottom = int(rng.integers(1, 6))
    left = int(rng.integers(1, 6))
    right = int(rng.integers(1, 6))
    h = outer_size + top + bottom
    w = outer_size + left + right
    if h > 30 or w > 30:
        return None

    outer_box = (top, left, top + outer_size - 1, left + outer_size - 1)
    colors = [int(x) for x in rng.choice(np.array(RING_COLORS), size=n_layers, replace=False)]
    sides = _choose_sides(rng, n_layers)
    inp, layer_specs, start = _build_input_from_specs(rng, (h, w), outer_box, colors, sides)

    if not _validate_scene(inp, layer_specs, start):
        return None

    output_sig = _hash_grid(main(inp))
    if output_sig in _SEEN_OUTPUT_SIGS:
        return None
    _SEEN_OUTPUT_SIGS.add(output_sig)
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        inp = _gen_attempt(rng)
        if inp is None:
            continue
        sig = _hash_grid(inp)
        if sig in _SEEN_INPUT_SIGS:
            continue
        _SEEN_INPUT_SIGS.add(sig)
        return inp
    raise RuntimeError("failed to generate a valid nested-ring ray input")
