from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np

try:
    _UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if _UTILS_DIR.exists():
        sys.path.append(str(_UTILS_DIR))
    from arc_utils import Color
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        GRAY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9


BG = int(Color.BLACK)
SEP = int(Color.GREY)
PALETTE = [int(c) for c in (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500
SHAPE_ATTEMPTS = 400


def _quadrants(n: int) -> list[dict[str, object]]:
    q = n // 2
    return [
        {"name": "TL", "r0": 0, "c0": 0, "r1": q, "c1": q},
        {"name": "TR", "r0": 0, "c0": q + 1, "r1": q, "c1": n},
        {"name": "BR", "r0": q + 1, "c0": q + 1, "r1": n, "c1": n},
        {"name": "BL", "r0": q + 1, "c0": 0, "r1": n, "c1": q},
    ]


def _to_global(local_coords: frozenset[tuple[int, int]], quad: dict[str, object]) -> frozenset[tuple[int, int]]:
    r0 = int(quad["r0"])
    c0 = int(quad["c0"])
    return frozenset((r0 + r, c0 + c) for r, c in local_coords)


def _rotate_local(coords: frozenset[tuple[int, int]], q: int, steps: int) -> frozenset[tuple[int, int]]:
    out = set(coords)
    for _ in range(steps % 4):
        out = {(c, q - 1 - r) for r, c in out}
    return frozenset(out)


def _rotate_global(coords: frozenset[tuple[int, int]], n: int, steps: int) -> frozenset[tuple[int, int]]:
    out = set(coords)
    for _ in range(steps % 4):
        out = {(c, n - 1 - r) for r, c in out}
    return frozenset(out)


def _bbox_dims(coords: frozenset[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _normalize(coords: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return frozenset((r - min_r, c - min_c) for r, c in coords)


def _is_connected(coords: frozenset[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nbr = (r + dr, c + dc)
            if nbr in coords and nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)
    return len(seen) == len(coords)


def _extract_quadrant_data(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    n, m = g.shape
    assert n == m and n % 2 == 1
    center = n // 2
    assert np.all(g[center, :] == SEP)
    assert np.all(g[:, center] == SEP)

    out: list[dict[str, object]] = []
    for quad in _quadrants(n):
        r0 = int(quad["r0"])
        c0 = int(quad["c0"])
        r1 = int(quad["r1"])
        c1 = int(quad["c1"])
        sub = g[r0:r1, c0:c1]
        mask = (sub != BG) & (sub != SEP)
        local = frozenset((int(r), int(c)) for r, c in np.argwhere(mask))
        assert 4 <= len(local) <= 7
        colors = sorted(set(map(int, np.unique(sub[mask]))))
        assert len(colors) == 1
        assert _is_connected(local)
        out.append(
            {
                "name": quad["name"],
                "color": colors[0],
                "coords_local": local,
                "coords_global": _to_global(local, quad),
            }
        )
    assert len({item["coords_local"] for item in out}) == 4
    return out


def _render_grid(n: int, local_shapes: list[frozenset[tuple[int, int]]], colors: list[int]) -> np.ndarray:
    grid = np.full((n, n), BG, dtype=int)
    center = n // 2
    grid[center, :] = SEP
    grid[:, center] = SEP
    for quad, shape, color in zip(_quadrants(n), local_shapes, colors):
        for r, c in _to_global(shape, quad):
            grid[r, c] = int(color)
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate the unique-color quadrant's shape to the other quadrants by center rotation and recolor them.

    Input structure:
    - The grid is square with odd side length and a grey center row plus center column that split it into four equal quadrants.
    - Each quadrant contains exactly one small 4-connected monochrome shape on black background.
    - Three quadrants share one foreground color, while the fourth quadrant uses a different foreground color.

    Rule:
    - Identify the unique-color quadrant as the template quadrant.
    - Keep that template unchanged, but erase the other three quadrants' foreground pixels.
    - Rotate the template shape around the global grid center by 90, 180, and 270 degrees to fill the other quadrants, repainting those three rotated copies with the shared normal color.

    Key insight:
    - The odd-colored quadrant is not the answer object to isolate; it is the only reliable template whose geometry must be quarter-turned around the whole canvas and then recolored for the ordinary quadrants.
    """
    g = np.asarray(grid, dtype=int)
    n = int(g.shape[0])
    quad_data = _extract_quadrant_data(g)

    color_counts = Counter(int(item["color"]) for item in quad_data)
    assert sorted(color_counts.values()) == [1, 3]
    normal_color = next(color for color, count in color_counts.items() if count == 3)
    template_color = next(color for color, count in color_counts.items() if count == 1)
    template_idx = next(i for i, item in enumerate(quad_data) if int(item["color"]) == template_color)

    out = g.copy()
    quad_defs = _quadrants(n)
    for i, quad in enumerate(quad_defs):
        if i == template_idx:
            continue
        r0 = int(quad["r0"])
        c0 = int(quad["c0"])
        r1 = int(quad["r1"])
        c1 = int(quad["c1"])
        sub = out[r0:r1, c0:c1]
        sub[(sub != BG) & (sub != SEP)] = BG

    template_coords = quad_data[template_idx]["coords_global"]
    assert isinstance(template_coords, frozenset)
    for steps in range(1, 4):
        target_idx = (template_idx + steps) % 4
        target_quad = quad_defs[target_idx]
        rotated = _rotate_global(template_coords, n, steps)
        assert all(
            int(target_quad["r0"]) <= r < int(target_quad["r1"]) and int(target_quad["c0"]) <= c < int(target_quad["c1"])
            for r, c in rotated
        )
        for r, c in rotated:
            assert out[r, c] == BG
            out[r, c] = int(normal_color)

    return out


def _random_polyomino(area: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = {
            (r + dr, c + dc)
            for r, c in cells
            for dr, dc in DIR4
            if (r + dr, c + dc) not in cells
        }
        cells.add(random.choice(tuple(frontier)))
    return _normalize(cells)


def _choose_offset(limit: int) -> int:
    if limit <= 0:
        return 0
    if limit >= 2 and random.random() < 0.7:
        return random.randint(1, limit - 1)
    return random.randint(0, limit)


def _sample_local_shape(
    q: int,
    *,
    forbidden: set[frozenset[tuple[int, int]]] | None = None,
    require_unique_rotations: bool = False,
    target_area: int | None = None,
    allowed_dims: set[tuple[int, int]] | None = None,
) -> frozenset[tuple[int, int]]:
    forbidden = set() if forbidden is None else set(forbidden)
    for _ in range(SHAPE_ATTEMPTS):
        area = int(target_area) if target_area is not None else random.randint(4, 7)
        base = _random_polyomino(area)
        h, w = _bbox_dims(base)
        if h < 2 or w < 2 or area == h * w:
            continue
        if h > q or w > q:
            continue
        if allowed_dims is not None and (h, w) not in allowed_dims:
            continue
        top = _choose_offset(q - h)
        left = _choose_offset(q - w)
        placed = frozenset((top + r, left + c) for r, c in base)
        if placed in forbidden:
            continue
        if require_unique_rotations:
            rotations = {_rotate_local(placed, q, k) for k in range(4)}
            if len(rotations) != 4:
                continue
        return placed
    raise RuntimeError("failed to sample a valid local shape")


def _expected_output(
    n: int,
    template_idx: int,
    template_local: frozenset[tuple[int, int]],
    normal_color: int,
    template_color: int,
) -> np.ndarray:
    q = n // 2
    local_shapes = [frozenset() for _ in range(4)]
    for steps in range(4):
        target_idx = (template_idx + steps) % 4
        local_shapes[target_idx] = _rotate_local(template_local, q, steps)
    colors = [normal_color] * 4
    colors[template_idx] = template_color
    return _render_grid(n, local_shapes, colors)


def _gen_attempt() -> np.ndarray | None:
    q = random.randint(6, 11)
    n = 2 * q + 1
    normal_color, template_color = random.sample(PALETTE, 2)
    template_idx = random.randrange(4)

    try:
        template_local = _sample_local_shape(q, require_unique_rotations=True)
    except RuntimeError:
        return None
    template_rotations = {_rotate_local(template_local, q, k) for k in range(4)}
    template_dims = _bbox_dims(template_local)
    bbox_decoy_dims = {template_dims, (template_dims[1], template_dims[0])}

    ordinary_shapes: list[frozenset[tuple[int, int]]] = []
    forbidden = set(template_rotations)
    for _ in range(SHAPE_ATTEMPTS):
        ordinary_shapes.clear()
        local_forbidden = set(forbidden)
        try:
            ordinary_shapes.append(
                _sample_local_shape(
                    q,
                    forbidden=local_forbidden,
                    target_area=len(template_local),
                    allowed_dims=bbox_decoy_dims,
                )
            )
            local_forbidden.add(ordinary_shapes[-1])
            for _slot in range(2):
                shape = _sample_local_shape(q, forbidden=local_forbidden)
                ordinary_shapes.append(shape)
                local_forbidden.add(shape)
        except RuntimeError:
            continue
        if any(len(shape) == len(template_local) for shape in ordinary_shapes):
            break
    if len(ordinary_shapes) != 3:
        return None

    local_shapes = [frozenset() for _ in range(4)]
    ordinary_iter = iter(ordinary_shapes)
    for idx in range(4):
        if idx == template_idx:
            local_shapes[idx] = template_local
        else:
            local_shapes[idx] = next(ordinary_iter)

    colors = [normal_color] * 4
    colors[template_idx] = template_color
    inp = _render_grid(n, local_shapes, colors)
    out = main(inp)

    try:
        _validate_case(inp, out, template_idx, template_local, normal_color, template_color)
    except AssertionError:
        return None
    return inp


def _validate_case(
    inp: np.ndarray,
    out: np.ndarray,
    template_idx: int,
    template_local: frozenset[tuple[int, int]],
    normal_color: int,
    template_color: int,
) -> None:
    assert isinstance(inp, np.ndarray)
    assert isinstance(out, np.ndarray)
    assert inp.shape == out.shape
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert out.ndim == 2 and max(out.shape) <= 30
    assert not np.array_equal(inp, out)

    in_quads = _extract_quadrant_data(inp)
    out_quads = _extract_quadrant_data(out)
    in_colors = [int(item["color"]) for item in in_quads]
    out_colors = [int(item["color"]) for item in out_quads]

    assert Counter(in_colors)[normal_color] == 3
    assert Counter(in_colors)[template_color] == 1
    assert in_colors[template_idx] == template_color
    assert out_colors[template_idx] == template_color
    assert out_colors.count(normal_color) == 3

    q = inp.shape[0] // 2
    expected_locals = [frozenset() for _ in range(4)]
    for steps in range(4):
        expected_locals[(template_idx + steps) % 4] = _rotate_local(template_local, q, steps)

    assert in_quads[template_idx]["coords_local"] == template_local
    assert len(set(expected_locals)) == 4
    assert np.array_equal(out, _expected_output(inp.shape[0], template_idx, template_local, normal_color, template_color))

    ordinary_input_shapes = []
    for idx in range(4):
        coords_local = in_quads[idx]["coords_local"]
        if idx == template_idx:
            continue
        ordinary_input_shapes.append(coords_local)
        assert coords_local != expected_locals[idx]
        assert out_quads[idx]["coords_local"] == expected_locals[idx]
        assert out_quads[idx]["color"] == normal_color
    assert len(set(ordinary_input_shapes)) == 3
    assert any(len(shape) == len(template_local) for shape in ordinary_input_shapes)
    template_dims = _bbox_dims(template_local)
    bbox_decoy_dims = {template_dims, (template_dims[1], template_dims[0])}
    assert any(_bbox_dims(shape) in bbox_decoy_dims for shape in ordinary_input_shapes)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quadrant-rotation puzzle")
