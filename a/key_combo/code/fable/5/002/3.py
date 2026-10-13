from __future__ import annotations

from collections import Counter
import random

import numpy as np


SHAPE_SOLID = 0
SHAPE_X = 1
SHAPE_DIAMOND = 2

SOLID_CELLS = frozenset((r, c) for r in range(3) for c in range(3))
X_CELLS = frozenset({(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)})
DIAMOND_CELLS = frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)})
SHAPES = (SOLID_CELLS, X_CELLS, DIAMOND_CELLS)

GEN_ATTEMPTS = 400
OPTIONAL_TOKEN_ATTEMPTS = 200
_CALL_COUNT = 0


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _edge_runs(line: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    runs: list[tuple[int, int, int, int]] = []
    i = 0
    n = int(line.shape[0])
    while i < n:
        color = int(line[i])
        if color == bg:
            i += 1
            continue
        j = i + 1
        while j < n and int(line[j]) == color:
            j += 1
        runs.append((i, j - 1, color, j - i))
        i = j
    return runs


def _find_source(g: np.ndarray, bg: int) -> dict:
    H, W = g.shape
    candidates: list[dict] = []
    edge_specs = (
        ("top", g[0, :], 0),
        ("bottom", g[H - 1, :], H - 1),
        ("left", g[:, 0], 0),
        ("right", g[:, W - 1], W - 1),
    )
    for edge, line, fixed in edge_specs:
        for start, end, color, length in _edge_runs(line, bg):
            if length < 3:
                continue
            candidates.append(
                {
                    "edge": edge,
                    "line": fixed,
                    "start": int(start),
                    "end": int(end),
                    "color": int(color),
                    "n": int(length),
                }
            )

    assert len(candidates) == 1, "expected one visible border source segment"
    return candidates[0]


def _shape_index(cells: frozenset[tuple[int, int]]) -> int | None:
    for i, shape in enumerate(SHAPES):
        if cells == shape:
            return int(i)
    return None


def _find_tokens(g: np.ndarray, bg: int, source_color: int) -> list[dict]:
    tokens: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in {bg, source_color}):
        coords = np.argwhere(g == int(color))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox_coords(coords)
        if (r1 - r0, c1 - c0) != (2, 2):
            continue
        cells = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
        shape = _shape_index(cells)
        if shape is None:
            continue
        tokens.append(
            {
                "color": int(color),
                "shape": int(shape),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "center": (int(r0 + 1), int(c0 + 1)),
            }
        )

    assert tokens, "expected at least one 3x3 token"
    return tokens


def _token_distance_and_span(token: dict, source: dict, shape: tuple[int, int]) -> tuple[int, bool]:
    H, W = shape
    r, c = token["center"]
    edge = source["edge"]
    start, end = source["start"], source["end"]
    if edge == "top":
        return int(r), bool(start <= c <= end)
    if edge == "bottom":
        return int(H - 1 - r), bool(start <= c <= end)
    if edge == "left":
        return int(c), bool(start <= r <= end)
    if edge == "right":
        return int(W - 1 - c), bool(start <= r <= end)
    raise AssertionError("unknown edge")


def _paint_shape(g: np.ndarray, center: tuple[int, int], color: int, shape: int) -> None:
    r, c = map(int, center)
    g[r - 1 : r + 2, c - 1 : c + 2] = _get_bg(g)
    for dr, dc in SHAPES[int(shape)]:
        g[r - 1 + int(dr), c - 1 + int(dc)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance 3x3 token shapes by counting invisible wavefronts emitted from a border segment.

    Input structure:
    - A dominant background contains exactly one straight same-color segment
      lying on one grid edge; its visible length is the wave wavelength n.
    - The interior contains several color-distinct 3x3 tokens, each shaped as
      a solid square, an X, or a diamond made from the center plus four
      cardinal neighbors.

    Rule:
    - Imagine wavefronts parallel to the source edge at distances n, 2n, 3n,
      and so on, but only inside the row or column span covered by the source
      segment.
    - For every token whose center lies inside that span, count the wavefronts
      strictly between its center and the source edge: k = (distance - 1) // n.
      Replace the token in place by advancing its shape k steps through
      solid -> X -> diamond -> solid, keeping its color.
    - Tokens whose centers are outside the source span, the source segment,
      and all other pixels remain unchanged.

    Key insight:
    - The grid never draws the wavefronts; the only visible wavelength is the
      border segment length, and token centers act as distance counters whose
      effect is clipped by the segment span.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    source = _find_source(g, bg)
    tokens = _find_tokens(g, bg, source["color"])

    out = g.copy()
    n = int(source["n"])
    for token in tokens:
        distance, in_span = _token_distance_and_span(token, source, g.shape)
        if not in_span:
            continue
        k = max(0, (int(distance) - 1) // n)
        new_shape = (int(token["shape"]) + int(k)) % len(SHAPES)
        r0, c0, r1, c1 = token["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        for dr, dc in SHAPES[new_shape]:
            out[r0 + int(dr), c0 + int(dc)] = int(token["color"])
    return out


# ---------------- generator ----------------


def _token_box(center: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = center
    return int(r - 1), int(c - 1), int(r + 1), int(c + 1)


def _boxes_clear(box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]], margin: int = 1) -> bool:
    r0, c0, r1, c1 = box
    for br0, bc0, br1, bc1 in boxes:
        if r1 + margin < br0 or br1 + margin < r0 or c1 + margin < bc0 or bc1 + margin < c0:
            continue
        return False
    return True


def _center_from(edge: str, H: int, W: int, distance: int, span_pos: int) -> tuple[int, int]:
    if edge == "top":
        return int(distance), int(span_pos)
    if edge == "bottom":
        return int(H - 1 - distance), int(span_pos)
    if edge == "left":
        return int(span_pos), int(distance)
    if edge == "right":
        return int(span_pos), int(W - 1 - distance)
    raise AssertionError("unknown edge")


def _inside_grid_for_token(center: tuple[int, int], H: int, W: int) -> bool:
    r, c = center
    return 2 <= int(r) <= H - 3 and 2 <= int(c) <= W - 3


def _place_token(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    center: tuple[int, int],
    color: int,
    shape: int,
    *,
    margin: int = 1,
) -> bool:
    H, W = grid.shape
    if not _inside_grid_for_token(center, H, W):
        return False
    box = _token_box(center)
    if not _boxes_clear(box, boxes, margin=margin):
        return False
    r0, c0, r1, c1 = box
    if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == _get_bg(grid)):
        return False
    _paint_shape(grid, center, color, shape)
    boxes.append(box)
    return True


def _span_positions(start: int, n: int) -> tuple[int, int]:
    return int(start + 1), int(start + n - 2)


def _outside_span_positions(start: int, end: int, limit: int) -> list[int]:
    positions = []
    left = int(start - 4)
    right = int(end + 4)
    if 2 <= left <= limit - 3:
        positions.append(left)
    if 2 <= right <= limit - 3:
        positions.append(right)
    return positions


def _draw_source(grid: np.ndarray, edge: str, start: int, n: int, color: int) -> None:
    H, W = grid.shape
    end = int(start + n)
    if edge == "top":
        grid[0, start:end] = int(color)
    elif edge == "bottom":
        grid[H - 1, start:end] = int(color)
    elif edge == "left":
        grid[start:end, 0] = int(color)
    elif edge == "right":
        grid[start:end, W - 1] = int(color)
    else:
        raise AssertionError("unknown edge")


def _hypothetical_inside(source: dict, center: tuple[int, int]) -> bool:
    r, c = center
    if source["edge"] in {"top", "bottom"}:
        return bool(source["start"] <= c <= source["end"])
    return bool(source["start"] <= r <= source["end"])


def _semantic_valid(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bg = _get_bg(inp)
        source = _find_source(inp, bg)
        tokens = _find_tokens(inp, bg, source["color"])
        out = main(inp)
    except Exception:
        return False
    if not np.array_equal(out, expected):
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if len(tokens) < 6:
        return False

    inside: list[tuple[int, int]] = []
    outside_changes_blocked = 0
    k1_shapes: set[int] = set()
    has_strict_boundary = False
    has_k2 = False
    n = int(source["n"])

    for token in tokens:
        distance, in_span = _token_distance_and_span(token, source, inp.shape)
        k = max(0, (int(distance) - 1) // n)
        before_shape = int(token["shape"])
        if in_span:
            inside.append((k, before_shape))
            if distance == n and k == 0:
                has_strict_boundary = True
            if k == 1:
                k1_shapes.add(before_shape)
            if k == 2:
                has_k2 = True
        elif k % 3 != 0:
            r0, c0, r1, c1 = token["bbox"]
            if np.array_equal(inp[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1]):
                outside_changes_blocked += 1

    ks = {k for k, _shape in inside}
    return bool(
        len(inside) >= 5
        and {0, 1, 2}.issubset(ks)
        and k1_shapes == {SHAPE_SOLID, SHAPE_X, SHAPE_DIAMOND}
        and has_strict_boundary
        and has_k2
        and outside_changes_blocked >= 1
    )


def _gen_attempt(edge: str) -> np.ndarray | None:
    bg = random.randrange(10)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    source_color = palette[0]
    token_colors = palette[1:]

    n = random.randint(7, 10)
    parallel = random.randint(max(n + 12, 22), 30)
    depth = random.randint(max(2 * n + 5, 23), 30)
    if edge in {"top", "bottom"}:
        H, W = depth, parallel
        start = random.randint(5, W - n - 6)
    else:
        H, W = parallel, depth
        start = random.randint(5, H - n - 6)

    grid = np.full((H, W), int(bg), dtype=int)
    _draw_source(grid, edge, start, n, source_color)

    boxes: list[tuple[int, int, int, int]] = []
    a, b = _span_positions(start, n)
    mandatory = [
        (n, a, random.randrange(3)),
        (n + 1, b, SHAPE_SOLID),
        (n + 5, a, SHAPE_X),
        (2 * n, b, SHAPE_DIAMOND),
        (2 * n + 2, a, random.randrange(3)),
    ]
    if 2 * n + 2 > depth - 2:
        return None

    color_i = 0
    for distance, span_pos, shape in mandatory:
        center = _center_from(edge, H, W, distance, span_pos)
        if not _place_token(grid, boxes, center, token_colors[color_i], shape):
            return None
        color_i += 1

    source_end = start + n - 1
    outside_limit = W if edge in {"top", "bottom"} else H
    outside_positions = _outside_span_positions(start, source_end, outside_limit)
    if not outside_positions:
        return None
    outside_pos = random.choice(outside_positions)
    outside_center = _center_from(edge, H, W, n + 1, outside_pos)
    if not _place_token(grid, boxes, outside_center, token_colors[color_i], random.randrange(3)):
        return None
    color_i += 1

    optional_count = random.randint(0, min(2, len(token_colors) - color_i))
    for _ in range(optional_count):
        placed = False
        for _attempt in range(OPTIONAL_TOKEN_ATTEMPTS):
            distance = random.randint(2, depth - 2)
            if random.random() < 0.55:
                span_pos = random.randint(start, source_end)
            else:
                choices = _outside_span_positions(start, source_end, outside_limit)
                if not choices:
                    continue
                span_pos = random.choice(choices)
            center = _center_from(edge, H, W, distance, span_pos)
            if _place_token(grid, boxes, center, token_colors[color_i], random.randrange(3)):
                placed = True
                color_i += 1
                break
        if not placed:
            break

    expected = main(grid)
    if not _semantic_valid(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _CALL_COUNT
    edges = ("top", "bottom", "left", "right")
    first_edge = edges[_CALL_COUNT % len(edges)]
    _CALL_COUNT += 1
    edge_order = [first_edge] + random.sample([e for e in edges if e != first_edge], 3)
    for edge in edge_order:
        for _ in range(GEN_ATTEMPTS // len(edges)):
            grid = _gen_attempt(edge)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid invisible-wavefront input")
