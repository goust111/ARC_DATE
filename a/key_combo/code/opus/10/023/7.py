from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GEN_RECT_COUNT = 4
GEN_RETRY_LIMIT = 500
GEN_VALIDATION_SAMPLES = 20
PALETTE = tuple(range(1, 10))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _valid_connector_positions(h: int, w: int) -> list[tuple[int, int, str]]:
    positions: list[tuple[int, int, str]] = []
    if w % 2 == 1:
        positions.append((0, w // 2, "top"))
        positions.append((h - 1, w // 2, "bottom"))
    if h % 2 == 1:
        positions.append((h // 2, 0, "left"))
        positions.append((h // 2, w - 1, "right"))
    return positions


def _parse_legend(row: np.ndarray) -> list[int]:
    legend: list[int] = []
    c = 0
    while c < row.shape[0]:
        color = int(row[c])
        if color == BG:
            c += 1
            continue
        legend.append(color)
        c += 1
        if c < row.shape[0] and int(row[c]) != BG:
            raise AssertionError("legend entries must be single pixels separated by background")
    assert 4 <= len(legend) <= 5
    assert len(legend) == len(set(legend))
    return legend


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g[1:] != BG)
    rects: list[dict] = []
    for coords in comps:
        coords = coords.copy()
        coords[:, 0] += 1
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert not np.any(patch == BG), "rectangle bbox must be fully filled"
        counts = Counter(map(int, patch.ravel()))
        assert len(counts) == 2, "each rectangle must have one connector color and one body color"
        body_color, body_count = counts.most_common(1)[0]
        connector_items = [(color, count) for color, count in counts.items() if color != body_color]
        assert len(connector_items) == 1
        connector_color, connector_count = connector_items[0]
        assert connector_count == 1
        assert body_count == patch.size - 1
        connector_cells = np.argwhere(patch == connector_color)
        assert connector_cells.shape == (1, 2)
        cr, cc = map(int, connector_cells[0])
        valid_positions = {(rr, cc_) for rr, cc_, _side in _valid_connector_positions(*patch.shape)}
        assert (cr, cc) in valid_positions, "connector must sit at the midpoint of one edge"
        rects.append(
            {
                "body_color": int(body_color),
                "connector_color": int(connector_color),
                "height": int(patch.shape[0]),
                "width": int(patch.shape[1]),
                "bbox": (r0, c0, r1, c1),
            }
        )
    assert len(rects) >= 1
    return rects


def _render_nested(rectangles: list[dict]) -> np.ndarray:
    outer = rectangles[0]
    out = np.zeros((int(outer["height"]), int(outer["width"])), dtype=int)
    out[:, :] = int(outer["body_color"])
    region = (0, 0, int(outer["height"]), int(outer["width"]))
    for rect in rectangles[1:]:
        rr, cc, rh, rw = region
        h = int(rect["height"])
        w = int(rect["width"])
        assert h <= rh and w <= rw
        r = rr + rh - h
        c = cc
        out[r : r + h, c : c + w] = int(rect["body_color"])
        region = (r, c, h, w)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Order connector-marked rectangles by the legend row, then rebuild them as a lower-left nesting.

    Input structure:
    - The top row is a legend strip: 4-5 single colored pixels separated by black cells, read left-to-right.
    - Below the legend are scattered solid rectangles on a black background; each rectangle has exactly one edge-midpoint pixel recolored to a legend color.
    - Rectangle body colors are distinct and disjoint from the legend colors.

    Rule:
    - Read the legend row and map each legend color to its left-to-right priority.
    - For each rectangle, find its unique off-color connector pixel and use that connector color to look up the rectangle's priority.
    - Recolor the connector back to the rectangle body color, then sort rectangles by increasing legend priority.
    - Rebuild the rectangles as a lower-left nesting: the first priority rectangle is the outer crop, and each later rectangle is placed left-aligned and bottom-aligned inside the previous one.

    Key insight:
    - The connector colors are not placement colors; they are lookup keys into the legend row, and the legend order determines the nesting order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    legend = _parse_legend(g[0])
    order = {color: idx for idx, color in enumerate(legend)}
    rects = _extract_rectangles(g)
    assert len(rects) == len(legend)
    for rect in rects:
        assert rect["connector_color"] in order
    rects.sort(key=lambda rect: order[int(rect["connector_color"])])
    return _render_nested(rects)


def _build_dim_chain(count: int) -> list[tuple[int, int]] | None:
    outer_candidates = [
        (h, w)
        for h in range(7, 11)
        for w in range(7, 12)
        if _valid_connector_positions(h, w)
    ]

    def extend(chain: list[tuple[int, int]], used_areas: set[int]) -> list[tuple[int, int]] | None:
        if len(chain) == count:
            return chain
        prev_h, prev_w = chain[-1]
        remaining = count - len(chain)
        candidates = [
            (h, w)
            for h in range(2, prev_h + 1)
            for w in range(2, prev_w + 1)
            if (h, w) != (prev_h, prev_w)
            and h * w not in used_areas
            and h * w < prev_h * prev_w
            and _valid_connector_positions(h, w)
            and max(h, w) >= 3
            and h + remaining - 1 >= 2
            and w + remaining - 1 >= 2
        ]
        random.shuffle(candidates)
        for dims in candidates:
            result = extend(chain + [dims], used_areas | {dims[0] * dims[1]})
            if result is not None:
                return result
        return None

    random.shuffle(outer_candidates)
    for outer in outer_candidates:
        result = extend([outer], {outer[0] * outer[1]})
        if result is not None:
            return result
    return None


def _legend_columns(width: int, count: int) -> list[int] | None:
    candidates = list(range(width))
    random.shuffle(candidates)
    chosen: list[int] = []
    for col in sorted(candidates):
        if all(abs(col - prev) >= 2 for prev in chosen):
            chosen.append(col)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _make_patch(h: int, w: int, body_color: int, connector_color: int) -> tuple[np.ndarray, str]:
    patch = np.full((h, w), int(body_color), dtype=int)
    connector_r, connector_c, side = random.choice(_valid_connector_positions(h, w))
    patch[connector_r, connector_c] = int(connector_color)
    return patch, side


def _free_positions(occupied: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    rows, cols = occupied.shape
    positions: list[tuple[int, int]] = []
    for r in range(0, rows - h + 1):
        for c in range(0, cols - w + 1):
            if occupied[r : r + h, c : c + w].any():
                continue
            positions.append((r, c))
    return positions


def _scatter_patches(
    grid_shape: tuple[int, int], patches: list[np.ndarray]
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    occupied = np.zeros((h - 1, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
    for idx in order:
        ph, pw = patches[idx].shape
        positions = _free_positions(occupied, ph + 2, pw + 2)
        if not positions:
            return None
        r_pad, c_pad = random.choice(positions)
        r = r_pad + 1
        c = c_pad + 1
        placements[idx] = (r, c - 1)
        occupied[r - 1 : r + ph + 1, c - 1 : c + pw + 1] = True
    return [pos for pos in placements if pos is not None]


def _expected_from_chain(
    dims: list[tuple[int, int]], body_colors: list[int], connector_colors: list[int]
) -> np.ndarray:
    rects = [
        {
            "body_color": body,
            "connector_color": connector,
            "height": h,
            "width": w,
        }
        for (h, w), body, connector in zip(dims, body_colors, connector_colors)
    ]
    return _render_nested(rects)


def _semantic_checks(
    grid: np.ndarray,
    dims: list[tuple[int, int]],
    body_colors: list[int],
    connector_colors: list[int],
    sides: list[str],
) -> bool:
    try:
        rects = _extract_rectangles(grid)
        legend = _parse_legend(grid[0])
    except Exception:
        return False
    if legend != connector_colors:
        return False
    if len(set(sides)) < 2:
        return False
    if sorted(body_colors) == body_colors or sorted(body_colors, reverse=True) == body_colors:
        return False
    ordered_by_scan = [rect["connector_color"] for rect in sorted(rects, key=lambda rect: rect["bbox"][:2])]
    if ordered_by_scan == connector_colors:
        return False
    if [rect["height"] * rect["width"] for rect in rects].count(max(h * w for h, w in dims)) != 1:
        return False
    expected = _expected_from_chain(dims, body_colors, connector_colors)
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    dims = _build_dim_chain(GEN_RECT_COUNT)
    if dims is None:
        return None

    colors = random.sample(PALETTE, 2 * GEN_RECT_COUNT)
    connector_colors = colors[:GEN_RECT_COUNT]
    body_colors = colors[GEN_RECT_COUNT:]
    if sorted(body_colors) == body_colors or sorted(body_colors, reverse=True) == body_colors:
        return None

    grid_h = random.randint(max(16, dims[0][0] + 4), 20)
    grid_w = random.randint(max(17, dims[0][1] + 5), 22)
    legend_cols = _legend_columns(grid_w, GEN_RECT_COUNT)
    if legend_cols is None:
        return None

    patches: list[np.ndarray] = []
    sides: list[str] = []
    for (h, w), body, connector in zip(dims, body_colors, connector_colors):
        patch, side = _make_patch(h, w, body, connector)
        patches.append(patch)
        sides.append(side)

    placements = _scatter_patches((grid_h, grid_w), patches)
    if placements is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for col, color in zip(legend_cols, connector_colors):
        grid[0, col] = int(color)
    for patch, (r, c) in zip(patches, placements):
        ph, pw = patch.shape
        grid[r : r + ph, c : c + pw] = patch

    if not _semantic_checks(grid, dims, body_colors, connector_colors, sides):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to produce a valid input")
