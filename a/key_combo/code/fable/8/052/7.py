import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        cells: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _find_frame(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int, int]:
    candidates: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    for color in sorted(map(int, np.unique(grid))):
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        ring = np.zeros_like(box, dtype=bool)
        ring[0, :] = True
        ring[-1, :] = True
        ring[:, 0] = True
        ring[:, -1] = True
        if not np.array_equal(box == color, ring):
            continue
        interior = box[1:-1, 1:-1]
        if interior.size == 0:
            continue
        interior_colors = np.unique(interior)
        if len(interior_colors) != 1:
            continue
        fill_color = int(interior_colors[0])
        if fill_color == color:
            continue
        area = int(box.shape[0] * box.shape[1])
        candidates.append((area, (r0, c0, r1, c1), color, fill_color))
    assert len(candidates) == 1, "expected exactly one rectangular frame"
    _, frame_bbox, frame_color, fill_color = candidates[0]
    return frame_bbox, int(frame_color), int(fill_color)


def _parse_bars(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    bg: int,
    frame_color: int,
    fill_color: int,
) -> tuple[list[dict], list[dict]]:
    r0, c0, r1, c1 = frame_bbox
    out_h = r1 - r0 - 1
    out_w = c1 - c0 - 1
    mask = ~np.isin(grid, [bg, frame_color, fill_color])
    comps = _find_components(mask)

    hbars: list[dict] = []
    vbars: list[dict] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    lengths: list[int] = []

    for coords in comps:
        color = int(grid[coords[0, 0], coords[0, 1]])
        rows = sorted(set(map(int, coords[:, 0])))
        cols = sorted(set(map(int, coords[:, 1])))
        matches: list[tuple[str, int, int, int]] = []

        if len(rows) == 1 and len(coords) == cols[-1] - cols[0] + 1:
            row = rows[0]
            if r0 < row < r1:
                if cols[-1] == c0 - 1:
                    matches.append(("h", row - (r0 + 1), len(coords), color))
                if cols[0] == c1 + 1:
                    matches.append(("h", row - (r0 + 1), len(coords), color))

        if len(cols) == 1 and len(coords) == rows[-1] - rows[0] + 1:
            col = cols[0]
            if c0 < col < c1:
                if rows[-1] == r0 - 1:
                    matches.append(("v", col - (c0 + 1), len(coords), color))
                if rows[0] == r1 + 1:
                    matches.append(("v", col - (c0 + 1), len(coords), color))

        assert len(matches) == 1, "each external component must be one unambiguous bar"
        axis, pos, length, color = matches[0]
        assert length >= 1
        if axis == "h":
            assert 0 <= pos < out_h and pos not in used_rows, "projected rows must be unique"
            used_rows.add(pos)
            hbars.append({"pos": pos, "length": length, "color": color})
        else:
            assert 0 <= pos < out_w and pos not in used_cols, "projected cols must be unique"
            used_cols.add(pos)
            vbars.append({"pos": pos, "length": length, "color": color})
        lengths.append(length)

    assert hbars and vbars, "expected both horizontal and vertical bars"
    assert len(set(lengths)) == len(lengths), "all external bar lengths must be distinct"
    return sorted(hbars, key=lambda x: x["pos"]), sorted(vbars, key=lambda x: x["pos"])


def _solve_from_scene(
    frame_bbox: tuple[int, int, int, int],
    fill_color: int,
    hbars: list[dict],
    vbars: list[dict],
) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    out = np.full((r1 - r0 - 1, c1 - c0 - 1), fill_color, dtype=int)
    for hbar in hbars:
        for vbar in vbars:
            if hbar["color"] == vbar["color"]:
                color = hbar["color"]
            elif hbar["length"] > vbar["length"]:
                color = hbar["color"]
            else:
                color = vbar["color"]
            out[hbar["pos"], vbar["pos"]] = color
    return out


def _extract_scene(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int, int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_bbox, frame_color, fill_color = _find_frame(g, bg)
    hbars, vbars = _parse_bars(g, frame_bbox, bg, frame_color, fill_color)
    return frame_bbox, frame_color, fill_color, hbars, vbars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint only the interior crossings implied by the outside bars around a framed area.

    Input structure:
    - A background canvas contains exactly one single-color rectangular frame.
    - The frame border encloses a uniformly filled interior of another color.
    - Outside the frame sit several monochrome horizontal bars on the left/right and vertical bars on the top/bottom; each bar touches one outer frame side with one endpoint.

    Rule:
    - Crop the output to the frame interior and fill it with the frame's interior color.
    - Each outside horizontal bar projects to one interior row, and each outside vertical bar projects to one interior column.
    - For every row/column projection pair, color only their intersection cell: same-colored bars keep that color, otherwise the longer outside bar wins.
    - All non-intersection interior cells stay in the fill color.

    Key insight:
    - The outside bars never draw full lines inside the frame; they matter only through their projected crossing points and the visible bar-length comparison.
    """
    frame_bbox, _frame_color, fill_color, hbars, vbars = _extract_scene(grid)
    return _solve_from_scene(frame_bbox, fill_color, hbars, vbars)


_GEN_RETRIES = 500


def _draw_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], frame_color: int, fill_color: int) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = fill_color
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color


def _render_input(
    grid_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    bg: int,
    frame_color: int,
    fill_color: int,
    hbars: list[dict],
    vbars: list[dict],
) -> np.ndarray:
    grid = np.full(grid_shape, bg, dtype=int)
    _draw_frame(grid, frame_bbox, frame_color, fill_color)
    r0, c0, r1, c1 = frame_bbox
    for bar in hbars:
        row = r0 + 1 + bar["pos"]
        if bar["side"] == "left":
            grid[row, c0 - bar["length"] : c0] = bar["color"]
        else:
            grid[row, c1 + 1 : c1 + 1 + bar["length"]] = bar["color"]
    for bar in vbars:
        col = c0 + 1 + bar["pos"]
        if bar["side"] == "top":
            grid[r0 - bar["length"] : r0, col] = bar["color"]
        else:
            grid[r1 + 1 : r1 + 1 + bar["length"], col] = bar["color"]
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    frame_bbox, _frame_color, fill_color, hbars, vbars = _extract_scene(grid)
    out = _solve_from_scene(frame_bbox, fill_color, hbars, vbars)
    colored = out[out != fill_color]
    if colored.size < 4:
        return False
    if len(set(map(int, colored))) < 2:
        return False
    same_color = any(h["color"] == v["color"] for h in hbars for v in vbars)
    h_wins = any(h["color"] != v["color"] and h["length"] > v["length"] for h in hbars for v in vbars)
    v_wins = any(h["color"] != v["color"] and v["length"] > h["length"] for h in hbars for v in vbars)
    if not (same_color and h_wins and v_wins):
        return False
    bg = _get_bg(grid)
    if int(np.sum(grid == bg)) <= max(np.sum(grid == fill_color), np.sum(grid == _frame_color)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    palette = list(range(10))
    bg, frame_color, fill_color = random.sample(palette, 3)
    usable = [c for c in palette if c not in {bg, frame_color, fill_color}]
    h_count = random.randint(2, 3)
    v_count = random.randint(2, 3)
    inner_h = random.randint(max(5, h_count + 2), 8)
    inner_w = random.randint(max(5, v_count + 2), 8)

    total = h_count + v_count
    for _ in range(80):
        max_length = total + random.randint(2, 3)
        lengths = random.sample(range(1, max_length + 1), total)
        random.shuffle(lengths)
        h_lengths = lengths[:h_count]
        v_lengths = lengths[h_count:]
        if min(h_lengths) < max(v_lengths) and min(v_lengths) < max(h_lengths):
            break
    else:
        return None

    shared_color = random.choice(usable)
    others = [c for c in usable if c != shared_color]
    if len(others) < 2:
        return None

    h_colors = [shared_color]
    v_colors = [shared_color]
    while len(h_colors) < h_count:
        h_colors.append(random.choice(others))
    while len(v_colors) < v_count:
        v_colors.append(random.choice(others))
    random.shuffle(h_colors)
    random.shuffle(v_colors)
    if shared_color not in h_colors:
        h_colors[0] = shared_color
    if shared_color not in v_colors:
        v_colors[0] = shared_color

    rows = random.sample(range(inner_h), h_count)
    cols = random.sample(range(inner_w), v_count)
    h_sides = [random.choice(["left", "right"]) for _ in range(h_count)]
    v_sides = [random.choice(["top", "bottom"]) for _ in range(v_count)]

    left_margin = max([1] + [length for length, side in zip(h_lengths, h_sides) if side == "left"])
    right_margin = max([1] + [length for length, side in zip(h_lengths, h_sides) if side == "right"])
    top_margin = max([1] + [length for length, side in zip(v_lengths, v_sides) if side == "top"])
    bottom_margin = max([1] + [length for length, side in zip(v_lengths, v_sides) if side == "bottom"])
    left_margin += random.randint(0, 2)
    right_margin += random.randint(0, 2)
    top_margin += random.randint(0, 2)
    bottom_margin += random.randint(0, 2)

    frame_h = inner_h + 2
    frame_w = inner_w + 2
    grid_h = top_margin + frame_h + bottom_margin
    grid_w = left_margin + frame_w + right_margin
    if max(grid_h, grid_w) > 30:
        return None

    frame_bbox = (top_margin, left_margin, top_margin + frame_h - 1, left_margin + frame_w - 1)
    hbars = [
        {"pos": pos, "length": length, "color": color, "side": side}
        for pos, length, color, side in zip(rows, h_lengths, h_colors, h_sides)
    ]
    vbars = [
        {"pos": pos, "length": length, "color": color, "side": side}
        for pos, length, color, side in zip(cols, v_lengths, v_colors, v_sides)
    ]

    grid = _render_input((grid_h, grid_w), frame_bbox, bg, frame_color, fill_color, hbars, vbars)
    try:
        expected = _solve_from_scene(frame_bbox, fill_color, sorted(hbars, key=lambda x: x["pos"]), sorted(vbars, key=lambda x: x["pos"]))
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
