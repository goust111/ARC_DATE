import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _rotate_cw(arr: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(arr, -(turns % 4))


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, int, int, dict[int, int], np.ndarray]:
    best = None
    h, w = grid.shape
    search_rows = min(h, max(4, h // 3))
    for r in range(search_rows):
        row = grid[r]
        non_bg = np.flatnonzero(row != bg)
        if non_bg.size == 0:
            continue
        c0 = int(non_bg[0])
        c1 = int(non_bg[-1])
        if np.any(row[c0 : c1 + 1] == bg):
            continue
        runs: list[tuple[int, int, int]] = []
        seen_colors: set[int] = set()
        c = c0
        while c <= c1:
            color = int(row[c])
            end = c
            while end + 1 <= c1 and int(row[end + 1]) == color:
                end += 1
            width = end - c + 1
            if width < 1 or width > 3 or color in seen_colors:
                runs = []
                break
            seen_colors.add(color)
            runs.append((color, width, c))
            c = end + 1
        if len(runs) < 3:
            continue
        score = (r, -len(runs), c1 - c0)
        if best is None or score < best[0]:
            best = (score, r, c0, c1 + 1, runs)

    assert best is not None, "legend row not found"
    _, row, start, end, runs = best
    mapping = {color: width - 1 for color, width, _ in runs}
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[row, start:end] = True
    return int(row), int(start), int(end), mapping, legend_mask


def _find_frame_inner(grid: np.ndarray, bg: int, marker: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = marker
    h, w = grid.shape
    assert r0 + 1 < h and c0 + 1 < w, "marker cannot sit on the outer border"

    border = int(grid[r0, c0 + 1])
    if border == bg:
        border = int(grid[r0 + 1, c0])
    assert border != bg and border != int(grid[r0, c0]), "frame border color not found"
    assert int(grid[r0, c0 + 1]) == border and int(grid[r0 + 1, c0]) == border

    c1 = c0 + 1
    while c1 + 1 < w and int(grid[r0, c1 + 1]) == border:
        c1 += 1
    r1 = r0 + 1
    while r1 + 1 < h and int(grid[r1 + 1, c0]) == border:
        r1 += 1

    assert c1 >= c0 + 2 and r1 >= r0 + 2, "frame must enclose a non-empty interior"
    assert np.all(grid[r0, c0 + 1 : c1 + 1] == border)
    assert np.all(grid[r0 + 1 : r1 + 1, c0] == border)
    assert np.all(grid[r1, c0 : c1 + 1] == border)
    assert np.all(grid[r0 : r1 + 1, c1] == border)
    assert np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg)

    return r0 + 1, c0 + 1, r1 - r0 - 1, c1 - c0 - 1


def _mask_from_points(points: list[tuple[int, int]]) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask, r0, c0


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color
    grid[top : top + h, left : left + w] = patch


def _clear_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, bg: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = bg
    grid[top : top + h, left : left + w] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each single-color fragment with the frame whose top-left marker has the same color.
    The legend row encodes the clockwise rotation count through each segment width:
    1 -> 0 turns, 2 -> 1 turn, 3 -> 2 turns.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    _, _, _, color_to_turns, legend_mask = _find_legend(g, bg)
    for color, turns in color_to_turns.items():
        comps = _components((g == color) & ~legend_mask)
        marker_comps = [comp for comp in comps if len(comp) == 1]
        fragment_comps = [comp for comp in comps if len(comp) > 1]
        assert len(marker_comps) == 1, "each color must have one frame marker"
        assert len(fragment_comps) == 1, "each color must have one fragment"

        marker = marker_comps[0][0]
        fragment = fragment_comps[0]
        inner_r, inner_c, inner_h, inner_w = _find_frame_inner(g, bg, marker)
        mask, frag_r, frag_c = _mask_from_points(fragment)
        rotated = _rotate_cw(mask, turns)
        rot_h, rot_w = rotated.shape
        assert rot_h <= inner_h and rot_w <= inner_w, "rotated fragment does not fit its frame"

        _clear_mask(out, frag_r, frag_c, mask, bg)
        _paint_mask(out, inner_r, inner_c, rotated, color)

    return out


def _l_shape() -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    return mask


def _t_shape() -> np.ndarray:
    height = random.randint(2, 4)
    width = random.choice((3, 5))
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[:, width // 2] = True
    return mask


def _random_fragment_mask() -> np.ndarray:
    base = _l_shape() if random.random() < 0.5 else _t_shape()
    return _rotate_cw(base, random.randint(0, 3))


def _can_place(occ: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occ.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not occ[r0:r1, c0:c1].any()


def _place_box(occ: np.ndarray, height: int, width: int, row_min: int) -> tuple[int, int] | None:
    h, w = occ.shape
    candidates: list[tuple[int, int]] = []
    for r in range(row_min, h - height + 1):
        for c in range(0, w - width + 1):
            if _can_place(occ, r, c, height, width):
                candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _mark_box(occ: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = occ.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    occ[r0:r1, c0:c1] = True


def _render_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, border: int, marker: int) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid[top, left] = marker
    grid[top, left + 1 : left + outer_w] = border
    grid[top + 1 : top + outer_h, left] = border
    grid[top + outer_h - 1, left : left + outer_w] = border
    grid[top : top + outer_h, left + outer_w - 1] = border


def _make_expected(inp: np.ndarray, specs: list[dict], bg: int) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        frag_top, frag_left = spec["fragment_pos"]
        _clear_mask(out, frag_top, frag_left, spec["fragment_mask"], bg)
        frame_top, frame_left = spec["frame_pos"]
        _paint_mask(out, frame_top + 1, frame_left + 1, spec["placed_mask"], spec["color"])
    return out


def _build_candidate() -> np.ndarray | None:
    h = random.randint(20, 28)
    w = random.randint(20, 30)
    bg = random.randint(0, 9)
    palette = [color for color in range(10) if color != bg]
    border = random.choice(palette)
    palette.remove(border)
    colors = random.sample(palette, 3)

    turns_by_color = {colors[0]: 0, colors[1]: 1, colors[2]: 2}
    legend_order = colors[:]
    random.shuffle(legend_order)
    legend_length = sum(turns_by_color[color] + 1 for color in legend_order)
    legend_start = random.randint(1, w - legend_length - 1)

    specs: list[dict] = []
    for color in colors:
        fragment_mask = _random_fragment_mask()
        placed_mask = _rotate_cw(fragment_mask, turns_by_color[color])
        extra_h = random.randint(0, 2)
        extra_w = random.randint(0, 2)
        inner_h = placed_mask.shape[0] + extra_h
        inner_w = placed_mask.shape[1] + extra_w
        specs.append(
            {
                "color": color,
                "turns": turns_by_color[color],
                "fragment_mask": fragment_mask,
                "placed_mask": placed_mask,
                "inner_h": inner_h,
                "inner_w": inner_w,
            }
        )

    occ = np.zeros((h, w), dtype=bool)
    occ[:3, :] = True

    for spec in sorted(specs, key=lambda item: (item["inner_h"] + 2) * (item["inner_w"] + 2), reverse=True):
        pos = _place_box(occ, spec["inner_h"] + 2, spec["inner_w"] + 2, row_min=3)
        if pos is None:
            return None
        spec["frame_pos"] = pos
        _mark_box(occ, pos[0], pos[1], spec["inner_h"] + 2, spec["inner_w"] + 2)

    for spec in sorted(specs, key=lambda item: item["fragment_mask"].size, reverse=True):
        frag_h, frag_w = spec["fragment_mask"].shape
        pos = _place_box(occ, frag_h, frag_w, row_min=3)
        if pos is None:
            return None
        spec["fragment_pos"] = pos
        _mark_box(occ, pos[0], pos[1], frag_h, frag_w)

    inp = np.full((h, w), bg, dtype=int)
    col = legend_start
    for color in legend_order:
        width = turns_by_color[color] + 1
        inp[0, col : col + width] = color
        col += width

    for spec in specs:
        frame_top, frame_left = spec["frame_pos"]
        _render_frame(inp, frame_top, frame_left, spec["inner_h"], spec["inner_w"], border, spec["color"])

        frag_top, frag_left = spec["fragment_pos"]
        _paint_mask(inp, frag_top, frag_left, spec["fragment_mask"], spec["color"])

    values, counts = np.unique(inp, return_counts=True)
    if counts[int(np.argmax(counts))] <= max(count for value, count in zip(values, counts) if int(value) != bg):
        return None

    expected = _make_expected(inp, specs, bg)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
