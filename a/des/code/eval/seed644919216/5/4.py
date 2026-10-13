from __future__ import annotations

import random
from itertools import combinations

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _legend_from_corner(
    grid: np.ndarray, bg: int, corner: str
) -> tuple[int, dict[tuple[int, int], int], np.ndarray]:
    h, w = grid.shape
    if h < 3 or w < 3:
        return 0, {}, np.zeros_like(grid, dtype=bool)

    c0 = 0 if corner in {"tl", "bl"} else w - 3
    count = 0
    mapping: dict[tuple[int, int], int] = {}
    mask = np.zeros_like(grid, dtype=bool)

    while True:
        if corner in {"tl", "tr"}:
            r0 = 3 * count
            if r0 + 3 > h:
                break
        else:
            r0 = h - 3 * (count + 1)
            if r0 < 0:
                break

        block = grid[r0 : r0 + 3, c0 : c0 + 3]
        if block.shape != (3, 3):
            break

        nz = np.argwhere(block != bg)
        if nz.shape[0] != 3:
            break
        if {tuple(x) for x in nz.tolist()} != {(0, 0), (0, 2), (1, 1)}:
            break

        a = int(block[0, 0])
        b = int(block[0, 2])
        mix = int(block[1, 1])
        if a == bg or b == bg or mix == bg or a == b:
            break

        key = tuple(sorted((a, b)))
        mapping[key] = mix
        mask[r0 : r0 + 3, c0 : c0 + 3] = True
        count += 1

    return count, mapping, mask


def _legend_data(grid: np.ndarray, bg: int) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    best_count = 0
    best_mapping: dict[tuple[int, int], int] = {}
    best_mask = np.zeros_like(grid, dtype=bool)

    for corner in ("tl", "tr", "bl", "br"):
        count, mapping, mask = _legend_from_corner(grid, bg, corner)
        if count > best_count:
            best_count = count
            best_mapping = mapping
            best_mask = mask

    return best_mapping, best_mask


def _run_length(
    grid: np.ndarray, mask: np.ndarray, r: int, c: int, dr: int, dc: int, bg: int
) -> tuple[int, int]:
    h, w = grid.shape
    color = int(grid[r, c])
    if color == bg or mask[r, c]:
        return 0, bg

    length = 0
    rr, cc = r, c
    while 0 <= rr < h and 0 <= cc < w and not mask[rr, cc] and int(grid[rr, cc]) == color:
        length += 1
        rr += dr
        cc += dc
    return length, color


def _choose_gap_fill(color_a: int, len_a: int, color_b: int, len_b: int) -> int | None:
    if len_a > len_b:
        return color_a
    if len_b > len_a:
        return color_b
    if color_a == color_b:
        return color_a
    return None


def _gap_fill_plan(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    fills: list[tuple[int, int, int]] = []

    for r in range(h):
        for c in range(w):
            if legend_mask[r, c] or int(grid[r, c]) != bg:
                continue

            candidates: list[tuple[str, int, int]] = []

            if 0 < c < w - 1:
                left = int(grid[r, c - 1])
                right = int(grid[r, c + 1])
                if left != bg and right != bg and not legend_mask[r, c - 1] and not legend_mask[r, c + 1]:
                    left_len, left_color = _run_length(grid, legend_mask, r, c - 1, 0, -1, bg)
                    right_len, right_color = _run_length(grid, legend_mask, r, c + 1, 0, 1, bg)
                    if left_len >= 2 and right_len >= 2:
                        fill = _choose_gap_fill(left_color, left_len, right_color, right_len)
                        if fill is not None:
                            candidates.append(("h", fill, max(left_len, right_len)))

            if 0 < r < h - 1:
                up = int(grid[r - 1, c])
                down = int(grid[r + 1, c])
                if up != bg and down != bg and not legend_mask[r - 1, c] and not legend_mask[r + 1, c]:
                    up_len, up_color = _run_length(grid, legend_mask, r - 1, c, -1, 0, bg)
                    down_len, down_color = _run_length(grid, legend_mask, r + 1, c, 1, 0, bg)
                    if up_len >= 2 and down_len >= 2:
                        fill = _choose_gap_fill(up_color, up_len, down_color, down_len)
                        if fill is not None:
                            candidates.append(("v", fill, max(up_len, down_len)))

            if len(candidates) == 1:
                fills.append((r, c, candidates[0][1]))
            elif len(candidates) == 2:
                first, second = candidates
                if first[1] == second[1]:
                    fills.append((r, c, first[1]))
                elif first[2] > second[2]:
                    fills.append((r, c, first[1]))
                elif second[2] > first[2]:
                    fills.append((r, c, second[1]))

    return fills


def _axis_color(
    grid: np.ndarray, legend_mask: np.ndarray, r: int, c: int, bg: int, axis: str
) -> int | None:
    h, w = grid.shape
    deltas = ((0, -1), (0, 1)) if axis == "h" else ((-1, 0), (1, 0))
    colors: list[int] = []

    for dr, dc in deltas:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and not legend_mask[rr, cc]:
            color = int(grid[rr, cc])
            if color != bg:
                colors.append(color)

    if not colors:
        return None
    if len(set(colors)) == 1:
        return colors[0]
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect one-cell gaps between aligned pipe fragments, then recolor crossings
    via a corner legend of pairwise color mixes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    legend_map, legend_mask = _legend_data(g, bg)

    out = g.copy()
    for r, c, color in _gap_fill_plan(g, bg, legend_mask):
        out[r, c] = color

    h, w = out.shape
    for r in range(h):
        for c in range(w):
            if legend_mask[r, c] or int(out[r, c]) == bg:
                continue

            neighbors = 0
            has_horizontal = False
            has_vertical = False
            for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and not legend_mask[rr, cc] and int(out[rr, cc]) != bg:
                    neighbors += 1
                    if dr == 0:
                        has_horizontal = True
                    else:
                        has_vertical = True

            if neighbors < 3 or not has_horizontal or not has_vertical:
                continue

            h_color = _axis_color(out, legend_mask, r, c, bg, "h")
            v_color = _axis_color(out, legend_mask, r, c, bg, "v")
            if h_color is None or v_color is None:
                continue

            mix = legend_map.get(tuple(sorted((h_color, v_color))))
            if mix is not None:
                out[r, c] = mix

    return out


def _random_pipe_colors(bg: int, count: int = 4) -> tuple[list[int], list[int]]:
    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    return colors[:count], colors[count:]


def _through_lengths() -> tuple[int, int]:
    return random.choice(((2, 2), (2, 3), (3, 2)))


def _build_gap_sprite(bg: int, colors: list[int], orient: str, same_color: bool) -> tuple[np.ndarray, dict]:
    if same_color:
        color = random.choice(colors)
        color_a = color_b = color
    else:
        color_a, color_b = random.sample(colors, 2)

    len_a = random.randint(2, 5)
    len_b = random.randint(2, 5)
    if not same_color and len_a == len_b:
        len_b = 2 + ((len_b - 1) % 4)
        if len_b == len_a:
            len_b = 5 if len_a != 5 else 2

    fill = _choose_gap_fill(color_a, len_a, color_b, len_b)
    assert fill is not None

    if orient == "h":
        sprite = np.full((1, len_a + 1 + len_b), bg, dtype=int)
        sprite[0, :len_a] = color_a
        sprite[0, len_a + 1 :] = color_b
        gap = (0, len_a)
    else:
        sprite = np.full((len_a + 1 + len_b, 1), bg, dtype=int)
        sprite[:len_a, 0] = color_a
        sprite[len_a + 1 :, 0] = color_b
        gap = (len_a, 0)

    meta = {"kind": "gap", "gap": gap, "fill": fill}
    return sprite, meta


def _build_intersection_sprite(
    bg: int, horiz_color: int, vert_color: int, shape: str, visible_axis: str
) -> tuple[np.ndarray, dict]:
    if visible_axis == "h":
        left_len, right_len = _through_lengths()
        if shape == "cross":
            up_len = random.randint(2, 3)
            down_len = random.randint(2, 3)
        else:
            if random.random() < 0.5:
                up_len, down_len = random.randint(2, 3), 0
            else:
                up_len, down_len = 0, random.randint(2, 3)

        sprite = np.full((up_len + 1 + down_len, left_len + 1 + right_len), bg, dtype=int)
        cr, cc = up_len, left_len
        sprite[cr, :] = horiz_color
        if up_len:
            sprite[:cr, cc] = vert_color
        if down_len:
            sprite[cr + 1 :, cc] = vert_color
        center_color = horiz_color
    else:
        up_len, down_len = _through_lengths()
        if shape == "cross":
            left_len = random.randint(2, 3)
            right_len = random.randint(2, 3)
        else:
            if random.random() < 0.5:
                left_len, right_len = random.randint(2, 3), 0
            else:
                left_len, right_len = 0, random.randint(2, 3)

        sprite = np.full((up_len + 1 + down_len, left_len + 1 + right_len), bg, dtype=int)
        cr, cc = up_len, left_len
        sprite[:, cc] = vert_color
        if left_len:
            sprite[cr, :cc] = horiz_color
        if right_len:
            sprite[cr, cc + 1 :] = horiz_color
        center_color = vert_color

    meta = {
        "kind": "intersection",
        "center": (cr, cc),
        "pair": tuple(sorted((horiz_color, vert_color))),
        "center_color": center_color,
    }
    return sprite, meta


def _reserve_corner(blocked: np.ndarray, corner: str) -> None:
    h, w = blocked.shape
    reserve_h = min(12, h)
    reserve_w = min(4, w)

    if corner == "tl":
        blocked[:reserve_h, :reserve_w] = True
    elif corner == "tr":
        blocked[:reserve_h, w - reserve_w :] = True
    elif corner == "bl":
        blocked[h - reserve_h :, :reserve_w] = True
    else:
        blocked[h - reserve_h :, w - reserve_w :] = True


def _place_sprite(
    grid: np.ndarray, blocked: np.ndarray, sprite: np.ndarray, bg: int
) -> tuple[int, int] | None:
    h, w = grid.shape
    sh, sw = sprite.shape
    if sh + 2 > h or sw + 2 > w:
        return None

    row_lo, row_hi = 1, h - sh - 1
    col_lo, col_hi = 1, w - sw - 1
    if row_hi < row_lo or col_hi < col_lo:
        return None

    for _ in range(200):
        r0 = random.randint(row_lo, row_hi)
        c0 = random.randint(col_lo, col_hi)

        rr0 = max(0, r0 - 1)
        rr1 = min(h, r0 + sh + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(w, c0 + sw + 1)
        if blocked[rr0:rr1, cc0:cc1].any():
            continue

        mask = sprite != bg
        grid[r0 : r0 + sh, c0 : c0 + sw][mask] = sprite[mask]
        blocked[rr0:rr1, cc0:cc1] = True
        return r0, c0

    return None


def _render_legend(
    grid: np.ndarray, bg: int, corner: str, entries: list[tuple[tuple[int, int], int]]
) -> dict[tuple[int, int], int]:
    h, w = grid.shape
    c0 = 0 if corner in {"tl", "bl"} else w - 3
    mapping: dict[tuple[int, int], int] = {}

    for idx, (pair, mix) in enumerate(entries):
        if corner in {"tl", "tr"}:
            r0 = 3 * idx
        else:
            r0 = h - 3 * (idx + 1)

        a, b = pair
        if random.random() < 0.5:
            a, b = b, a

        block = np.full((3, 3), bg, dtype=int)
        block[0, 0] = a
        block[0, 2] = b
        block[1, 1] = mix
        grid[r0 : r0 + 3, c0 : c0 + 3] = block
        mapping[tuple(sorted(pair))] = mix

    return mapping


def _validate_scene(
    grid: np.ndarray,
    gap_points: list[tuple[tuple[int, int], int]],
    inter_points: list[tuple[tuple[int, int], tuple[int, int], int]],
    legend_map: dict[tuple[int, int], int],
) -> bool:
    bg = _bg_color(grid)
    if int(np.count_nonzero(grid == bg)) <= grid.size // 2:
        return False

    out = main(grid)
    for (r, c), fill in gap_points:
        if int(out[r, c]) != fill:
            return False

    for (r, c), pair, center_color in inter_points:
        mix = legend_map.get(pair)
        if mix is None:
            if int(out[r, c]) != center_color:
                return False
        else:
            if int(out[r, c]) != mix:
                return False

    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    bg = random.randint(0, 9)
    pipe_colors, mix_pool = _random_pipe_colors(bg, count=4)
    if len(mix_pool) < 2:
        return None

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    corner = random.choice(("tl", "tr", "bl", "br"))
    _reserve_corner(blocked, corner)

    gap_specs: list[tuple[str, bool]] = [("h", False), ("v", True)]
    for _ in range(random.randint(0, 2)):
        gap_specs.append((random.choice(("h", "v")), random.random() < 0.5))
    random.shuffle(gap_specs)

    inter_shapes = ["cross", "tee"]
    for _ in range(random.randint(0, 2)):
        inter_shapes.append(random.choice(("cross", "tee")))
    random.shuffle(inter_shapes)

    pair_pool = random.sample(list(combinations(pipe_colors, 2)), k=len(inter_shapes))

    gap_points: list[tuple[tuple[int, int], int]] = []
    inter_points: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    seen_pairs: list[tuple[int, int]] = []

    for orient, same_color in gap_specs:
        sprite, meta = _build_gap_sprite(bg, pipe_colors, orient, same_color)
        placed = _place_sprite(grid, blocked, sprite, bg)
        if placed is None:
            return None
        r0, c0 = placed
        gr, gc = meta["gap"]
        gap_points.append(((r0 + gr, c0 + gc), int(meta["fill"])))

    for shape, pair in zip(inter_shapes, pair_pool, strict=True):
        if random.random() < 0.5:
            horiz_color, vert_color = pair
        else:
            vert_color, horiz_color = pair
        sprite, meta = _build_intersection_sprite(
            bg, horiz_color, vert_color, shape, random.choice(("h", "v"))
        )
        placed = _place_sprite(grid, blocked, sprite, bg)
        if placed is None:
            return None
        r0, c0 = placed
        cr, cc = meta["center"]
        pair_key = tuple(sorted(meta["pair"]))
        inter_points.append(((r0 + cr, c0 + cc), pair_key, int(meta["center_color"])))
        seen_pairs.append(pair_key)

    distinct_pairs = sorted(set(seen_pairs))
    if len(distinct_pairs) < 2:
        return None

    max_mapped = min(3, len(distinct_pairs) - 1)
    if max_mapped < 1:
        return None
    mapped_count = random.randint(1, max_mapped)
    mapped_pairs = random.sample(distinct_pairs, mapped_count)
    if len(mix_pool) < mapped_count:
        return None

    mixes = random.sample(mix_pool, mapped_count)
    legend_entries = list(zip(mapped_pairs, mixes, strict=True))
    random.shuffle(legend_entries)
    legend_map = _render_legend(grid, bg, corner, legend_entries)

    if not _validate_scene(grid, gap_points, inter_points, legend_map):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
