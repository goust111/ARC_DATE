from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    border = np.concatenate((g[0], g[-1], g[1:-1, 0], g[1:-1, -1]))
    return int(Counter(map(int, border)).most_common(1)[0][0])


def _intervals(flags: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, val in enumerate(map(bool, flags)):
        if val and start is None:
            start = i
        elif not val and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def _extract_regions(grid: np.ndarray, bg: int) -> dict[str, tuple[int, int, int, int]]:
    cols = np.any(grid != bg, axis=0)
    col_intervals = _intervals(cols)
    assert len(col_intervals) == 2, "expected exactly two separated non-background regions"

    regions: list[tuple[int, int, int, int]] = []
    for c0, c1 in col_intervals:
        rows = np.any(grid[:, c0 : c1 + 1] != bg, axis=1)
        row_ids = np.where(rows)[0]
        assert row_ids.size > 0
        r0 = int(row_ids.min())
        r1 = int(row_ids.max())
        regions.append((r0, c0, r1, c1))

    left, right = sorted(regions, key=lambda b: b[1])
    return {"left": left, "right": right}


def _parse_left_groups(patch: np.ndarray, bg: int) -> list[dict[str, int | None]]:
    h, w = patch.shape
    assert h >= 1 and w >= 1
    for r in range(h):
        assert np.all(patch[r] == patch[r, 0]), "left rows must be full-width uniform stripes or gaps"

    groups: list[dict[str, int | None]] = []
    r = h - 1
    while r >= 0:
        col = int(patch[r, 0])
        size = 1
        rr = r - 1
        while rr >= 0 and int(patch[rr, 0]) == col:
            size += 1
            rr -= 1
        groups.append({"color": None if col == bg else col, "size": int(size), "mode": "sum"})
        r = rr

    assert groups[0]["color"] is not None, "bottom stripe must stay visible to localize the stack"
    assert groups[-1]["color"] is not None, "top stripe must stay visible to localize the stack"
    return groups


def _band_is_color(square: np.ndarray, offset: int, color: int) -> bool:
    n = square.shape[0]
    lo = int(offset)
    hi = int(n - 1 - offset)
    if lo > hi:
        return False
    if not np.all(square[lo, lo : hi + 1] == color):
        return False
    if not np.all(square[hi, lo : hi + 1] == color):
        return False
    if not np.all(square[lo : hi + 1, lo] == color):
        return False
    if not np.all(square[lo : hi + 1, hi] == color):
        return False
    return True


def _parse_right_groups(patch: np.ndarray, bg: int) -> list[dict[str, int | None]]:
    n, m = patch.shape
    assert n == m, "right object must be a square"
    groups_outer_to_inner: list[dict[str, int | None]] = []
    lo = 0
    hi = n - 1

    while lo <= hi:
        cur = patch[lo : hi + 1, lo : hi + 1]
        side = int(cur.shape[0])
        uniq = np.unique(cur)
        if uniq.size == 1:
            col = int(uniq[0])
            groups_outer_to_inner.append(
                {"color": None if col == bg else col, "size": side, "mode": "sum"}
            )
            break

        col = int(cur[0, 0])
        thickness = 0
        while 2 * thickness < side and _band_is_color(cur, thickness, col):
            thickness += 1
        assert thickness > 0, "failed to parse a right-side layer thickness"
        groups_outer_to_inner.append(
            {"color": None if col == bg else col, "size": int(thickness), "mode": "sum"}
        )
        lo += thickness
        hi -= thickness

    groups = list(reversed(groups_outer_to_inner))
    assert groups[-1]["color"] is not None, "outer square frame must stay visible to localize the square"
    if groups and groups[0]["color"] is None:
        groups[0]["mode"] = "center_square"
    return groups


def _count_decodings(
    left_groups: list[dict[str, int | None]],
    right_groups: list[dict[str, int | None]],
    limit: int = 2,
) -> int:
    left = tuple((g["color"], int(g["size"])) for g in left_groups)
    right = tuple((g["color"], int(g["size"]), str(g["mode"])) for g in right_groups)
    memo: dict[tuple[int, int, int, int, str], int] = {}

    def rec(i_l: int, rem_l: int, i_r: int, rem_r: int, mode_r: str) -> int:
        key = (i_l, rem_l, i_r, rem_r, mode_r)
        if key in memo:
            return memo[key]
        if i_l == len(left) and i_r == len(right):
            return 1
        if i_l == len(left) or i_r == len(right):
            return 0

        color_l = left[i_l][0]
        color_r = right[i_r][0]
        candidates: list[tuple[int, int]] = []
        if color_l is not None:
            candidates.append((int(color_l), int(rem_l)))
        if color_r is not None:
            candidates.append((int(color_r), int(rem_r)))
        if not candidates:
            memo[key] = 0
            return 0

        first = candidates[0]
        if any(c != first for c in candidates[1:]):
            memo[key] = 0
            return 0

        color, height = first
        if height <= 0:
            memo[key] = 0
            return 0

        next_i_l = i_l
        next_rem_l = rem_l
        if color_l is None:
            if rem_l < height:
                memo[key] = 0
                return 0
            next_rem_l = rem_l - height
        else:
            if rem_l != height:
                memo[key] = 0
                return 0
            next_rem_l = 0
        if next_rem_l == 0:
            next_i_l += 1
            if next_i_l < len(left):
                next_rem_l = int(left[next_i_l][1])

        next_i_r = i_r
        next_rem_r = rem_r
        next_mode_r = mode_r
        if color_r is None:
            if mode_r == "center_square":
                if rem_r < height or (rem_r - height) % 2 != 0:
                    memo[key] = 0
                    return 0
                next_rem_r = (rem_r - height) // 2
                next_mode_r = "sum"
            else:
                if rem_r < height:
                    memo[key] = 0
                    return 0
                next_rem_r = rem_r - height
        else:
            if rem_r != height:
                memo[key] = 0
                return 0
            next_rem_r = 0
            next_mode_r = "sum"
        if next_rem_r == 0:
            next_i_r += 1
            if next_i_r < len(right):
                next_rem_r = int(right[next_i_r][1])
                next_mode_r = str(right[next_i_r][2])

        total = rec(next_i_l, next_rem_l, next_i_r, next_rem_r, next_mode_r)
        memo[key] = min(limit, total)
        return memo[key]

    return rec(0, int(left[0][1]), 0, int(right[0][1]), str(right[0][2]))


def _decode_layers(
    left_groups: list[dict[str, int | None]],
    right_groups: list[dict[str, int | None]],
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    left = [dict(g) for g in left_groups]
    right = [dict(g) for g in right_groups]
    assert _count_decodings(left, right) == 1, "layer sequence must be uniquely forced"

    layers: list[tuple[int, int]] = []
    i_l = 0
    i_r = 0
    rem_l = int(left[0]["size"])
    rem_r = int(right[0]["size"])
    repair_left = 0
    repair_right = 0

    while i_l < len(left) and i_r < len(right):
        group_l = left[i_l]
        group_r = right[i_r]
        candidates: list[tuple[int, int]] = []
        if group_l["color"] is not None:
            candidates.append((int(group_l["color"]), int(rem_l)))
        if group_r["color"] is not None:
            candidates.append((int(group_r["color"]), int(rem_r)))
        assert candidates, "a layer cannot be missing on both sides"

        color, height = candidates[0]
        assert all(candidate == (color, height) for candidate in candidates[1:])
        assert height > 0

        if group_l["color"] is None:
            repair_left += 1
            rem_l -= height
            assert rem_l >= 0
        else:
            assert rem_l == height
            rem_l = 0

        if group_r["color"] is None:
            repair_right += 1
            if str(group_r["mode"]) == "center_square":
                assert rem_r >= height and (rem_r - height) % 2 == 0
                rem_r = (rem_r - height) // 2
                group_r["mode"] = "sum"
            else:
                rem_r -= height
                assert rem_r >= 0
        else:
            assert rem_r == height
            rem_r = 0

        layers.append((int(color), int(height)))

        if rem_l == 0:
            i_l += 1
            if i_l < len(left):
                rem_l = int(left[i_l]["size"])
        if rem_r == 0:
            i_r += 1
            if i_r < len(right):
                rem_r = int(right[i_r]["size"])

    assert i_l == len(left) and i_r == len(right), "both encodings must finish together"
    return layers, {"repair_left": repair_left, "repair_right": repair_right}


def _build_left_stack(layers: list[tuple[int, int]], width: int) -> np.ndarray:
    stripes = [np.full((height, width), color, dtype=int) for color, height in reversed(layers)]
    return np.vstack(stripes)


def _build_right_square(layers: list[tuple[int, int]]) -> np.ndarray:
    color0, h0 = layers[0]
    cur = np.full((h0, h0), color0, dtype=int)
    for color, thickness in layers[1:]:
        nxt = np.full((cur.shape[0] + 2 * thickness, cur.shape[1] + 2 * thickness), color, dtype=int)
        nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
        cur = nxt
    return cur


def _layer_mask(layers: list[tuple[int, int]], idx: int) -> np.ndarray:
    heights = [height for _color, height in layers]
    h0 = heights[0]
    total_side = int(h0 + 2 * sum(heights[1:]))
    mask = np.zeros((total_side, total_side), dtype=bool)
    outer_pad = int(sum(heights[idx + 1 :]))
    if idx == 0:
        side = int(h0)
        mask[outer_pad : outer_pad + side, outer_pad : outer_pad + side] = True
        return mask

    outer_side = int(h0 + 2 * sum(heights[1 : idx + 1]))
    inner_side = int(h0 + 2 * sum(heights[1:idx]))
    mask[outer_pad : outer_pad + outer_side, outer_pad : outer_pad + outer_side] = True
    inner_pad = outer_pad + heights[idx]
    mask[inner_pad : inner_pad + inner_side, inner_pad : inner_pad + inner_side] = False
    return mask


def _paint_missing_left(stack: np.ndarray, bg: int, layers: list[tuple[int, int]], missing: set[int]) -> np.ndarray:
    out = stack.copy()
    total_h = int(sum(height for _color, height in layers))
    consumed = 0
    for idx, (_color, height) in enumerate(layers):
        r1 = total_h - consumed
        r0 = r1 - height
        if idx in missing:
            out[r0:r1, :] = bg
        consumed += height
    return out


def _paint_missing_right(square: np.ndarray, bg: int, layers: list[tuple[int, int]], missing: set[int]) -> np.ndarray:
    out = square.copy()
    for idx in missing:
        out[_layer_mask(layers, idx)] = bg
    return out


def _has_adjacent(indices: set[int]) -> bool:
    ordered = sorted(indices)
    return any(b == a + 1 for a, b in zip(ordered, ordered[1:]))


def _solve_core(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    regions = _extract_regions(g, bg)

    lr0, lc0, lr1, lc1 = regions["left"]
    rr0, rc0, rr1, rc1 = regions["right"]
    left_patch = g[lr0 : lr1 + 1, lc0 : lc1 + 1]
    right_patch = g[rr0 : rr1 + 1, rc0 : rc1 + 1]

    left_groups = _parse_left_groups(left_patch, bg)
    right_groups = _parse_right_groups(right_patch, bg)
    layers, usage = _decode_layers(left_groups, right_groups)

    full_left = _build_left_stack(layers, left_patch.shape[1])
    full_right = _build_right_square(layers)
    assert full_right.shape == right_patch.shape

    out = g.copy()
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = full_left
    out[rr0 : rr1 + 1, rc0 : rc1 + 1] = full_right

    meta = {
        "bg": bg,
        "layers": layers,
        "left_groups": left_groups,
        "right_groups": right_groups,
        "repair_left": int(usage["repair_left"]),
        "repair_right": int(usage["repair_right"]),
        "regions": regions,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the left stripe stack and right concentric square by cross-decoding their missing layers.

    Input structure:
    - The left side is a stack of full-width horizontal stripes, read bottom-to-top.
    - The right side is an axis-aligned concentric square pattern, read inner-to-outer.
    - Some stripe layers and some square layers are replaced by background, but the same layer
      is never missing on both sides.

    Rule:
    - Parse the left stack into bottom-up stripe groups and the right square into inner-out layer groups.
    - Use the visible side of each layer to force its color and thickness, while background groups on the
      opposite side merely contribute remaining total thickness.
    - Reconstruct the full inner-to-outer layer sequence, then redraw the complete stripe stack and square.

    Key insight:
    - Both halves encode the same ordered layer list: stripe height equals square-layer thickness, and
      the missing boundaries inside a background run can only be recovered from the opposite encoding.
    """
    return _solve_core(grid)[0]


def _choose_layers() -> list[tuple[int, int]] | None:
    n_layers = random.randint(4, 6)
    colors = random.sample(range(1, 10), n_layers)
    heights = [random.randint(1, 4)]
    heights.extend(random.randint(1, 3) for _ in range(n_layers - 1))
    total_square = heights[0] + 2 * sum(heights[1:])
    total_stack = sum(heights)
    if total_square > 16 or total_stack > 15:
        return None
    return [(int(color), int(height)) for color, height in zip(colors, heights)]


def _choose_missing_sets(n_layers: int) -> tuple[set[int], set[int]] | None:
    patterns = [("left", False), ("right", False), ("both", False), ("left", True), ("right", True), ("both", True)]
    for _ in range(200):
        left: set[int] = set()
        right: set[int] = set()
        merged_side, allow_center_chain = random.choice(patterns)

        left_run_len = 2 if merged_side in {"left", "both"} else 1
        right_run_len = 2 if merged_side in {"right", "both"} else 1

        left_starts = [s for s in range(1, n_layers - left_run_len) if s + left_run_len - 1 <= n_layers - 2]
        if not left_starts:
            continue
        left_start = random.choice(left_starts)
        left.update(range(left_start, left_start + left_run_len))

        right_starts = []
        for s in range(0, n_layers):
            end = s + right_run_len - 1
            if end > n_layers - 2:
                continue
            run = set(range(s, end + 1))
            if run & left:
                continue
            if not allow_center_chain and 0 in run and len(run) > 1:
                continue
            right_starts.append(s)
        if not right_starts:
            continue
        right_start = random.choice(right_starts)
        right.update(range(right_start, right_start + right_run_len))

        remaining_left = [i for i in range(1, n_layers - 1) if i not in left and i not in right]
        remaining_right = [i for i in range(0, n_layers - 1) if i not in left and i not in right]

        if remaining_left and random.random() < 0.45:
            left.add(random.choice(remaining_left))
            remaining_right = [i for i in remaining_right if i not in left]
        if remaining_right and random.random() < 0.45:
            extra = random.choice(remaining_right)
            if extra != 0 or allow_center_chain or 1 not in right:
                right.add(extra)

        if not left or not right or left & right:
            continue
        if 0 in left or (n_layers - 1) in left:
            continue
        if (n_layers - 1) in right:
            continue
        if n_layers - len(left) < 2 or n_layers - len(right) < 2:
            continue
        if n_layers - len(left | right) < 1:
            continue
        if not (_has_adjacent(left) or _has_adjacent(right)):
            continue
        return left, right
    return None


def _render_instance(
    layers: list[tuple[int, int]],
    left_missing: set[int],
    right_missing: set[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    bg = 0
    left_width = random.randint(4, 8)
    gap = random.randint(2, 5)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    top_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)

    clean_left = _build_left_stack(layers, left_width)
    clean_right = _build_right_square(layers)
    input_left = _paint_missing_left(clean_left, bg, layers, left_missing)
    input_right = _paint_missing_right(clean_right, bg, layers, right_missing)

    rows = max(clean_left.shape[0], clean_right.shape[0]) + top_margin + bottom_margin
    cols = left_margin + left_width + gap + clean_right.shape[1] + right_margin
    if rows > 30 or cols > 30:
        return None

    out = np.full((rows, cols), bg, dtype=int)
    inp = np.full((rows, cols), bg, dtype=int)

    left_r0 = random.randint(0, rows - clean_left.shape[0])
    right_r0 = random.randint(0, rows - clean_right.shape[0])
    left_c0 = left_margin
    right_c0 = cols - right_margin - clean_right.shape[1]

    out[left_r0 : left_r0 + clean_left.shape[0], left_c0 : left_c0 + left_width] = clean_left
    out[right_r0 : right_r0 + clean_right.shape[0], right_c0 : right_c0 + clean_right.shape[1]] = clean_right
    inp[left_r0 : left_r0 + input_left.shape[0], left_c0 : left_c0 + left_width] = input_left
    inp[right_r0 : right_r0 + input_right.shape[0], right_c0 : right_c0 + clean_right.shape[1]] = input_right

    return inp, out


def _validate_instance(
    inp: np.ndarray,
    out: np.ndarray,
    layers: list[tuple[int, int]],
    left_missing: set[int],
    right_missing: set[int],
) -> bool:
    try:
        solved, meta = _solve_core(inp)
    except Exception:
        return False

    if not np.array_equal(solved, out):
        return False
    if meta["layers"] != layers:
        return False
    if _count_decodings(meta["left_groups"], meta["right_groups"]) != 1:
        return False
    if int(meta["repair_left"]) < 1 or int(meta["repair_right"]) < 1:
        return False
    if not left_missing or not right_missing or left_missing & right_missing:
        return False
    if not (_has_adjacent(left_missing) or _has_adjacent(right_missing)):
        return False
    if np.array_equal(inp, out):
        return False

    bg = int(meta["bg"])
    lr0, lc0, lr1, lc1 = meta["regions"]["left"]
    rr0, rc0, rr1, rc1 = meta["regions"]["right"]
    if not np.all(inp[lr0, lc0 : lc1 + 1] != bg):
        return False
    if not np.all(inp[lr1, lc0 : lc1 + 1] != bg):
        return False
    right_patch = inp[rr0 : rr1 + 1, rc0 : rc1 + 1]
    if not np.all(right_patch[0] != bg):
        return False
    if not np.all(right_patch[-1] != bg):
        return False
    if not np.all(right_patch[:, 0] != bg):
        return False
    if not np.all(right_patch[:, -1] != bg):
        return False
    if out.shape[0] > 30 or out.shape[1] > 30 or inp.shape[0] > 30 or inp.shape[1] > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layers = _choose_layers()
    if layers is None:
        return None
    missing = _choose_missing_sets(len(layers))
    if missing is None:
        return None
    left_missing, right_missing = missing
    rendered = _render_instance(layers, left_missing, right_missing)
    if rendered is None:
        return None
    inp, out = rendered
    if not _validate_instance(inp, out, layers, left_missing, right_missing):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
