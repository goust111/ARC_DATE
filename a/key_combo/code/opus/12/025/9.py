from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


ROD_COLOR = 6
HINGE_COLOR = 4
ENDPOINT_COLOR = 9
FILLER_A = 5
FILLER_B = 7
ROD_PALETTE = {ROD_COLOR, HINGE_COLOR, ENDPOINT_COLOR}
HIDDEN_COLOR_POOL = [0, 1, 2, 3, 8]
MAX_GENERATION_ATTEMPTS = 600
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _colors_present(grid: np.ndarray) -> list[int]:
    return sorted(set(map(int, grid.ravel())))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_hidden_pair(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    candidates = []
    for a, b in itertools.combinations([c for c in _colors_present(g) if c not in ROD_PALETTE], 2):
        mask = np.isin(g, [a, b])
        rows = mask.any(axis=1)
        cols = mask.any(axis=0)
        if int(rows.sum()) < 5 or int(cols.sum()) < 5:
            continue
        sub = g[np.ix_(rows, cols)]
        sub_colors = set(map(int, sub.ravel()))
        if not sub_colors.issubset({a, b, ROD_COLOR, HINGE_COLOR, ENDPOINT_COLOR}):
            continue
        if not np.all(np.any(np.isin(sub, [a, b]), axis=1)):
            continue
        if not np.all(np.any(np.isin(sub, [a, b]), axis=0)):
            continue
        pair_counts = Counter(map(int, sub[np.isin(sub, [a, b])].ravel()))
        if len(pair_counts) != 2:
            continue
        majority_count = max(pair_counts.values())
        if list(pair_counts.values()).count(majority_count) != 1:
            continue
        candidates.append((rows, cols, (a, b)))
    assert len(candidates) == 1, f"expected one hidden pair, got {len(candidates)}"
    return candidates[0]


def _reconstruct_hidden_image(grid: np.ndarray, rows: np.ndarray, cols: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
    sub = np.asarray(grid, dtype=int)[np.ix_(rows, cols)]
    pair_mask = np.isin(sub, pair)
    pair_counts = Counter(map(int, sub[pair_mask].ravel()))
    base_color, base_count = pair_counts.most_common(1)[0]
    assert sum(count == base_count for count in pair_counts.values()) == 1
    other_color = pair[0] if pair[1] == base_color else pair[1]
    out = np.full(sub.shape, base_color, dtype=int)
    out[sub == other_color] = other_color
    return out


def _fold_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> set[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    coord_set = set(comp)
    hinge_cells = [p for p in comp if int(g[p]) == HINGE_COLOR]
    endpoint_cells = [p for p in comp if int(g[p]) == ENDPOINT_COLOR]
    magenta_cells = [p for p in comp if int(g[p]) == ROD_COLOR]
    assert len(hinge_cells) == 1 and len(endpoint_cells) == 1
    assert len(magenta_cells) == len(comp) - 2
    hinge = hinge_cells[0]
    endpoint = endpoint_cells[0]

    rows = {r for r, _ in comp}
    cols = {c for _, c in comp}
    assert len(rows) == 1 or len(cols) == 1

    final_magenta: set[tuple[int, int]] = set()
    if len(rows) == 1:
        row = next(iter(rows))
        sorted_cols = sorted(c for _r, c in comp)
        assert sorted_cols == list(range(sorted_cols[0], sorted_cols[-1] + 1))
        fixed_sign = 1 if endpoint[1] > hinge[1] else -1
        assert endpoint[1] != hinge[1]
        cursor = hinge[1] + fixed_sign
        while (row, cursor) in coord_set:
            if (row, cursor) == endpoint:
                break
            assert int(g[row, cursor]) == ROD_COLOR
            final_magenta.add((row, cursor))
            cursor += fixed_sign
        assert (row, cursor) == endpoint

        tail_sign = -fixed_sign
        tail_len = 0
        cursor = hinge[1] + tail_sign
        while (row, cursor) in coord_set:
            assert int(g[row, cursor]) == ROD_COLOR
            tail_len += 1
            cursor += tail_sign
        assert tail_len >= 1

        up_space = hinge[0]
        down_space = h - 1 - hinge[0]
        assert up_space != down_space
        fold_step = (-1, 0) if up_space > down_space else (1, 0)
    else:
        col = next(iter(cols))
        sorted_rows = sorted(r for r, _c in comp)
        assert sorted_rows == list(range(sorted_rows[0], sorted_rows[-1] + 1))
        fixed_sign = 1 if endpoint[0] > hinge[0] else -1
        assert endpoint[0] != hinge[0]
        cursor = hinge[0] + fixed_sign
        while (cursor, col) in coord_set:
            if (cursor, col) == endpoint:
                break
            assert int(g[cursor, col]) == ROD_COLOR
            final_magenta.add((cursor, col))
            cursor += fixed_sign
        assert (cursor, col) == endpoint

        tail_sign = -fixed_sign
        tail_len = 0
        cursor = hinge[0] + tail_sign
        while (cursor, col) in coord_set:
            assert int(g[cursor, col]) == ROD_COLOR
            tail_len += 1
            cursor += tail_sign
        assert tail_len >= 1

        left_space = hinge[1]
        right_space = w - 1 - hinge[1]
        assert left_space != right_space
        fold_step = (0, -1) if left_space > right_space else (0, 1)

    for step in range(1, tail_len + 1):
        nr = hinge[0] + fold_step[0] * step
        nc = hinge[1] + fold_step[1] * step
        assert _in_bounds(g.shape, nr, nc)
        final_magenta.add((nr, nc))

    return final_magenta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the hidden two-color image, fold each marked rod in the original grid, and overwrite the folded arm in compressed coordinates.

    Input structure:
    - A mixed grid contains one hidden two-color image whose two colors determine a set of active rows and active columns.
    - Inside that active row/column product there are 1-2 straight magenta rods, each marked by one maroon endpoint and one yellow hinge.
    - Grey and orange filler occupy the deleted separator rows and columns outside the hidden image support.

    Rule:
    - Find the unique color pair whose cells identify the active rows and active columns, then compress the grid to those rows and columns.
    - Reconstruct the hidden two-color image on that compressed support, treating rod and marker occlusions as overwritten cells of the image.
    - For each rod, keep the endpoint-to-hinge side fixed, fold the tail 90 degrees toward the side with more space to the nearest grid edge, map the folded magenta cells into the compressed coordinates, and overwrite those cells with magenta.

    Key insight:
    - The rod is folded in the original coordinate system, but its effect must be projected through the row/column compression induced by the hidden two-color mask.
    """
    g = np.asarray(grid, dtype=int)
    rows, cols, pair = _find_hidden_pair(g)
    out = _reconstruct_hidden_image(g, rows, cols, pair)
    row_map = {r: i for i, r in enumerate(np.flatnonzero(rows))}
    col_map = {c: i for i, c in enumerate(np.flatnonzero(cols))}

    rod_comps = _components(np.isin(g, [ROD_COLOR, HINGE_COLOR, ENDPOINT_COLOR]))
    assert 1 <= len(rod_comps) <= 2
    folded_targets: set[tuple[int, int]] = set()
    for comp in rod_comps:
        for r, c in _fold_component(g, comp):
            assert r in row_map and c in col_map
            folded_targets.add((row_map[r], col_map[c]))

    for r, c in folded_targets:
        out[r, c] = ROD_COLOR
    return out


def _sample_block_sizes() -> list[int]:
    count = random.choice([2, 2, 3])
    for _ in range(200):
        sizes = [random.randint(2, 7) for _ in range(count)]
        if sum(sizes) < 5:
            continue
        if max(sizes) < 4:
            continue
        if sum(sizes) > 14:
            continue
        return sizes
    raise RuntimeError("failed to sample block sizes")


def _build_axis(block_sizes: list[int]):
    seps = [random.randint(1, 2) for _ in range(len(block_sizes) - 1)]
    before = random.randint(1, 3)
    after = random.randint(1, 3)
    active: list[int] = []
    blocks = []
    comp_cursor = 0
    orig_cursor = before
    for i, size in enumerate(block_sizes):
        comp0 = comp_cursor
        comp1 = comp_cursor + size - 1
        orig0 = orig_cursor
        orig1 = orig_cursor + size - 1
        active.extend(range(orig0, orig1 + 1))
        blocks.append(
            {
                "comp0": comp0,
                "comp1": comp1,
                "orig0": orig0,
                "orig1": orig1,
                "size": size,
            }
        )
        comp_cursor += size
        orig_cursor += size
        if i < len(seps):
            orig_cursor += seps[i]
    total = orig_cursor + after
    return np.array(active, dtype=int), blocks, total


def _choose_spanning_interval(blocks: list[dict], which: str) -> tuple[int, int]:
    first = blocks[0]
    last = blocks[-1]
    if which == "row":
        lo0 = first["comp0"]
        hi0 = min(first["comp1"], first["comp0"] + max(0, first["size"] - 2))
        lo1 = max(last["comp0"], last["comp1"] - max(0, last["size"] - 2))
        hi1 = last["comp1"]
    else:
        lo0 = first["comp0"]
        hi0 = min(first["comp1"], first["comp0"] + max(0, first["size"] - 2))
        lo1 = max(last["comp0"], last["comp1"] - max(0, last["size"] - 2))
        hi1 = last["comp1"]
    a = random.randint(lo0, hi0)
    b = random.randint(lo1, hi1)
    if b - a < 2:
        a = first["comp0"]
        b = last["comp1"]
    return a, b


def _make_loop_mask(h: int, w: int, row_blocks: list[dict], col_blocks: list[dict]) -> np.ndarray | None:
    if h < 5 or w < 5:
        return None
    r0, r1 = _choose_spanning_interval(row_blocks, "row")
    c0, c1 = _choose_spanning_interval(col_blocks, "col")
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _random_from_block(block: dict) -> int:
    return random.randint(block["comp0"], block["comp1"])


def _make_branch_mask(h: int, w: int, row_blocks: list[dict], col_blocks: list[dict]) -> np.ndarray | None:
    if h < 5 or w < 5:
        return None
    top_row = _random_from_block(row_blocks[0])
    bottom_row = _random_from_block(row_blocks[-1])
    if top_row == bottom_row:
        return None
    left_col = _random_from_block(col_blocks[0])
    right_col = _random_from_block(col_blocks[-1])
    if left_col == right_col:
        return None

    trunk_col = random.randint(min(left_col, right_col), max(left_col, right_col))
    candidate_rows = []
    if len(row_blocks) >= 3:
        candidate_rows.extend(range(row_blocks[1]["comp0"], row_blocks[-2]["comp1"] + 1))
    candidate_rows.extend([top_row, bottom_row])
    main_row = random.choice(candidate_rows)

    mask = np.zeros((h, w), dtype=bool)
    mask[min(top_row, bottom_row) : max(top_row, bottom_row) + 1, trunk_col] = True
    mask[main_row, min(left_col, right_col) : max(left_col, right_col) + 1] = True

    extra_rows = []
    for block in row_blocks:
        for r in range(block["comp0"], block["comp1"] + 1):
            if abs(r - main_row) > 1:
                extra_rows.append(r)
    random.shuffle(extra_rows)
    for row in extra_rows[: random.randint(0, min(3, len(extra_rows)))]:
        side = random.choice([-1, 1])
        if side < 0:
            reach = random.randint(1, max(1, trunk_col - min(left_col, right_col)))
            mask[row, trunk_col - reach : trunk_col + 1] = True
        else:
            reach = random.randint(1, max(1, max(left_col, right_col) - trunk_col))
            mask[row, trunk_col : trunk_col + reach + 1] = True
    return mask


def _generate_hidden_image(h: int, w: int, row_blocks: list[dict], col_blocks: list[dict], bg: int, fg: int) -> np.ndarray | None:
    mask = None
    if random.random() < 0.4:
        mask = _make_loop_mask(h, w, row_blocks, col_blocks)
    if mask is None:
        mask = _make_branch_mask(h, w, row_blocks, col_blocks)
    if mask is None:
        return None
    density = float(mask.mean())
    if not 0.10 <= density <= 0.35:
        return None
    comp_count = len(_components(mask))
    if comp_count != 1:
        return None
    out = np.full((h, w), bg, dtype=int)
    out[mask] = fg
    return out


def _expanded(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for nr, nc in _neighbors4(r, c):
            out.add((nr, nc))
    return out


def _enumerate_rods(
    base: np.ndarray,
    bg_color: int,
    row_blocks: list[dict],
    col_blocks: list[dict],
    row_to_orig: np.ndarray,
    col_to_orig: np.ndarray,
    grid_shape: tuple[int, int],
):
    candidates = []
    h, w = base.shape
    for rb in row_blocks:
        for cb in col_blocks:
            for r in range(rb["comp0"], rb["comp1"] + 1):
                for c in range(cb["comp0"], cb["comp1"] + 1):
                    orig_r = int(row_to_orig[r])
                    orig_c = int(col_to_orig[c])
                    up_space = orig_r
                    down_space = grid_shape[0] - 1 - orig_r
                    left_space = orig_c
                    right_space = grid_shape[1] - 1 - orig_c

                    for fixed_sign in (-1, 1):
                        if up_space == down_space:
                            continue
                        max_fold = (r - rb["comp0"]) if up_space > down_space else (rb["comp1"] - r)
                        max_fixed = (cb["comp1"] - c + 1) if fixed_sign > 0 else (c - cb["comp0"] + 1)
                        max_tail = (c - cb["comp0"]) if fixed_sign > 0 else (cb["comp1"] - c)
                        for fixed_len in range(2, max_fixed + 1):
                            for tail_len in range(1, min(max_tail, max_fold) + 1):
                                total_len = fixed_len + tail_len
                                if not 4 <= total_len <= 7:
                                    continue
                                input_comp = {(r, c)}
                                final_comp = set()
                                ok = True
                                endpoint = (r, c + fixed_sign * (fixed_len - 1))
                                input_comp.add(endpoint)
                                for step in range(1, fixed_len - 1):
                                    cell = (r, c + fixed_sign * step)
                                    input_comp.add(cell)
                                    final_comp.add(cell)
                                for step in range(1, tail_len + 1):
                                    tail_cell = (r, c - fixed_sign * step)
                                    input_comp.add(tail_cell)
                                fold_dir = -1 if up_space > down_space else 1
                                for step in range(1, tail_len + 1):
                                    final_comp.add((r + fold_dir * step, c))
                                for rr, cc in input_comp:
                                    if base[rr, cc] != bg_color:
                                        ok = False
                                        break
                                if not ok:
                                    continue
                                input_orig = {(int(row_to_orig[rr]), int(col_to_orig[cc])) for rr, cc in input_comp}
                                final_orig = {(int(row_to_orig[rr]), int(col_to_orig[cc])) for rr, cc in final_comp}
                                candidates.append(
                                    {
                                        "orientation": "h",
                                        "input_comp": input_comp,
                                        "final_comp": final_comp,
                                        "input_orig": input_orig,
                                        "final_orig": final_orig,
                                        "envelope_orig": input_orig | final_orig,
                                        "endpoint_comp": endpoint,
                                        "hinge_comp": (r, c),
                                        "tail_len": tail_len,
                                    }
                                )

                    for fixed_sign in (-1, 1):
                        if left_space == right_space:
                            continue
                        max_fold = (c - cb["comp0"]) if left_space > right_space else (cb["comp1"] - c)
                        max_fixed = (rb["comp1"] - r + 1) if fixed_sign > 0 else (r - rb["comp0"] + 1)
                        max_tail = (r - rb["comp0"]) if fixed_sign > 0 else (rb["comp1"] - r)
                        for fixed_len in range(2, max_fixed + 1):
                            for tail_len in range(1, min(max_tail, max_fold) + 1):
                                total_len = fixed_len + tail_len
                                if not 4 <= total_len <= 7:
                                    continue
                                input_comp = {(r, c)}
                                final_comp = set()
                                ok = True
                                endpoint = (r + fixed_sign * (fixed_len - 1), c)
                                input_comp.add(endpoint)
                                for step in range(1, fixed_len - 1):
                                    cell = (r + fixed_sign * step, c)
                                    input_comp.add(cell)
                                    final_comp.add(cell)
                                for step in range(1, tail_len + 1):
                                    tail_cell = (r - fixed_sign * step, c)
                                    input_comp.add(tail_cell)
                                fold_dir = -1 if left_space > right_space else 1
                                for step in range(1, tail_len + 1):
                                    final_comp.add((r, c + fold_dir * step))
                                for rr, cc in input_comp:
                                    if base[rr, cc] != bg_color:
                                        ok = False
                                        break
                                if not ok:
                                    continue
                                input_orig = {(int(row_to_orig[rr]), int(col_to_orig[cc])) for rr, cc in input_comp}
                                final_orig = {(int(row_to_orig[rr]), int(col_to_orig[cc])) for rr, cc in final_comp}
                                candidates.append(
                                    {
                                        "orientation": "v",
                                        "input_comp": input_comp,
                                        "final_comp": final_comp,
                                        "input_orig": input_orig,
                                        "final_orig": final_orig,
                                        "envelope_orig": input_orig | final_orig,
                                        "endpoint_comp": endpoint,
                                        "hinge_comp": (r, c),
                                        "tail_len": tail_len,
                                    }
                                )
    return candidates


def _compatible(candidate: dict, chosen: list[dict]) -> bool:
    expanded_input = _expanded(set(candidate["input_orig"]))
    envelope = set(candidate["envelope_orig"])
    for rec in chosen:
        if expanded_input & _expanded(set(rec["input_orig"])):
            return False
        if envelope & set(rec["envelope_orig"]):
            return False
    return True


def _choose_rods(candidates: list[dict]) -> list[dict] | None:
    if not candidates:
        return None
    rod_count = random.choice([1, 1, 2])
    random.shuffle(candidates)
    by_orientation = {"h": [c for c in candidates if c["orientation"] == "h"], "v": [c for c in candidates if c["orientation"] == "v"]}

    for _ in range(120):
        chosen: list[dict] = []
        target_orientations = []
        if rod_count == 2 and by_orientation["h"] and by_orientation["v"] and random.random() < 0.6:
            target_orientations = random.sample(["h", "v"], 2)
        else:
            target_orientations = [random.choice(["h", "v"]) for _ in range(rod_count)]
        pools = []
        for orient in target_orientations:
            pool = by_orientation[orient] or candidates
            pools.append(random.sample(pool, min(len(pool), 40)))
        ok = True
        for pool in pools:
            found = None
            random.shuffle(pool)
            for candidate in pool:
                if _compatible(candidate, chosen):
                    found = candidate
                    break
            if found is None:
                ok = False
                break
            chosen.append(found)
        if ok:
            return chosen
    return None


def _render_input(
    base: np.ndarray,
    row_to_orig: np.ndarray,
    col_to_orig: np.ndarray,
    grid_shape: tuple[int, int],
    rods: list[dict],
) -> np.ndarray:
    h, w = grid_shape
    grid = np.full((h, w), FILLER_A, dtype=int)

    active_rows = set(map(int, row_to_orig))
    active_cols = set(map(int, col_to_orig))
    first_row, last_row = int(row_to_orig[0]), int(row_to_orig[-1])
    first_col, last_col = int(col_to_orig[0]), int(col_to_orig[-1])
    for r in range(first_row, last_row + 1):
        if r not in active_rows:
            grid[r, :] = FILLER_B
    for c in range(first_col, last_col + 1):
        if c not in active_cols:
            grid[:, c] = FILLER_B

    for rr, orig_r in enumerate(row_to_orig):
        for cc, orig_c in enumerate(col_to_orig):
            grid[int(orig_r), int(orig_c)] = int(base[rr, cc])

    for rec in rods:
        hr, hc = rec["hinge_comp"]
        er, ec = rec["endpoint_comp"]
        grid[int(row_to_orig[hr]), int(col_to_orig[hc])] = HINGE_COLOR
        grid[int(row_to_orig[er]), int(col_to_orig[ec])] = ENDPOINT_COLOR
        for rr, cc in rec["input_comp"]:
            if (rr, cc) in {rec["hinge_comp"], rec["endpoint_comp"]}:
                continue
            grid[int(row_to_orig[rr]), int(col_to_orig[cc])] = ROD_COLOR
    return grid


def _expected_output(base: np.ndarray, rods: list[dict]) -> np.ndarray:
    out = base.copy()
    for rec in rods:
        for r, c in rec["final_comp"]:
            out[r, c] = ROD_COLOR
    return out


def _semantic_ok(inp: np.ndarray, expected: np.ndarray, rods: list[dict]) -> bool:
    try:
        rows, cols, pair = _find_hidden_pair(inp)
    except Exception:
        return False
    if tuple(expected.shape) != (int(rows.sum()), int(cols.sum())):
        return False
    if len(_components(np.isin(inp, [ROD_COLOR, HINGE_COLOR, ENDPOINT_COLOR]))) != len(rods):
        return False
    if not any(rec["final_comp"] - rec["input_comp"] for rec in rods):
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    sub = inp[np.ix_(rows, cols)]
    visible = sub[np.isin(sub, pair)]
    counts = Counter(map(int, visible.ravel()))
    if len(counts) != 2:
        return False
    majority_count = max(counts.values())
    if list(counts.values()).count(majority_count) != 1:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    row_sizes = _sample_block_sizes()
    col_sizes = _sample_block_sizes()
    row_to_orig, row_blocks, total_h = _build_axis(row_sizes)
    col_to_orig, col_blocks, total_w = _build_axis(col_sizes)
    if max(total_h, total_w) > 30:
        return None

    hidden_h = len(row_to_orig)
    hidden_w = len(col_to_orig)
    if hidden_h < 5 or hidden_w < 5:
        return None

    hidden_bg, hidden_fg = random.sample(HIDDEN_COLOR_POOL, 2)
    base = _generate_hidden_image(hidden_h, hidden_w, row_blocks, col_blocks, hidden_bg, hidden_fg)
    if base is None:
        return None

    candidates = _enumerate_rods(
        base=base,
        bg_color=hidden_bg,
        row_blocks=row_blocks,
        col_blocks=col_blocks,
        row_to_orig=row_to_orig,
        col_to_orig=col_to_orig,
        grid_shape=(total_h, total_w),
    )
    rods = _choose_rods(candidates)
    if rods is None:
        return None

    inp = _render_input(base, row_to_orig, col_to_orig, (total_h, total_w), rods)
    expected = _expected_output(base, rods)
    return inp if _semantic_ok(inp, expected, rods) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
