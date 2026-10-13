import random
from collections import Counter

import numpy as np


GENERATION_RETRIES = 500
BACKGROUND = 0


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _mirror_variants(mask: np.ndarray) -> list[np.ndarray]:
    arr = np.asarray(mask, dtype=bool)
    return [arr, np.fliplr(arr), np.flipud(arr), np.flipud(np.fliplr(arr))]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = _mirror_variants(mask)
    payloads = [(tuple(v.shape), v.astype(np.uint8).tobytes()) for v in variants]
    return min(payloads)


def _ranges_between(size: int, dividers: list[int]) -> list[tuple[int, int]]:
    ranges = []
    start = 0
    for cut in list(dividers) + [size]:
        if start < cut:
            ranges.append((start, cut - 1))
        start = cut + 1
    return ranges


def _detect_divider(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == color)]
        cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == color)]
        if rows and cols:
            candidates.append((int(color), rows, cols))
    assert len(candidates) == 1, "expected exactly one divider color"
    return candidates[0]


def _cell_slices(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    bg = get_bg(grid)
    divider, divider_rows, divider_cols = _detect_divider(grid, bg)
    row_ranges = _ranges_between(grid.shape[0], divider_rows)
    col_ranges = _ranges_between(grid.shape[1], divider_cols)
    assert len(row_ranges) >= 2 and len(col_ranges) >= 2
    heights = {r1 - r0 + 1 for r0, r1 in row_ranges}
    widths = {c1 - c0 + 1 for c0, c1 in col_ranges}
    assert len(heights) == 1 and len(widths) == 1, "all cells must be equal-sized"
    return divider, row_ranges, col_ranges


def _cell_contents(
    grid: np.ndarray,
    divider: int,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    bg = get_bg(grid)
    cell_colors = np.full((len(row_ranges), len(col_ranges)), -1, dtype=int)
    cell_masks: dict[tuple[int, int], np.ndarray] = {}

    for rr, (r0, r1) in enumerate(row_ranges):
        for cc, (c0, c1) in enumerate(col_ranges):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = sorted(set(map(int, np.unique(cell))) - {bg, divider})
            if not colors:
                cell_masks[(rr, cc)] = np.zeros(cell.shape, dtype=bool)
                continue
            assert len(colors) == 1, "each occupied cell must contain exactly one sprite color"
            color = int(colors[0])
            mask = cell == color
            assert np.all((cell == bg) | mask), "cell content must be a monochrome sprite"
            cell_colors[rr, cc] = color
            cell_masks[(rr, cc)] = mask
    return cell_colors, cell_masks


def _transform_between(mask: np.ndarray, src: tuple[int, int], dst: tuple[int, int]) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    if src[0] != dst[0]:
        out = np.flipud(out)
    if src[1] != dst[1]:
        out = np.fliplr(out)
    return out


def _infer_missing_mask(
    present: list[tuple[int, int]],
    missing: tuple[int, int],
    cell_masks: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    predictions = [_transform_between(cell_masks[pos], pos, missing) for pos in present]
    first = predictions[0]
    assert all(np.array_equal(first, pred) for pred in predictions[1:]), "mirror inference must agree"
    return first


def _find_blocks(cell_colors: np.ndarray) -> tuple[list[dict], set[tuple[int, int]]]:
    blocks = []
    used_present: set[tuple[int, int]] = set()
    rows, cols = cell_colors.shape
    for r in range(rows - 1):
        for c in range(cols - 1):
            cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            vals = [int(cell_colors[pos]) for pos in cells]
            present = [cells[i] for i, value in enumerate(vals) if value != -1]
            if len(present) != 3:
                continue
            colors = [int(cell_colors[pos]) for pos in present]
            if len(set(colors)) != 1:
                continue
            if any(pos in used_present for pos in present):
                continue
            missing = next(pos for pos in cells if int(cell_colors[pos]) == -1)
            blocks.append({"color": colors[0], "present": present, "missing": missing})
            used_present.update(present)
    assert blocks, "expected at least one three-of-four mirror block"
    return blocks, used_present


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each missing mirror-quartet sprite, then move the uniquely matching loose sprite into that cell with its own color.

    Input structure:
    - The background is the most common color.
    - One non-background color forms complete horizontal and vertical divider lines,
      partitioning the canvas into equal-sized lattice cells.
    - Some 2-by-2 cell blocks contain exactly three same-colored sprites and one empty cell;
      the three visible sprites are horizontal/vertical mirror variants around the block center.
    - Other distant occupied cells each contain one loose monochrome sprite of a unique color.

    Rule:
    - Split the grid by the divider lines into lattice cells.
    - For every three-of-four 2-by-2 block, infer the missing cell-local sprite mask by reflecting
      the visible masks across the block center; all three reflections must agree.
    - Among the loose sprites, find the unique one whose full cell-local shape is mirror-equivalent
      to that inferred mask, reflect it as needed, paint it into the empty cell using the loose sprite's
      own color, and clear its original cell to background.
    - Leave divider lines, the existing block sprites, and every non-matching decoy sprite unchanged.

    Key insight:
    - The three same-colored sprites provide only the missing shape and orientation; the actual fill color
      comes from a separate loose sprite whose mirror-equivalent mask acts as the transferable key.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    divider, row_ranges, col_ranges = _cell_slices(g)
    cell_colors, cell_masks = _cell_contents(g, divider, row_ranges, col_ranges)
    blocks, block_cells = _find_blocks(cell_colors)

    loose_entries = []
    for r in range(cell_colors.shape[0]):
        for c in range(cell_colors.shape[1]):
            color = int(cell_colors[r, c])
            if color == -1 or (r, c) in block_cells:
                continue
            loose_entries.append({"pos": (r, c), "color": color, "mask": cell_masks[(r, c)]})
    assert loose_entries, "expected loose sprites outside the mirror blocks"

    loose_by_key: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for entry in loose_entries:
        loose_by_key.setdefault(_mask_key(entry["mask"]), []).append(entry)

    out = g.copy()
    used_loose_positions: set[tuple[int, int]] = set()
    for block in blocks:
        missing = block["missing"]
        target_mask = _infer_missing_mask(block["present"], missing, cell_masks)
        key = _mask_key(target_mask)
        candidates = [entry for entry in loose_by_key.get(key, []) if entry["pos"] not in used_loose_positions]
        assert len(candidates) == 1, "each missing cell must have exactly one matching loose sprite"
        source = candidates[0]
        used_loose_positions.add(source["pos"])

        source_variants = [variant for variant in _mirror_variants(source["mask"]) if np.array_equal(variant, target_mask)]
        assert source_variants, "matching loose sprite must be orientable to the target mask"
        oriented = source_variants[0]

        sr0, sr1 = row_ranges[source["pos"][0]]
        sc0, sc1 = col_ranges[source["pos"][1]]
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = bg

        tr0, tr1 = row_ranges[missing[0]]
        tc0, tc1 = col_ranges[missing[1]]
        out[tr0 : tr1 + 1, tc0 : tc1 + 1] = bg
        target_window = out[tr0 : tr1 + 1, tc0 : tc1 + 1]
        target_window[oriented] = int(source["color"])

    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    cells = {tuple(map(int, pt)) for pt in coords}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            nxt = (nr, nc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    for _ in range(200):
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = {start}
        while len(cells) < area:
            candidates = []
            for r, c in frontier:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        candidates.append((nr, nc))
            if not candidates:
                break
            nxt = random.choice(candidates)
            cells.add(nxt)
            frontier.add(nxt)
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _is_connected(mask):
            continue
        if len({variant.astype(np.uint8).tobytes() for variant in _mirror_variants(mask)}) != 4:
            continue
        return mask
    raise RuntimeError("failed to sample an asymmetric connected mask")


def _choose_palette(group_count: int, decoy_count: int) -> tuple[int, list[int], list[int]]:
    divider = random.choice(list(range(1, 10)))
    remaining = [color for color in range(1, 10) if color != divider]
    random.shuffle(remaining)
    block_colors = remaining[:group_count]
    loose_colors = remaining[group_count : group_count + group_count + decoy_count]
    return divider, block_colors, loose_colors


def _sample_block_positions(rows: int, cols: int, count: int) -> list[tuple[int, int]]:
    candidates = [(r, c) for r in range(rows - 1) for c in range(cols - 1)]
    random.shuffle(candidates)

    def search(start: int, chosen: list[tuple[int, int]], used_cells: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(chosen) == count:
            return chosen[:]
        remaining_needed = count - len(chosen)
        if len(candidates) - start < remaining_needed:
            return None
        for idx in range(start, len(candidates)):
            r, c = candidates[idx]
            block_cells = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
            if used_cells & block_cells:
                continue
            chosen.append((r, c))
            result = search(idx + 1, chosen, used_cells | block_cells)
            if result is not None:
                return result
            chosen.pop()
        return None

    result = search(0, [], set())
    if result is None:
        raise RuntimeError("failed to place non-overlapping 2x2 blocks")
    return result


def _pick_loose_cells(
    rows: int,
    cols: int,
    blocked: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]]:
    free = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in blocked]
    assert len(free) >= count
    ranked = []
    for cell in free:
        dist = min(abs(cell[0] - br) + abs(cell[1] - bc) for br, bc in blocked)
        ranked.append((dist, random.random(), cell))
    ranked.sort(reverse=True)
    top = [cell for _dist, _noise, cell in ranked[: max(count, min(len(ranked), count + 6))]]
    return random.sample(top, count)


def _paint_cell(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    cell: tuple[int, int],
    mask: np.ndarray,
    color: int,
) -> None:
    r0, r1 = row_ranges[cell[0]]
    c0, c1 = col_ranges[cell[1]]
    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    window[mask] = int(color)


def _build_canvas(
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    bg: int,
    divider: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = rows * cell_h + (rows - 1)
    width = cols * cell_w + (cols - 1)
    canvas = np.full((height, width), bg, dtype=int)
    row_ranges = []
    col_ranges = []
    cursor = 0
    for r in range(rows):
        row_ranges.append((cursor, cursor + cell_h - 1))
        cursor += cell_h
        if r != rows - 1:
            canvas[cursor, :] = divider
            cursor += 1
    cursor = 0
    for c in range(cols):
        col_ranges.append((cursor, cursor + cell_w - 1))
        cursor += cell_w
        if c != cols - 1:
            canvas[:, cursor] = divider
            cursor += 1
    return canvas, row_ranges, col_ranges


def _random_shape_class(
    cell_h: int,
    cell_w: int,
    forbidden: set[tuple[tuple[int, int], bytes]],
    area_choices: list[int] | None = None,
) -> np.ndarray:
    max_area = min(cell_h * cell_w - 1, 7)
    areas = area_choices or list(range(4, max_area + 1))
    for _ in range(300):
        area = random.choice(areas)
        mask = _random_connected_mask(cell_h, cell_w, area)
        key = _mask_key(mask)
        if key not in forbidden:
            return mask
    raise RuntimeError("failed to sample a unique shape class")


def _generate_instance() -> tuple[np.ndarray, np.ndarray, dict]:
    group_count = random.randint(1, 3)
    max_decoys = min(3, 8 - 2 * group_count)
    decoy_count = random.randint(1, max_decoys)

    for _ in range(200):
        rows = random.randint(4, 5)
        cols = random.randint(4, 5)
        if rows * cols >= 4 * group_count + decoy_count:
            break
    else:
        raise RuntimeError("failed to size lattice")

    cell_h = random.randint(4, 5)
    cell_w = random.randint(4, 5)
    divider, block_colors, loose_colors = _choose_palette(group_count, decoy_count)
    input_grid, row_ranges, col_ranges = _build_canvas(rows, cols, cell_h, cell_w, BACKGROUND, divider)
    output_grid = input_grid.copy()

    block_origins = _sample_block_positions(rows, cols, group_count)
    blocked_cells = set()
    for r, c in block_origins:
        blocked_cells |= {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}

    loose_cells = _pick_loose_cells(rows, cols, blocked_cells, group_count + decoy_count)
    random.shuffle(loose_cells)
    source_cells = loose_cells[:group_count]
    decoy_cells = loose_cells[group_count:]

    shape_keys: set[tuple[tuple[int, int], bytes]] = set()
    target_keys = []
    target_areas = []
    details = {"groups": [], "decoys": []}
    forced_rotated_group = random.randrange(group_count)

    for idx, ((br, bc), block_color, loose_color, source_cell) in enumerate(
        zip(block_origins, block_colors, loose_colors[:group_count], source_cells)
    ):
        base_mask = _random_shape_class(cell_h, cell_w, shape_keys)
        shape_keys.add(_mask_key(base_mask))
        missing_index = random.randrange(4)
        block_cells = [(br, bc), (br, bc + 1), (br + 1, bc), (br + 1, bc + 1)]
        missing_cell = block_cells[missing_index]

        if idx == forced_rotated_group:
            source_transform = random.choice([1, 2, 3])
        else:
            source_transform = random.randrange(4)

        target_mask = _transform_between(base_mask, (br, bc), missing_cell)
        target_keys.append(_mask_key(target_mask))
        target_areas.append(int(target_mask.sum()))
        source_mask = _mirror_variants(target_mask)[source_transform]

        for cell in block_cells:
            if cell == missing_cell:
                continue
            mask = _transform_between(base_mask, (br, bc), cell)
            _paint_cell(input_grid, row_ranges, col_ranges, cell, mask, block_color)
            _paint_cell(output_grid, row_ranges, col_ranges, cell, mask, block_color)

        _paint_cell(input_grid, row_ranges, col_ranges, source_cell, source_mask, loose_color)
        _paint_cell(output_grid, row_ranges, col_ranges, missing_cell, target_mask, loose_color)

        details["groups"].append(
            {
                "block_origin": (br, bc),
                "missing_cell": missing_cell,
                "source_cell": source_cell,
                "target_key": _mask_key(target_mask),
                "target_area": int(target_mask.sum()),
                "source_transform": source_transform,
            }
        )

    forbidden = set(target_keys)
    for decoy_color, decoy_cell in zip(loose_colors[group_count:], decoy_cells):
        decoy_mask = _random_shape_class(cell_h, cell_w, shape_keys | forbidden, area_choices=target_areas)
        decoy_key = _mask_key(decoy_mask)
        shape_keys.add(decoy_key)
        _paint_cell(input_grid, row_ranges, col_ranges, decoy_cell, decoy_mask, decoy_color)
        _paint_cell(output_grid, row_ranges, col_ranges, decoy_cell, decoy_mask, decoy_color)
        details["decoys"].append({"cell": decoy_cell, "key": decoy_key, "area": int(decoy_mask.sum())})

    return input_grid, output_grid, details


def _validate_instance(inp: np.ndarray, expected: np.ndarray, details: dict) -> bool:
    if inp.shape != expected.shape or inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if get_bg(inp) != BACKGROUND or get_bg(expected) != BACKGROUND:
        return False
    if np.array_equal(inp, expected):
        return False
    if not details["decoys"]:
        return False
    if not any(group["source_transform"] != 0 for group in details["groups"]):
        return False
    target_keys = [group["target_key"] for group in details["groups"]]
    if len(target_keys) != len(set(target_keys)):
        return False
    decoy_keys = [decoy["key"] for decoy in details["decoys"]]
    if set(target_keys) & set(decoy_keys):
        return False
    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        try:
            inp, expected, details = _generate_instance()
        except Exception:
            continue
        if _validate_instance(inp, expected, details):
            return inp
    raise RuntimeError("failed to generate a valid mirrored-key transfer instance")
