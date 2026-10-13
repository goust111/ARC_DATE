from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(int(v) for v in grid.flat).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_table_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    comps = _components_of_mask(grid != bg)
    assert comps, "expected a non-background color table"
    table_coords = max(comps, key=lambda coords: int(coords.shape[0]))
    bbox = _bbox_coords(table_coords)
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert patch.shape[0] == patch.shape[1], "color table must be square"
    assert int(table_coords.shape[0]) == patch.shape[0] * patch.shape[1], "table must be solid"
    assert bool(np.all(patch != bg)), "table cells must all be non-background"
    return bbox


def _arrow_tip_from_bbox(bbox: tuple[int, int, int, int], direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    dr, dc = direction
    tip_r = r0 - 1 if dr < 0 else r1 + 1
    tip_c = c0 - 1 if dc < 0 else c1 + 1
    return int(tip_r), int(tip_c)


def _parse_arrows(
    grid: np.ndarray, bg: int, table_bbox: tuple[int, int, int, int]
) -> list[dict[str, int | tuple[int, int]]]:
    r0, c0, r1, c1 = table_bbox
    outside_mask = grid != bg
    outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    arrows = []
    for coords in _components_of_mask(outside_mask):
        bbox = _bbox_coords(coords)
        pr0, pc0, pr1, pc1 = bbox
        patch = grid[pr0 : pr1 + 1, pc0 : pc1 + 1]
        assert patch.shape == (2, 2), "arrows must occupy 2x2 bboxes"
        assert int(coords.shape[0]) == 3, "arrows must have exactly three colored cells"
        colors = sorted(set(int(v) for v in patch.flat if int(v) != int(bg)))
        assert len(colors) == 1, "each arrow must be monochromatic"
        miss = np.argwhere(patch == bg)
        assert miss.shape == (1, 2), "each arrow must have one missing corner"
        mr, mc = map(int, miss[0])
        direction = (-1 if mr == 0 else 1, -1 if mc == 0 else 1)
        arrows.append(
            {
                "bbox": bbox,
                "color": int(colors[0]),
                "direction": direction,
                "tip": _arrow_tip_from_bbox(bbox, direction),
            }
        )
    arrows.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    assert arrows, "expected at least one arrow"
    return arrows


def _apply_arrow_clues(
    grid: np.ndarray, table_bbox: tuple[int, int, int, int], arrows: list[dict[str, int | tuple[int, int]]]
) -> np.ndarray:
    r0, c0, r1, c1 = table_bbox
    table = _crop_bbox(grid, table_bbox)
    h, w = grid.shape
    for arrow in arrows:
        dr, dc = arrow["direction"]
        rr, cc = arrow["tip"]
        hit = False
        while 0 <= rr + dr < h and 0 <= cc + dc < w:
            rr += dr
            cc += dc
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                tr = rr - r0
                tc = cc - c0
                if int(table[tr, tc]) == 0:
                    table[tr, tc] = int(arrow["color"])
                    hit = True
                    break
        assert hit, "each arrow must hit a blank cell in the table"
    return table


def _latin_solution_count(
    partial: np.ndarray, palette: list[int], limit: int = 2
) -> tuple[int, np.ndarray | None]:
    n = partial.shape[0]
    assert partial.shape == (n, n)
    assert len(palette) == n
    color_to_bit = {int(color): idx for idx, color in enumerate(palette)}
    full_mask = (1 << n) - 1
    grid = partial.copy()
    row_used = [0] * n
    col_used = [0] * n
    blanks: list[tuple[int, int]] = []

    for r in range(n):
        for c in range(n):
            value = int(grid[r, c])
            if value == 0:
                blanks.append((r, c))
                continue
            assert value in color_to_bit, "non-blank table colors must belong to the palette"
            bit = 1 << color_to_bit[value]
            if row_used[r] & bit or col_used[c] & bit:
                return 0, None
            row_used[r] |= bit
            col_used[c] |= bit

    count = 0
    first_solution: np.ndarray | None = None

    def dfs() -> None:
        nonlocal count, first_solution
        if count >= limit:
            return

        best_cell: tuple[int, int] | None = None
        best_mask = 0
        best_bits = n + 1
        for r, c in blanks:
            if int(grid[r, c]) != 0:
                continue
            mask = full_mask & ~row_used[r] & ~col_used[c]
            bits = mask.bit_count()
            if bits == 0:
                return
            if bits < best_bits:
                best_bits = bits
                best_mask = mask
                best_cell = (r, c)
                if bits == 1:
                    break

        if best_cell is None:
            count += 1
            if first_solution is None:
                first_solution = grid.copy()
            return

        r, c = best_cell
        for idx, color in enumerate(palette):
            if (best_mask >> idx) & 1 == 0:
                continue
            bit = 1 << idx
            grid[r, c] = int(color)
            row_used[r] |= bit
            col_used[c] |= bit
            dfs()
            row_used[r] ^= bit
            col_used[c] ^= bit
            grid[r, c] = 0
            if count >= limit:
                return

    dfs()
    return count, first_solution


def main(grid: np.ndarray) -> np.ndarray:
    """
    Nail arrow colors into the first blank cells they hit, then uniquely complete the central Latin square.

    Input structure:
    - A square color table sits in the middle as one solid non-background block; some table cells are black blanks.
    - Every non-black table entry already respects row and column non-repetition.
    - Outside the table are small monochromatic 2x2 arrowhead sprites with one missing corner.

    Rule:
    - For each arrow, infer its diagonal direction from the missing corner and cast a ray from the outer tip toward the table.
    - Skip filled cells on that ray and fill the first black table cell hit with the arrow's color.
    - After all arrows have nailed their cells, complete the remaining black cells so each row and column contains the full table palette exactly once.

    Key insight:
    - The table alone is underconstrained, and the arrows alone fill only a few cells; uniqueness comes from combining first-blank ray hits with Latin row/column constraints.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    table_bbox = _find_table_bbox(g, bg)
    arrows = _parse_arrows(g, bg, table_bbox)
    nailed = _apply_arrow_clues(g, table_bbox, arrows)

    palette = sorted(
        set(int(v) for v in nailed.flat if int(v) != 0)
        | {int(arrow["color"]) for arrow in arrows}
    )
    assert len(palette) == nailed.shape[0], "palette size must match the table order"
    count, solution = _latin_solution_count(nailed, palette, limit=2)
    assert count == 1 and solution is not None, "table must have a unique completion"
    return solution


def _build_latin_square(n: int, palette: list[int]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    row_perm = list(range(n))
    col_perm = list(range(n))
    random.shuffle(row_perm)
    random.shuffle(col_perm)
    color_cycle = palette[:]
    random.shuffle(color_cycle)
    square = np.array(
        [[color_cycle[(row_perm[r] + col_perm[c]) % n] for c in range(n)] for r in range(n)],
        dtype=int,
    )

    row_pairs: list[tuple[int, int]] = []
    seen_rows: set[tuple[int, int]] = set()
    for i in range(n):
        partner = next(j for j in range(n) if (row_perm[j] - row_perm[i]) % n == n // 2)
        pair = tuple(sorted((i, partner)))
        if pair not in seen_rows:
            seen_rows.add(pair)
            row_pairs.append(pair)

    col_pairs: list[tuple[int, int]] = []
    seen_cols: set[tuple[int, int]] = set()
    for i in range(n):
        partner = next(j for j in range(n) if (col_perm[j] - col_perm[i]) % n == n // 2)
        pair = tuple(sorted((i, partner)))
        if pair not in seen_cols:
            seen_cols.add(pair)
            col_pairs.append(pair)

    return square, row_pairs, col_pairs


def _first_blank_target(blank_mask: np.ndarray, cell: tuple[int, int], direction: tuple[int, int]) -> bool:
    n = blank_mask.shape[0]
    r, c = cell
    dr, dc = direction
    rr, cc = r, c
    while 0 <= rr - dr < n and 0 <= cc - dc < n:
        rr -= dr
        cc -= dc
    while 0 <= rr < n and 0 <= cc < n:
        if bool(blank_mask[rr, cc]):
            return (rr, cc) == (r, c)
        rr += dr
        cc += dc
    return False


def _entry_cell(cell: tuple[int, int], direction: tuple[int, int], n: int) -> tuple[int, int]:
    r, c = cell
    dr, dc = direction
    rr, cc = r, c
    while 0 <= rr - dr < n and 0 <= cc - dc < n:
        rr -= dr
        cc -= dc
    return int(rr), int(cc)


def _arrow_bbox_from_tip(tip: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int, int, int]:
    tr, tc = tip
    dr, dc = direction
    r0 = tr - 2 if dr > 0 else tr + 1
    c0 = tc - 2 if dc > 0 else tc + 1
    return int(r0), int(c0), int(r0 + 1), int(c0 + 1)


def _choose_arrow_specs(
    blank_mask: np.ndarray, blocks: list[list[tuple[int, int]]]
) -> list[dict[str, tuple[int, int] | int]] | None:
    chosen: list[dict[str, tuple[int, int] | int]] = []
    used_dirs: set[tuple[int, int]] = set()
    for block_id, cells in enumerate(blocks):
        candidates = []
        for cell in cells:
            for direction in DIRS:
                if _first_blank_target(blank_mask, cell, direction):
                    score = (1 if direction not in used_dirs else 0, random.random())
                    candidates.append((score, cell, direction))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        _score, cell, direction = candidates[0]
        chosen.append({"block": block_id, "cell": cell, "direction": direction})
        used_dirs.add(direction)
    return chosen


def _render_arrow_patch(color: int, bg: int, direction: tuple[int, int]) -> np.ndarray:
    patch = np.full((2, 2), int(color), dtype=int)
    miss_r = 0 if direction[0] < 0 else 1
    miss_c = 0 if direction[1] < 0 else 1
    patch[miss_r, miss_c] = int(bg)
    return patch


def _place_arrows(
    partial: np.ndarray,
    bg: int,
    specs: list[dict[str, tuple[int, int] | int]],
    top: int,
    left: int,
) -> list[dict[str, tuple[int, int] | int]] | None:
    n = partial.shape[0]
    h = n + top + random.randint(4, 6)
    w = n + left + random.randint(4, 6)
    if h > 30 or w > 30:
        return None

    occupied = np.zeros((h, w), dtype=bool)
    occupied[top : top + n, left : left + n] = True
    placed: list[dict[str, tuple[int, int] | int]] = []

    for spec in specs:
        cell = spec["cell"]
        direction = spec["direction"]
        entry_r, entry_c = _entry_cell(cell, direction, n)
        tip0 = (top + entry_r - direction[0], left + entry_c - direction[1])
        gaps = list(range(8))
        random.shuffle(gaps)
        found = False
        for gap in gaps:
            tip = (tip0[0] - gap * direction[0], tip0[1] - gap * direction[1])
            bbox = _arrow_bbox_from_tip(tip, direction)
            r0, c0, r1, c1 = bbox
            if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
                continue
            pad_r0 = max(0, r0 - 1)
            pad_c0 = max(0, c0 - 1)
            pad_r1 = min(h - 1, r1 + 1)
            pad_c1 = min(w - 1, c1 + 1)
            if bool(occupied[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1].any()):
                continue

            rr, cc = tip
            clear_path = False
            while 0 <= rr + direction[0] < h and 0 <= cc + direction[1] < w:
                rr += direction[0]
                cc += direction[1]
                inside_table = top <= rr < top + n and left <= cc < left + n
                if inside_table:
                    clear_path = True
                    break
                if occupied[rr, cc]:
                    clear_path = False
                    break
            if not clear_path:
                continue

            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            placed.append({**spec, "tip": tip, "bbox": bbox, "shape": (h, w)})
            found = True
            break
        if not found:
            return None
    return placed


def _check_arrow_necessity(
    partial: np.ndarray, solution: np.ndarray, palette: list[int], specs: list[dict[str, tuple[int, int] | int]]
) -> bool:
    filled = partial.copy()
    for spec in specs:
        cell = spec["cell"]
        filled[cell] = int(solution[cell])
    count, recovered = _latin_solution_count(filled, palette, limit=2)
    if count != 1 or recovered is None or not np.array_equal(recovered, solution):
        return False

    for skip_idx in range(len(specs)):
        probe = partial.copy()
        for idx, spec in enumerate(specs):
            if idx == skip_idx:
                continue
            cell = spec["cell"]
            probe[cell] = int(solution[cell])
        count, _ = _latin_solution_count(probe, palette, limit=2)
        if count <= 1:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n = 6 if random.random() < 0.65 else 4
        chosen = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], n + 1)
        bg = chosen[0]
        palette = chosen[1:]
        solution, row_pairs, col_pairs = _build_latin_square(n, palette)

        random.shuffle(row_pairs)
        random.shuffle(col_pairs)
        block_count = 1 if n == 4 else (2 if random.random() < 0.7 else 1)
        blocks: list[list[tuple[int, int]]] = []
        blank_mask = np.zeros((n, n), dtype=bool)
        for row_pair, col_pair in zip(row_pairs[:block_count], col_pairs[:block_count]):
            r1, r2 = row_pair
            c1, c2 = col_pair
            cells = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
            blocks.append(cells)
            for cell in cells:
                blank_mask[cell] = True

        partial = solution.copy()
        partial[blank_mask] = 0
        latin_only_count, _ = _latin_solution_count(partial, palette, limit=3)
        if latin_only_count <= 1:
            continue

        specs = _choose_arrow_specs(blank_mask, blocks)
        if specs is None:
            continue
        if not _check_arrow_necessity(partial, solution, palette, specs):
            continue

        visible_palette = set(int(v) for v in partial.flat if int(v) != 0)
        visible_palette |= {int(solution[spec["cell"]]) for spec in specs}
        if len(visible_palette) != n:
            continue

        top = random.randint(4, 6)
        left = random.randint(4, 6)
        placed = _place_arrows(partial, bg, specs, top, left)
        if placed is None:
            continue
        h, w = placed[0]["shape"]
        grid = np.full((h, w), int(bg), dtype=int)
        grid[top : top + n, left : left + n] = partial
        for spec in placed:
            r0, c0, r1, c1 = spec["bbox"]
            patch = _render_arrow_patch(int(solution[spec["cell"]]), bg, spec["direction"])
            grid[r0 : r1 + 1, c0 : c1 + 1] = np.where(patch == bg, grid[r0 : r1 + 1, c0 : c1 + 1], patch)

        table_bbox = (top, left, top + n - 1, left + n - 1)
        arrows = _parse_arrows(grid, bg, table_bbox)
        after_arrows = _apply_arrow_clues(grid, table_bbox, arrows)
        if not bool(np.any(after_arrows == 0)):
            continue
        if not _check_arrow_necessity(partial, solution, palette, specs):
            continue
        if not np.array_equal(main(grid), solution):
            continue
        return grid

    raise RuntimeError("generate_input failed")
