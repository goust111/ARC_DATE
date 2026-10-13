from __future__ import annotations

import random
from collections import Counter

import numpy as np


OPS = ("identity", "hflip", "vflip", "rot180")
OP_SHAPES = {
    "identity": [(0, 0)],
    "hflip": [(0, 0), (0, 1)],
    "vflip": [(0, 0), (1, 0)],
    "rot180": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "seed": [(0, 0), (0, 1), (1, 0), (1, 1)],
}
GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bands(lines: list[int]) -> list[tuple[int, int]]:
    return [(int(a + 1), int(b)) for a, b in zip(lines, lines[1:]) if b - a > 1]


def _payload_mask(cell: np.ndarray, bg: int, sep: int) -> np.ndarray:
    return (cell != bg) & (cell != sep)


def _single_payload_color(cell: np.ndarray, bg: int, sep: int) -> int:
    mask = _payload_mask(cell, bg, sep)
    colors = np.unique(cell[mask])
    assert colors.size == 1
    return int(colors[0])


def _decode_op(cell: np.ndarray, bg: int, sep: int) -> str:
    mask = _payload_mask(cell, bg, sep)
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    tight = mask[r0 : r1 + 1, c0 : c1 + 1]
    count = int(coords.shape[0])
    h, w = tight.shape
    if count == 1:
        return "identity"
    if count == 2 and h == 1 and w == 2 and tight.all():
        return "hflip"
    if count == 2 and h == 2 and w == 1 and tight.all():
        return "vflip"
    if count == 4 and h == 2 and w == 2 and tight.all():
        return "rot180"
    raise AssertionError("unknown header shape")


def _apply_op(arr: np.ndarray, op: str) -> np.ndarray:
    if op == "identity":
        return arr.copy()
    if op == "hflip":
        return np.fliplr(arr)
    if op == "vflip":
        return np.flipud(arr)
    if op == "rot180":
        return np.rot90(arr, 2)
    raise AssertionError(op)


def _parse_reference(top: np.ndarray, bg: int, sep: int) -> tuple[dict[tuple[int, int], str], list[int]]:
    sep_coords = np.argwhere(top == sep)
    assert sep_coords.size > 0
    r0, c0, r1, c1 = _bbox(sep_coords)
    ref = top[r0 : r1 + 1, c0 : c1 + 1]

    line_rows = [r for r in range(ref.shape[0]) if np.all(ref[r, :] == sep)]
    line_cols = [c for c in range(ref.shape[1]) if np.all(ref[:, c] == sep)]
    row_bands = _bands(line_rows)
    col_bands = _bands(line_cols)
    assert len(row_bands) == len(col_bands) and len(row_bands) >= 2

    def get_cell(i: int, j: int) -> np.ndarray:
        rr = row_bands[i]
        cc = col_bands[j]
        return ref[rr[0] : rr[1], cc[0] : cc[1]]

    header_ops = [_decode_op(get_cell(0, j), bg, sep) for j in range(1, len(col_bands))]
    row_colors = [_single_payload_color(get_cell(i, 0), bg, sep) for i in range(1, len(row_bands))]

    lookup: dict[tuple[int, int], str] = {}
    for i, body_color in enumerate(row_colors, start=1):
        seen: set[int] = set()
        for j, op in enumerate(header_ops, start=1):
            seed_color = _single_payload_color(get_cell(i, j), bg, sep)
            assert seed_color not in seen
            seen.add(seed_color)
            lookup[(body_color, seed_color)] = op
    return lookup, row_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the upper reference table, then transform each lower-canvas object in place.

    Rows are labeled by body color in the first column. The first row encodes operation
    types by shape. Inside each body-color row, the colored seed in a column tells which
    instruction-marker color selects that column.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    h, _w = g.shape

    split_rows = [r for r in range(h) if g[r, 0] != bg and np.all(g[r, :] == g[r, 0])]
    assert split_rows
    split = int(split_rows[0])
    sep = int(g[split, 0])

    lookup, row_colors = _parse_reference(g[:split, :], bg, sep)
    label_set = set(row_colors)

    out = g.copy()
    canvas = g[split + 1 :, :]
    canvas_out = out[split + 1 :, :]
    comps = _components(canvas != bg)

    for comp in comps:
        comp_tuples = [tuple(map(int, rc)) for rc in comp]
        values = [int(canvas[r, c]) for r, c in comp_tuples]
        counts = Counter(values)
        label_counts = {color: count for color, count in counts.items() if color in label_set}
        if not label_counts:
            continue

        body_color = max(label_counts.items(), key=lambda kv: kv[1])[0]
        marker_candidates = [color for color, count in label_counts.items() if color != body_color and count == 1]
        if len(marker_candidates) != 1:
            marker_candidates = [color for color in label_counts if color != body_color]
        assert len(marker_candidates) == 1
        marker_color = int(marker_candidates[0])

        marker_cells = [(r, c) for r, c in comp_tuples if int(canvas[r, c]) == marker_color]
        assert len(marker_cells) == 1
        marker = marker_cells[0]

        obj_coords = np.array([(r, c) for r, c in comp_tuples if (r, c) != marker], dtype=int)
        r0, c0, r1, c1 = _bbox(obj_coords)
        sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for r, c in obj_coords:
            sprite[r - r0, c - c0] = int(canvas[r, c])

        op = lookup[(body_color, marker_color)]
        transformed = _apply_op(sprite, op)

        for r, c in comp_tuples:
            canvas_out[r, c] = bg
        region = canvas_out[r0 : r1 + 1, c0 : c1 + 1]
        region[transformed != bg] = transformed[transformed != bg]

    return out


def _center_offsets(cell_size: int, shape: list[tuple[int, int]]) -> tuple[int, int]:
    max_r = max(r for r, _c in shape) + 1
    max_c = max(c for _r, c in shape) + 1
    return (cell_size - max_r) // 2, (cell_size - max_c) // 2


def _draw_shape(cell: np.ndarray, shape: list[tuple[int, int]], color: int) -> None:
    dr, dc = _center_offsets(cell.shape[0], shape)
    for r, c in shape:
        cell[dr + r, dc + c] = int(color)


def _sample_row_assignments(label_colors: list[int], header_ops: list[str]) -> list[list[int]]:
    while True:
        rows = [random.sample(label_colors, len(label_colors)) for _ in label_colors]
        if len({tuple(row) for row in rows}) != len(rows):
            continue
        if any(all(row[col] == label_colors[i] for i, row in enumerate(rows)) for col in range(len(header_ops))):
            continue
        varied = True
        for seed_color in label_colors:
            seen_ops = {header_ops[row.index(seed_color)] for row in rows}
            if len(seen_ops) < 2:
                varied = False
                break
        if varied:
            return rows


def _random_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        target = random.randint(max(5, h + w - 2), min(h * w - 1, 10))
        cells = {(random.randrange(h), random.randrange(w))}
        for _ in range(target * 8):
            if len(cells) >= target:
                break
            base_r, base_c = random.choice(tuple(cells))
            nbrs = []
            for nr, nc in _neighbors4(base_r, base_c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    nbrs.append((nr, nc))
            if nbrs:
                cells.add(random.choice(nbrs))
        if len(cells) < target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox(coords)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if min(mask.shape) < 2:
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return mask
    raise RuntimeError("failed to sample asymmetric mask")


def _sample_sprite(body_color: int, accent_pool: list[int], bg: int) -> np.ndarray:
    for _ in range(200):
        mask = _random_mask()
        sprite = np.full(mask.shape, bg, dtype=int)
        sprite[mask] = int(body_color)

        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        accent_count = 1 if len(coords) < 7 else random.choice([1, 2])
        accent_color = random.choice(accent_pool)
        for r, c in random.sample(coords, accent_count):
            sprite[r, c] = int(accent_color)

        if all(not np.array_equal(sprite, _apply_op(sprite, op)) for op in OPS[1:]):
            return sprite
    raise RuntimeError("failed to sample asymmetric sprite")


def _attach_marker(sprite: np.ndarray, marker_color: int, bg: int) -> np.ndarray:
    mask = sprite != bg
    h, w = sprite.shape
    options: list[tuple[int, int]] = []
    for c in range(w):
        if mask[0, c]:
            options.append((-1, c))
        if mask[h - 1, c]:
            options.append((h, c))
    for r in range(h):
        if mask[r, 0]:
            options.append((r, -1))
        if mask[r, w - 1]:
            options.append((r, w))
    mr, mc = random.choice(options)

    min_r, max_r = min(0, mr), max(h - 1, mr)
    min_c, max_c = min(0, mc), max(w - 1, mc)
    comp = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
    body_r0 = -min_r
    body_c0 = -min_c
    body_region = comp[body_r0 : body_r0 + h, body_c0 : body_c0 + w]
    body_region[sprite != bg] = sprite[sprite != bg]
    comp[body_r0 + mr, body_c0 + mc] = int(marker_color)
    return comp


def _place_components(components: list[np.ndarray], height: int, width: int, bg: int) -> np.ndarray | None:
    canvas = np.full((height, width), bg, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    for comp in sorted(components, key=lambda arr: -int(np.count_nonzero(arr != bg))):
        ch, cw = comp.shape
        placed = False
        for _ in range(300):
            top = random.randint(0, height - ch)
            left = random.randint(0, width - cw)
            rr0 = max(0, top - 1)
            rr1 = min(height, top + ch + 1)
            cc0 = max(0, left - 1)
            cc1 = min(width, left + cw + 1)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            region = canvas[top : top + ch, left : left + cw]
            if np.any((comp != bg) & (region != bg)):
                continue
            region[comp != bg] = comp[comp != bg]
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return canvas


def _render_reference(
    grid: np.ndarray,
    top: int,
    left: int,
    cell_size: int,
    sep: int,
    bg: int,
    header_color: int,
    label_colors: list[int],
    header_ops: list[str],
    row_assignments: list[list[int]],
) -> None:
    n = len(label_colors) + 1
    size = n * cell_size + (n + 1)

    for k in range(n + 1):
        r = top + k * (cell_size + 1)
        c = left + k * (cell_size + 1)
        grid[r, left : left + size] = int(sep)
        grid[top : top + size, c] = int(sep)

    def cell_view(i: int, j: int) -> np.ndarray:
        r0 = top + 1 + i * (cell_size + 1)
        c0 = left + 1 + j * (cell_size + 1)
        return grid[r0 : r0 + cell_size, c0 : c0 + cell_size]

    cell_view(0, 0)[:, :] = int(bg)
    for j, op in enumerate(header_ops, start=1):
        _draw_shape(cell_view(0, j), OP_SHAPES[op], header_color)
    for i, color in enumerate(label_colors, start=1):
        _draw_shape(cell_view(i, 0), OP_SHAPES["identity"], color)
    for i, row in enumerate(row_assignments, start=1):
        for j, seed_color in enumerate(row, start=1):
            _draw_shape(cell_view(i, j), OP_SHAPES["seed"], seed_color)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        chosen = random.sample(list(range(10)), 9)
        bg = chosen[0]
        sep = chosen[1]
        header_color = chosen[2]
        accent_pool = [chosen[3], chosen[4]]
        label_colors = chosen[5:9]

        header_ops = random.sample(list(OPS), len(OPS))
        row_assignments = _sample_row_assignments(label_colors, header_ops)

        cell_size = random.randint(4, 5)
        table_cells = len(label_colors) + 1
        ref_size = table_cells * cell_size + (table_cells + 1)
        left_margin = random.randint(1, 4)
        right_margin = random.randint(1, 4)
        top_margin = random.randint(1, 3)
        gap = random.randint(1, 3)
        width = ref_size + left_margin + right_margin
        bottom_height = random.randint(16, 22)
        split = top_margin + ref_size + gap
        height = split + 1 + bottom_height

        grid = np.full((height, width), bg, dtype=int)
        _render_reference(
            grid,
            top_margin,
            left_margin,
            cell_size,
            sep,
            bg,
            header_color,
            label_colors,
            header_ops,
            row_assignments,
        )
        grid[split, :] = int(sep)

        all_pairs: list[tuple[int, int, str]] = []
        for i, body_color in enumerate(label_colors):
            for col, op in enumerate(header_ops):
                marker_color = row_assignments[i][col]
                if marker_color != body_color:
                    all_pairs.append((body_color, marker_color, op))

        required: list[tuple[int, int, str]] = []
        for op in OPS:
            candidates = [pair for pair in all_pairs if pair[2] == op]
            if not candidates:
                required = []
                break
            required.append(random.choice(candidates))
        if len(required) != len(OPS):
            continue

        object_count = 4 + random.randint(0, 2)
        pairs = required[:]
        while len(pairs) < object_count:
            pairs.append(random.choice(all_pairs))
        random.shuffle(pairs)

        components = []
        try:
            for body_color, marker_color, _op in pairs:
                sprite = _sample_sprite(body_color, accent_pool, bg)
                components.append(_attach_marker(sprite, marker_color, bg))
        except RuntimeError:
            continue

        bottom = _place_components(components, bottom_height, width, bg)
        if bottom is None:
            continue
        grid[split + 1 :, :] = bottom

        if _bg_color(grid) != bg:
            continue
        out = main(grid)
        if np.array_equal(out, grid):
            continue
        if not np.array_equal(out[: split + 1, :], grid[: split + 1, :]):
            continue
        return grid

    raise RuntimeError(f"failed to generate input after {GEN_RETRIES} attempts")
