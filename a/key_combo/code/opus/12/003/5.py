from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SEP = 1
GREY = 5
PALETTE_POOL = [2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _split_sections(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(sep_cols) == 2, "expected exactly two blue separator columns"
    c0, c1 = sep_cols
    assert c0 + 1 < c1, "sections must have non-zero width"
    return grid[:, :c0], grid[:, c0 + 1 : c1], grid[:, c1 + 1 :]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _count_components(mask: np.ndarray) -> int:
    return len(_components(mask))


def _extract_palette(section: np.ndarray) -> list[int]:
    rows = [r for r in range(section.shape[0]) if np.any(section[r] != BG)]
    assert rows, "palette section must contain a colored row"
    row = rows[0]
    cols = np.flatnonzero(section[row] != BG)
    assert cols.size >= 2, "palette row must contain at least two colors"
    assert np.all(np.diff(cols) == 1), "palette row must be contiguous"
    palette = [int(section[row, c]) for c in cols]
    assert len(set(palette)) == len(palette), "palette colors must be unique"
    return palette


def _paint_band(out: np.ndarray, coords: np.ndarray, palette: list[int]) -> None:
    rows_by_col: dict[int, list[int]] = {}
    for r, c in coords:
        rows_by_col.setdefault(int(c), []).append(int(r))

    c0 = min(rows_by_col)
    c1 = max(rows_by_col)
    left_rows = sorted(rows_by_col[c0])
    right_rows = sorted(rows_by_col[c1])
    assert left_rows == list(range(left_rows[0], left_rows[-1] + 1)), "left cap must be contiguous"

    slopes_down = (sum(right_rows) / len(right_rows)) > (sum(left_rows) / len(left_rows))
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())

    norm_left = sorted(r - r0 if slopes_down else r1 - r for r in left_rows)
    cap_height = len(norm_left)
    assert norm_left == list(range(cap_height)), "left cap must be the normalized start"

    extent = 0
    normalized: list[tuple[int, int, int, int]] = []
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        row_offset = rr - r0 if slopes_down else r1 - rr
        col_offset = cc - c0
        normalized.append((rr, cc, row_offset, col_offset))
        extent = max(extent, row_offset, col_offset)

    last_layer = extent - cap_height + 1
    assert last_layer >= 0, "band must have at least one layer"
    for r, c, row_offset, col_offset in normalized:
        layer = min(row_offset, col_offset, last_layer)
        out[r, c] = int(palette[layer % len(palette)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey diagonal bands in the right section using only the first N palette colors, where N is the component count in the left section.

    Input structure:
    - Two solid blue separator columns split the grid into three vertical sections A, B, and C.
    - Section A contains several separated non-black blobs on a black background; section B contains one contiguous palette row near the top; section C contains one or more grey diagonal bands on black.
    - The separator columns remain blue, and any non-grey pixels in section C are already final.

    Rule:
    - Count the 4-connected non-black components in section A and call that number N.
    - Read the palette row in section B from left to right and keep only its first N colors as the active palette.
    - For each grey band in section C, treat its left flat cap as layer 0, assign each grey pixel layer min(row_offset_from_cap, col_offset_from_left_edge), clip the far end to the last layer, and recolor with active_palette[layer mod N].

    Key insight:
    - Section A does not describe geometry for the output at all; it only supplies the palette length, so the same band shape changes both its color set and its repetition period when the component count changes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    section_a, section_b, section_c = _split_sections(g)
    n = _count_components(section_a != BG)
    palette = _extract_palette(section_b)
    assert 1 <= n <= len(palette), "component count must fit inside the palette row"
    active = palette[:n]

    out = g.copy()
    for coords in _components(section_c == GREY):
        global_coords = coords.copy()
        global_coords[:, 1] += section_a.shape[1] + section_b.shape[1] + 2
        _paint_band(out, global_coords, active)
    return out


def _random_polyomino(size: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < size:
        r, c = random.choice(tuple(cells))
        nr, nc = random.choice(_neighbors4(r, c))
        cells.add((nr, nc))
    arr = np.array(sorted(cells), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return arr


def _expanded(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_components(height: int, width: int, count: int, color: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    for _ in range(count):
        shape = _random_polyomino(random.randint(2, 5))
        sh, sw = int(shape[:, 0].max()) + 1, int(shape[:, 1].max()) + 1
        if sh > height or sw > width:
            return None
        blocked = _expanded(occupied)
        placed = False
        anchors = [(r, c) for r in range(height - sh + 1) for c in range(width - sw + 1)]
        random.shuffle(anchors)
        for r0, c0 in anchors:
            coords = shape + np.array([r0, c0], dtype=int)
            if np.any(blocked[coords[:, 0], coords[:, 1]]):
                continue
            grid[coords[:, 0], coords[:, 1]] = color
            occupied[coords[:, 0], coords[:, 1]] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _band_cells(size: int, cap_height: int, slope: int) -> tuple[np.ndarray, np.ndarray]:
    coords: list[tuple[int, int]] = []
    layers: list[int] = []
    last_layer = size - cap_height
    grow = cap_height - 1
    for col in range(size):
        row_lo = max(0, col - grow)
        row_hi = min(cap_height - 1 + col, size - 1)
        for row in range(row_lo, row_hi + 1):
            rr = row if slope > 0 else size - 1 - row
            coords.append((rr, col))
            layers.append(min(row, col, last_layer))
    return np.array(coords, dtype=int), np.array(layers, dtype=int)


def _place_bands(height: int, width: int, specs: list[dict[str, int]]) -> list[tuple[np.ndarray, np.ndarray]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[np.ndarray, np.ndarray]] = []
    for spec in sorted(specs, key=lambda x: x["size"], reverse=True):
        local_coords, layers = _band_cells(spec["size"], spec["cap_height"], spec["slope"])
        sh = sw = spec["size"]
        blocked = _expanded(occupied)
        anchors = [(r, c) for r in range(height - sh + 1) for c in range(width - sw + 1)]
        random.shuffle(anchors)
        placed = False
        for r0, c0 in anchors:
            coords = local_coords + np.array([r0, c0], dtype=int)
            if np.any(blocked[coords[:, 0], coords[:, 1]]):
                continue
            occupied[coords[:, 0], coords[:, 1]] = True
            placements.append((coords, layers))
            placed = True
            break
        if not placed:
            return None
    return placements


def _make_band_specs(n: int, num_bands: int) -> list[dict[str, int]]:
    specs: list[dict[str, int]] = []
    slopes = random.sample([1, -1], k=2) if num_bands == 2 else [random.choice([1, -1])]
    for i in range(num_bands):
        cap_height = random.randint(2, 3)
        if num_bands == 1:
            cycles = random.choices([1, 2], weights=[1, 3])[0]
        else:
            choices = [1, 2] if n <= 3 else [1]
            cycles = random.choice(choices)
        size = cycles * n + cap_height - 1
        specs.append(
            {
                "cap_height": cap_height,
                "cycles": cycles,
                "size": size,
                "slope": slopes[i],
            }
        )
    return specs


def _assemble_input(
    section_a: np.ndarray,
    section_b: np.ndarray,
    section_c: np.ndarray,
) -> np.ndarray:
    h = section_a.shape[0]
    aw = section_a.shape[1]
    bw = section_b.shape[1]
    cw = section_c.shape[1]
    grid = np.zeros((h, aw + 1 + bw + 1 + cw), dtype=int)
    grid[:, :aw] = section_a
    grid[:, aw] = SEP
    grid[:, aw + 1 : aw + 1 + bw] = section_b
    grid[:, aw + 1 + bw] = SEP
    grid[:, aw + 2 + bw :] = section_c
    return grid


def _build_expected(
    inp: np.ndarray,
    placements: list[tuple[np.ndarray, np.ndarray]],
    active_palette: list[int],
    c_start: int,
) -> np.ndarray:
    out = inp.copy()
    pal = np.array(active_palette, dtype=int)
    for coords, layers in placements:
        global_coords = coords.copy()
        global_coords[:, 1] += c_start
        out[global_coords[:, 0], global_coords[:, 1]] = pal[layers % len(active_palette)]
    return out


def _attempt_generate() -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    palette_len = random.randint(3, 5)
    n = random.randint(2, palette_len)
    num_bands = random.choice([1, 2])
    palette = random.sample(PALETTE_POOL, palette_len)
    shape_color = random.choice([c for c in PALETTE_POOL if c not in palette])
    specs = _make_band_specs(n, num_bands)

    max_size = max(spec["size"] for spec in specs)
    b_width = random.randint(max(5, palette_len), max(6, palette_len + 2))
    a_width = random.randint(5, 7)
    c_width_max = 30 - a_width - b_width - 2
    if c_width_max < max_size:
        return None

    c_width = random.randint(max_size, min(13, c_width_max))
    height = random.randint(max(max_size, 9), min(13, max(max_size, 9) + 2))

    section_a = _place_components(height, a_width, n, shape_color)
    if section_a is None:
        return None

    section_b = np.zeros((height, b_width), dtype=int)
    palette_start = random.randint(0, b_width - palette_len)
    section_b[0, palette_start : palette_start + palette_len] = np.array(palette, dtype=int)

    placements = _place_bands(height, c_width, specs)
    if placements is None:
        return None

    section_c = np.zeros((height, c_width), dtype=int)
    for coords, _layers in placements:
        section_c[coords[:, 0], coords[:, 1]] = GREY

    inp = _assemble_input(section_a, section_b, section_c)
    c_start = a_width + 1 + b_width + 1
    expected = _build_expected(inp, placements, palette[:n], c_start)

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    out_c = expected[:, c_start:]
    used_output_colors = set(map(int, out_c[out_c != BG]))
    if len(used_output_colors) < 2:
        return None

    meta = {
        "n": n,
        "palette_len": palette_len,
        "num_bands": num_bands,
        "height": height,
        "c_width": c_width,
    }
    return inp, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _attempt_generate()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
