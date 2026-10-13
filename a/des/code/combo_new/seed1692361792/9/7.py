from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SEPARATOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_CHOICES = ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2))
CELL_SIZE_MIN = 4
CELL_SIZE_MAX = 6
GENERATION_RETRIES = 400


def _base_shapes() -> list[np.ndarray]:
    bases = [
        np.array([[1]], dtype=bool),
        np.array([[1, 1]], dtype=bool),
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    ]
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in bases:
        for turns in range(4):
            rotated = np.rot90(base, turns)
            for candidate in (rotated, np.fliplr(rotated)):
                key = candidate.tobytes() + bytes(candidate.shape)
                if key in seen:
                    continue
                seen.add(key)
                variants.append(candidate.copy())
    return variants


SHAPES = _base_shapes()


def _bands(indices: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        bands.append((start, prev))
        start = idx
        prev = idx
    bands.append((start, prev))
    return bands


def _separator_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    raw_rows = [r for r in range(h) if not np.any(g[r] == BG)]
    raw_cols = [c for c in range(w) if not np.any(g[:, c] == BG)]
    assert len(raw_rows) >= 2 and len(raw_cols) >= 2

    border_values: list[int] = []
    for r in raw_rows:
        border_values.extend(int(v) for v in g[r] if int(v) != BG)
    for c in raw_cols:
        border_values.extend(int(v) for v in g[:, c] if int(v) != BG)
    separator = Counter(border_values).most_common(1)[0][0]

    rows = [r for r in raw_rows if int(np.count_nonzero(g[r] == separator)) > w // 2]
    cols = [c for c in raw_cols if int(np.count_nonzero(g[:, c] == separator)) > h // 2]

    row_bands = _bands(rows)
    col_bands = _bands(cols)
    assert row_bands[0][0] == 0 and row_bands[-1][1] == h - 1
    assert col_bands[0][0] == 0 and col_bands[-1][1] == w - 1
    assert all(end - start == 1 for start, end in row_bands)
    assert all(end - start == 1 for start, end in col_bands)
    return int(separator), row_bands, col_bands


def _collect_labels(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    separator: int,
) -> list[tuple[int, int, int]]:
    labels: list[tuple[int, int, int]] = []
    for r, c in coords:
        value = int(grid[r, c])
        if value not in (BG, separator):
            labels.append((r, c, value))
    return labels


def _regions(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    separator, row_bands, col_bands = _separator_structure(g)
    regions: list[dict[str, object]] = []

    for ri, (top_band, bottom_band) in enumerate(zip(row_bands, row_bands[1:])):
        for ci, (left_band, right_band) in enumerate(zip(col_bands, col_bands[1:])):
            inner_top = top_band[1] + 1
            inner_left = left_band[1] + 1
            inner_h = bottom_band[0] - inner_top
            inner_w = right_band[0] - inner_left
            assert inner_h >= 1 and inner_w >= 1

            top_side = [(top_band[1], c) for c in range(inner_left, inner_left + inner_w)]
            bottom_side = [(bottom_band[0], c) for c in range(inner_left, inner_left + inner_w)]
            left_side = [(r, left_band[1]) for r in range(inner_top, inner_top + inner_h)]
            right_side = [(r, right_band[0]) for r in range(inner_top, inner_top + inner_h)]

            senders = _collect_labels(g, top_side + left_side, separator)
            receivers = _collect_labels(g, bottom_side + right_side, separator)
            assert len(senders) == 1
            assert len(receivers) == 1

            patch = g[inner_top : inner_top + inner_h, inner_left : inner_left + inner_w].copy()
            assert not np.any(patch == separator)

            regions.append(
                {
                    "row": int(ri),
                    "col": int(ci),
                    "inner_top": int(inner_top),
                    "inner_left": int(inner_left),
                    "inner_h": int(inner_h),
                    "inner_w": int(inner_w),
                    "send": int(senders[0][2]),
                    "receive": int(receivers[0][2]),
                    "patch": patch,
                }
            )

    assert regions
    return separator, row_bands, col_bands, regions


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    separator, row_bands, col_bands, regions = _regions(g)

    send_to_region: dict[int, int] = {}
    receive_to_region: dict[int, int] = {}
    for idx, region in enumerate(regions):
        send_color = int(region["send"])
        receive_color = int(region["receive"])
        assert send_color != receive_color
        assert send_color not in send_to_region
        assert receive_color not in receive_to_region
        send_to_region[send_color] = int(idx)
        receive_to_region[receive_color] = int(idx)

    assert set(send_to_region) == set(receive_to_region)

    out = np.full_like(g, BG)
    for start, end in row_bands:
        out[start : end + 1, :] = separator
    for start, end in col_bands:
        out[:, start : end + 1] = separator

    for region in regions:
        target = regions[receive_to_region[int(region["send"])]]
        patch = np.asarray(region["patch"], dtype=int)
        target_h = int(target["inner_h"])
        target_w = int(target["inner_w"])
        for dr, dc in np.argwhere(patch != BG):
            dr = int(dr)
            dc = int(dc)
            assert dr < target_h and dc < target_w
            out[int(target["inner_top"]) + dr, int(target["inner_left"]) + dc] = int(patch[dr, dc])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reroute each subregion's interior pattern to the region whose receive label matches its send label.

    Input structure:
    - Black background with a full grey outer frame and full grey separator rows and columns that partition the canvas into a rectangular grid of subregions.
    - Each subregion contains one non-black, non-grey foreground pattern inside its interior.
    - Each subregion border has exactly two single-pixel colored labels placed on grey border cells.
    - A label on the top or left border is that region's send label, while a label on the bottom or right border is its receive label.

    Rule:
    - Detect the grey separator lattice and split the grid into rectangular subregions.
    - Read each region's send-label color from its top/left border and its receive-label color from its bottom/right border.
    - For every region A, find the unique region B whose receive-label color equals A's send-label color.
    - Copy A's interior pattern into B while preserving the pattern's exact color, shape, and offset relative to the subregion's top-left corner.
    - Restore all label pixels to grey and keep the grey separator lattice unchanged.

    Key insight:
    - The colored border pixels are routing markers rather than part of the shapes: border position determines send versus receive, and matching label colors determines where each pattern must move.
    """
    return _solve(np.asarray(grid, dtype=int))


def _balanced_sides(count: int, first: str, second: str) -> list[str]:
    extra_first = random.choice([True, False])
    n_first = count // 2 + int(extra_first and count % 2 == 1)
    n_second = count - n_first
    sides = [first] * n_first + [second] * n_second
    random.shuffle(sides)
    return sides


def _random_permutation(n: int) -> list[int]:
    indices = list(range(n))
    random.shuffle(indices)

    if random.random() < 0.5:
        groups = [indices]
    else:
        cycle_count = random.randint(2, n // 2)
        sizes = [2] * cycle_count
        for _ in range(n - 2 * cycle_count):
            sizes[random.randrange(cycle_count)] += 1
        random.shuffle(sizes)
        groups = []
        cursor = 0
        for size in sizes:
            groups.append(indices[cursor : cursor + size])
            cursor += size

    perm = [0] * n
    for group in groups:
        for src, dst in zip(group, group[1:] + group[:1]):
            perm[src] = dst
    assert all(i != perm[i] for i in range(n))
    return perm


def _random_payload(cell_h: int, cell_w: int, forbidden: set[int]) -> np.ndarray:
    valid_shapes = [shape for shape in SHAPES if shape.shape[0] < cell_h and shape.shape[1] < cell_w]
    assert valid_shapes
    mask = random.choice(valid_shapes)
    color_choices = [color for color in PALETTE if color not in forbidden]
    color = random.choice(color_choices)

    top = random.randint(0, cell_h - mask.shape[0])
    left = random.randint(0, cell_w - mask.shape[1])

    patch = np.full((cell_h, cell_w), BG, dtype=int)
    window = patch[top : top + mask.shape[0], left : left + mask.shape[1]]
    window[mask] = int(color)
    return patch


def _label_position(
    side: str,
    top: int,
    left: int,
    cell_h: int,
    cell_w: int,
) -> tuple[int, int]:
    if side == "top":
        return top + 1, left + 2 + random.randrange(cell_w)
    if side == "left":
        return top + 2 + random.randrange(cell_h), left + 1
    if side == "bottom":
        return top + cell_h + 2, left + 2 + random.randrange(cell_w)
    assert side == "right"
    return top + 2 + random.randrange(cell_h), left + cell_w + 2


def _render_lattice(rows: int, cols: int, cell_h: int, cell_w: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    height = rows * cell_h + 2 * (rows + 1)
    width = cols * cell_w + 2 * (cols + 1)
    grid = np.full((height, width), BG, dtype=int)

    row_bands = [idx * (cell_h + 2) for idx in range(rows + 1)]
    col_bands = [idx * (cell_w + 2) for idx in range(cols + 1)]
    for row in row_bands:
        grid[row : row + 2, :] = SEPARATOR
    for col in col_bands:
        grid[:, col : col + 2] = SEPARATOR

    region_origins: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            region_origins.append((r * (cell_h + 2), c * (cell_w + 2)))
    return grid, region_origins


def _expected_output(
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    region_origins: list[tuple[int, int]],
    payloads: list[np.ndarray],
    perm: list[int],
) -> np.ndarray:
    out, _ = _render_lattice(rows, cols, cell_h, cell_w)
    for src, dst in enumerate(perm):
        top, left = region_origins[dst]
        patch = payloads[src]
        mask = patch != BG
        out[top + 2 : top + 2 + cell_h, left + 2 : left + 2 + cell_w][mask] = patch[mask]
    return out


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(GRID_CHOICES)
    max_h = min(CELL_SIZE_MAX, (30 - 2 * (rows + 1)) // rows)
    max_w = min(CELL_SIZE_MAX, (30 - 2 * (cols + 1)) // cols)
    if max_h < CELL_SIZE_MIN or max_w < CELL_SIZE_MIN:
        return None
    cell_h = random.randint(CELL_SIZE_MIN, max_h)
    cell_w = random.randint(CELL_SIZE_MIN, max_w)
    region_count = rows * cols

    grid, region_origins = _render_lattice(rows, cols, cell_h, cell_w)
    send_colors = random.sample(PALETTE, region_count)
    perm = _random_permutation(region_count)

    receive_colors = [0] * region_count
    for src, dst in enumerate(perm):
        receive_colors[dst] = send_colors[src]

    payloads: list[np.ndarray] = []
    seen_payloads: set[bytes] = set()
    for idx in range(region_count):
        forbidden = {SEPARATOR, send_colors[idx], receive_colors[idx]}
        for _ in range(80):
            patch = _random_payload(cell_h, cell_w, forbidden)
            key = patch.tobytes()
            if key in seen_payloads:
                continue
            seen_payloads.add(key)
            payloads.append(patch)
            break
        else:
            return None

    send_sides = _balanced_sides(region_count, "top", "left")
    receive_sides = _balanced_sides(region_count, "bottom", "right")

    for idx, (top, left) in enumerate(region_origins):
        patch = payloads[idx]
        mask = patch != BG
        grid[top + 2 : top + 2 + cell_h, left + 2 : left + 2 + cell_w][mask] = patch[mask]

        sr, sc = _label_position(send_sides[idx], top, left, cell_h, cell_w)
        rr, rc = _label_position(receive_sides[idx], top, left, cell_h, cell_w)
        grid[sr, sc] = send_colors[idx]
        grid[rr, rc] = receive_colors[idx]

    expected = _expected_output(rows, cols, cell_h, cell_w, region_origins, payloads, perm)

    try:
        separator, row_bands, col_bands, regions = _regions(grid)
    except AssertionError:
        return None

    if separator != SEPARATOR:
        return None
    if len(row_bands) != rows + 1 or len(col_bands) != cols + 1:
        return None
    if len(regions) != region_count:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 7")
