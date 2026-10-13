from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SYMMETRY_TYPES = ("horizontal", "vertical", "both", "none")
CORNERS = ("tl", "tr", "bl", "br")
VERTICAL_OPPOSITE = {"tl": "bl", "tr": "br", "bl": "tl", "br": "tr"}
HORIZONTAL_OPPOSITE = {"tl": "tr", "tr": "tl", "bl": "br", "br": "bl"}
DIAGONAL_OPPOSITE = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}

REGION_ROW_COUNT_RANGE = (2, 3)
REGION_COL_COUNT_RANGE = (2, 3)
REGION_HEIGHT_RANGE = (7, 10)
REGION_WIDTH_RANGE = (7, 10)
MAX_GENERATION_ATTEMPTS = 400
MASK_ATTEMPTS = 200


def _separator_lines(g: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    height, width = map(int, g.shape)
    candidates = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        rows = [r for r in range(height) if np.all(g[r, :] == color)]
        cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected one separator color"
    return candidates[0]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for sep in sorted(separators):
        assert start < sep < length - 1, "separator must leave nonempty regions"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "expected a final region"
    spans.append((start, length - 1))
    return spans


def _classify_patch(patch: np.ndarray) -> str:
    horizontal = np.array_equal(patch, np.flipud(patch))
    vertical = np.array_equal(patch, np.fliplr(patch))
    if horizontal and vertical:
        return "both"
    if horizontal:
        return "horizontal"
    if vertical:
        return "vertical"
    return "none"


def _extract_seed(region: np.ndarray, bg: int) -> tuple[np.ndarray, str]:
    mask = region != bg
    assert bool(mask.any()), "each region must contain a seed"
    colors = set(map(int, np.unique(region[mask])))
    assert len(colors) == 1, "each region must use one seed color"

    coords = np.argwhere(mask)
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    height, width = map(int, region.shape)

    top = r0 == 0
    bottom = r1 == height - 1
    left = c0 == 0
    right = c1 == width - 1
    assert top != bottom and left != right, "seed must occupy exactly one corner"

    if top and left:
        corner = "tl"
    elif top and right:
        corner = "tr"
    elif bottom and left:
        corner = "bl"
    else:
        corner = "br"

    seed = region[r0 : r1 + 1, c0 : c1 + 1].copy()
    return seed, corner


def _corner_origin(region_shape: tuple[int, int], patch_shape: tuple[int, int], corner: str) -> tuple[int, int]:
    region_h, region_w = map(int, region_shape)
    patch_h, patch_w = map(int, patch_shape)
    row = 0 if corner in ("tl", "tr") else region_h - patch_h
    col = 0 if corner in ("tl", "bl") else region_w - patch_w
    return row, col


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, row: int, col: int, bg: int) -> None:
    patch_h, patch_w = map(int, patch.shape)
    window = canvas[row : row + patch_h, col : col + patch_w]
    mask = patch != bg
    assert np.all((window[mask] == bg) | (window[mask] == patch[mask]))
    window[mask] = patch[mask]


def _paste_corner(canvas: np.ndarray, patch: np.ndarray, corner: str, bg: int) -> None:
    row, col = _corner_origin(canvas.shape, patch.shape, corner)
    _paste_patch(canvas, patch, row, col, bg)


def _transform_between_corners(patch: np.ndarray, source_corner: str, target_corner: str) -> np.ndarray:
    out = patch
    if source_corner[0] != target_corner[0]:
        out = np.flipud(out)
    if source_corner[1] != target_corner[1]:
        out = np.fliplr(out)
    return out.copy()


def _apply_region_rule(region: np.ndarray, seed: np.ndarray, source_corner: str, symmetry_type: str, bg: int) -> None:
    if symmetry_type == "horizontal":
        targets = [VERTICAL_OPPOSITE[source_corner]]
    elif symmetry_type == "vertical":
        targets = [HORIZONTAL_OPPOSITE[source_corner]]
    elif symmetry_type == "both":
        targets = [corner for corner in CORNERS if corner != source_corner]
    else:
        targets = [DIAGONAL_OPPOSITE[source_corner]]

    for target_corner in targets:
        if symmetry_type == "none":
            patch = np.rot90(seed, 2)
        else:
            patch = _transform_between_corners(seed, source_corner, target_corner)
        _paste_corner(region, patch, target_corner, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each partitioned region's corners according to the seed pattern's own axis symmetries.

    Input structure:
    - Black background with one-pixel grey separator rows and columns.
    - Each rectangular region contains one single-color seed pattern in exactly one corner.

    Rule:
    - Detect the seed corner and crop its tight bounding box.
    - Classify the cropped seed as horizontal-only, vertical-only, both-axis, or no-axis symmetric.
    - Horizontal-only: copy a vertically flipped version to the opposite vertical corner.
    - Vertical-only: copy a horizontally flipped version to the opposite horizontal corner.
    - Both-axis: fill the other three corners using the corresponding flips.
    - No-axis: place the 180-degree rotation in the diagonal corner only.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = int(np.bincount(g.reshape(-1)).argmax())
    separator, sep_rows, sep_cols = _separator_lines(g, bg)
    assert separator != bg

    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            seed, corner = _extract_seed(region, bg)
            symmetry_type = _classify_patch(seed)
            _apply_region_rule(out[r0 : r1 + 1, c0 : c1 + 1], seed, corner, symmetry_type, bg)
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    height, width = map(int, mask.shape)
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _mask_valid(mask: np.ndarray, symmetry_type: str) -> bool:
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return False
    if not bool(mask[0, 0]):
        return False
    if not bool(mask[-1, :].any()) or not bool(mask[:, -1].any()):
        return False
    count = int(mask.sum())
    if count < 3 or count == int(mask.size):
        return False
    if not _is_connected(mask):
        return False
    return _classify_patch(mask.astype(int)) == symmetry_type


def _fallback_mask(height: int, width: int, symmetry_type: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if symmetry_type == "both":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
    elif symmetry_type == "horizontal":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
    elif symmetry_type == "vertical":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
    else:
        mask[0, :] = True
        mask[:, 0] = True
    assert _mask_valid(mask, symmetry_type)
    return mask


def _random_mask(height: int, width: int, symmetry_type: str) -> np.ndarray:
    density = random.uniform(0.28, 0.62)
    if symmetry_type == "both":
        base = np.random.random(((height + 1) // 2, (width + 1) // 2)) < density
        base[0, 0] = True
        mask = np.zeros((height, width), dtype=bool)
        for r in range(height):
            for c in range(width):
                mask[r, c] = bool(base[min(r, height - 1 - r), min(c, width - 1 - c)])
        return mask
    if symmetry_type == "horizontal":
        base = np.random.random(((height + 1) // 2, width)) < density
        base[0, 0] = True
        base[random.randrange(base.shape[0]), -1] = True
        mask = np.zeros((height, width), dtype=bool)
        for r in range(height):
            mask[r, :] = base[min(r, height - 1 - r), :]
        return mask
    if symmetry_type == "vertical":
        base = np.random.random((height, (width + 1) // 2)) < density
        base[0, 0] = True
        base[-1, random.randrange(base.shape[1])] = True
        mask = np.zeros((height, width), dtype=bool)
        for c in range(width):
            mask[:, c] = base[:, min(c, width - 1 - c)]
        return mask

    mask = np.random.random((height, width)) < density
    mask[0, 0] = True
    mask[-1, random.randrange(width)] = True
    mask[random.randrange(height), -1] = True
    return mask


def _sample_seed_mask(height: int, width: int, symmetry_type: str) -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        mask = _random_mask(height, width, symmetry_type)
        if _mask_valid(mask, symmetry_type):
            return mask
    return _fallback_mask(height, width, symmetry_type)


def _layout_from_lengths(lengths: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    spans = []
    separators = []
    cursor = 0
    for i, length in enumerate(lengths):
        spans.append((cursor, cursor + int(length) - 1))
        cursor += int(length)
        if i + 1 < len(lengths):
            separators.append(cursor)
            cursor += 1
    return spans, separators, cursor


def _expected_from_metadata(inp: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = inp.copy()
    for item in regions:
        r0, r1 = item["row_span"]
        c0, c1 = item["col_span"]
        _apply_region_rule(
            out[r0 : r1 + 1, c0 : c1 + 1],
            item["seed"],
            item["corner"],
            item["symmetry_type"],
            BG,
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    n_row_regions = random.randint(*REGION_ROW_COUNT_RANGE)
    n_col_regions = random.randint(*REGION_COL_COUNT_RANGE)
    region_count = n_row_regions * n_col_regions

    row_lengths = [random.randint(*REGION_HEIGHT_RANGE) for _ in range(n_row_regions)]
    col_lengths = [random.randint(*REGION_WIDTH_RANGE) for _ in range(n_col_regions)]
    row_spans, sep_rows, height = _layout_from_lengths(row_lengths)
    col_spans, sep_cols, width = _layout_from_lengths(col_lengths)

    grid = np.full((height, width), BG, dtype=int)
    grid[sep_rows, :] = SEPARATOR
    grid[:, sep_cols] = SEPARATOR

    symmetry_types = list(SYMMETRY_TYPES)
    symmetry_types.extend(random.choice(SYMMETRY_TYPES) for _ in range(region_count - len(SYMMETRY_TYPES)))
    random.shuffle(symmetry_types)

    regions = []
    index = 0
    for row_span in row_spans:
        for col_span in col_spans:
            region_h = int(row_span[1] - row_span[0] + 1)
            region_w = int(col_span[1] - col_span[0] + 1)
            max_seed_h = (region_h - 1) // 2
            max_seed_w = (region_w - 1) // 2
            if max_seed_h < 3 or max_seed_w < 3:
                return None

            seed_h = random.randint(3, max_seed_h)
            seed_w = random.randint(3, max_seed_w)
            symmetry_type = symmetry_types[index]
            index += 1

            mask = _sample_seed_mask(seed_h, seed_w, symmetry_type)
            color = random.choice(COLORS)
            seed = np.full((seed_h, seed_w), BG, dtype=int)
            seed[mask] = color
            corner = random.choice(CORNERS)
            seed = _transform_between_corners(seed, "tl", corner)

            r0, r1 = row_span
            c0, c1 = col_span
            _paste_corner(grid[r0 : r1 + 1, c0 : c1 + 1], seed, corner, BG)
            regions.append(
                {
                    "row_span": row_span,
                    "col_span": col_span,
                    "seed": seed,
                    "corner": corner,
                    "symmetry_type": symmetry_type,
                }
            )

    expected = _expected_from_metadata(grid, regions)
    if np.array_equal(expected, grid):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
