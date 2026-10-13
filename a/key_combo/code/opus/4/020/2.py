from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, find_components


BBox = tuple[int, int, int, int]


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _bboxes_share_row_or_col(a: BBox, b: BBox) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return _spans_overlap(ar0, ar1, br0, br1) or _spans_overlap(ac0, ac1, bc0, bc1)


def _merged_fragment_groups(components: list[dict]) -> list[dict]:
    parent = list(range(len(components)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if _bboxes_share_row_or_col(components[i]["bbox"], components[j]["bbox"]):
                union(i, j)

    grouped: dict[int, list[np.ndarray]] = {}
    for idx, component in enumerate(components):
        grouped.setdefault(find(idx), []).append(component["coords"])

    merged = []
    for coord_groups in grouped.values():
        coords = np.vstack(coord_groups)
        merged.append({"coords": coords, "bbox": bbox_coords(coords)})

    merged.sort(key=lambda group: tuple(int(v) for v in group["bbox"]))
    return merged


def _is_horizontally_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, mask[::-1, :]))


def main(grid: np.ndarray) -> np.ndarray:
    """Merge each fragmented teal rectangle, then fill its bbox solid red or checkerboard red based on horizontal symmetry.

    Input structure:
    - The grid has a black background and teal foreground fragments.
    - Each target rectangle is split into 2 to 4 separate 4-connected fragments.
    - Fragments from the same rectangle are not 4-connected, but their bboxes overlap in row span or column span transitively.
    - Different rectangles have disjoint row spans and disjoint column spans at the bbox level.

    Rule:
    - Find all teal 4-connected fragments.
    - Merge fragments transitively when their bboxes overlap in row span or in column span.
    - For each merged group, take the inclusive bbox of all its teal cells and inspect only the original teal pattern inside that bbox.
    - If that teal pattern is mirror-symmetric across the bbox's horizontal midline, fill every black cell in the bbox red.
    - Otherwise, fill only the black bbox cells with even local parity `(r-r0 + c-c0) % 2 == 0` red, leaving odd-parity black cells unchanged.
    - Keep all original teal cells unchanged.

    Key insight:
    - The branch condition depends on the fragment distribution inside the reconstructed rectangle's bbox, not on the rectangle outline itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    fragments = find_components(g, bg=Color.BLACK, colors=[Color.TEAL], connectivity=4)
    assert fragments, "expected teal fragments"

    out = g.copy()
    for group in _merged_fragment_groups(fragments):
        r0, c0, r1, c1 = map(int, group["bbox"])
        teal_mask = g[r0 : r1 + 1, c0 : c1 + 1] == Color.TEAL
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        background_mask = region == Color.BLACK

        if _is_horizontally_symmetric(teal_mask):
            region[background_mask] = Color.RED
        else:
            rr, cc = np.indices(region.shape)
            fill_mask = background_mask & (((rr + cc) % 2) == 0)
            region[fill_mask] = Color.RED

    return out


GENERATION_ATTEMPTS = 500
MAX_GRID_SIZE = 30
GROUP_COUNT_CHOICES = (2, 2, 3)
RECT_HEIGHT_RANGE = (4, 7)
RECT_WIDTH_RANGE = (4, 8)
GAP_RANGE = (1, 3)
BORDER_RANGE = (1, 3)
PATCH_ATTEMPTS = 24


def _sym_top_bottom_rows(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[height - 1, :] = True
    return mask


def _sym_side_columns(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[:, 0] = True
    mask[:, width - 1] = True
    return mask


def _sym_four_corners(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, 0] = True
    mask[0, width - 1] = True
    mask[height - 1, 0] = True
    mask[height - 1, width - 1] = True
    return mask


def _asym_top_row_bottom_segment(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    segment_len = random.randint(1, width - 2)
    start = random.randint(0, width - segment_len)
    mask[height - 1, start : start + segment_len] = True
    return mask


def _asym_left_column_top_right_segment(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[:, 0] = True
    start = random.randint(2, width - 1)
    mask[0, start:width] = True
    return mask


def _asym_three_corners(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, 0] = True
    mask[0, width - 1] = True
    mask[height - 1, 0] = True
    return mask


SYMMETRIC_BUILDERS = (
    _sym_top_bottom_rows,
    _sym_side_columns,
    _sym_four_corners,
)

ASYMMETRIC_BUILDERS = (
    _asym_top_row_bottom_segment,
    _asym_left_column_top_right_segment,
    _asym_three_corners,
)


def _patch_fragment_count(mask: np.ndarray) -> int:
    patch = np.where(mask, Color.TEAL, Color.BLACK)
    return len(find_components(patch, bg=Color.BLACK, colors=[Color.TEAL], connectivity=4))


def _patch_valid(mask: np.ndarray, symmetric: bool) -> bool:
    height, width = mask.shape
    components = find_components(np.where(mask, Color.TEAL, Color.BLACK), bg=Color.BLACK, colors=[Color.TEAL], connectivity=4)
    if not (2 <= len(components) <= 4):
        return False
    merged = _merged_fragment_groups(components)
    if len(merged) != 1:
        return False
    if tuple(int(v) for v in merged[0]["bbox"]) != (0, 0, height - 1, width - 1):
        return False
    return _is_horizontally_symmetric(mask) == bool(symmetric)


def _sample_patch(height: int, width: int, symmetric: bool) -> np.ndarray | None:
    builders = SYMMETRIC_BUILDERS if symmetric else ASYMMETRIC_BUILDERS
    for _ in range(PATCH_ATTEMPTS):
        mask = random.choice(builders)(height, width)
        if _patch_valid(mask, symmetric):
            return mask
    return None


def _sample_layout(sizes: list[tuple[int, int]]) -> tuple[int, int, list[BBox]] | None:
    heights = [int(h) for h, _w in sizes]
    widths = [int(w) for _h, w in sizes]

    for _ in range(PATCH_ATTEMPTS):
        top = random.randint(*BORDER_RANGE)
        left = random.randint(*BORDER_RANGE)
        bottom = random.randint(*BORDER_RANGE)
        right = random.randint(*BORDER_RANGE)
        row_gaps = [random.randint(*GAP_RANGE) for _ in range(len(sizes) - 1)]
        col_gaps = [random.randint(*GAP_RANGE) for _ in range(len(sizes) - 1)]

        total_rows = top + sum(heights) + sum(row_gaps) + bottom
        total_cols = left + sum(widths) + sum(col_gaps) + right
        if total_rows > MAX_GRID_SIZE or total_cols > MAX_GRID_SIZE:
            continue

        bboxes: list[BBox] = []
        row = top
        col = left
        for idx, (height, width) in enumerate(sizes):
            bboxes.append((row, col, row + height - 1, col + width - 1))
            row += height
            col += width
            if idx < len(sizes) - 1:
                row += row_gaps[idx]
                col += col_gaps[idx]
        return total_rows, total_cols, bboxes

    return None


def _render_expected(mask: np.ndarray, symmetric: bool) -> np.ndarray:
    out = np.full(mask.shape, Color.BLACK, dtype=int)
    out[mask] = Color.TEAL
    if symmetric:
        out[~mask] = Color.RED
    else:
        rr, cc = np.indices(mask.shape)
        out[(~mask) & (((rr + cc) % 2) == 0)] = Color.RED
    return out


def _groups_well_separated(bboxes: list[BBox]) -> bool:
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            a = bboxes[i]
            b = bboxes[j]
            if _spans_overlap(a[0], a[2], b[0], b[2]):
                return False
            if _spans_overlap(a[1], a[3], b[1], b[3]):
                return False
    return True


def _validate_candidate(inp: np.ndarray, specs: list[dict], bboxes: list[BBox], expected: np.ndarray) -> bool:
    if not _groups_well_separated(bboxes):
        return False
    if not any(spec["symmetric"] for spec in specs):
        return False
    if not any(not spec["symmetric"] for spec in specs):
        return False

    fragments = find_components(inp, bg=Color.BLACK, colors=[Color.TEAL], connectivity=4)
    merged = _merged_fragment_groups(fragments)
    intended = sorted(zip(bboxes, specs), key=lambda item: item[0])
    if len(merged) != len(intended):
        return False

    for actual_group, (bbox, spec) in zip(merged, intended):
        if tuple(int(v) for v in actual_group["bbox"]) != tuple(int(v) for v in bbox):
            return False
        r0, c0, r1, c1 = bbox
        local = inp[r0 : r1 + 1, c0 : c1 + 1]
        local_mask = local == Color.TEAL
        if _patch_fragment_count(local_mask) != int(spec["fragment_count"]):
            return False
        if not (2 <= int(spec["fragment_count"]) <= 4):
            return False
        if _is_horizontally_symmetric(local_mask) != bool(spec["symmetric"]):
            return False

    return bool(np.array_equal(main(inp), expected))


def _gen_attempt() -> np.ndarray | None:
    group_count = random.choice(GROUP_COUNT_CHOICES)
    symmetries = [True, False] + [random.choice([True, False]) for _ in range(group_count - 2)]
    random.shuffle(symmetries)

    specs = []
    sizes = []
    for symmetric in symmetries:
        height = random.randint(*RECT_HEIGHT_RANGE)
        width = random.randint(*RECT_WIDTH_RANGE)
        mask = _sample_patch(height, width, symmetric)
        if mask is None:
            return None
        specs.append(
            {
                "symmetric": symmetric,
                "mask": mask,
                "fragment_count": _patch_fragment_count(mask),
            }
        )
        sizes.append(mask.shape)

    layout = _sample_layout(sizes)
    if layout is None:
        return None

    rows, cols, bboxes = layout
    inp = np.full((rows, cols), Color.BLACK, dtype=int)
    expected = np.full((rows, cols), Color.BLACK, dtype=int)

    for spec, bbox in zip(specs, bboxes):
        r0, c0, r1, c1 = bbox
        mask = spec["mask"]
        inp[r0 : r1 + 1, c0 : c1 + 1][mask] = Color.TEAL
        expected[r0 : r1 + 1, c0 : c1 + 1] = _render_expected(mask, bool(spec["symmetric"]))

    if _validate_candidate(inp, specs, bboxes, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
