from __future__ import annotations

import random

import numpy as np


GEN_RETRIES = 400
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
INSTRUCTION_CASES = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _anti_diagonal_flip(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, 2).T.copy()


def _apply_transform(arr: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    if offset in {(0, 0), (-1, 0)}:
        return arr.copy()
    if offset == (0, 1):
        return np.rot90(arr, -1).copy()
    if offset == (1, 0):
        return np.rot90(arr, 2).copy()
    if offset == (0, -1):
        return np.rot90(arr, 1).copy()
    if offset == (-1, 1):
        return arr.T.copy()
    if offset == (-1, -1):
        return np.fliplr(arr).copy()
    if offset == (1, -1):
        return _anti_diagonal_flip(arr)
    if offset == (1, 1):
        return np.flipud(arr).copy()
    raise AssertionError(f"unsupported instruction offset: {offset}")


def _find_divider(grid: np.ndarray, bg: int) -> int:
    full_cols = [
        c
        for c in range(grid.shape[1])
        if int(grid[0, c]) != int(bg) and bool(np.all(grid[:, c] == grid[0, c]))
    ]
    assert len(full_cols) == 1, "expected one full-height divider column"
    return int(full_cols[0])


def _extract_template_patch(grid: np.ndarray, divider_col: int, bg: int) -> np.ndarray:
    left = grid[:, :divider_col]
    mask = left != int(bg)
    assert bool(mask.any()), "template region is empty"
    colors = np.unique(left[mask])
    assert len(colors) == 1, "template must use one foreground color"
    return _crop_to_mask(left, mask)


def _instruction_offset(grid: np.ndarray, divider_col: int, bg: int) -> tuple[int, int]:
    right = grid[:, divider_col + 1 :]
    mask = right != int(bg)
    assert bool(mask.any()), "instruction region is empty"
    sub = _crop_to_mask(right, mask)

    colors, counts = np.unique(sub[sub != int(bg)], return_counts=True)
    frame_color = int(colors[int(np.argmax(counts))])

    full_rows = [r for r in range(sub.shape[0]) if bool(np.all(sub[r, :] == frame_color))]
    full_cols = [c for c in range(sub.shape[1]) if bool(np.all(sub[:, c] == frame_color))]
    assert len(full_rows) == 4 and len(full_cols) == 4, "instruction frame must be 3x3"

    cell_rows = [(full_rows[i] + full_rows[i + 1]) // 2 for i in range(3)]
    cell_cols = [(full_cols[i] + full_cols[i + 1]) // 2 for i in range(3)]

    marker = np.argwhere((sub != int(bg)) & (sub != frame_color))
    assert marker.shape[0] == 1, "expected exactly one instruction marker"
    mr, mc = map(int, marker[0])
    assert mr in cell_rows and mc in cell_cols, "marker must lie in one frame cell"

    return cell_rows.index(mr) - 1, cell_cols.index(mc) - 1


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[str, ...]]:
    return mask.shape, tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _dihedral_keys(mask: np.ndarray) -> set[tuple[tuple[int, int], tuple[str, ...]]]:
    keys: set[tuple[tuple[int, int], tuple[str, ...]]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        keys.add(_mask_key(rotated))
        keys.add(_mask_key(np.fliplr(rotated)))
    return keys


def _random_connected_mask() -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        max_cells = h * w - 1
        if max_cells < 3:
            continue
        target = random.randint(3, max_cells)

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = {
            (start[0] + dr, start[1] + dc)
            for dr, dc in NEIGHBORS4
            if 0 <= start[0] + dr < h and 0 <= start[1] + dc < w
        }

        while int(mask.sum()) < target and frontier:
            r, c = random.choice(tuple(frontier))
            frontier.remove((r, c))
            if mask[r, c]:
                continue
            mask[r, c] = True
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        tight = _crop_to_mask(mask.astype(int), mask).astype(bool)
        if tight.shape[0] < 2 or tight.shape[1] < 2:
            continue
        if int(tight.sum()) == tight.size:
            continue
        if len(_dihedral_keys(tight)) != 8:
            continue
        return tight
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform a left-side template according to a compass-like marker inside a right-side 3x3 instruction frame.

    Input structure:
    - The dominant color is the background.
    - One full-height non-background column acts as the vertical separator between a template area on the left and an instruction area on the right.
    - The left area contains one small single-color template pattern, surrounded by background and tight-fit recoverable by its bounding box.
    - The right area contains one grey 3x3 grid frame with exactly one non-grey marker pixel placed in one of the nine cells.

    Rule:
    - Crop the left template to its minimal bounding box.
    - Read the marker position relative to the instruction frame center.
    - Apply the corresponding transform to the cropped template:
      up or center = identity; right = 90 degree clockwise; down = 180 degree; left = 90 degree counterclockwise;
      upper-right = main-diagonal flip; upper-left = horizontal flip as a left-right mirror;
      lower-left = anti-diagonal flip; lower-right = vertical flip as a top-bottom mirror.
    - Return the transformed template using its own minimal bounding box as the full output grid.

    Key insight:
    - The right-hand frame is not a drawable object but a spatial codebook: the unique marker cell encodes which dihedral transform must be applied to the asymmetric left-hand template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    divider_col = _find_divider(g, bg)
    template = _extract_template_patch(g, divider_col, bg)
    offset = _instruction_offset(g, divider_col, bg)
    transformed = _apply_transform(template, offset)

    mask = transformed != int(bg)
    assert bool(mask.any()), "transformed template must contain foreground pixels"
    return _crop_to_mask(transformed, mask)


def _build_input(
    bg: int,
    frame_color: int,
    template_color: int,
    template_mask: np.ndarray,
    instruction: tuple[int, int],
) -> np.ndarray:
    template_h, template_w = template_mask.shape

    left_w = random.randint(template_w + 3, 14)
    right_w = random.randint(9, min(14, 29 - left_w))
    height = random.randint(max(template_h + 2, 9), 22)
    width = left_w + 1 + right_w

    grid = np.full((height, width), bg, dtype=int)
    divider_col = left_w
    grid[:, divider_col] = frame_color

    template_r = random.randint(1, height - template_h - 1)
    template_c = random.randint(1, left_w - template_w - 1)
    template_patch = np.where(template_mask, template_color, bg).astype(int)
    area = grid[template_r : template_r + template_h, template_c : template_c + template_w]
    area[template_mask] = template_patch[template_mask]

    frame_r = random.randint(1, height - 8)
    frame_c = divider_col + 1 + random.randint(1, right_w - 8)
    grid[frame_r : frame_r + 7 : 2, frame_c : frame_c + 7] = frame_color
    grid[frame_r : frame_r + 7, frame_c : frame_c + 7 : 2] = frame_color
    marker_r = frame_r + 1 + 2 * (instruction[0] + 1)
    marker_c = frame_c + 1 + 2 * (instruction[1] + 1)
    grid[marker_r, marker_c] = template_color

    return grid


def _gen_attempt() -> np.ndarray | None:
    template_mask = _random_connected_mask()
    if template_mask is None:
        return None

    bg, frame_color, template_color = random.sample(range(10), 3)
    instruction = random.choice(INSTRUCTION_CASES)
    grid = _build_input(bg, frame_color, template_color, template_mask, instruction)

    try:
        out = main(grid)
    except AssertionError:
        return None

    expected = _crop_to_mask(
        _apply_transform(np.where(template_mask, template_color, bg).astype(int), instruction),
        _apply_transform(template_mask.astype(int), instruction) != 0,
    )
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
