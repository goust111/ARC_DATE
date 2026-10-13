from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
BROWN = 9

ROTATION_TURNS_CW = {
    RED: 0,
    BLUE: 1,
    GREEN: 2,
    YELLOW: 3,
}

FLIP_FLAGS = {
    MAGENTA: False,
    ORANGE: True,
}

PATTERN_COLORS = [BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE, BROWN]
TRANSFORM_CODES = [(rot, flip) for rot in ROTATION_TURNS_CW for flip in FLIP_FLAGS]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 300


def _transform_patch(patch: np.ndarray, turns_cw: int, flip_horizontal: bool) -> np.ndarray:
    out = np.rot90(patch, -int(turns_cw))
    if flip_horizontal:
        out = np.fliplr(out)
    return out


def _find_template_patch(grid: np.ndarray) -> np.ndarray:
    grey_pixels = np.argwhere(grid == GREY)
    assert grey_pixels.size > 0, "expected a grey template frame"

    r0, c0 = map(int, grey_pixels.min(axis=0))
    r1, c1 = map(int, grey_pixels.max(axis=0))
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "template frame must enclose an interior"

    box = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(box[0, :] == GREY)
    assert np.all(box[-1, :] == GREY)
    assert np.all(box[:, 0] == GREY)
    assert np.all(box[:, -1] == GREY)
    assert not np.any(box[1:-1, 1:-1] == GREY)
    return box[1:-1, 1:-1].copy()


def _parse_instruction_row(row: np.ndarray) -> list[tuple[int, int, bool]]:
    instructions: list[tuple[int, int, bool]] = []
    c = 0
    w = int(row.shape[0])
    while c < w:
        if int(row[c]) == BLACK:
            c += 1
            continue

        assert c + 2 < w, "instruction groups must contain rotation, flip, and azure marker"
        rot_color = int(row[c])
        flip_color = int(row[c + 1])
        azure_color = int(row[c + 2])
        assert rot_color in ROTATION_TURNS_CW
        assert flip_color in FLIP_FLAGS
        assert azure_color == AZURE

        instructions.append((c + 2, ROTATION_TURNS_CW[rot_color], FLIP_FLAGS[flip_color]))
        c += 3

    assert instructions, "expected at least one instruction group"
    return instructions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the grey-framed template and the bottom instruction strip, then stamp transformed
    copies of the template into a black output canvas whose width matches the input.
    Later instruction groups overwrite earlier ones.
    """
    g = np.asarray(grid, dtype=int)
    template = _find_template_patch(g)
    instructions = _parse_instruction_row(g[-1, :])

    out_h = int(template.shape[0])
    out_w = int(g.shape[1])
    out = np.full((out_h, out_w), BLACK, dtype=int)

    for left_col, turns_cw, flip_horizontal in instructions:
        placed = _transform_patch(template, turns_cw, flip_horizontal)
        ph, pw = map(int, placed.shape)
        if left_col >= out_w:
            continue
        clip_h = min(out_h, ph)
        clip_w = min(out_w - left_col, pw)
        if clip_h <= 0 or clip_w <= 0:
            continue
        out[:clip_h, left_col : left_col + clip_w] = placed[:clip_h, :clip_w]

    return out


def _all_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for turns_cw in range(4):
        rotated = np.rot90(mask, -turns_cw)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _is_fully_asymmetric(mask: np.ndarray) -> bool:
    signatures = {(variant.shape, variant.tobytes()) for variant in _all_variants(mask)}
    return len(signatures) == 8


def _random_connected_mask(size: int) -> np.ndarray:
    min_area = max(4, size)
    max_area = min(size * size - 2, (size * size * 2) // 3 + 1)

    for _ in range(400):
        target_area = random.randint(min_area, max_area)
        mask = np.zeros((size, size), dtype=bool)
        seed = (random.randrange(size), random.randrange(size))
        mask[seed] = True
        frontier = set()
        frontier.add(seed)

        while int(mask.sum()) < target_area and frontier:
            base_r, base_c = random.choice(tuple(frontier))
            candidates = []
            for dr, dc in NEIGHBORS4:
                nr = base_r + dr
                nc = base_c + dc
                if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                    candidates.append((nr, nc))
            if not candidates:
                frontier.remove((base_r, base_c))
                continue
            nr, nc = random.choice(candidates)
            mask[nr, nc] = True
            frontier.add((nr, nc))

        if int(mask.sum()) != target_area:
            continue

        coords = np.argwhere(mask)
        rows = coords[:, 0]
        cols = coords[:, 1]
        if int(rows.max()) == int(rows.min()) or int(cols.max()) == int(cols.min()):
            continue
        if not _is_fully_asymmetric(mask):
            continue
        return mask

    raise RuntimeError("failed to generate an asymmetric template mask")


def _stamp_full_patch(out: np.ndarray, patch: np.ndarray, left_col: int) -> None:
    ph, pw = map(int, patch.shape)
    oh, ow = map(int, out.shape)
    if left_col >= ow:
        return
    clip_h = min(oh, ph)
    clip_w = min(ow - left_col, pw)
    if clip_h <= 0 or clip_w <= 0:
        return
    out[:clip_h, left_col : left_col + clip_w] = patch[:clip_h, :clip_w]


def _build_expected_output(template: np.ndarray, width: int, instructions: list[tuple[int, int]]) -> np.ndarray:
    out = np.full((template.shape[0], width), BLACK, dtype=int)
    for start_col, (rot_color, flip_color) in instructions:
        patch = _transform_patch(template, ROTATION_TURNS_CW[int(rot_color)], FLIP_FLAGS[int(flip_color)])
        _stamp_full_patch(out, patch, start_col + 2)
    return out


def _draw_template_frame(grid: np.ndarray, top: int, left: int, template: np.ndarray) -> None:
    size_h, size_w = map(int, template.shape)
    bottom = top + size_h + 1
    right = left + size_w + 1
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    grid[top + 1 : bottom, left + 1 : right] = template


def _sample_instruction_codes(count: int) -> list[tuple[int, int]]:
    base = random.sample(TRANSFORM_CODES, k=min(count, len(TRANSFORM_CODES)))
    while len(base) < count:
        base.append(random.choice(TRANSFORM_CODES))
    random.shuffle(base)
    return base


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    size = random.choice([3, 4, 5, 5])
    template_mask = _random_connected_mask(size)
    template_color = random.choice(PATTERN_COLORS)
    template = np.full((size, size), BLACK, dtype=int)
    template[template_mask] = int(template_color)

    width = random.randint(max(16, size + 10), 28)
    max_groups = 1 + (width - size - 2) // 4
    if max_groups < 2:
        raise RuntimeError("grid width too small for two instruction groups")
    group_count = random.randint(2, min(5, max_groups))
    lead_max = width - size - 2 - 4 * (group_count - 1)
    lead = random.randint(0, lead_max)
    starts = [lead + 4 * i for i in range(group_count)]
    codes = _sample_instruction_codes(group_count)

    frame_top = random.randint(0, 1)
    frame_left = random.randint(0, 1)
    min_height = frame_top + size + 4
    height = random.randint(max(min_height, 8), max(min_height + 4, 12))

    grid = np.full((height, width), BLACK, dtype=int)
    _draw_template_frame(grid, frame_top, frame_left, template)

    for start_col, (rot_color, flip_color) in zip(starts, codes):
        grid[-1, start_col] = int(rot_color)
        grid[-1, start_col + 1] = int(flip_color)
        grid[-1, start_col + 2] = AZURE

    expected = _build_expected_output(template, width, list(zip(starts, codes)))
    return grid, expected


def generate_input() -> np.ndarray:
    """
    Generate a black grid with a grey-framed square template near the top-left and a bottom
    instruction strip. The template shape is asymmetric under all non-trivial dihedral transforms.
    """
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid, expected = _generate_case()
        except RuntimeError:
            continue
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
