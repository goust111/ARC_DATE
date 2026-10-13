from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
ALL_NON_RED = [0, 1, 3, 4, 5, 6, 7, 8, 9]

GEN_RETRIES = 500


def _divider_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == RED)]
    assert len(cols) == 1, f"expected one full-height red divider, got {cols}"
    return int(cols[0])


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    div = _divider_col(grid)
    left = np.asarray(grid[:, :div], dtype=int)
    right = np.asarray(grid[:, div + 1 :], dtype=int)
    assert left.size and right.size, "divider must separate two non-empty panels"
    return left, right


def _translation_matches(src: np.ndarray, dst: np.ndarray) -> list[tuple[int, int]]:
    src = np.asarray(src, dtype=int)
    dst = np.asarray(dst, dtype=int)
    coords = np.argwhere(src != BLACK)
    if coords.size == 0:
        return []

    vals = src[coords[:, 0], coords[:, 1]]
    r_min = -int(coords[:, 0].min())
    r_max = int(dst.shape[0] - 1 - coords[:, 0].max())
    c_min = -int(coords[:, 1].min())
    c_max = int(dst.shape[1] - 1 - coords[:, 1].max())

    matches: list[tuple[int, int]] = []
    for dr in range(r_min, r_max + 1):
        rows = coords[:, 0] + dr
        for dc in range(c_min, c_max + 1):
            cols = coords[:, 1] + dc
            if np.array_equal(dst[rows, cols], vals):
                matches.append((int(dr), int(dc)))
    return matches


def _recover_background(fg_panel: np.ndarray, composite: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    out = np.array(composite, dtype=int, copy=True)
    dr, dc = shift
    coords = np.argwhere(fg_panel != BLACK)
    width = composite.shape[1]
    for r, c in coords:
        rr = int(r) + dr
        cc = int(c) + dc
        mirror_c = width - 1 - cc
        out[rr, cc] = int(composite[rr, mirror_c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the symmetric background by removing the translated foreground and mirroring across the composite panel's vertical centerline.

    Input structure:
    - A full-height red separator column splits the grid into two rectangular panels.
    - One panel is a composite image: a left-right symmetric background with an opaque foreground pasted on top, where black is transparent only in the standalone foreground panel.
    - The other panel shows the foreground pattern alone on black, but at a different unknown location.

    Rule:
    - Test both panel directions and identify the unique case where all non-black cells of one panel can be translated together to exact color matches inside the other panel.
    - Treat the matched cells in the composite panel as foreground-covered cells.
    - Keep all other composite cells, and replace each covered cell by the value at its left-right mirror position in the composite panel.

    Key insight:
    - The standalone panel tells you what the foreground is but not where it lands; only after aligning it can you know which cells to subtract, and the background's vertical symmetry then reveals every hidden value from the visible mirror partner.
    """
    g = np.asarray(grid, dtype=int)
    left, right = _split_panels(g)

    candidates: list[tuple[np.ndarray, np.ndarray, tuple[int, int]]] = []
    for src, dst in ((left, right), (right, left)):
        matches = _translation_matches(src, dst)
        if len(matches) == 1:
            candidates.append((src, dst, matches[0]))

    assert len(candidates) == 1, f"expected exactly one foreground/composite orientation, got {len(candidates)}"
    fg_panel, composite, shift = candidates[0]
    return _recover_background(fg_panel, composite, shift)


def _weighted_choice(values: list[int], zero_weight: int = 5, color_weight: int = 2) -> int:
    weights = [zero_weight if value == BLACK else color_weight for value in values]
    return int(random.choices(values, weights=weights, k=1)[0])


def _build_background(height: int, width: int) -> np.ndarray:
    nonzero = random.sample(PALETTE, random.randint(3, 5))
    palette = [BLACK] + nonzero
    half = (width + 1) // 2
    grid = np.zeros((height, width), dtype=int)

    for r in range(height):
        for c in range(half):
            choices = []
            if r > 0:
                choices.append(int(grid[r - 1, c]))
            if c > 0:
                choices.append(int(grid[r, c - 1]))
            if choices and random.random() < 0.58:
                color = random.choice(choices)
            else:
                color = _weighted_choice(palette)
            grid[r, c] = color
            grid[r, width - 1 - c] = color

    for _ in range(random.randint(3, 6)):
        color = random.choice(nonzero)
        fh = random.randint(1, min(4, height))
        fw = random.randint(1, min(3, half))
        r0 = random.randint(0, height - fh)
        c0 = random.randint(0, half - fw)
        grid[r0 : r0 + fh, c0 : c0 + fw] = color
        mc0 = width - (c0 + fw)
        grid[r0 : r0 + fh, mc0 : mc0 + fw] = color

    for _ in range(random.randint(1, 3)):
        fh = random.randint(1, min(3, height))
        fw = random.randint(1, min(2, half))
        r0 = random.randint(0, height - fh)
        c0 = random.randint(0, half - fw)
        grid[r0 : r0 + fh, c0 : c0 + fw] = BLACK
        mc0 = width - (c0 + fw)
        grid[r0 : r0 + fh, mc0 : mc0 + fw] = BLACK

    present = sorted(set(map(int, grid.ravel())))
    if len([c for c in present if c != BLACK]) < 3:
        return np.empty((0, 0), dtype=int)
    nonblack = int(np.count_nonzero(grid))
    area = height * width
    if not (area // 6 <= nonblack <= (4 * area) // 5):
        return np.empty((0, 0), dtype=int)
    if len({tuple(map(int, row)) for row in grid}) < max(3, height // 2):
        return np.empty((0, 0), dtype=int)
    return grid


def _side_of_col(col: int, width: int) -> int:
    if width % 2 == 1 and col == width // 2:
        return 0
    return -1 if col < width / 2 else 1


def _spanning_anchors(height: int, width: int, box_h: int, box_w: int) -> list[tuple[int, int]]:
    anchors = []
    if width % 2 == 0:
        left_need = width // 2 - 1
        right_need = width // 2
    else:
        left_need = width // 2 - 1
        right_need = width // 2 + 1

    for r0 in range(height - box_h + 1):
        for c0 in range(width - box_w + 1):
            c1 = c0 + box_w - 1
            if c0 <= left_need and c1 >= right_need:
                anchors.append((r0, c0))
    return anchors


def _random_mask(
    height: int,
    width: int,
    box_h: int,
    box_w: int,
    anchor: tuple[int, int],
) -> list[tuple[int, int]] | None:
    r0, c0 = anchor
    allowed: list[tuple[int, int]] = []
    left_cells: list[tuple[int, int]] = []
    right_cells: list[tuple[int, int]] = []
    for lr in range(box_h):
        for lc in range(box_w):
            gc = c0 + lc
            if width % 2 == 1 and gc == width // 2:
                continue
            pos = (lr, lc)
            allowed.append(pos)
            side = _side_of_col(gc, width)
            if side < 0:
                left_cells.append(pos)
            elif side > 0:
                right_cells.append(pos)

    if len(left_cells) < 2 or len(right_cells) < 2:
        return None

    for _ in range(80):
        target = random.randint(7, min(len(allowed), max(9, (box_h * box_w * 2) // 3)))
        chosen: set[tuple[int, int]] = set()

        seeds = random.sample(left_cells, 2) + random.sample(right_cells, 2)
        ok = True
        for lr, lc in seeds:
            gc = c0 + lc
            mlc = (width - 1 - gc) - c0
            if (lr, mlc) in chosen:
                ok = False
                break
            chosen.add((lr, lc))
        if not ok:
            continue

        shuffled = allowed[:]
        random.shuffle(shuffled)
        for lr, lc in shuffled:
            if len(chosen) >= target:
                break
            gc = c0 + lc
            mlc = (width - 1 - gc) - c0
            if (lr, mlc) in chosen:
                continue
            chosen.add((lr, lc))

        if len(chosen) < target:
            continue

        left_count = sum(1 for _lr, lc in chosen if _side_of_col(c0 + lc, width) < 0)
        right_count = sum(1 for _lr, lc in chosen if _side_of_col(c0 + lc, width) > 0)
        if left_count >= 2 and right_count >= 2:
            return sorted(chosen)
    return None


def _assign_sprite_colors(
    background: np.ndarray,
    mask_cells: list[tuple[int, int]],
    anchor: tuple[int, int],
) -> tuple[np.ndarray, int, set[tuple[int, int]], set[tuple[int, int]]] | None:
    r0, c0 = anchor
    box_h = max(r for r, _c in mask_cells) + 1
    box_w = max(c for _r, c in mask_cells) + 1
    sprite = np.zeros((box_h, box_w), dtype=int)

    same_candidates = []
    left_cells = []
    right_cells = []
    for lr, lc in mask_cells:
        br = r0 + lr
        bc = c0 + lc
        bg_color = int(background[br, bc])
        side = _side_of_col(bc, background.shape[1])
        if bg_color != BLACK:
            same_candidates.append((lr, lc))
        if side < 0:
            left_cells.append((lr, lc))
        elif side > 0:
            right_cells.append((lr, lc))

    if len(same_candidates) < 1 or not left_cells or not right_cells:
        return None

    for _ in range(80):
        max_same = min(3, len(same_candidates), len(mask_cells) - 2)
        same_count = random.randint(1, max_same)
        same_cells = set(random.sample(same_candidates, same_count))
        changed_left = [cell for cell in left_cells if cell not in same_cells]
        changed_right = [cell for cell in right_cells if cell not in same_cells]
        if changed_left and changed_right:
            break
    else:
        return None

    preferred = random.sample(PALETTE, random.randint(2, 4))
    used_colors = set()
    changed_cells: set[tuple[int, int]] = set()
    visible_mismatch_cells: set[tuple[int, int]] = set()
    for lr, lc in mask_cells:
        br = r0 + lr
        bc = c0 + lc
        bg_color = int(background[br, bc])
        if (lr, lc) in same_cells:
            sprite[lr, lc] = bg_color
            used_colors.add(bg_color)
            continue

        choices = [c for c in preferred if c != bg_color]
        if not choices:
            choices = [c for c in PALETTE if c != bg_color]
        color = int(random.choice(choices))
        sprite[lr, lc] = color
        used_colors.add(color)
        changed_cells.add((lr, lc))
        visible_mismatch_cells.add((lr, lc))

    if len(used_colors) < 2:
        return None

    left_mismatch = [cell for cell in visible_mismatch_cells if _side_of_col(c0 + cell[1], background.shape[1]) < 0]
    right_mismatch = [cell for cell in visible_mismatch_cells if _side_of_col(c0 + cell[1], background.shape[1]) > 0]
    if not left_mismatch or not right_mismatch:
        return None

    return sprite, len(same_cells), changed_cells, visible_mismatch_cells


def _stamp(grid: np.ndarray, sprite: np.ndarray, anchor: tuple[int, int]) -> None:
    r0, c0 = anchor
    mask = sprite != BLACK
    h, w = sprite.shape
    grid[r0 : r0 + h, c0 : c0 + w][mask] = sprite[mask]


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    fg_panel: np.ndarray,
    composite: np.ndarray,
    expected_shift: tuple[int, int],
) -> bool:
    left, right = _split_panels(inp)
    lr = _translation_matches(left, right)
    rl = _translation_matches(right, left)
    one_way = (len(lr) == 1 and len(rl) == 0) or (len(rl) == 1 and len(lr) == 0)
    if not one_way:
        return False
    if not np.array_equal(main(inp), expected):
        return False
    matches = _translation_matches(fg_panel, composite)
    return matches == [expected_shift]


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(8, 14)
    width = random.randint(8, 14)

    background = _build_background(height, width)
    if background.size == 0:
        return None

    max_box_h = min(7, height)
    max_box_w = min(8, width)
    box_h = random.randint(3, max_box_h)
    box_w = random.randint(4, max_box_w)
    anchors = _spanning_anchors(height, width, box_h, box_w)
    if not anchors:
        return None

    final_anchor = random.choice(anchors)
    all_anchors = [(r, c) for r in range(height - box_h + 1) for c in range(width - box_w + 1)]
    other_anchors = [anchor for anchor in all_anchors if anchor != final_anchor]
    if not other_anchors:
        return None
    other_anchor = random.choice(other_anchors)

    mask_cells = _random_mask(height, width, box_h, box_w, final_anchor)
    if not mask_cells:
        return None

    assigned = _assign_sprite_colors(background, mask_cells, final_anchor)
    if assigned is None:
        return None
    sprite, same_count, _changed_cells, visible_mismatch_cells = assigned
    if same_count < 1:
        return None

    final_nonblack_bg = int(np.count_nonzero(background)) - sum(
        1 for lr, lc in mask_cells if background[final_anchor[0] + lr, final_anchor[1] + lc] != BLACK
    )
    if final_nonblack_bg < 6:
        return None

    composite = np.array(background, copy=True)
    _stamp(composite, sprite, final_anchor)

    fg_panel = np.zeros((height, width), dtype=int)
    _stamp(fg_panel, sprite, other_anchor)

    expected_shift = (final_anchor[0] - other_anchor[0], final_anchor[1] - other_anchor[1])
    if expected_shift == (0, 0):
        return None

    separator = np.full((height, 1), RED, dtype=int)
    if random.random() < 0.5:
        inp = np.concatenate([composite, separator, fg_panel], axis=1)
    else:
        inp = np.concatenate([fg_panel, separator, composite], axis=1)

    if inp.shape[1] > 30 or inp.shape[0] > 30:
        return None

    if not _semantic_ok(inp, background, fg_panel, composite, expected_shift):
        return None

    left_mismatch = 0
    right_mismatch = 0
    for lr, lc in visible_mismatch_cells:
        side = _side_of_col(final_anchor[1] + lc, width)
        if side < 0:
            left_mismatch += 1
        elif side > 0:
            right_mismatch += 1
    if left_mismatch == 0 or right_mismatch == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
