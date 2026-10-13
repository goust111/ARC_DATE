from __future__ import annotations

import random
from typing import Iterable

import numpy as np


ALL_COLORS = tuple(range(10))
GEN_RETRIES = 400
PAIR_SHAPE_COUNT = (6, 10)
SHAPE_SIDE_RANGE = (2, 6)
GRID_SIDE_RANGE = (15, 24)
OCCLUDER_SIDE_RANGE = (3, 7)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _box_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded_box(
    box: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _component_coords(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _solid_rectangle_candidates(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    candidates = []
    for color in map(int, np.unique(grid)):
        mask = grid == color
        box = _bbox(mask)
        if box is None:
            continue
        if int(mask.sum()) != _box_area(box):
            continue
        if np.all(_crop(mask, box)):
            candidates.append((color, box))
    return candidates


def _inside_box(r: int, c: int, box: tuple[int, int, int, int]) -> bool:
    return box[0] <= r <= box[2] and box[1] <= c <= box[3]


def _analyze_visible_pairs(
    grid: np.ndarray, occ_box: tuple[int, int, int, int]
) -> tuple[set[tuple[int, int]], int, set[int]]:
    h, w = grid.shape
    mismatches: set[tuple[int, int]] = set()
    mismatch_count = 0
    equal_colors: set[int] = set()
    for r in range(h):
        for c in range(w):
            rr, cc = h - 1 - r, w - 1 - c
            if (r, c) > (rr, cc):
                continue
            if _inside_box(r, c, occ_box) or _inside_box(rr, cc, occ_box):
                continue
            a = int(grid[r, c])
            b = int(grid[rr, cc])
            if a == b:
                equal_colors.add(a)
            else:
                mismatches.add(tuple(sorted((a, b))))
                mismatch_count += 1
    return mismatches, mismatch_count, equal_colors


def _infer_occluder_and_swap_pair(
    grid: np.ndarray,
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    candidates = _solid_rectangle_candidates(grid)
    valid = []
    for occ_color, occ_box in candidates:
        mismatches, mismatch_count, _equal_colors = _analyze_visible_pairs(grid, occ_box)
        if mismatch_count == 0 or len(mismatches) != 1:
            continue
        pair = next(iter(mismatches))
        if occ_color in pair:
            continue

        okay = True
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                rr, cc = h - 1 - r, w - 1 - c
                if (r, c) > (rr, cc):
                    continue
                if _inside_box(r, c, occ_box) or _inside_box(rr, cc, occ_box):
                    continue
                a = int(grid[r, c])
                b = int(grid[rr, cc])
                if a == b:
                    if a in pair:
                        okay = False
                        break
                elif tuple(sorted((a, b))) != pair:
                    okay = False
                    break
            if not okay:
                break
        if okay:
            valid.append((occ_box, pair))

    assert len(valid) == 1, f"expected exactly one solid occluder, got {valid}"
    return valid[0]


def _swap_colors(grid: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
    a, b = pair
    out = grid.copy()
    a_mask = grid == a
    b_mask = grid == b
    out[a_mask] = b
    out[b_mask] = a
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the hidden patch by rotating the scene 180 degrees and swapping the unique complementary color pair.

    Input structure:
    - A dominant background and several colored motifs form one large pattern.
    - Exactly one solid rectangular occluder uses a color that does not appear in the underlying pattern.
    - Away from the occluder, 180-degree-opposite cells are either the same neutral color or one of two complementary colors.

    Rule:
    - Locate the unique solid rectangle and treat it as the occluder bbox.
    - Compare visible 180-degree-opposite cell pairs to infer the only color pair that always appears as a mismatch.
    - Rotate the whole grid by 180 degrees, swap that complementary color pair everywhere, and crop the occluder bbox.

    Key insight:
    - The scene is not plain rotational symmetry: the hidden content is supplied by the opposite side only after composing the half-turn with one specific two-color swap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    occ_box, pair = _infer_occluder_and_swap_pair(g)
    restored = _swap_colors(np.rot90(g, 2), pair)
    out = _crop(restored, occ_box)
    occ_color = int(g[occ_box[0], occ_box[1]])
    assert not np.any(out == occ_color)
    return out


def _random_shape_mask(h: int, w: int) -> np.ndarray:
    for _ in range(120):
        density = random.uniform(0.38, 0.78)
        mask = np.random.rand(h, w) < density
        if int(mask.sum()) < 3 or bool(mask.all()):
            continue
        comps = _component_coords(mask, connectivity=8)
        if not comps:
            continue
        largest = max(comps, key=len)
        blob = np.zeros((h, w), dtype=bool)
        blob[largest[:, 0], largest[:, 1]] = True
        if int(blob.sum()) < 3 or bool(blob.all()):
            continue
        if _bbox(blob) != (0, 0, h - 1, w - 1):
            continue
        return blob
    raise RuntimeError("failed to generate shape mask")


def _paint_pair(
    grid: np.ndarray,
    shape_mask: np.ndarray,
    top_left: tuple[int, int],
    colors: tuple[int, int],
    same_color: bool,
) -> None:
    h, w = shape_mask.shape
    r, c = top_left
    rr, cc = grid.shape[0] - h - r, grid.shape[1] - w - c
    left_mask = shape_mask
    right_mask = np.rot90(shape_mask, 2)

    if same_color:
        left_color = right_color = colors[0]
    else:
        left_color, right_color = colors

    grid[r : r + h, c : c + w][left_mask] = left_color
    grid[rr : rr + h, cc : cc + w][right_mask] = right_color


def _try_place_shape_pair(
    blocked: np.ndarray, h: int, w: int
) -> tuple[int, int] | None:
    gh, gw = blocked.shape
    choices = []
    for r in range(gh - h + 1):
        for c in range(gw - w + 1):
            rr, cc = gh - h - r, gw - w - c
            box1 = (r, c, r + h - 1, c + w - 1)
            box2 = (rr, cc, rr + h - 1, cc + w - 1)
            if _boxes_overlap(box1, box2):
                continue
            exp1 = _expanded_box(box1, blocked.shape, pad=1)
            exp2 = _expanded_box(box2, blocked.shape, pad=1)
            if np.any(_crop(blocked, exp1)) or np.any(_crop(blocked, exp2)):
                continue
            choices.append((r, c))
    return random.choice(choices) if choices else None


def _mark_blocked(blocked: np.ndarray, boxes: Iterable[tuple[int, int, int, int]]) -> None:
    for box in boxes:
        r0, c0, r1, c1 = _expanded_box(box, blocked.shape, pad=1)
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _build_pattern() -> tuple[np.ndarray, tuple[int, int], list[int], int]:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    bg, comp_a, comp_b, occ_color = random.sample(ALL_COLORS, 4)
    neutral_count = random.randint(2, 3)
    neutral_colors = random.sample([c for c in ALL_COLORS if c not in {bg, comp_a, comp_b, occ_color}], neutral_count)

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    target_comp = random.randint(3, 5)
    target_neutral = random.randint(2, 4)
    extra_budget = random.randint(0, 2)

    placed_comp = 0
    placed_neutral = 0
    attempts = 0
    while attempts < 80 and (
        placed_comp < target_comp
        or placed_neutral < target_neutral
        or extra_budget > 0
    ):
        attempts += 1
        if placed_comp < target_comp and placed_neutral < target_neutral:
            kind = random.choice(["comp", "neutral"])
        elif placed_comp < target_comp:
            kind = "comp"
        elif placed_neutral < target_neutral:
            kind = "neutral"
        elif extra_budget > 0:
            kind = random.choice(["comp", "neutral"])
        else:
            break

        side_h = random.randint(*SHAPE_SIDE_RANGE)
        side_w = random.randint(*SHAPE_SIDE_RANGE)
        pos = _try_place_shape_pair(blocked, side_h, side_w)
        if pos is None:
            continue
        shape_mask = _random_shape_mask(side_h, side_w)
        r, c = pos
        rr, cc = h - side_h - r, w - side_w - c
        box1 = (r, c, r + side_h - 1, c + side_w - 1)
        box2 = (rr, cc, rr + side_h - 1, cc + side_w - 1)

        if kind == "comp":
            if random.choice([False, True]):
                _paint_pair(grid, shape_mask, pos, (comp_a, comp_b), same_color=False)
            else:
                _paint_pair(grid, shape_mask, pos, (comp_b, comp_a), same_color=False)
            placed_comp += 1
        else:
            neutral = random.choice(neutral_colors)
            _paint_pair(grid, shape_mask, pos, (neutral, neutral), same_color=True)
            placed_neutral += 1

        _mark_blocked(blocked, (box1, box2))
        if placed_comp >= target_comp and placed_neutral >= target_neutral and extra_budget > 0:
            extra_budget -= 1

    if placed_comp < 3 or placed_neutral < 2:
        raise RuntimeError("insufficient branch coverage in generated pattern")
    return grid, (comp_a, comp_b), neutral_colors, occ_color


def _choose_occluder_box(
    pattern: np.ndarray, pair: tuple[int, int], neutral_colors: list[int], occ_color: int
) -> tuple[int, int, int, int] | None:
    h, w = pattern.shape
    comp_set = set(pair)
    neutral_set = set(neutral_colors)
    min_oh = OCCLUDER_SIDE_RANGE[0]
    max_oh = min(OCCLUDER_SIDE_RANGE[1], h - 2)
    min_ow = OCCLUDER_SIDE_RANGE[0]
    max_ow = min(OCCLUDER_SIDE_RANGE[1], w - 2)
    if min_oh > max_oh or min_ow > max_ow:
        return None

    comp_cells = np.argwhere(np.isin(pattern, list(comp_set)))
    if comp_cells.size == 0:
        return None

    for _ in range(220):
        oh = random.randint(min_oh, max_oh)
        ow = random.randint(min_ow, max_ow)
        anchor_r, anchor_c = map(int, comp_cells[random.randrange(len(comp_cells))])
        r_lo = max(0, anchor_r - oh + 1)
        r_hi = min(anchor_r, h - oh)
        c_lo = max(0, anchor_c - ow + 1)
        c_hi = min(anchor_c, w - ow)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        r = random.randint(r_lo, r_hi)
        c = random.randint(c_lo, c_hi)
        rr, cc = h - oh - r, w - ow - c
        box = (r, c, r + oh - 1, c + ow - 1)
        opp = (rr, cc, rr + oh - 1, cc + ow - 1)
        if _boxes_overlap(box, opp):
            continue

        hidden = _crop(pattern, box)
        visible = _crop(pattern, opp)
        hidden_colors = set(map(int, np.unique(hidden)))
        if hidden_colors == {int(pattern[0, 0])}:
            continue
        if not hidden_colors & comp_set:
            continue
        if len(hidden_colors) < 2:
            continue
        if not hidden_colors & neutral_set:
            continue
        if np.array_equal(hidden, visible):
            continue

        inp = pattern.copy()
        inp[r : r + oh, c : c + ow] = occ_color
        try:
            occ_box, inferred_pair = _infer_occluder_and_swap_pair(inp)
        except AssertionError:
            continue
        if occ_box != box or set(inferred_pair) != comp_set:
            continue
        out = main(inp)
        if not np.array_equal(out, hidden):
            continue

        mismatches, mismatch_count, equal_colors = _analyze_visible_pairs(inp, box)
        if mismatch_count < 4 or len(mismatches) != 1:
            continue
        if not (equal_colors & neutral_set):
            continue
        return box

    return None


def _generate_attempt() -> np.ndarray | None:
    try:
        pattern, pair, neutral_colors, occ_color = _build_pattern()
    except RuntimeError:
        return None

    box = _choose_occluder_box(pattern, pair, neutral_colors, occ_color)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    inp = pattern.copy()
    inp[r0 : r1 + 1, c0 : c1 + 1] = occ_color

    if len(_solid_rectangle_candidates(inp)) != 1:
        return None

    out = main(inp)
    expected = _crop(pattern, box)
    if np.array_equal(out, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid anti-symmetric occlusion input")
