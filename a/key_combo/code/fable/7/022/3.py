from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste, random_sprite


BG = Color.BLACK
CLUE = Color.GRAY
SEPARATOR = Color.MAROON
SPRITE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL]

MAX_SIDE = 30
MAX_GENERATION_ATTEMPTS = 400


@dataclass(frozen=True)
class Candidate:
    r: int
    c: int
    row_mask: int
    col_mask: int
    occ_mask: int


def _interval_mask(start: int, length: int) -> int:
    return ((1 << int(length)) - 1) << int(start)


def _vector_to_mask(vec: np.ndarray) -> int:
    mask = 0
    for i, flag in enumerate(np.asarray(vec, dtype=bool)):
        if bool(flag):
            mask |= 1 << i
    return mask


def _mask_to_segments(mask: int, length: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    i = 0
    while i < length:
        if not ((mask >> i) & 1):
            i += 1
            continue
        j = i
        while j + 1 < length and ((mask >> (j + 1)) & 1):
            j += 1
        segments.append((i, j))
        i = j + 1
    return segments


def _bbox_masks_from_solution(
    placements: list[tuple[np.ndarray, Candidate]], shape: tuple[int, int]
) -> tuple[int, int]:
    height, width = map(int, shape)
    row_mask = 0
    col_mask = 0
    for sprite, cand in placements:
        h, w = sprite.shape
        row_mask |= _interval_mask(cand.r, h)
        col_mask |= _interval_mask(cand.c, w)
    row_mask &= (1 << height) - 1
    col_mask &= (1 << width) - 1
    return row_mask, col_mask


def _paint_solution(shape: tuple[int, int], placements: list[tuple[np.ndarray, Candidate]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for sprite, cand in placements:
        paste(out, sprite, cand.r, cand.c, transparent=BG)
    return out


def _find_separator_row(grid: np.ndarray) -> int:
    rows = []
    for r in range(grid.shape[0]):
        vals = np.unique(grid[r])
        if vals.size == 1 and int(vals[0]) == SEPARATOR:
            rows.append(r)
    assert len(rows) == 1, "expected exactly one full-width separator row"
    return int(rows[0])


def _extract_sprites(legend: np.ndarray) -> list[np.ndarray]:
    comps = find_components(legend, bg=BG, connectivity=4, monochromatic=True)
    assert comps, "expected at least one legend sprite"

    sprites: list[tuple[int, np.ndarray]] = []
    seen_colors: set[int] = set()
    for comp in comps:
        color = int(comp["color"])
        assert color not in {BG, CLUE, SEPARATOR}, "legend uses reserved colors"
        assert color not in seen_colors, "legend sprite colors must be unique"
        seen_colors.add(color)
        sprite = crop_bbox(legend, comp["bbox"])
        assert np.any(sprite == color), "cropped sprite must contain foreground"
        sprites.append((color, sprite))

    sprites.sort(key=lambda pair: pair[0])
    return [sprite for _color, sprite in sprites]


def _parse_input(grid: np.ndarray) -> tuple[tuple[int, int], list[np.ndarray], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_r = _find_separator_row(g)
    assert 0 < sep_r < g.shape[0] - 1, "separator must split scene and legend"
    scene = g[:sep_r].copy()
    legend = g[sep_r + 1 :].copy()
    assert scene.size > 0 and legend.size > 0, "scene and legend must both be present"

    assert set(map(int, np.unique(scene))) <= {BG, CLUE}, "scene may only contain black and gray clues"
    interior = scene.copy()
    interior[0, :] = BG
    interior[:, 0] = BG
    assert np.all(interior == BG), "only the top row and left column may contain clues"

    row_mask = _vector_to_mask(scene[:, 0] == CLUE)
    col_mask = _vector_to_mask(scene[0, :] == CLUE)
    assert row_mask != 0 and col_mask != 0, "both clue axes must be non-empty"

    sprites = _extract_sprites(legend)
    return scene.shape, sprites, row_mask, col_mask


def _sprite_coords(sprite: np.ndarray) -> np.ndarray:
    return np.argwhere(np.asarray(sprite) != BG)


def _enumerate_candidates(
    scene_shape: tuple[int, int],
    sprite: np.ndarray,
    row_target: int,
    col_target: int,
) -> list[Candidate]:
    height, width = map(int, scene_shape)
    h, w = map(int, sprite.shape)
    coords = _sprite_coords(sprite)
    bit_offsets = [int(rr) * width + int(cc) for rr, cc in coords]
    row_limit = height - h + 1
    col_limit = width - w + 1
    candidates: list[Candidate] = []

    for r in range(row_limit):
        row_mask = _interval_mask(r, h)
        if row_mask & ~row_target:
            continue
        row_base = r * width
        for c in range(col_limit):
            col_mask = _interval_mask(c, w)
            if col_mask & ~col_target:
                continue
            base = row_base + c
            occ_mask = 0
            for bit_offset in bit_offsets:
                occ_mask |= 1 << (base + bit_offset)
            candidates.append(Candidate(r=r, c=c, row_mask=row_mask, col_mask=col_mask, occ_mask=occ_mask))
    return candidates


def _solve_with_count(
    scene_shape: tuple[int, int],
    sprites: list[np.ndarray],
    row_target: int,
    col_target: int,
    *,
    max_solutions: int = 2,
) -> tuple[int, list[Candidate] | None]:
    raw_candidates = [_enumerate_candidates(scene_shape, sprite, row_target, col_target) for sprite in sprites]
    if any(len(cands) == 0 for cands in raw_candidates):
        return 0, None

    order = sorted(range(len(sprites)), key=lambda i: (len(raw_candidates[i]), -sprites[i].size, i))
    ordered_candidates = [raw_candidates[i] for i in order]

    suffix_row_possible = [0] * (len(sprites) + 1)
    suffix_col_possible = [0] * (len(sprites) + 1)
    for i in range(len(sprites) - 1, -1, -1):
        row_union = 0
        col_union = 0
        for cand in ordered_candidates[i]:
            row_union |= cand.row_mask
            col_union |= cand.col_mask
        suffix_row_possible[i] = suffix_row_possible[i + 1] | row_union
        suffix_col_possible[i] = suffix_col_possible[i + 1] | col_union

    solution_count = 0
    first_solution: list[Candidate] | None = None
    chosen: list[Candidate | None] = [None] * len(sprites)

    def dfs(index: int, occ_mask: int, row_cover: int, col_cover: int) -> None:
        nonlocal solution_count, first_solution
        if solution_count >= max_solutions:
            return
        if index == len(sprites):
            if row_cover == row_target and col_cover == col_target:
                solution_count += 1
                if first_solution is None:
                    first_solution = [cand for cand in chosen if cand is not None]
            return

        if (row_target & ~row_cover) & ~suffix_row_possible[index]:
            return
        if (col_target & ~col_cover) & ~suffix_col_possible[index]:
            return

        remaining_row = row_target & ~row_cover
        remaining_col = col_target & ~col_cover
        candidates = sorted(
            ordered_candidates[index],
            key=lambda cand: (
                -(((cand.row_mask & remaining_row).bit_count()) + ((cand.col_mask & remaining_col).bit_count())),
                cand.r,
                cand.c,
            ),
        )
        for cand in candidates:
            if cand.occ_mask & occ_mask:
                continue
            chosen[order[index]] = cand
            dfs(index + 1, occ_mask | cand.occ_mask, row_cover | cand.row_mask, col_cover | cand.col_mask)
            chosen[order[index]] = None
            if solution_count >= max_solutions:
                return

    dfs(0, 0, 0, 0)
    return solution_count, first_solution


def _assemble_from_solution(
    scene_shape: tuple[int, int], sprites: list[np.ndarray], solution: list[Candidate]
) -> np.ndarray:
    placements = list(zip(sprites, solution))
    return _paint_solution(scene_shape, placements)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the legend sprites into the hidden scene so their bbox shadow unions match the gray clues.

    Input structure:
    - A full-width separator row splits the grid into an upper scene area and a lower legend area.
    - In the scene area, only the top row may contain gray horizontal clue segments and only the left column may contain gray vertical clue segments; all other scene cells are black.
    - The legend area contains several non-overlapping sprites with distinct colors and fixed orientations.

    Rule:
    - Place each legend sprite into the scene exactly once without rotating, reflecting, or overlapping.
    - For every placed sprite, project its bounding-box row interval onto the scene's left column and its bounding-box column interval onto the scene's top row.
    - The unions of those row and column intervals must match the gray clue segments exactly; then clear the clue cells to black and draw the placed sprites in the scene.

    Key insight:
    - The gray marks are global unions of 1D bbox shadows, so the sprites must be assembled jointly rather than assigned to clue segments one by one.
    """
    scene_shape, sprites, row_target, col_target = _parse_input(grid)
    count, solution = _solve_with_count(scene_shape, sprites, row_target, col_target, max_solutions=2)
    assert count == 1 and solution is not None, "expected a unique valid assembly"
    return _assemble_from_solution(scene_shape, sprites, solution)


def _count_segments(mask: int, length: int) -> int:
    return len(_mask_to_segments(mask, length))


def _build_scene_input(scene_shape: tuple[int, int], row_mask: int, col_mask: int) -> np.ndarray:
    height, width = map(int, scene_shape)
    scene = np.full((height, width), BG, dtype=int)
    for r in range(height):
        if (row_mask >> r) & 1:
            scene[r, 0] = CLUE
    for c in range(width):
        if (col_mask >> c) & 1:
            scene[0, c] = CLUE
    return scene


def _layout_legend(sprites: list[np.ndarray], width: int, max_height: int) -> np.ndarray | None:
    n = len(sprites)
    indices = list(range(n))
    random.shuffle(indices)
    max_rows = min(3, n)

    for row_count in range(1, max_rows + 1):
        assignments = list(itertools.product(range(row_count), repeat=n))
        random.shuffle(assignments)
        for assign in assignments:
            if set(assign) != set(range(row_count)):
                continue
            rows: list[list[np.ndarray]] = [[] for _ in range(row_count)]
            for idx, row_id in zip(indices, assign):
                rows[row_id].append(sprites[idx])

            row_gap = 1
            col_gap = random.randint(1, 2)
            row_widths = []
            row_heights = []
            for row_sprites in rows:
                row_width = sum(sprite.shape[1] for sprite in row_sprites) + col_gap * (len(row_sprites) - 1)
                row_height = max(sprite.shape[0] for sprite in row_sprites)
                row_widths.append(int(row_width))
                row_heights.append(int(row_height))

            legend_h = sum(row_heights) + row_gap * (row_count - 1)
            if legend_h > max_height or any(row_width > width for row_width in row_widths):
                continue

            legend = np.full((legend_h, width), BG, dtype=int)
            top = 0
            for row_sprites, row_height, row_width in zip(rows, row_heights, row_widths):
                left_margin = random.randint(0, width - row_width)
                c = left_margin
                for sprite in row_sprites:
                    r = top + random.randint(0, row_height - sprite.shape[0])
                    paste(legend, sprite, r, c, transparent=BG)
                    c += sprite.shape[1] + col_gap
                top += row_height + row_gap
            return legend
    return None


def _sample_sprite(color: int) -> np.ndarray:
    for _ in range(80):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        try:
            sprite = random_sprite(
                h,
                w,
                density=random.uniform(0.45, 0.8),
                symmetry=None,
                color_palette=[color],
                connectivity=4,
                background=BG,
            )
        except AssertionError:
            continue
        if 4 <= int(np.count_nonzero(sprite)) <= h * w - 1:
            return sprite.astype(int)
    return np.full((2, 2), color, dtype=int)


def _sample_scene_shape(sprites: list[np.ndarray]) -> tuple[int, int]:
    max_h = max(sprite.shape[0] for sprite in sprites)
    max_w = max(sprite.shape[1] for sprite in sprites)
    min_h = max(max_h + 2, 8)
    max_h_scene = min(MAX_SIDE - 6, max(max_h + 4, min_h))
    min_w = max(max_w + 2, 9)
    max_w_scene = min(MAX_SIDE - 6, max(max_w + 5, min_w))
    height = random.randint(min_h, max_h_scene)
    width = random.randint(min_w, max_w_scene)
    return int(height), int(width)


def _sample_solution(scene_shape: tuple[int, int], sprites: list[np.ndarray]) -> list[Candidate] | None:
    height, width = map(int, scene_shape)
    modes = ["row_merge", "col_merge", "both_merge"]
    desired_mode = random.choice(modes)
    all_rows = (1 << height) - 1
    all_cols = (1 << width) - 1
    all_candidates = [_enumerate_candidates(scene_shape, sprite, all_rows, all_cols) for sprite in sprites]

    for _ in range(160):
        occ_mask = 0
        placements: list[Candidate] = []
        order = sorted(range(len(sprites)), key=lambda i: -int(np.count_nonzero(sprites[i])))
        ok = True

        for idx in order:
            candidate_positions = [cand for cand in all_candidates[idx] if not (cand.occ_mask & occ_mask)]
            if not candidate_positions:
                ok = False
                break
            cand = random.choice(candidate_positions)
            occ_mask |= cand.occ_mask
            placements.append(cand)

        if not ok:
            continue

        placements_by_sprite = [None] * len(sprites)
        for idx, cand in zip(order, placements):
            placements_by_sprite[idx] = cand

        if any(cand is None for cand in placements_by_sprite):
            continue

        row_mask = 0
        col_mask = 0
        for sprite, cand in zip(sprites, placements_by_sprite):
            h, w = sprite.shape
            row_mask |= _interval_mask(cand.r, h)
            col_mask |= _interval_mask(cand.c, w)

        row_segments = _count_segments(row_mask, height)
        col_segments = _count_segments(col_mask, width)
        if desired_mode == "row_merge" and row_segments >= len(sprites):
            continue
        if desired_mode == "col_merge" and col_segments >= len(sprites):
            continue
        if desired_mode == "both_merge" and not (row_segments < len(sprites) and col_segments < len(sprites)):
            continue
        return [cand for cand in placements_by_sprite if cand is not None]
    return None


def _validate_generated_sample(inp: np.ndarray, *, require_axis_ambiguity: bool = True) -> tuple[np.ndarray, dict]:
    scene_shape, sprites, row_target, col_target = _parse_input(inp)
    full_count, full_solution = _solve_with_count(scene_shape, sprites, row_target, col_target, max_solutions=2)
    assert full_count == 1 and full_solution is not None, "generated sample must have a unique solution"
    out = _assemble_from_solution(scene_shape, sprites, full_solution)

    placements = list(zip(sprites, full_solution))
    row_mask, col_mask = _bbox_masks_from_solution(placements, scene_shape)
    assert row_mask == row_target and col_mask == col_target, "solution must match the clue unions exactly"

    row_segments = _count_segments(row_target, scene_shape[0])
    col_segments = _count_segments(col_target, scene_shape[1])
    assert row_segments < len(sprites) or col_segments < len(sprites), "at least one clue axis must merge shadows"

    info = {
        "row_segments": row_segments,
        "col_segments": col_segments,
        "sprite_count": len(sprites),
    }

    if require_axis_ambiguity:
        all_rows = (1 << scene_shape[0]) - 1
        all_cols = (1 << scene_shape[1]) - 1
        count_without_vertical, _ = _solve_with_count(scene_shape, sprites, all_rows, col_target, max_solutions=2)
        count_without_horizontal, _ = _solve_with_count(scene_shape, sprites, row_target, all_cols, max_solutions=2)
        assert count_without_vertical != 1, "vertical clue axis should be necessary"
        assert count_without_horizontal != 1, "horizontal clue axis should be necessary"
        info["without_vertical"] = count_without_vertical
        info["without_horizontal"] = count_without_horizontal

    return out, info


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sprite_count = random.choices([2, 3], weights=[5, 4])[0]
        colors = random.sample(SPRITE_COLORS, sprite_count)
        sprites = [_sample_sprite(color) for color in colors]
        scene_shape = _sample_scene_shape(sprites)
        scene_h, scene_w = scene_shape

        solution = _sample_solution(scene_shape, sprites)
        if solution is None:
            continue

        row_mask, col_mask = _bbox_masks_from_solution(list(zip(sprites, solution)), scene_shape)
        scene = _build_scene_input(scene_shape, row_mask, col_mask)

        max_legend_h = MAX_SIDE - scene_h - 1
        if max_legend_h < 5:
            continue
        legend = _layout_legend(sprites, scene_w, max_legend_h)
        if legend is None:
            continue

        inp = np.vstack([scene, np.full((1, scene_w), SEPARATOR, dtype=int), legend])
        if max(inp.shape) > MAX_SIDE:
            continue

        try:
            out, _info = _validate_generated_sample(inp, require_axis_ambiguity=True)
        except AssertionError:
            continue

        if not np.array_equal(out, _paint_solution(scene_shape, list(zip(sprites, solution)))):
            continue
        return inp

    raise RuntimeError("failed to generate a valid input for puzzle 3")
