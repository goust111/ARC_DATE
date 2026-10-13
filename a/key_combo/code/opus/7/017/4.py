from __future__ import annotations

from collections import Counter
import random

import numpy as np


MAIN_COLORS = [1, 2, 3, 4, 5]
SECONDARY_COLORS = [6, 7, 8, 9]
BG = 0
BLOCK_COUNT = 5
HEIGHT = 5
SEPARATOR_WIDTH = 1
BLOCK_WIDTH_CHOICES = (3, 4, 5)
GEN_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(comp, dtype=int))
    return out


def _non_separator_intervals(grid: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for c in range(grid.shape[1]):
        is_sep = bool(np.all(grid[:, c] == BG))
        if is_sep:
            if start is not None:
                intervals.append((start, c))
                start = None
        elif start is None:
            start = c
    if start is not None:
        intervals.append((start, grid.shape[1]))
    return intervals


def _dominant_color(block: np.ndarray) -> int:
    counts = Counter(map(int, block.reshape(-1)))
    counts.pop(BG, None)
    ranked = counts.most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]), "expected a unique dominant color"
    return int(ranked[0][0])


def _parse_blocks(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == HEIGHT
    intervals = _non_separator_intervals(g)
    assert len(intervals) == BLOCK_COUNT, "expected five payload blocks"
    assert len({c1 - c0 for c0, c1 in intervals}) == 1, "all blocks must have the same width"

    raw_blocks = [g[:, c0:c1].copy() for c0, c1 in intervals]
    mains = [_dominant_color(block) for block in raw_blocks]
    assert len(set(mains)) == BLOCK_COUNT, "main colors must be unique"
    assert set(mains) == set(MAIN_COLORS), "expected the five designated main colors"
    main_set = set(mains)

    infos: list[dict] = []
    for idx, ((c0, c1), block, main) in enumerate(zip(intervals, raw_blocks, mains)):
        top = block[0]
        marker_cells = [c for c, value in enumerate(top) if int(value) in main_set and int(value) != main]
        assert len(marker_cells) == 1, "each block needs exactly one foreign main-color marker in row 0"
        marker_c = marker_cells[0]
        marker = int(top[marker_c])

        secondary_colors = sorted(
            {
                int(v)
                for v in np.unique(block)
                if int(v) not in {BG, main, marker}
            }
        )
        assert len(secondary_colors) == 1, "each block needs exactly one secondary color"
        secondary = secondary_colors[0]
        secondary_mask = block == secondary
        comps = _components_mask(secondary_mask)
        assert len(comps) == 1, "secondary pixels must form one 4-connected sprite"
        assert 2 <= int(secondary_mask.sum()) <= 4, "sprite size must be 2..4"
        sprite_box = _bbox_coords(comps[0])
        assert sprite_box[0] >= 1, "sprite must stay below the marker row"
        assert not np.any(block[0] == secondary), "secondary color cannot touch the marker row"

        infos.append(
            {
                "index": idx,
                "c0": c0,
                "c1": c1,
                "block": block,
                "main": main,
                "secondary": secondary,
                "marker": marker,
                "marker_pos": (0, marker_c),
            }
        )

    markers = [info["marker"] for info in infos]
    assert len(set(markers)) == BLOCK_COUNT and set(markers) == main_set
    return infos


def _transform_block(info: dict, position_index: int) -> np.ndarray:
    block = info["block"].copy()
    main = int(info["main"])
    secondary = int(info["secondary"])
    marker_r, marker_c = info["marker_pos"]

    if position_index % 2 == 0:
        block[block == secondary] = main
        final_fill = main
    else:
        main_mask = block == main
        secondary_mask = block == secondary
        block[main_mask] = secondary
        block[secondary_mask] = main
        final_fill = secondary

    block[marker_r, marker_c] = final_fill
    return block


def _sorted_infos(grid: np.ndarray) -> list[dict]:
    infos = _parse_blocks(grid)
    by_main = {int(info["main"]): info for info in infos}
    order = [int(info["marker"]) for info in infos]
    assert order != [int(info["main"]) for info in infos], "marker order must differ from current order"
    return [by_main[color] for color in order]


def _render_output(sorted_infos: list[dict], width: int) -> np.ndarray:
    block_width = int(sorted_infos[0]["block"].shape[1])
    out = np.zeros((HEIGHT, width), dtype=int)
    cursor = 0
    for pos, info in enumerate(sorted_infos):
        out[:, cursor : cursor + block_width] = _transform_block(info, pos)
        cursor += block_width
        if pos != len(sorted_infos) - 1:
            cursor += SEPARATOR_WIDTH
    return out


def _sort_only_output(grid: np.ndarray) -> np.ndarray:
    sorted_infos = _sorted_infos(grid)
    width = grid.shape[1]
    block_width = int(sorted_infos[0]["block"].shape[1])
    out = np.zeros((HEIGHT, width), dtype=int)
    cursor = 0
    for pos, info in enumerate(sorted_infos):
        block = info["block"].copy()
        block[info["marker_pos"]] = int(info["main"])
        out[:, cursor : cursor + block_width] = block
        cursor += block_width
        if pos != len(sorted_infos) - 1:
            cursor += SEPARATOR_WIDTH
    return out


def _wrong_original_parity_output(grid: np.ndarray) -> np.ndarray:
    infos = _parse_blocks(grid)
    transformed_by_origin = {
        int(info["main"]): _transform_block(info, info["index"])
        for info in infos
    }
    sorted_infos = _sorted_infos(grid)
    block_width = int(sorted_infos[0]["block"].shape[1])
    out = np.zeros_like(grid)
    cursor = 0
    for pos, info in enumerate(sorted_infos):
        out[:, cursor : cursor + block_width] = transformed_by_origin[int(info["main"])]
        cursor += block_width
        if pos != len(sorted_infos) - 1:
            cursor += SEPARATOR_WIDTH
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder marked vertical blocks, then reverse each block's encoding according to its sorted odd/even position.

    Input structure:
    - The grid has height 5 and is split by black separator columns into five equal-width vertical blocks.
    - Each block has one unique dominant main color, one secondary sprite color, and exactly one foreign main-color marker pixel in row 0.
    - The secondary color forms one 4-connected 2..4-pixel sprite entirely below the top row.

    Rule:
    - Read the row-0 marker inside each block: its color names the main color that should occupy that slot after sorting.
    - Reorder whole blocks so each slot receives the block whose main color matches that slot's marker.
    - Number the sorted slots from left to right starting at 1. Odd-numbered slots erase their secondary sprite into solid main color; even-numbered slots swap main and secondary colors.
    - Finally replace the marker pixel in every sorted block with that block's final dominant fill color, while keeping separator columns black.

    Key insight:
    - The marker row does more than permute blocks: the sorted destination parity decides which reverse-coding rule each block uses, so sorting and per-block inversion are causally coupled.
    """
    g = np.asarray(grid, dtype=int)
    sorted_infos = _sorted_infos(g)
    return _render_output(sorted_infos, g.shape[1])


SHAPES = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
]


def _placed_shape(block_w: int) -> tuple[tuple[int, int], ...]:
    candidates: list[tuple[tuple[int, int], ...]] = []
    for shape in SHAPES:
        rs = [r for r, _ in shape]
        cs = [c for _, c in shape]
        sh = max(rs) + 1
        sw = max(cs) + 1
        for dr in range(1, HEIGHT - sh + 1):
            for dc in range(0, block_w - sw + 1):
                coords = tuple((r + dr, c + dc) for r, c in shape)
                candidates.append(coords)
    return random.choice(candidates)


def _build_block(block_w: int, main: int, secondary: int) -> np.ndarray:
    block = np.full((HEIGHT, block_w), main, dtype=int)
    for r, c in _placed_shape(block_w):
        block[r, c] = secondary
    return block


def _make_secondary_assignment() -> list[int]:
    pool = SECONDARY_COLORS + [random.choice(SECONDARY_COLORS)]
    random.shuffle(pool)
    return pool[:BLOCK_COUNT]


def _has_cross_parity_move(current_order: list[int], target_order: list[int]) -> bool:
    current_pos = {color: idx for idx, color in enumerate(current_order)}
    return any((current_pos[color] % 2) != (idx % 2) for idx, color in enumerate(target_order))


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        infos = _parse_blocks(grid)
        current_order = [int(info["main"]) for info in infos]
        target_order = [int(info["marker"]) for info in infos]
        if target_order == current_order:
            return False
        if not _has_cross_parity_move(current_order, target_order):
            return False

        final = main(grid)
        sort_only = _sort_only_output(grid)
        wrong = _wrong_original_parity_output(grid)
        if np.array_equal(final, sort_only):
            return False
        if np.array_equal(final, wrong):
            return False

        sorted_infos = _sorted_infos(grid)
        for pos, info in enumerate(sorted_infos):
            rendered = final[:, pos * (info["block"].shape[1] + SEPARATOR_WIDTH) : pos * (info["block"].shape[1] + SEPARATOR_WIDTH) + info["block"].shape[1]]
            if pos % 2 == 0 and len(set(map(int, np.unique(rendered)))) != 1:
                return False
            if pos % 2 == 1 and int(info["main"]) not in rendered:
                return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        block_w = random.choice(BLOCK_WIDTH_CHOICES)
        current_order = random.sample(MAIN_COLORS, BLOCK_COUNT)
        target_order = random.sample(MAIN_COLORS, BLOCK_COUNT)
        if target_order == current_order:
            continue
        if not _has_cross_parity_move(current_order, target_order):
            continue

        secondaries = _make_secondary_assignment()
        by_main = {
            main_color: _build_block(block_w, main_color, secondary)
            for main_color, secondary in zip(current_order, secondaries)
        }

        blocks = []
        for slot_idx, slot_main in enumerate(current_order):
            marker = target_order[slot_idx]
            block = by_main[slot_main].copy()
            marker_col = random.randrange(block_w)
            block[0, marker_col] = marker
            blocks.append(block)

        width = BLOCK_COUNT * block_w + (BLOCK_COUNT - 1) * SEPARATOR_WIDTH
        grid = np.zeros((HEIGHT, width), dtype=int)
        cursor = 0
        for i, block in enumerate(blocks):
            grid[:, cursor : cursor + block_w] = block
            cursor += block_w
            if i != BLOCK_COUNT - 1:
                cursor += SEPARATOR_WIDTH

        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
