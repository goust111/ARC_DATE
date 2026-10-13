from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from arc_utils import Color, bbox, colors_present


BLACK = int(Color.BLACK)
ERASED = int(Color.GREY)
GRID_N = 4
BLOCK = 3
STEP = BLOCK + 1
PANEL = GRID_N * BLOCK + (GRID_N - 1)
PAYLOAD_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _block_origin(br: int, bc: int) -> tuple[int, int]:
    return int(br * STEP), int(bc * STEP)


def _grid_candidates(g: np.ndarray) -> list[tuple[int, int]]:
    H, W = map(int, g.shape)
    candidates: list[tuple[int, int]] = []
    sep_idx = {BLOCK, BLOCK + STEP, BLOCK + 2 * STEP}

    for r0 in range(H - PANEL + 1):
        for c0 in range(W - PANEL + 1):
            sub = g[r0 : r0 + PANEL, c0 : c0 + PANEL]
            ok = True
            for rr in sep_idx:
                if not bool(np.isin(sub[rr, :], [BLACK, ERASED]).all()):
                    ok = False
                    break
            if not ok:
                continue
            for cc in sep_idx:
                if not bool(np.isin(sub[:, cc], [BLACK, ERASED]).all()):
                    ok = False
                    break
            if not ok:
                continue

            for br in range(GRID_N):
                for bc in range(GRID_N):
                    r, c = _block_origin(br, bc)
                    block = sub[r : r + BLOCK, c : c + BLOCK]
                    if bool((block == BLACK).any()):
                        ok = False
                        break
                    if not bool(((block != BLACK) & (block != ERASED)).any()):
                        ok = False
                        break
                if not ok:
                    break

            if ok:
                candidates.append((r0, c0))
    return candidates


def _extract_blocks(sub: np.ndarray) -> list[list[np.ndarray]]:
    blocks: list[list[np.ndarray]] = []
    for br in range(GRID_N):
        row: list[np.ndarray] = []
        for bc in range(GRID_N):
            r, c = _block_origin(br, bc)
            row.append(sub[r : r + BLOCK, c : c + BLOCK].copy())
        blocks.append(row)
    return blocks


def _infer_sprite(sub: np.ndarray, blocks: list[list[np.ndarray]]) -> tuple[np.ndarray, int, int]:
    payload = colors_present(sub, exclude={BLACK, ERASED})
    assert len(payload) == 2, "expected exactly two payload colors inside the coded panel"
    payload_set = set(payload)

    sprite_samples: list[np.ndarray] = []
    for row in blocks:
        for block in row:
            if ERASED in set(map(int, block.reshape(-1))):
                continue
            if set(colors_present(block)) == payload_set:
                sprite_samples.append(block.copy())

    assert sprite_samples, "expected at least one fully visible mixed sprite block"
    sprite = sprite_samples[0]
    for sample in sprite_samples[1:]:
        assert np.array_equal(sample, sprite), "fully visible sprite blocks must agree"

    counts = {color: int(np.count_nonzero(sprite == color)) for color in payload}
    assert len(set(counts.values())) == 2, "sprite colors must have a strict majority/minority split"
    majority = max(counts, key=counts.get)
    minority = min(counts, key=counts.get)
    return sprite, int(majority), int(minority)


def _classify_block(block: np.ndarray, sprite: np.ndarray, blank: int) -> bool:
    known = block != ERASED
    assert bool(known.any()), "each block must expose at least one non-grey pixel"
    matches_sprite = bool(np.all(block[known] == sprite[known]))
    matches_blank = bool(np.all(block[known] == blank))
    assert matches_sprite != matches_blank, "each block must match exactly one encoded state"
    return matches_sprite


def _render_panel(
    blocks: list[list[np.ndarray]],
    sprite: np.ndarray,
    majority: int,
    minority: int,
    *,
    invert: bool,
) -> np.ndarray:
    out = np.zeros((PANEL, PANEL), dtype=int)
    for br in range(GRID_N):
        for bc in range(GRID_N):
            block = blocks[br][bc]
            is_sprite = _classify_block(block, sprite, minority)
            if invert:
                patch = np.full((BLOCK, BLOCK), majority, dtype=int) if is_sprite else sprite
            else:
                patch = sprite if is_sprite else np.full((BLOCK, BLOCK), minority, dtype=int)
            r, c = _block_origin(br, bc)
            out[r : r + BLOCK, c : c + BLOCK] = patch
    return out


def _legend_info(g: np.ndarray, panel_origin: tuple[int, int]) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    mask = g != BLACK
    r0, c0 = panel_origin
    mask[r0 : r0 + PANEL, c0 : c0 + PANEL] = False
    legend_bbox = bbox(mask)
    assert legend_bbox is not None, "expected one legend outside the coded panel"
    lr0, lc0, lr1, lc1 = legend_bbox
    assert lc1 - lc0 + 1 == 2, "legend must be exactly two columns wide"
    assert 2 <= lr1 - lr0 + 1 <= 3, "legend must have two or three rows"

    region = g[lr0 : lr1 + 1, lc0 : lc1 + 1]
    assert bool((region != BLACK).all()), "legend rows must be contiguous and fully occupied"
    assert int(mask.sum()) == int(region.size), "legend must be the only non-panel foreground outside the block grid"

    pairs = [(int(g[r, lc0]), int(g[r, lc0 + 1])) for r in range(lr0, lr1 + 1)]
    return (int(lr0), int(lc0), int(lr1), int(lc1)), pairs


def _apply_chain(panel: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    out = panel.copy()
    for src, dst in pairs:
        out[out == int(src)] = int(dst)
    return out


def _apply_parallel(panel: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    out = panel.copy()
    base = panel.copy()
    for src, dst in pairs:
        out[base == int(src)] = int(dst)
    return out


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    candidates = _grid_candidates(g)
    assert len(candidates) == 1, "expected exactly one 4x4 coded block panel"
    panel_r0, panel_c0 = candidates[0]
    sub = g[panel_r0 : panel_r0 + PANEL, panel_c0 : panel_c0 + PANEL]
    blocks = _extract_blocks(sub)
    sprite, majority, minority = _infer_sprite(sub, blocks)
    legend_bbox, pairs = _legend_info(g, (panel_r0, panel_c0))

    reversed_panel = _render_panel(blocks, sprite, majority, minority, invert=True)
    recolored_panel = _apply_chain(reversed_panel, pairs)

    out = g.copy()
    out[panel_r0 : panel_r0 + PANEL, panel_c0 : panel_c0 + PANEL] = recolored_panel
    return out, {
        "panel_origin": (panel_r0, panel_c0),
        "legend_bbox": legend_bbox,
        "pairs": pairs,
        "sprite": sprite,
        "majority": majority,
        "minority": minority,
        "blocks": blocks,
        "reversed_panel": reversed_panel,
        "recolored_panel": recolored_panel,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Invert a sprite/blank block code and then recolor the decoded panel by a top-to-bottom legend chain.

    Input structure:
    - One 15x15 coded panel contains a 4x4 array of 3x3 blocks separated by black rows and columns.
    - Each block is either a repeated two-color sprite or a solid minority-color blank block; grey pixels are erasures/wildcards.
    - A separate two-column legend on the right lists 2 or 3 color substitutions from top to bottom.

    Rule:
    - Detect the coded panel, infer the repeated sprite from fully visible mixed blocks, and treat grey block pixels as wildcards while classifying each block.
    - Reverse the encoding: sprite blocks become solid majority-color blocks, solid minority-color blocks become the full sprite, and grey separator pixels become black.
    - Then scan the legend rows from top to bottom; after each row, recolor every non-black pixel in the decoded panel from the left color to the right color, allowing one pixel to change multiple times in the same pass.
    - Keep the legend and all background outside the coded panel unchanged.

    Key insight:
    - The legend acts on the pixels created by the block-code inversion, so inversion and recoloring are not interchangeable and the legend row order matters.
    """
    return _solve_details(grid)[0]


def _random_sprite(majority: int, minority: int) -> np.ndarray:
    for _ in range(200):
        minority_count = random.choice([3, 4])
        cells = random.sample(range(BLOCK * BLOCK), minority_count)
        mask = np.zeros((BLOCK, BLOCK), dtype=bool)
        for cell in cells:
            mask[cell // BLOCK, cell % BLOCK] = True
        row_counts = mask.sum(axis=1)
        col_counts = mask.sum(axis=0)
        if not bool(((row_counts > 0) & (row_counts < BLOCK)).all()):
            continue
        if not bool(((col_counts > 0) & (col_counts < BLOCK)).all()):
            continue
        if (
            np.array_equal(mask, np.fliplr(mask))
            or np.array_equal(mask, np.flipud(mask))
            or np.array_equal(mask, np.rot90(mask, 2))
        ):
            continue
        return np.where(mask, minority, majority).astype(int)
    raise RuntimeError("failed to build a distinctive sprite")


def _state_matrix() -> np.ndarray:
    sprite_count = random.randint(5, 11)
    picks = set(random.sample(range(GRID_N * GRID_N), sprite_count))
    return np.array(
        [[(br * GRID_N + bc) in picks for bc in range(GRID_N)] for br in range(GRID_N)],
        dtype=bool,
    )


def _choose_pairs(majority: int, minority: int) -> list[tuple[int, int]]:
    chain_source = random.choice([majority, minority])
    other = minority if chain_source == majority else majority
    aux = random.sample([c for c in PAYLOAD_POOL if c not in {majority, minority}], 2)
    pairs = [(chain_source, aux[0]), (aux[0], aux[1])]
    if random.random() < 0.5:
        pairs.append((other, chain_source))
    return [(int(src), int(dst)) for src, dst in pairs]


def _greyed_sprite(sprite: np.ndarray, force_visible: bool) -> np.ndarray:
    block = sprite.copy()
    if force_visible:
        return block

    target = random.choice([0, 1, 1, 2, 2])
    positions = [(r, c) for r in range(BLOCK) for c in range(BLOCK)]
    random.shuffle(positions)
    chosen: list[tuple[int, int]] = []
    for r, c in positions:
        if len(chosen) == target:
            break
        trial = chosen + [(r, c)]
        masked = block.copy()
        for rr, cc in trial:
            masked[rr, cc] = ERASED
        visible = masked[masked != ERASED]
        if len(set(map(int, visible))) < 2:
            continue
        chosen = trial
    for r, c in chosen:
        block[r, c] = ERASED
    return block


def _greyed_blank(blank: int) -> np.ndarray:
    block = np.full((BLOCK, BLOCK), blank, dtype=int)
    target = random.choice([0, 0, 1, 1, 2])
    cells = random.sample(range(BLOCK * BLOCK), target)
    for cell in cells:
        block[cell // BLOCK, cell % BLOCK] = ERASED
    if bool((block == ERASED).all()):
        block[random.randrange(BLOCK), random.randrange(BLOCK)] = blank
    return block


def _separator_cells() -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    sep_idx = [BLOCK, BLOCK + STEP, BLOCK + 2 * STEP]
    for r in sep_idx:
        for c in range(PANEL):
            pts.append((r, c))
    for c in sep_idx:
        for r in range(PANEL):
            pts.append((r, c))
    return sorted(set(pts))


def _decorate_with_grey(panel: np.ndarray) -> np.ndarray:
    out = panel.copy()
    sep_cells = _separator_cells()
    grey_count = random.choice([0, 1, 2, 3, 4])
    for r, c in random.sample(sep_cells, grey_count):
        out[r, c] = ERASED
    return out


def _build_input() -> tuple[np.ndarray, np.ndarray]:
    panel_r0 = random.randint(0, 3)
    panel_c0 = random.randint(0, 2)
    bottom_margin = random.randint(0, 4)
    gap = random.randint(1, 3)
    right_margin = random.randint(0, 3)

    H = panel_r0 + PANEL + bottom_margin
    majority, minority = random.sample(PAYLOAD_POOL, 2)
    sprite = _random_sprite(majority, minority)
    states = _state_matrix()
    sprite_positions = [(br, bc) for br in range(GRID_N) for bc in range(GRID_N) if states[br, bc]]
    full_visible = random.choice(sprite_positions)

    coded_panel = np.zeros((PANEL, PANEL), dtype=int)
    grey_used = False
    for br in range(GRID_N):
        for bc in range(GRID_N):
            r, c = _block_origin(br, bc)
            if states[br, bc]:
                block = _greyed_sprite(sprite, force_visible=(br, bc) == full_visible)
            else:
                block = _greyed_blank(minority)
            grey_used = grey_used or bool((block == ERASED).any())
            coded_panel[r : r + BLOCK, c : c + BLOCK] = block

    coded_panel = _decorate_with_grey(coded_panel)
    grey_used = grey_used or bool((coded_panel == ERASED).any())

    pairs = _choose_pairs(majority, minority)
    legend_rows = len(pairs)
    legend_c0 = panel_c0 + PANEL + gap
    W = legend_c0 + 2 + right_margin
    legend_r0 = random.randint(0, H - legend_rows)

    inp = np.zeros((H, W), dtype=int)
    inp[panel_r0 : panel_r0 + PANEL, panel_c0 : panel_c0 + PANEL] = coded_panel
    for i, (src, dst) in enumerate(pairs):
        inp[legend_r0 + i, legend_c0] = src
        inp[legend_r0 + i, legend_c0 + 1] = dst

    if not grey_used:
        rr, cc = random.choice(_separator_cells())
        inp[panel_r0 + rr, panel_c0 + cc] = ERASED

    reversed_panel = _render_panel(_extract_blocks(coded_panel), sprite, majority, minority, invert=True)
    expected_panel = _apply_chain(reversed_panel, pairs)

    out = inp.copy()
    out[panel_r0 : panel_r0 + PANEL, panel_c0 : panel_c0 + PANEL] = expected_panel

    return inp, out


def _validate(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solved, info = _solve_details(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    panel_r0, panel_c0 = info["panel_origin"]
    panel_slice = (slice(panel_r0, panel_r0 + PANEL), slice(panel_c0, panel_c0 + PANEL))
    expected_panel = expected[panel_slice]
    pairs = info["pairs"]
    sprite = info["sprite"]
    majority = info["majority"]
    minority = info["minority"]
    blocks = info["blocks"]
    clean_panel = _render_panel(blocks, sprite, majority, minority, invert=False)
    reversed_panel = info["reversed_panel"]

    if np.array_equal(reversed_panel, expected_panel):
        return False
    if np.array_equal(_apply_chain(clean_panel, pairs), expected_panel):
        return False
    if np.array_equal(_apply_parallel(reversed_panel, pairs), expected_panel):
        return False
    if np.array_equal(_apply_chain(reversed_panel, list(reversed(pairs))), expected_panel):
        return False

    src0, mid = pairs[0]
    if not bool((reversed_panel == src0).any()):
        return False
    after_first = reversed_panel.copy()
    after_first[after_first == src0] = mid
    if not bool((after_first == mid).any()):
        return False
    if len(pairs) < 2 or pairs[1][0] != mid:
        return False
    if np.array_equal(_apply_chain(reversed_panel, pairs[:2]), _apply_parallel(reversed_panel, pairs[:2])):
        return False

    candidates = _grid_candidates(inp)
    if len(candidates) != 1:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp, expected = _build_input()
        if _validate(inp, expected):
            return inp
    raise RuntimeError("failed to generate a non-degenerate block-inversion chain-recolor puzzle")
