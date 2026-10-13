from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
if _ARC_UTILS_PATH.exists():
    _ARC_UTILS = SourceFileLoader("_arc_utils_fable_8_003_5", str(_ARC_UTILS_PATH)).load_module()
    colors_present = _ARC_UTILS.colors_present
else:
    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        cols = set(map(int, np.asarray(grid).ravel()))
        if exclude is not None:
            cols -= set(map(int, exclude))
        return sorted(cols)


ARC_COLORS = tuple(range(1, 10))
BLOCK_COUNT_RANGE = (3, 6)
GRID_HEIGHT_RANGE = (8, 18)
BLOCK_WIDTH_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 500


def _dominant_color(values: np.ndarray) -> int:
    colors, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    order = np.argsort(counts)[::-1]
    assert len(order) >= 1
    if len(order) >= 2:
        assert int(counts[order[0]]) > int(counts[order[1]]), "dominant color must be unique"
    return int(colors[order[0]])


def _parse_blocks(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 2 and w >= 3

    column_mains = [_dominant_color(g[:, c]) for c in range(w)]
    runs: list[tuple[int, int, int]] = []
    c0 = 0
    while c0 < w:
        main = int(column_mains[c0])
        c1 = c0 + 1
        while c1 < w and int(column_mains[c1]) == main:
            c1 += 1
        runs.append((c0, c1, main))
        c0 = c1

    assert len(runs) >= 3, "expected at least three vertical blocks"
    widths = {c1 - c0 for c0, c1, _ in runs}
    assert len(widths) == 1, "all blocks must share one width"

    blocks = []
    seen_main_colors = set()
    for c0, c1, main in runs:
        patch = g[:, c0:c1].copy()
        patch_colors = colors_present(patch)
        assert main in patch_colors
        assert len(patch_colors) in (1, 2), "each block must contain one main color and at most one guest color"
        guest_colors = [color for color in patch_colors if int(color) != main]
        guest = int(guest_colors[0]) if guest_colors else None
        if guest is not None:
            assert guest != main
        assert main not in seen_main_colors, "block main colors must be unique"
        seen_main_colors.add(main)
        blocks.append(
            {
                "main": int(main),
                "guest": guest,
                "patch": patch,
                "c0": int(c0),
                "marker_count": int(np.count_nonzero(patch == guest)) if guest is not None else 0,
            }
        )
    return blocks


def _chain_order(blocks: list[dict]) -> list[dict]:
    by_main = {int(block["main"]): block for block in blocks}
    assert len(by_main) == len(blocks)

    guest_colors = [int(block["guest"]) for block in blocks if block["guest"] is not None]
    assert len(guest_colors) == len(blocks) - 1, "exactly one block must be unmarked"
    assert len(set(guest_colors)) == len(guest_colors), "guest colors must be unique"
    assert all(color in by_main for color in guest_colors), "every guest color must match another block main color"

    heads = [block for block in blocks if int(block["main"]) not in set(guest_colors)]
    assert len(heads) == 1, "there must be one unreferenced head block"

    tails = [block for block in blocks if block["guest"] is None]
    assert len(tails) == 1, "there must be one unmarked tail block"

    order = []
    seen = set()
    cur = heads[0]
    while True:
        main = int(cur["main"])
        assert main not in seen, "successor links must form one simple chain"
        seen.add(main)
        order.append(cur)
        if cur["guest"] is None:
            break
        cur = by_main[int(cur["guest"])]

    assert len(order) == len(blocks), "the successor chain must visit every block"
    assert order[-1]["guest"] is None
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder equal-width vertical blocks by following the guest-color successor chain encoded inside them.

    Input structure:
    - The whole grid is tiled by adjacent equal-width vertical blocks, each with a unique main color.
    - Inside each block, all off-main pixels, if any, share one guest color that equals another block's main color.
    - Exactly one block has no guest pixels; no block points to itself; guest colors are pairwise distinct.

    Rule:
    - Read "block A contains guest color B" as "the block whose main color is B is A's successor".
    - Find the head block as the unique main color that is never used as any guest color.
    - Starting from that head, follow successor links until the unique unmarked block, then concatenate the whole blocks from left to right in that chain order.

    Key insight:
    - The foreign-color pixels are not noise to erase: each block contains a pointer to its successor, and the missing incoming reference is the only clue that reveals the chain head.
    """
    blocks = _parse_blocks(np.asarray(grid, dtype=int))
    order = _chain_order(blocks)
    return np.concatenate([block["patch"] for block in order], axis=1)


def _place_guest_markers(patch: np.ndarray, guest: int) -> None:
    h, w = patch.shape
    count = random.randint(1, min(h, w))
    rows = random.sample(range(h), count)
    cols = random.sample(range(w), count)
    for r, c in zip(rows, cols):
        patch[r, c] = int(guest)


def _describe_instance(grid: np.ndarray) -> dict:
    blocks = _parse_blocks(grid)
    order = _chain_order(blocks)
    return {
        "blocks": blocks,
        "input_order": [int(block["main"]) for block in blocks],
        "chain_order": [int(block["main"]) for block in order],
        "guest_order": [block["guest"] for block in blocks],
        "marker_counts": [int(block["marker_count"]) for block in blocks],
    }


def _anti_shortcut_checks(inp: np.ndarray) -> bool:
    meta = _describe_instance(inp)
    chain_order = meta["chain_order"]
    input_order = meta["input_order"]
    blocks = meta["blocks"]

    if chain_order == input_order or chain_order == input_order[::-1]:
        return False
    if chain_order == sorted(chain_order):
        return False
    if chain_order == sorted(chain_order, reverse=True):
        return False

    by_guest = [int(block["main"]) for block in sorted(blocks, key=lambda block: (10 if block["guest"] is None else int(block["guest"]), int(block["main"])))]
    if chain_order == by_guest:
        return False

    by_count_asc = [int(block["main"]) for block in sorted(blocks, key=lambda block: (int(block["marker_count"]), int(block["main"])))]
    by_count_desc = [int(block["main"]) for block in sorted(blocks, key=lambda block: (-int(block["marker_count"]), int(block["main"])))]
    if chain_order == by_count_asc or chain_order == by_count_desc:
        return False

    first_rows = []
    for block in blocks:
        guest = block["guest"]
        if guest is None:
            first_rows.append((10**6, int(block["main"])))
            continue
        coords = np.argwhere(block["patch"] == int(guest))
        first_rows.append((int(coords[:, 0].min()), int(block["main"])))
    by_first_row = [main for _, main in sorted(first_rows)]
    if chain_order == by_first_row:
        return False

    return True


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    n_blocks = random.randint(*BLOCK_COUNT_RANGE)
    block_width = random.randint(BLOCK_WIDTH_RANGE[0], min(BLOCK_WIDTH_RANGE[1], 30 // n_blocks))
    height = random.randint(*GRID_HEIGHT_RANGE)
    if max(height, n_blocks * block_width) > 30:
        return None

    main_colors = random.sample(ARC_COLORS, n_blocks)
    chain_colors = random.sample(main_colors, len(main_colors))
    input_order = random.sample(main_colors, len(main_colors))
    if input_order == chain_colors:
        return None

    guest_of: dict[int, int | None] = {}
    for i, main_color in enumerate(chain_colors):
        guest_of[main_color] = chain_colors[i + 1] if i + 1 < len(chain_colors) else None

    patches: dict[int, np.ndarray] = {}
    for main_color in main_colors:
        patch = np.full((height, block_width), int(main_color), dtype=int)
        guest = guest_of[main_color]
        if guest is not None:
            _place_guest_markers(patch, int(guest))
        patches[int(main_color)] = patch

    inp = np.concatenate([patches[int(main_color)] for main_color in input_order], axis=1)
    expected = np.concatenate([patches[int(main_color)] for main_color in chain_colors], axis=1)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        inp, expected = built

        try:
            actual = main(inp)
        except Exception:
            continue

        if not np.array_equal(actual, expected):
            continue
        if np.array_equal(inp, expected):
            continue
        if not _anti_shortcut_checks(inp):
            continue

        meta = _describe_instance(inp)
        counts = Counter(meta["marker_counts"])
        if len(counts) == 1:
            continue

        return inp

    raise RuntimeError("failed to generate a valid guest-pointer sorting input")
