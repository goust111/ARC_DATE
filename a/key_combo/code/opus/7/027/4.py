import random

import numpy as np


PALETTE = (1, 2, 3, 4, 6)
GREY = 5
BLACK = 0

TOP_BAND_HEIGHT = 5
SYMBOL_SIZE = 3
SLOT_STEP = 4
SLOT_COUNTS = (4, 5, 6)
MAX_ATTEMPTS = 500


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            key = candidate.astype(np.uint8).tobytes()
            if key not in seen:
                seen.add(key)
                variants.append(candidate.copy())
    return variants


def _build_mask_library() -> dict[int, list[np.ndarray]]:
    base_masks = {
        1: [
            np.array(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=bool,
            ),
        ],
        2: [
            np.array(
                [
                    [1, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                ],
                dtype=bool,
            ),
        ],
        3: [
            np.array(
                [
                    [1, 1, 1],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 0, 0],
                ],
                dtype=bool,
            ),
        ],
        4: [
            np.array(
                [
                    [1, 1, 0],
                    [1, 1, 0],
                    [0, 0, 0],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [1, 1, 1],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                dtype=bool,
            ),
        ],
        5: [
            np.array(
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                ],
                dtype=bool,
            ),
        ],
        6: [
            np.array(
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [0, 0, 0],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                ],
                dtype=bool,
            ),
        ],
        7: [
            np.array(
                [
                    [0, 1, 1],
                    [1, 1, 1],
                    [1, 1, 0],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 0, 0],
                ],
                dtype=bool,
            ),
        ],
        8: [
            np.array(
                [
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                ],
                dtype=bool,
            ),
        ],
        9: [
            np.ones((3, 3), dtype=bool),
        ],
    }

    library: dict[int, list[np.ndarray]] = {}
    for density, bases in base_masks.items():
        variants: list[np.ndarray] = []
        seen: set[bytes] = set()
        for base in bases:
            for candidate in _unique_transforms(base):
                key = candidate.astype(np.uint8).tobytes()
                if key not in seen:
                    seen.add(key)
                    variants.append(candidate.copy())
        library[density] = variants
    return library


MASK_LIBRARY = _build_mask_library()


def _row_clusters(grid: np.ndarray) -> list[tuple[int, int]]:
    rows = np.flatnonzero(np.any(grid != BLACK, axis=1))
    assert rows.size > 0, "expected foreground symbols"
    clusters: list[tuple[int, int]] = []
    start = int(rows[0])
    prev = int(rows[0])
    for row in rows[1:]:
        row = int(row)
        if row == prev + 1:
            prev = row
            continue
        clusters.append((start, prev))
        start = prev = row
    clusters.append((start, prev))
    return clusters


def _slot_count(width: int) -> int:
    assert (width + 1) % SLOT_STEP == 0, f"unexpected width {width}"
    count = (width + 1) // SLOT_STEP
    assert count in SLOT_COUNTS, f"unexpected slot count {count}"
    return count


def _compress_patch(patch: np.ndarray) -> tuple[int, int]:
    colors = {int(v) for v in patch.ravel() if int(v) != BLACK}
    assert len(colors) <= 1, "each slot must be monochrome"
    if not colors:
        return BLACK, 0
    color = next(iter(colors))
    density = int(np.count_nonzero(patch))
    return color, density


def _extract_slots(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    clusters = _row_clusters(g)
    assert len(clusters) == 2, "expected exactly two symbol bands"
    (top_r0, top_r1), (bot_r0, bot_r1) = clusters
    assert top_r1 - top_r0 == SYMBOL_SIZE - 1, "top symbol rows must span 3 rows"
    assert bot_r1 - bot_r0 == SYMBOL_SIZE - 1, "bottom symbol rows must span 3 rows"

    slot_count = _slot_count(g.shape[1])
    top: list[tuple[int, int]] = []
    bottom: list[tuple[int, int]] = []
    for slot_idx in range(slot_count):
        c0 = slot_idx * SLOT_STEP
        c1 = c0 + SYMBOL_SIZE
        top.append(_compress_patch(g[top_r0 : top_r0 + SYMBOL_SIZE, c0:c1]))
        bottom.append(_compress_patch(g[bot_r0 : bot_r0 + SYMBOL_SIZE, c0:c1]))
    return top, bottom


def _composite(top: list[tuple[int, int]], bottom: list[tuple[int, int]]) -> np.ndarray:
    assert len(top) == len(bottom)
    out = np.zeros((1, len(top)), dtype=int)
    for idx, ((top_color, top_density), (bot_color, bot_density)) in enumerate(zip(top, bottom)):
        if top_density == 0 and bot_density == 0:
            out[0, idx] = BLACK
        elif bot_density == 0:
            out[0, idx] = top_color
        elif top_density == 0:
            out[0, idx] = bot_color
        elif top_density > bot_density:
            out[0, idx] = top_color
        elif bot_density > top_density:
            out[0, idx] = bot_color
        else:
            out[0, idx] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compress aligned 3x3 symbols and overlay them using pre-compression density as the conflict rule.

    Input structure:
    - The grid has black background and two horizontally aligned symbol bands stacked vertically.
    - Each band contains the same number of 3x3 slots laid out left-to-right with one black separator column between neighboring slots.
    - Every non-empty slot contains a monochrome 3x3 symbol, while an empty slot is all black.

    Rule:
    - Compress each 3x3 slot to one pixel of its symbol color, or black if the slot is empty.
    - For every aligned top/bottom slot pair, keep the lone non-black color if only one band is occupied.
    - If both bands are occupied, compare the symbols' original foreground-pixel counts: keep the denser symbol's color, and output gray when the densities are equal.

    Key insight:
    - The visible output is a one-pixel-per-slot overlay, but conflict resolution depends on hidden information preserved from the discarded 3x3 symbols: their original densities.
    """
    top, bottom = _extract_slots(np.asarray(grid, dtype=int))
    return _composite(top, bottom)


def _render_symbol(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros((SYMBOL_SIZE, SYMBOL_SIZE), dtype=int)
    patch[mask] = color
    return patch


def _choose_mask(density: int) -> np.ndarray:
    return random.choice(MASK_LIBRARY[density]).copy()


def _choose_color(exclude: set[int] | None = None) -> int:
    forbidden = set() if exclude is None else set(exclude)
    choices = [color for color in PALETTE if color not in forbidden]
    return random.choice(choices)


def _make_slot_spec(kind: str) -> dict[str, int]:
    if kind == "top_wins":
        bot_density = random.randint(1, 8)
        top_density = random.randint(bot_density + 1, 9)
        top_color, bot_color = random.sample(PALETTE, 2)
        return {
            "top_density": top_density,
            "bottom_density": bot_density,
            "top_color": top_color,
            "bottom_color": bot_color,
        }
    if kind == "bottom_wins":
        top_density = random.randint(1, 8)
        bot_density = random.randint(top_density + 1, 9)
        top_color, bot_color = random.sample(PALETTE, 2)
        return {
            "top_density": top_density,
            "bottom_density": bot_density,
            "top_color": top_color,
            "bottom_color": bot_color,
        }
    if kind == "tie":
        density = random.randint(1, 9)
        top_color, bot_color = random.sample(PALETTE, 2)
        return {
            "top_density": density,
            "bottom_density": density,
            "top_color": top_color,
            "bottom_color": bot_color,
        }
    if kind == "top_only":
        return {
            "top_density": random.randint(1, 9),
            "bottom_density": 0,
            "top_color": _choose_color(),
            "bottom_color": BLACK,
        }
    if kind == "bottom_only":
        return {
            "top_density": 0,
            "bottom_density": random.randint(1, 9),
            "top_color": BLACK,
            "bottom_color": _choose_color(),
        }
    if kind == "both_empty":
        return {
            "top_density": 0,
            "bottom_density": 0,
            "top_color": BLACK,
            "bottom_color": BLACK,
        }
    raise ValueError(f"unknown slot kind {kind}")


def _slot_kind(top_slot: tuple[int, int], bottom_slot: tuple[int, int]) -> str:
    top_color, top_density = top_slot
    bot_color, bot_density = bottom_slot
    if top_density == 0 and bot_density == 0:
        return "both_empty"
    if top_density == 0:
        return "bottom_only"
    if bot_density == 0:
        return "top_only"
    if top_density == bot_density:
        return "tie"
    if top_density > bot_density:
        assert top_color != BLACK and bot_color != BLACK
        return "top_wins"
    return "bottom_wins"


def _priority_overlay(top: list[tuple[int, int]], bottom: list[tuple[int, int]], prefer_top: bool, ties_to_grey: bool) -> np.ndarray:
    out = np.zeros((1, len(top)), dtype=int)
    for idx, ((top_color, top_density), (bot_color, bot_density)) in enumerate(zip(top, bottom)):
        if top_density == 0 and bot_density == 0:
            out[0, idx] = BLACK
        elif bot_density == 0:
            out[0, idx] = top_color
        elif top_density == 0:
            out[0, idx] = bot_color
        elif top_density == bot_density and ties_to_grey:
            out[0, idx] = GREY
        else:
            out[0, idx] = top_color if prefer_top else bot_color
    return out


def _density_overlay_with_tie_choice(
    top: list[tuple[int, int]],
    bottom: list[tuple[int, int]],
    choose_top_on_tie: bool,
) -> np.ndarray:
    out = np.zeros((1, len(top)), dtype=int)
    for idx, ((top_color, top_density), (bot_color, bot_density)) in enumerate(zip(top, bottom)):
        if top_density == 0 and bot_density == 0:
            out[0, idx] = BLACK
        elif bot_density == 0:
            out[0, idx] = top_color
        elif top_density == 0:
            out[0, idx] = bot_color
        elif top_density > bot_density:
            out[0, idx] = top_color
        elif bot_density > top_density:
            out[0, idx] = bot_color
        else:
            out[0, idx] = top_color if choose_top_on_tie else bot_color
    return out


def _validate_semantics(grid: np.ndarray, expected: np.ndarray) -> bool:
    top, bottom = _extract_slots(grid)
    actual = _composite(top, bottom)
    if not np.array_equal(actual, expected):
        return False

    kinds = [_slot_kind(t, b) for t, b in zip(top, bottom)]
    if kinds.count("tie") != 1:
        return False
    if "top_wins" not in kinds or "bottom_wins" not in kinds:
        return False
    if not any(kind in ("top_only", "bottom_only") for kind in kinds):
        return False

    if np.array_equal(expected, _priority_overlay(top, bottom, prefer_top=True, ties_to_grey=False)):
        return False
    if np.array_equal(expected, _priority_overlay(top, bottom, prefer_top=False, ties_to_grey=False)):
        return False
    if np.array_equal(expected, _priority_overlay(top, bottom, prefer_top=True, ties_to_grey=True)):
        return False
    if np.array_equal(expected, _priority_overlay(top, bottom, prefer_top=False, ties_to_grey=True)):
        return False
    if np.array_equal(expected, _density_overlay_with_tie_choice(top, bottom, choose_top_on_tie=True)):
        return False
    if np.array_equal(expected, _density_overlay_with_tie_choice(top, bottom, choose_top_on_tie=False)):
        return False
    return True


def _place_band_rows(slot_masks: list[np.ndarray | None]) -> set[int]:
    rows: set[int] = set()
    for mask in slot_masks:
        if mask is None:
            continue
        rows.update(int(r) for r in np.flatnonzero(np.any(mask, axis=1)))
    return rows


def _make_layout(slot_specs: list[dict[str, int]], separator_rows: int) -> np.ndarray | None:
    slot_count = len(slot_specs)
    height = TOP_BAND_HEIGHT * 2 + separator_rows
    width = slot_count * SLOT_STEP - 1
    grid = np.zeros((height, width), dtype=int)

    top_row0 = 1
    bottom_row0 = TOP_BAND_HEIGHT + separator_rows + 1
    top_masks: list[np.ndarray | None] = []
    bottom_masks: list[np.ndarray | None] = []

    for spec in slot_specs:
        top_masks.append(_choose_mask(spec["top_density"]) if spec["top_density"] else None)
        bottom_masks.append(_choose_mask(spec["bottom_density"]) if spec["bottom_density"] else None)

    if _place_band_rows(top_masks) != {0, 1, 2}:
        return None
    if _place_band_rows(bottom_masks) != {0, 1, 2}:
        return None

    for idx, spec in enumerate(slot_specs):
        c0 = idx * SLOT_STEP
        if top_masks[idx] is not None:
            grid[top_row0 : top_row0 + SYMBOL_SIZE, c0 : c0 + SYMBOL_SIZE] = _render_symbol(
                top_masks[idx],
                spec["top_color"],
            )
        if bottom_masks[idx] is not None:
            grid[bottom_row0 : bottom_row0 + SYMBOL_SIZE, c0 : c0 + SYMBOL_SIZE] = _render_symbol(
                bottom_masks[idx],
                spec["bottom_color"],
            )
    return grid


def _sample_slot_kinds(slot_count: int) -> list[str]:
    kinds = ["top_wins", "bottom_wins", "tie"]
    if slot_count == 4:
        kinds.append(random.choice(["top_only", "bottom_only"]))
    elif slot_count == 5:
        kinds.extend(["top_only", "bottom_only"])
    else:
        kinds.extend(["top_only", "bottom_only"])
        kinds.append(random.choice(["top_wins", "bottom_wins", "both_empty"]))
    random.shuffle(kinds)
    return kinds


def _attempt_generate() -> np.ndarray | None:
    slot_count = random.choice(SLOT_COUNTS)
    separator_rows = random.choice((0, 1))
    slot_specs = [_make_slot_spec(kind) for kind in _sample_slot_kinds(slot_count)]
    grid = _make_layout(slot_specs, separator_rows)
    if grid is None:
        return None

    top, bottom = _extract_slots(grid)
    expected = _composite(top, bottom)
    if len(set(expected.ravel().tolist())) < 3:
        return None
    if not _validate_semantics(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-4 input")
