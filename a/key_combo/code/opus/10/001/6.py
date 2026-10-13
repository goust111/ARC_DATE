import random

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
MAROON = 9

PANEL_SIZE = 3
GRID_HEIGHT = 6
GEN_PANEL_COUNTS = (4, 5, 6)
GEN_RETRY_LIMIT = 400


def _arr(rows: tuple[tuple[int, ...], ...]) -> np.ndarray:
    return np.array(rows, dtype=int)


SYMBOL_PATTERNS = (
    _arr(((1, 1, 1), (1, 0, 1), (1, 1, 1))),
    _arr(((0, 0, 0), (0, 1, 0), (0, 0, 0))),
    _arr(((0, 0, 1), (0, 1, 0), (1, 0, 0))),
    _arr(((0, 0, 0), (0, 0, 0), (1, 1, 1))),
    _arr(((1, 1, 1), (0, 0, 0), (0, 0, 0))),
)

SYMBOL_COLORS = (GREEN, YELLOW, MAROON, BLUE, PINK)
SYMBOL_TO_COLOR = {
    tuple(tuple(int(v) for v in row) for row in pattern): color
    for pattern, color in zip(SYMBOL_PATTERNS, SYMBOL_COLORS)
}
KEY_TO_SYMBOL_ID = {
    tuple(tuple(int(v) for v in row) for row in pattern): idx
    for idx, pattern in enumerate(SYMBOL_PATTERNS)
}
GEN_SYMBOL_IDS = (0, 1, 2, 3)

MASK_LIBRARY = (
    _arr(((1, 0, 0), (0, 0, 0), (0, 0, 1))),
    _arr(((0, 1, 0), (0, 1, 0), (0, 0, 0))),
    _arr(((1, 0, 0), (1, 1, 0), (0, 0, 0))),
    _arr(((0, 0, 1), (0, 1, 0), (1, 0, 0))),
    _arr(((0, 1, 0), (1, 1, 1), (0, 0, 0))),
    _arr(((1, 0, 1), (0, 0, 0), (1, 0, 1))),
    _arr(((0, 1, 0), (1, 1, 1), (0, 1, 0))),
    _arr(((1, 1, 0), (0, 1, 1), (0, 1, 0))),
)


def _panel_key(block: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in block)


def _decode_symbol(block: np.ndarray) -> int:
    key = _panel_key((block == GREY).astype(int))
    assert key in SYMBOL_TO_COLOR, "encountered an unknown 3x3 symbol"
    return SYMBOL_TO_COLOR[key]


def _panel_col0(panel_index: int) -> int:
    return PANEL_SIZE * int(panel_index)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask)


def _mask_component_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < 3 and 0 <= nc < 3 and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _apply_dihedral(mask: np.ndarray, variant: int) -> np.ndarray:
    out = np.rot90(mask, variant % 4)
    if variant >= 4:
        out = np.fliplr(out)
    return out.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each top-row symbol panel and use the decoded solid blocks to expose the lower mask's empty cells.

    Input structure:
    - The grid is 6 rows high and split into two aligned 3-row bands of adjacent 3x3 panels.
    - The upper band contains black/grey symbol masks; each 3x3 mask belongs to the fixed symbol codebook.
    - The lower band contains black background with magenta foreground pixels inside the same panel layout.

    Rule:
    - Decode every upper 3x3 grey mask into its codebook color and replace the whole upper panel by that solid color.
    - Compare each lower panel cell with the decoded upper panel at the same offset.
    - Wherever the decoded upper cell is non-black and the lower cell is black background, paint the lower cell yellow.
    - Keep lower magenta pixels unchanged.

    Key insight:
    - The top symbols do not stay as drawings: they first become solid color panels, and those panels then reveal the lower mask's negative space by marking only the aligned background cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == GRID_HEIGHT, "expected a 6-row grid"
    assert width >= PANEL_SIZE and width % PANEL_SIZE == 0, "expected adjacent 3x3 panels"

    top = g[:PANEL_SIZE]
    bottom = g[PANEL_SIZE:]
    assert set(map(int, np.unique(top))) <= {BLACK, GREY}, "top band must be black/grey"
    assert set(map(int, np.unique(bottom))) <= {BLACK, PINK}, "bottom band must be black/pink"

    out = g.copy()
    for panel_index in range(width // PANEL_SIZE):
        c0 = _panel_col0(panel_index)
        cols = slice(c0, c0 + PANEL_SIZE)
        decoded_color = _decode_symbol(top[:, cols])
        out[:PANEL_SIZE, cols] = decoded_color
        lower = bottom[:, cols]
        panel_out = lower.copy()
        if decoded_color != BLACK:
            panel_out[lower == BLACK] = YELLOW
        out[PANEL_SIZE:, cols] = panel_out
    return out


def _sample_symbol_ids(panel_count: int) -> list[int]:
    symbols = list(GEN_SYMBOL_IDS)
    if panel_count > len(symbols):
        extras = [random.choice(GEN_SYMBOL_IDS) for _ in range(panel_count - len(symbols))]
        symbols.extend(extras)
    random.shuffle(symbols)
    return symbols


def _random_mask() -> np.ndarray:
    base = random.choice(MASK_LIBRARY)
    return _apply_dihedral(base, random.randrange(8))


def _sample_bottom_masks(symbol_ids: list[int]) -> list[np.ndarray] | None:
    panel_count = len(symbol_ids)
    for _ in range(120):
        masks = [_random_mask() for _ in range(panel_count)]

        different_symbol_pairs = [
            (i, j)
            for i in range(panel_count)
            for j in range(i + 1, panel_count)
            if symbol_ids[i] != symbol_ids[j]
        ]
        if not different_symbol_pairs:
            return None

        i, j = random.choice(different_symbol_pairs)
        repeated = _random_mask()
        masks[i] = repeated.copy()
        masks[j] = repeated.copy()

        mask_keys = [_mask_key(mask) for mask in masks]
        distinct_masks = len(set(mask_keys))
        pink_counts = [int(mask.sum()) for mask in masks]
        component_counts = [_mask_component_count(mask.astype(bool)) for mask in masks]

        repeated_symbol_has_different_masks = True
        if len(set(symbol_ids)) < panel_count:
            repeated_symbol_has_different_masks = False
            for symbol_id in set(symbol_ids):
                positions = [idx for idx, value in enumerate(symbol_ids) if value == symbol_id]
                if len(positions) < 2:
                    continue
                if len({_mask_key(masks[idx]) for idx in positions}) >= 2:
                    repeated_symbol_has_different_masks = True
                    break

        if distinct_masks < 3:
            continue
        if len(set(pink_counts)) < 3:
            continue
        if not any(count == 1 for count in component_counts):
            continue
        if not any(count > 1 for count in component_counts):
            continue
        if not repeated_symbol_has_different_masks:
            continue
        return masks
    return None


def _build_input(symbol_ids: list[int], bottom_masks: list[np.ndarray]) -> np.ndarray:
    panel_count = len(symbol_ids)
    grid = np.full((GRID_HEIGHT, PANEL_SIZE * panel_count), BLACK, dtype=int)
    for panel_index, (symbol_id, bottom_mask) in enumerate(zip(symbol_ids, bottom_masks)):
        c0 = _panel_col0(panel_index)
        cols = slice(c0, c0 + PANEL_SIZE)
        symbol = SYMBOL_PATTERNS[int(symbol_id)]
        grid[:PANEL_SIZE, cols] = np.where(symbol == 1, GREY, BLACK)
        grid[PANEL_SIZE:, cols] = np.where(bottom_mask == 1, PINK, BLACK)
    return grid


def _validate_generated_case(inp: np.ndarray, out: np.ndarray) -> None:
    assert inp.shape[0] == GRID_HEIGHT and inp.shape[1] % PANEL_SIZE == 0
    panel_count = inp.shape[1] // PANEL_SIZE
    assert panel_count in GEN_PANEL_COUNTS
    assert inp.ndim == 2 and out.ndim == 2
    assert inp.shape == out.shape
    assert max(inp.shape) <= 30 and max(out.shape) <= 30

    symbol_ids = []
    mask_keys = []
    connected_flags = []
    pink_counts = []

    for panel_index in range(panel_count):
        c0 = _panel_col0(panel_index)
        cols = slice(c0, c0 + PANEL_SIZE)
        top_block = inp[:PANEL_SIZE, cols]
        bottom_block = inp[PANEL_SIZE:, cols]
        out_top = out[:PANEL_SIZE, cols]
        out_bottom = out[PANEL_SIZE:, cols]

        assert set(map(int, np.unique(top_block))) <= {BLACK, GREY}
        assert set(map(int, np.unique(bottom_block))) <= {BLACK, PINK}
        color = _decode_symbol(top_block)
        assert color not in {BLACK, PINK}
        assert np.all(out_top == color)

        pink_count = int(np.sum(bottom_block == PINK))
        assert 2 <= pink_count <= 5
        assert np.any(bottom_block == BLACK)

        expected_bottom = np.where(bottom_block == PINK, PINK, YELLOW if color != BLACK else BLACK)
        assert np.array_equal(out_bottom, expected_bottom)

        symbol_key = _panel_key((top_block == GREY).astype(int))
        symbol_ids.append(KEY_TO_SYMBOL_ID[symbol_key])
        mask_binary = (bottom_block == PINK).astype(int)
        mask_keys.append(_mask_key(mask_binary))
        connected_flags.append(_mask_component_count(mask_binary.astype(bool)) == 1)
        pink_counts.append(pink_count)

    assert len(set(symbol_ids)) >= 3
    assert len(set(mask_keys)) >= 3
    assert len(set(pink_counts)) >= 3
    assert any(connected_flags) and any(not flag for flag in connected_flags)

    same_mask_diff_symbol = False
    for i in range(panel_count):
        for j in range(i + 1, panel_count):
            if mask_keys[i] == mask_keys[j] and symbol_ids[i] != symbol_ids[j]:
                same_mask_diff_symbol = True
    assert same_mask_diff_symbol

    if len(set(symbol_ids)) < panel_count:
        same_symbol_diff_mask = False
        for i in range(panel_count):
            for j in range(i + 1, panel_count):
                if symbol_ids[i] == symbol_ids[j] and mask_keys[i] != mask_keys[j]:
                    same_symbol_diff_mask = True
        assert same_symbol_diff_mask


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.choice(GEN_PANEL_COUNTS)
    symbol_ids = _sample_symbol_ids(panel_count)
    bottom_masks = _sample_bottom_masks(symbol_ids)
    if bottom_masks is None:
        return None

    inp = _build_input(symbol_ids, bottom_masks)
    out = main(inp)
    try:
        _validate_generated_case(inp, out)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
