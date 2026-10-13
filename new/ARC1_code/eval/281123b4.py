import numpy as np
import random as _gen_random
from arc_utils import components_mask as _gen_components_mask, paste as _gen_paste


PANEL_STARTS = (0, 5, 10, 15)
SEPARATOR_COLS = (4, 9, 14)
VISIBLE_ORDER = (2, 3, 0, 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the four separated 4x4 panels into one 4x4 image using a fixed front-to-back panel order.

    Input structure
    - The input is a 4x19 grid split by three full-height separator columns into four 4x4 panels.
    - Each panel contains background 0 plus exactly one foreground color, and the four panel colors are distinct.

    Rule
    - Treat 0 cells in each panel as transparent.
    - Stack the panels front-to-back in positional order: third panel, fourth panel, first panel, second panel.
    - The output cell is the first non-background color visible at that position, or 0 if all panels are empty there.

    Key insight
    - The four masks are already aligned; the only hidden information is their layer priority, which starts at the third panel and then proceeds cyclically.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (4, 19), "expected four 4x4 panels separated by three columns"

    separators = g[:, SEPARATOR_COLS]
    separator_color = int(separators[0, 0])
    assert separator_color != 0, "separator color must be non-background"
    assert np.all(separators == separator_color), "separator columns must be uniform"

    panels = []
    panel_colors = []
    for start in PANEL_STARTS:
        panel = g[:, start : start + 4]
        colors = set(map(int, np.unique(panel))) - {0}
        assert len(colors) == 1, "each panel must have exactly one foreground color"
        color = colors.pop()
        assert color != separator_color, "panel colors must differ from the separator color"
        panels.append(panel)
        panel_colors.append(color)
    assert len(set(panel_colors)) == len(panel_colors), "panel colors must be distinct"

    out = np.zeros((4, 4), dtype=int)
    filled = np.zeros((4, 4), dtype=bool)
    for panel_index in VISIBLE_ORDER:
        panel = panels[panel_index]
        mask = (panel != 0) & (~filled)
        out[mask] = panel[mask]
        filled |= mask

    return out

# --------------- generator (o5x) ---------------

# Grid geometry: four 4x4 panels separated by three uniform columns.
_GEN_PANEL_SIZE, _GEN_INPUT_WIDTH = 4, 19
_GEN_GRID_SHAPE = (_GEN_PANEL_SIZE, _GEN_PANEL_SIZE)
_GEN_PANEL_STARTS = (0, 5, 10, 15)
_GEN_SEPARATOR_COLS = (4, 9, 14)
_GEN_PANEL_COUNT = len(_GEN_PANEL_STARTS)

# Overlay structure: order positions map to panel indices front-to-back.
_GEN_VISIBLE_ORDER = (2, 3, 0, 1)
_GEN_BACK_ORDER_POSITIONS = (1, 2, 3)

# Palette: one separator color plus one distinct foreground color per panel.
_GEN_COLOR_VALUES = range(1, 10)
_GEN_COLOR_COUNT = 5

# Panel mask constraints observed in the task examples.
_GEN_PANEL_AREA_MIN, _GEN_PANEL_AREA_MAX = 5, 12
_GEN_COMPONENT_MIN, _GEN_COMPONENT_MAX = 1, 5
_GEN_COMPONENT_CONNECTIVITY = 4

# Output composition: 0-3 blank cells and usually 3-4 visible colors.
_GEN_ZERO_CELL_MIN, _GEN_ZERO_CELL_MAX = 0, 3
_GEN_PRESENT_COLOR_COUNTS = (3, 4)

# Generation retry limit.
_GEN_MAX_ATTEMPTS = 300

_GEN_ALL_CELLS = tuple((r, c) for r in range(_GEN_PANEL_SIZE) for c in range(_GEN_PANEL_SIZE))
_GEN_BIT_WEIGHTS = np.array(
    [1 << i for i in range(_GEN_PANEL_SIZE * _GEN_PANEL_SIZE)], dtype=np.uint32
)


def _gen_bit_to_mask(bit):
    return ((bit & _GEN_BIT_WEIGHTS) != 0).reshape(_GEN_GRID_SHAPE)


def _gen_mask_bit(mask):
    return int(np.sum(_GEN_BIT_WEIGHTS[np.asarray(mask, dtype=bool).ravel()]))


def _gen_count_components(mask):
    return len(_gen_components_mask(mask, connectivity=_GEN_COMPONENT_CONNECTIVITY))


def _gen_build_mask_bank():
    bank = []
    by_area = {}
    for bit in range(1, 1 << (_GEN_PANEL_SIZE * _GEN_PANEL_SIZE)):
        area = bit.bit_count()
        if not _GEN_PANEL_AREA_MIN <= area <= _GEN_PANEL_AREA_MAX:
            continue
        mask = _gen_bit_to_mask(bit)
        comp_count = _gen_count_components(mask)
        if _GEN_COMPONENT_MIN <= comp_count <= _GEN_COMPONENT_MAX:
            entry = (bit, mask, area, comp_count)
            bank.append(entry)
            by_area.setdefault(area, []).append(entry)
    return bank, by_area


_GEN_MASK_BANK, _GEN_MASKS_BY_AREA = _gen_build_mask_bank()


def _gen_copy_mask_entry(entry):
    bit, mask, _, _ = entry
    return bit, mask.copy()


def _gen_choose_entry_with_area_then_components(entries, area=None):
    if area is None:
        area = _gen_random.choice(sorted({entry[2] for entry in entries}))
        entries = [entry for entry in entries if entry[2] == area]

    comp_count = _gen_random.choice(sorted({entry[3] for entry in entries}))
    return _gen_random.choice([entry for entry in entries if entry[3] == comp_count])


def _gen_sample_bank_mask(area):
    entry = _gen_choose_entry_with_area_then_components(_GEN_MASKS_BY_AREA[area], area)
    return _gen_copy_mask_entry(entry)


def _gen_choose_panel_mask(required_bit, allowed_bit):
    candidates = [
        entry
        for entry in _GEN_MASK_BANK
        if (entry[0] & required_bit) == required_bit and (entry[0] & ~allowed_bit) == 0
    ]
    if not candidates:
        return None

    entry = _gen_choose_entry_with_area_then_components(candidates)
    return _gen_copy_mask_entry(entry)


def _gen_random_composition(total, parts):
    if parts == 0:
        return []
    choices = []
    for cuts in np.ndindex(*(total + 1 for _ in range(parts - 1))):
        vals = []
        prev = 0
        ok = True
        for cut in sorted(cuts):
            val = cut - prev
            if val < 1 or val > _GEN_PANEL_AREA_MAX:
                ok = False
                break
            vals.append(val)
            prev = cut
        last = total - prev
        if ok and 1 <= last <= _GEN_PANEL_AREA_MAX:
            vals.append(last)
            choices.append(vals)
    return _gen_random.choice(choices) if choices else None


def _gen_sample_palette():
    colors = _gen_random.sample(_GEN_COLOR_VALUES, _GEN_COLOR_COUNT)
    return colors[0], colors[1:]


def _gen_build_output_winners():
    zero_count = _gen_random.randint(_GEN_ZERO_CELL_MIN, _GEN_ZERO_CELL_MAX)
    filled_count = (_GEN_PANEL_SIZE * _GEN_PANEL_SIZE) - zero_count
    present_count = _gen_random.choice(_GEN_PRESENT_COLOR_COUNTS)
    front_area_hi = min(_GEN_PANEL_AREA_MAX, filled_count - (present_count - 1))
    front_area = _gen_random.randint(_GEN_PANEL_AREA_MIN, front_area_hi)
    _, front_mask = _gen_sample_bank_mask(front_area)

    selected_positions = _gen_random.sample(_GEN_BACK_ORDER_POSITIONS, present_count - 1)
    selected_positions.sort()
    other_counts = _gen_random_composition(filled_count - front_area, len(selected_positions))
    if other_counts is None:
        return None

    winners = np.full(_GEN_GRID_SHAPE, -1, dtype=int)
    winners[front_mask] = 0

    remaining = [cell for cell in _GEN_ALL_CELLS if not front_mask[cell]]
    _gen_random.shuffle(remaining)
    nonzero_remaining = remaining[zero_count:]
    cursor = 0
    for pos, count in zip(selected_positions, other_counts):
        for r, c in nonzero_remaining[cursor : cursor + count]:
            winners[r, c] = pos
        cursor += count
    return winners


def _gen_decompose_winners(winners):
    cumulative_front = 0
    panel_masks = [None] * _GEN_PANEL_COUNT
    for order_pos, panel_index in enumerate(_GEN_VISIBLE_ORDER):
        required_bit = _gen_mask_bit(winners == order_pos)
        allowed_bit = cumulative_front | required_bit
        chosen = _gen_choose_panel_mask(required_bit, allowed_bit)
        if chosen is None:
            return None
        bit, mask = chosen
        panel_masks[panel_index] = mask
        cumulative_front |= required_bit
    return panel_masks


def _gen_expected_output(winners, panel_colors):
    out = np.zeros(_GEN_GRID_SHAPE, dtype=int)
    for order_pos, panel_index in enumerate(_GEN_VISIBLE_ORDER):
        out[winners == order_pos] = panel_colors[panel_index]
    return out


def _gen_render_input(separator_color, panel_colors, panel_masks):
    inp = np.zeros((_GEN_PANEL_SIZE, _GEN_INPUT_WIDTH), dtype=int)
    inp[:, _GEN_SEPARATOR_COLS] = separator_color
    for panel_index, start in enumerate(_GEN_PANEL_STARTS):
        panel = np.where(panel_masks[panel_index], panel_colors[panel_index], 0)
        _gen_paste(inp, panel, 0, start)
    return inp


def _gen_attempt():
    separator_color, panel_colors = _gen_sample_palette()

    winners = _gen_build_output_winners()
    if winners is None:
        return None

    panel_masks = _gen_decompose_winners(winners)
    if panel_masks is None:
        return None

    inp = _gen_render_input(separator_color, panel_colors, panel_masks)
    expected = _gen_expected_output(winners, panel_colors)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 281123b4 input")
