from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
GREY = 5
ALL_COLORS = tuple(range(10))
PANEL_DIM_RANGE = (3, 5)
MASK_FILL_RATIO_RANGE = (0.30, 0.65)
STRUCTURE_FILL_RATIO_RANGE = (0.40, 0.75)
GENERATION_RETRIES = 500
MAX_TRACKED_SIGNATURES = 512

_SEEN_INPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _grid_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    g = np.asarray(grid, dtype=int)
    return tuple(map(int, g.shape)), g.tobytes()


def _dominant_non_grey_color(grid: np.ndarray) -> int:
    vals = [int(v) for v in grid.flat if int(v) != GREY]
    assert vals, "expected at least one non-grey cell"
    return Counter(vals).most_common(1)[0][0]


def _separator_specs(grid: np.ndarray) -> list[tuple[str, int]]:
    specs: list[tuple[str, int]] = []
    for r in range(grid.shape[0]):
        if np.all(grid[r, :] == GREY):
            specs.append(("row", int(r)))
    for c in range(grid.shape[1]):
        if np.all(grid[:, c] == GREY):
            specs.append(("col", int(c)))
    return specs


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    specs = _separator_specs(g)
    assert len(specs) == 1, "expected exactly one solid grey separator line"
    kind, idx = specs[0]
    bg = _dominant_non_grey_color(g)
    assert bg != GREY, "separator color cannot be the background"

    if kind == "row":
        assert 0 < idx < g.shape[0] - 1, "separator row must split two non-empty panels"
        first = g[:idx, :]
        second = g[idx + 1 :, :]
    else:
        assert 0 < idx < g.shape[1] - 1, "separator column must split two non-empty panels"
        first = g[:, :idx]
        second = g[:, idx + 1 :]

    return first.copy(), second.copy(), int(bg)


def _non_bg_colors(panel: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(panel) if int(v) != int(bg))


def _identify_roles(panel_a: np.ndarray, panel_b: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, int]:
    colors_a = _non_bg_colors(panel_a, bg)
    colors_b = _non_bg_colors(panel_b, bg)

    a_is_mask = len(colors_a) == 1
    b_is_mask = len(colors_b) == 1
    assert a_is_mask != b_is_mask, "exactly one panel must be monochrome"

    if a_is_mask:
        mask_panel = panel_a
        structure_panel = panel_b
        mask_color = colors_a[0]
        assert len(colors_b) >= 2, "the structure panel must contain multiple foreground colors"
    else:
        mask_panel = panel_b
        structure_panel = panel_a
        mask_color = colors_b[0]
        assert len(colors_a) >= 2, "the structure panel must contain multiple foreground colors"

    return structure_panel, mask_panel, int(mask_color)


def _block_from_mask(mask_fg: np.ndarray, color: int, bg: int, invert: bool) -> np.ndarray:
    if invert:
        return np.where(mask_fg, bg, color).astype(int)
    return np.where(mask_fg, color, bg).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the multicolor structure by the monochrome panel's mask, inverting the mask when colors match.

    Input structure
    - A single solid grey separator row or column splits the grid into exactly two non-empty panels.
    - One panel is monochrome apart from background, so its lone foreground color defines the unit mask color c.
    - The other panel contains background plus multiple foreground colors and therefore acts as the structure panel.

    Rule
    - Let the monochrome panel's non-background cells be the mask shape, and let the other panel be the coarse structure.
    - Each structure cell expands to one block whose size equals the mask panel size.
    - A background structure cell gives an all-background block; a non-background cell of color v != c gives the mask recolored to v; a non-background cell of color c gives the figure-ground inverse of the mask in color c.

    Key insight
    - Panel roles are identified by monochromy rather than position, and only structure cells that match the mask color trigger the inverted block branch.
    """
    panel_a, panel_b, bg = _split_panels(np.asarray(grid, dtype=int))
    structure, mask_panel, mask_color = _identify_roles(panel_a, panel_b, bg)
    mask_fg = mask_panel != bg

    out_h = int(structure.shape[0] * mask_panel.shape[0])
    out_w = int(structure.shape[1] * mask_panel.shape[1])
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for sr in range(structure.shape[0]):
        for sc in range(structure.shape[1]):
            value = int(structure[sr, sc])
            r0 = sr * mask_panel.shape[0]
            c0 = sc * mask_panel.shape[1]
            if value == bg:
                continue
            block = _block_from_mask(mask_fg, value, bg, invert=value == mask_color)
            out[r0 : r0 + mask_panel.shape[0], c0 : c0 + mask_panel.shape[1]] = block

    return out


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _random_connected_mask(h: int, w: int) -> np.ndarray:
    area = h * w
    low = max(2, int(np.ceil(area * MASK_FILL_RATIO_RANGE[0])))
    high = min(area - 2, int(np.floor(area * MASK_FILL_RATIO_RANGE[1])))
    if low > high:
        low, high = 2, area - 2

    for _ in range(200):
        target = random.randint(low, high)
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True

        while int(mask.sum()) < target:
            frontier = []
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c), h, w):
                    if not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True

        if int(mask.sum()) != target:
            continue
        if not mask[0, :].any() or not mask[-1, :].any():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if np.all(mask == mask[0, 0]):
            continue
        return mask

    raise RuntimeError("failed to build a non-degenerate connected mask")


def _random_structure(h: int, w: int, mask_color: int) -> np.ndarray:
    area = h * w
    low = max(3, int(np.ceil(area * STRUCTURE_FILL_RATIO_RANGE[0])))
    high = min(area - 1, int(np.floor(area * STRUCTURE_FILL_RATIO_RANGE[1])))
    if low > high:
        low, high = 3, area - 1

    palette = [c for c in ALL_COLORS if c not in {BACKGROUND, GREY, mask_color}]
    for _ in range(200):
        fg_count = random.randint(low, high)
        fg_positions = random.sample(range(area), fg_count)
        arr = np.full(area, BACKGROUND, dtype=int)

        extra_color_count = random.randint(1, min(2, len(palette)))
        active_colors = [mask_color] + random.sample(palette, extra_color_count)
        assigned = [mask_color] + active_colors[1:]
        assigned.extend(random.choices(active_colors, k=fg_count - len(assigned)))
        random.shuffle(assigned)

        for idx, color in zip(fg_positions, assigned):
            arr[idx] = int(color)

        panel = arr.reshape(h, w)
        non_bg = _non_bg_colors(panel, BACKGROUND)
        if len(non_bg) < 2:
            continue
        if not np.any(panel == BACKGROUND):
            continue
        if not np.any(panel == mask_color):
            continue
        if not np.any((panel != BACKGROUND) & (panel != mask_color)):
            continue
        return panel

    raise RuntimeError("failed to build a valid structure panel")


def _assemble_input(structure: np.ndarray, mask_panel: np.ndarray, orientation: str, mask_first: bool) -> np.ndarray:
    if orientation == "horizontal":
        top = mask_panel if mask_first else structure
        bottom = structure if mask_first else mask_panel
        sep = np.full((1, top.shape[1]), GREY, dtype=int)
        return np.vstack([top, sep, bottom]).astype(int)

    left = mask_panel if mask_first else structure
    right = structure if mask_first else mask_panel
    sep = np.full((left.shape[0], 1), GREY, dtype=int)
    return np.hstack([left, sep, right]).astype(int)


def _expected_block(mask_panel: np.ndarray, mask_color: int, color: int, bg: int) -> np.ndarray:
    mask_fg = mask_panel != bg
    return _block_from_mask(mask_fg, color, bg, invert=color == mask_color)


def _validate_semantics(inp: np.ndarray) -> bool:
    try:
        panel_a, panel_b, bg = _split_panels(inp)
        structure, mask_panel, mask_color = _identify_roles(panel_a, panel_b, bg)
        out = main(inp)
    except AssertionError:
        return False

    mask_fg = mask_panel != bg
    if not mask_fg.any() or np.all(mask_fg):
        return False
    if not np.any(structure == bg):
        return False
    if not np.any(structure == mask_color):
        return False
    if not np.any((structure != bg) & (structure != mask_color)):
        return False

    normal_c = _block_from_mask(mask_fg, mask_color, bg, invert=False)
    inverted_c = _block_from_mask(mask_fg, mask_color, bg, invert=True)
    if np.array_equal(normal_c, inverted_c):
        return False

    block_h, block_w = mask_panel.shape
    expected_shape = (structure.shape[0] * block_h, structure.shape[1] * block_w)
    if tuple(map(int, out.shape)) != tuple(map(int, expected_shape)):
        return False
    if max(out.shape) > 30:
        return False

    for sr in range(structure.shape[0]):
        for sc in range(structure.shape[1]):
            r0 = sr * block_h
            c0 = sc * block_w
            block = out[r0 : r0 + block_h, c0 : c0 + block_w]
            expected = _expected_block(mask_panel, mask_color, int(structure[sr, sc]), bg)
            if not np.array_equal(block, expected):
                return False

    return True


def _sample_panels() -> tuple[np.ndarray, np.ndarray, str, bool]:
    orientation = random.choice(("horizontal", "vertical"))
    mask_first = random.choice((False, True))

    if orientation == "horizontal":
        common_w = random.randint(*PANEL_DIM_RANGE)
        mask_h = random.randint(*PANEL_DIM_RANGE)
        structure_h = random.randint(*PANEL_DIM_RANGE)
        mask_shape = (mask_h, common_w)
        structure_shape = (structure_h, common_w)
    else:
        common_h = random.randint(*PANEL_DIM_RANGE)
        mask_w = random.randint(*PANEL_DIM_RANGE)
        structure_w = random.randint(*PANEL_DIM_RANGE)
        mask_shape = (common_h, mask_w)
        structure_shape = (common_h, structure_w)

    mask_color = random.choice([c for c in ALL_COLORS if c not in {BACKGROUND, GREY}])
    mask = np.where(_random_connected_mask(*mask_shape), mask_color, BACKGROUND).astype(int)
    structure = _random_structure(structure_shape[0], structure_shape[1], mask_color)
    return structure, mask, orientation, mask_first


def generate_input() -> np.ndarray:
    global _SEEN_INPUT_SIGNATURES, _SEEN_OUTPUT_SIGNATURES

    if len(_SEEN_INPUT_SIGNATURES) > MAX_TRACKED_SIGNATURES:
        _SEEN_INPUT_SIGNATURES.clear()
    if len(_SEEN_OUTPUT_SIGNATURES) > MAX_TRACKED_SIGNATURES:
        _SEEN_OUTPUT_SIGNATURES.clear()

    for _ in range(GENERATION_RETRIES):
        structure, mask_panel, orientation, mask_first = _sample_panels()
        inp = _assemble_input(structure, mask_panel, orientation, mask_first)
        out = main(inp)
        in_sig = _grid_signature(inp)
        out_sig = _grid_signature(out)
        if in_sig in _SEEN_INPUT_SIGNATURES or out_sig in _SEEN_OUTPUT_SIGNATURES:
            continue
        if max(inp.shape) > 30 or max(out.shape) > 30:
            continue
        if not _validate_semantics(inp):
            continue
        _SEEN_INPUT_SIGNATURES.add(in_sig)
        _SEEN_OUTPUT_SIGNATURES.add(out_sig)
        return inp

    raise RuntimeError("failed to generate a valid input for the hetero-kron inversion task")
