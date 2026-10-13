from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


MAX_GRID_SIZE = 30
GEN_MAX_ATTEMPTS = 400
SLOTS = ("tl", "tr", "bl", "br")

GLYPHS = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
)


def _quadrants(grid: np.ndarray, split_r: int, split_c: int) -> dict[str, np.ndarray]:
    return {
        "tl": grid[:split_r, :split_c].copy(),
        "tr": grid[:split_r, split_c + 1 :].copy(),
        "bl": grid[split_r + 1 :, :split_c].copy(),
        "br": grid[split_r + 1 :, split_c + 1 :].copy(),
    }


def _separator_spec(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    candidates: list[tuple[int, int, int]] = []

    for color in map(int, np.unique(g)):
        full_rows = [r for r in range(height) if np.all(g[r] == color)]
        full_cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if len(full_rows) != 1 or len(full_cols) != 1:
            continue

        split_r = int(full_rows[0])
        split_c = int(full_cols[0])
        top_h = split_r
        bottom_h = height - split_r - 1
        left_w = split_c
        right_w = width - split_c - 1
        if top_h <= 0 or left_w <= 0 or top_h != bottom_h or left_w != right_w:
            continue

        quadrants = _quadrants(g, split_r, split_c)
        sep_slots = [slot for slot, panel in quadrants.items() if np.all(panel == color)]
        if len(sep_slots) == 1:
            candidates.append((color, split_r, split_c))

    if len(candidates) != 1:
        raise ValueError("expected exactly one separator cross layout")
    return candidates[0]


def _extract_panels(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    sep_color, split_r, split_c = _separator_spec(grid)
    quadrants = _quadrants(np.asarray(grid, dtype=int), split_r, split_c)
    panels: list[dict[str, object]] = []
    for slot in SLOTS:
        panel = quadrants[slot]
        if np.all(panel == sep_color):
            continue
        panels.append({"slot": slot, "grid": panel})

    if len(panels) != 3:
        raise ValueError("expected exactly three occupied panels")
    return sep_color, panels


def _glyph_info(panel: np.ndarray) -> tuple[int, int, np.ndarray] | None:
    bg = int(get_bg(panel))
    fg_colors = colors_present(panel, exclude={bg})
    if len(fg_colors) != 1:
        return None

    mask = panel != bg
    if not mask.any():
        return None

    glyph_bbox = bbox(mask)
    if glyph_bbox is None:
        return None
    glyph = crop_bbox(mask, glyph_bbox).astype(bool)
    if glyph.shape != (3, 3):
        return None
    if int(mask.sum()) >= panel.size:
        return None
    if not glyph[0].any() or not glyph[-1].any() or not glyph[:, 0].any() or not glyph[:, -1].any():
        return None
    return bg, int(fg_colors[0]), glyph


def _decode_instance(input_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _sep_color, panels = _extract_panels(np.asarray(input_grid, dtype=int))

    clue_idx = None
    clue_data = None
    for idx, item in enumerate(panels):
        panel = np.asarray(item["grid"], dtype=int)
        info = _glyph_info(panel)
        if info is None:
            continue
        if clue_idx is not None:
            raise ValueError("ambiguous clue panel")
        clue_idx = int(idx)
        clue_data = info

    if clue_idx is None or clue_data is None:
        raise ValueError("missing clue panel")

    _clue_bg, glyph_color, glyph = clue_data
    source_candidates = [idx for idx in range(3) if idx != clue_idx]
    matching = [
        idx
        for idx in source_candidates
        if glyph_color in colors_present(np.asarray(panels[idx]["grid"], dtype=int))
    ]
    if len(matching) != 1:
        raise ValueError("expected exactly one source panel bound to the glyph color")

    src_true = np.asarray(panels[matching[0]]["grid"], dtype=int)
    src_false = np.asarray(panels[[idx for idx in source_candidates if idx != matching[0]][0]]["grid"], dtype=int)
    if src_true.shape != src_false.shape:
        raise ValueError("source panels must share the same shape")
    return glyph, src_true, src_false


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Use the 3x3 glyph as a periodic boolean sampling mask across the whole panel.
    True cells sample from the source panel whose palette contains the glyph color,
    and false cells sample from the other source panel.
    """
    grid = np.asarray(input_grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("expected a 2D grid")

    glyph, src_true, src_false = _decode_instance(grid)
    phase_mask = tile_fill(src_true.shape, glyph.astype(bool))
    return np.where(phase_mask, src_true, src_false).astype(int)


def _pattern_panel(
    rng: np.random.Generator,
    shape: tuple[int, int],
    palette: list[int],
    *,
    required_color: int | None = None,
) -> np.ndarray:
    height, width = map(int, shape)
    tile_h = int(rng.integers(2, min(5, height + 1)))
    tile_w = int(rng.integers(2, min(5, width + 1)))
    tile = rng.choice(np.array(palette, dtype=int), size=(tile_h, tile_w), replace=True).astype(int)

    if required_color is not None:
        tile[0, 0] = int(required_color)
    if len(np.unique(tile)) == 1:
        tile[-1, -1] = int(palette[1 % len(palette)])

    if required_color is not None and required_color not in tile:
        tile[0, 0] = int(required_color)

    offset = (int(rng.integers(tile_h)), int(rng.integers(tile_w)))
    return tile_fill((height, width), tile, offset=offset).astype(int)


def _place_panel(grid: np.ndarray, panel: np.ndarray, slot: str, split_r: int, split_c: int) -> None:
    if slot == "tl":
        grid[:split_r, :split_c] = panel
    elif slot == "tr":
        grid[:split_r, split_c + 1 :] = panel
    elif slot == "bl":
        grid[split_r + 1 :, :split_c] = panel
    elif slot == "br":
        grid[split_r + 1 :, split_c + 1 :] = panel
    else:
        raise ValueError(f"unknown slot: {slot}")


def _make_instance(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    panel_h = int(rng.integers(6, 15))
    panel_w = int(rng.integers(6, 15))
    if 2 * panel_h + 1 > MAX_GRID_SIZE or 2 * panel_w + 1 > MAX_GRID_SIZE:
        raise ValueError("panel size exceeds ARC limits")

    true_count = int(rng.integers(2, 4))
    false_count = int(rng.integers(2, 4))
    sample_count = 2 + true_count + false_count
    sampled = [int(v) for v in rng.choice(np.array(Color.ALL_COLORS, dtype=int), size=sample_count, replace=False)]

    sep_color = int(sampled[0])
    clue_bg = int(sampled[1])
    true_palette = sampled[2 : 2 + true_count]
    false_palette = sampled[2 + true_count :]
    glyph_color = int(rng.choice(np.array(true_palette, dtype=int)))

    glyph = GLYPHS[int(rng.integers(len(GLYPHS)))]
    src_true = _pattern_panel(rng, (panel_h, panel_w), true_palette, required_color=glyph_color)
    src_false = _pattern_panel(rng, (panel_h, panel_w), false_palette)

    clue_panel = np.full((panel_h, panel_w), clue_bg, dtype=int)
    glyph_r = int(rng.integers(0, panel_h - 2))
    glyph_c = int(rng.integers(0, panel_w - 2))
    clue_view = clue_panel[glyph_r : glyph_r + 3, glyph_c : glyph_c + 3]
    clue_view[glyph] = glyph_color

    expected = np.where(tile_fill((panel_h, panel_w), glyph.astype(bool)), src_true, src_false).astype(int)

    split_r = panel_h
    split_c = panel_w
    grid = np.full((2 * panel_h + 1, 2 * panel_w + 1), sep_color, dtype=int)
    missing_slot = SLOTS[int(rng.integers(len(SLOTS)))]
    occupied_slots = [slot for slot in SLOTS if slot != missing_slot]
    shuffled_slots = [occupied_slots[idx] for idx in rng.permutation(len(occupied_slots))]

    panels_by_role = [clue_panel, src_true, src_false]
    for slot, panel in zip(shuffled_slots, panels_by_role):
        _place_panel(grid, panel, slot, split_r, split_c)

    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid, expected = _make_instance(rng)
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
