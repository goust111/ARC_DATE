from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np

try:
    from arc_utils import Color, bbox
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_shared", "/home/lixinyu/snap/new/solvers/arc_utils.py"
    ).load_module()
    Color = _ARC_UTILS.Color
    bbox = _ARC_UTILS.bbox


_SYMBOL_TO_COLOR = {
    ((0, 1, 0), (1, 1, 1), (0, 1, 0)): Color.RED,
    ((0, 1, 0), (0, 1, 0), (0, 1, 0)): Color.BLUE,
    ((1, 0, 0), (1, 0, 0), (1, 1, 1)): Color.GREEN,
    ((1, 1, 0), (1, 1, 0), (0, 0, 0)): Color.YELLOW,
}
_SYMBOL_PATTERNS = tuple(_SYMBOL_TO_COLOR)
_BOOTSTRAP_SEQUENCES = (
    (_SYMBOL_PATTERNS[0], _SYMBOL_PATTERNS[1]),
    (_SYMBOL_PATTERNS[2], _SYMBOL_PATTERNS[3], _SYMBOL_PATTERNS[0]),
    (_SYMBOL_PATTERNS[1], _SYMBOL_PATTERNS[3], _SYMBOL_PATTERNS[2], _SYMBOL_PATTERNS[0]),
)
_GEN_CALL_COUNT = 0
_GEN_ATTEMPTS = 400


def _rectangular_frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    box = bbox(mask)
    if box is None:
        return None

    r0, c0, r1, c1 = box
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    expected = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(mask[r0 : r1 + 1, c0 : c1 + 1], expected):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _mask_key(block: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in block)


def _is_symbol_window(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if r < 0 or c < 0 or r + 3 > h or c + 3 > w:
        return False

    window = grid[r : r + 3, c : c + 3]
    colors = set(map(int, np.unique(window)))
    if not colors <= {Color.BLACK, Color.GREY} or Color.GREY not in colors:
        return False

    er0 = max(0, r - 1)
    ec0 = max(0, c - 1)
    er1 = min(h, r + 4)
    ec1 = min(w, c + 4)
    expanded = grid[er0:er1, ec0:ec1] == Color.GREY
    inner = np.zeros_like(expanded, dtype=bool)
    inner[r - er0 : r - er0 + 3, c - ec0 : c - ec0 + 3] = True
    if np.any(expanded & ~inner):
        return False

    return _mask_key(window == Color.GREY) in _SYMBOL_TO_COLOR


def _find_symbol_panels(grid: np.ndarray) -> list[tuple[int, int]]:
    candidates = []
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            if _is_symbol_window(grid, r, c):
                candidates.append((int(r), int(c)))

    assert candidates, "expected at least one grey symbol panel"
    candidates.sort()
    anchors = [candidates[0]]
    for r, c in candidates[1:]:
        prev_r, prev_c = anchors[-1]
        if (r, c) == (prev_r + 4, prev_c + 4):
            anchors.append((r, c))
        else:
            raise AssertionError("expected symbol panels to form a single down-right diagonal chain")

    return anchors


def _decode_palette(grid: np.ndarray, panel_positions: list[tuple[int, int]]) -> list[int]:
    palette = []
    for r, c in panel_positions:
        mask = _mask_key(grid[r : r + 3, c : c + 3] == Color.GREY)
        assert mask in _SYMBOL_TO_COLOR, "encountered an unknown symbol mask"
        palette.append(int(_SYMBOL_TO_COLOR[mask]))
    return palette


def _fill_concentric(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], palette: list[int]
) -> np.ndarray:
    assert palette, "expected at least one decoded color"
    out = grid.copy()
    r0, c0, r1, c1 = frame_bbox
    capacity = min(r1 - r0 + 1, c1 - c0 + 1) // 2
    assert len(palette) <= capacity, "expected the frame to have enough nested layers"

    for i, color in enumerate(palette):
        rr0 = r0 + 1 + i
        cc0 = c0 + 1 + i
        rr1 = r1 - 1 - i
        cc1 = c1 - 1 - i
        assert rr0 <= rr1 and cc0 <= cc1, "invalid inner ring geometry"

        if i == len(palette) - 1:
            out[rr0 : rr1 + 1, cc0 : cc1 + 1] = color
        else:
            out[rr0, cc0 : cc1 + 1] = color
            out[rr1, cc0 : cc1 + 1] = color
            out[rr0 : rr1 + 1, cc0] = color
            out[rr0 : rr1 + 1, cc1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode diagonal grey symbol panels into a color palette and use it to fill a hollow frame with nested bands.

    Input structure:
    - The background is black, with one orange one-pixel-thick hollow rectangle in the lower-right region.
    - The upper-left region contains 2 to 4 isolated 3x3 black/grey symbol panels whose top-left corners lie on a down-right diagonal with one blank cell gap between panels.
    - Each grey mask is one glyph from a fixed task codebook, while the frame interior is entirely background.

    Rule:
    - Read the symbol panels in diagonal order from upper-left to lower-right.
    - Convert each panel's grey mask into a color using the learned symbol-to-color codebook.
    - Use the decoded colors as an outer-to-inner palette for the frame interior: each color paints the next one-cell-thick rectangular ring, and the final decoded color fills the entire remaining center.
    - Keep the orange frame, the symbol panels, and the surrounding background unchanged.

    Key insight:
    - The diagonal grey symbols do not indicate pixels to repaint directly; they encode the ordered palette that drives the concentric fill geometry inside the distant frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_bbox = _rectangular_frame_bbox(g == Color.ORANGE)
    assert frame_bbox is not None, "expected a single orange hollow frame"
    r0, c0, r1, c1 = frame_bbox
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == Color.BLACK), "expected an empty frame interior"

    panels = _find_symbol_panels(g)
    palette = _decode_palette(g, panels)
    return _fill_concentric(g, frame_bbox, palette)


def _oracle_output(
    inp: np.ndarray,
    panel_positions: list[tuple[int, int]],
    symbol_sequence: list[tuple[tuple[int, ...], ...]],
    frame_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    palette = [int(_SYMBOL_TO_COLOR[symbol]) for symbol in symbol_sequence]
    for (r, c), symbol in zip(panel_positions, symbol_sequence):
        observed = _mask_key(inp[r : r + 3, c : c + 3] == Color.GREY)
        assert observed == symbol, "rendered panel does not match its intended symbol"
    return _fill_concentric(inp, frame_bbox, palette)


def _sample_symbol_sequence() -> list[tuple[tuple[int, ...], ...]]:
    global _GEN_CALL_COUNT

    if _GEN_CALL_COUNT < len(_BOOTSTRAP_SEQUENCES):
        sequence = list(_BOOTSTRAP_SEQUENCES[_GEN_CALL_COUNT])
        _GEN_CALL_COUNT += 1
        return sequence

    _GEN_CALL_COUNT += 1
    k = random.randint(2, 4)
    return random.sample(list(_SYMBOL_PATTERNS), k=k)


def _draw_panel(
    grid: np.ndarray, top_left: tuple[int, int], symbol: tuple[tuple[int, ...], ...]
) -> None:
    r0, c0 = top_left
    mask = np.array(symbol, dtype=int)
    block = np.where(mask == 1, Color.GREY, Color.BLACK)
    grid[r0 : r0 + 3, c0 : c0 + 3] = block


def _draw_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = Color.ORANGE
    grid[r1, c0 : c1 + 1] = Color.ORANGE
    grid[r0 : r1 + 1, c0] = Color.ORANGE
    grid[r0 : r1 + 1, c1] = Color.ORANGE


def _semantic_check(
    inp: np.ndarray,
    out: np.ndarray,
    panel_positions: list[tuple[int, int]],
    symbol_sequence: list[tuple[tuple[int, ...], ...]],
    frame_bbox: tuple[int, int, int, int],
) -> bool:
    if _rectangular_frame_bbox(inp == Color.ORANGE) != frame_bbox:
        return False
    r0, c0, r1, c1 = frame_bbox
    if not np.all(inp[r0 + 1 : r1, c0 + 1 : c1] == Color.BLACK):
        return False
    if _find_symbol_panels(inp) != panel_positions:
        return False
    if _decode_palette(inp, panel_positions) != [int(_SYMBOL_TO_COLOR[s]) for s in symbol_sequence]:
        return False

    oracle = _oracle_output(inp, panel_positions, symbol_sequence, frame_bbox)
    if not np.array_equal(out, oracle):
        return False

    for i, symbol in enumerate(symbol_sequence[:-1]):
        color = int(_SYMBOL_TO_COLOR[symbol])
        rr0 = r0 + 1 + i
        cc0 = c0 + 1 + i
        rr1 = r1 - 1 - i
        cc1 = c1 - 1 - i
        ring = np.zeros_like(inp, dtype=bool)
        ring[rr0, cc0 : cc1 + 1] = True
        ring[rr1, cc0 : cc1 + 1] = True
        ring[rr0 : rr1 + 1, cc0] = True
        ring[rr0 : rr1 + 1, cc1] = True
        if not np.all(out[ring] == color):
            return False

    final_color = int(_SYMBOL_TO_COLOR[symbol_sequence[-1]])
    k = len(symbol_sequence) - 1
    center = out[r0 + 1 + k : r1 - k, c0 + 1 + k : c1 - k]
    if center.size == 0 or not np.all(center == final_color):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    symbol_sequence = _sample_symbol_sequence()
    k = len(symbol_sequence)
    diag_offset = random.randint(0, 2)
    diag_span = 4 * k - 1
    frame_h = random.randint(9, 11)
    frame_w = random.randint(9, 11)
    gap_r = random.randint(2, 4)
    gap_c = random.randint(2, 4)
    bottom_pad = random.randint(0, 3)
    right_pad = random.randint(0, 3)
    extra_h = random.randint(0, 3)
    extra_w = random.randint(0, 3)

    min_h = diag_offset + diag_span + gap_r + frame_h + bottom_pad
    min_w = diag_offset + diag_span + gap_c + frame_w + right_pad
    height = min_h + extra_h
    width = min_w + extra_w
    if max(height, width) > 30:
        return None

    frame_r0 = height - frame_h - bottom_pad
    frame_c0 = width - frame_w - right_pad
    if frame_r0 < diag_offset + diag_span + gap_r or frame_c0 < diag_offset + diag_span + gap_c:
        return None

    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)
    inp = np.full((height, width), Color.BLACK, dtype=int)
    _draw_frame(inp, frame_bbox)

    panel_positions = []
    for i, symbol in enumerate(symbol_sequence):
        top_left = (diag_offset + 4 * i, diag_offset + 4 * i)
        panel_positions.append(top_left)
        _draw_panel(inp, top_left, symbol)

    oracle = _oracle_output(inp, panel_positions, symbol_sequence, frame_bbox)
    if not _semantic_check(inp, oracle, panel_positions, symbol_sequence, frame_bbox):
        return None
    if not np.array_equal(main(inp), oracle):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
