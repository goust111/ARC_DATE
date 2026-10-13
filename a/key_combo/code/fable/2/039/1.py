import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components


BLACK = Color.BLACK
LEGEND_BG = Color.GREY
LEGEND_H = 3
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_RETRIES = 300


def _parse_legend(g: np.ndarray) -> tuple[int, tuple[int, ...]]:
    assert g.ndim == 2
    rows, cols = g.shape
    legend_h = 0
    while legend_h < rows and np.any(g[legend_h] == LEGEND_BG):
        legend_h += 1

    assert 0 < legend_h < rows, "a nonempty grey legend band must sit above the main grid"
    assert not np.any(g[legend_h:] == LEGEND_BG), "grey is reserved for the top legend band"

    band = g[:legend_h, :]
    cycle: list[int] = []
    for c in range(cols):
        non_grey = band[:, c][band[:, c] != LEGEND_BG]
        if non_grey.size == 0:
            continue
        assert non_grey.size == 1, "legend colors must be single cells"
        color = int(non_grey[0])
        assert color not in (BLACK, LEGEND_BG), "legend entries must be visible non-background colors"
        cycle.append(color)

    assert len(cycle) >= 2, "legend must contain a color cycle"
    assert len(set(cycle)) == len(cycle), "legend colors must be unique so every offset is determined"
    return legend_h, tuple(cycle)


def _frame_border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_hollow_frames(base: np.ndarray, cycle: tuple[int, ...]) -> list[tuple[int, int, int, int, int]]:
    allowed = np.isin(base, np.array((BLACK, *cycle), dtype=int))
    assert bool(np.all(allowed)), "main grid may contain only black and legend colors"

    components = find_components(
        base,
        bg=BLACK,
        connectivity=4,
        monochromatic=True,
        colors=list(cycle),
        return_format="dict",
    )
    assert components, "expected at least one hollow rectangle frame"

    covered = np.zeros_like(base, dtype=bool)
    frames: list[tuple[int, int, int, int, int]] = []
    for comp in components:
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h, w = r1 - r0 + 1, c1 - c0 + 1
        assert h >= 3 and w >= 3, "a hollow rectangle needs a black interior"

        local = base[r0 : r1 + 1, c0 : c1 + 1]
        border = _frame_border_mask(h, w)
        assert np.all(local[border] == color), "frame border must be complete and monochrome"
        assert np.all(local[~border] == BLACK), "frame interior must be black in the input"
        assert np.array_equal(local == color, border), "foreground component must be exactly a one-pixel frame"

        covered[r0 : r1 + 1, c0 : c1 + 1] |= border
        frames.append((r0, c0, r1, c1, color))

    assert np.array_equal(covered, base != BLACK), "all foreground pixels must belong to one frame"
    return frames


def _paint_cycle_shells(
    shape: tuple[int, int],
    frames: list[tuple[int, int, int, int, int]],
    cycle: tuple[int, ...],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    index = {color: i for i, color in enumerate(cycle)}
    for r0, c0, r1, c1, color in frames:
        start = index[color]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                depth = min(r - r0, r1 - r, c - c0, c1 - c)
                out[r, c] = cycle[(start + depth) % len(cycle)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow rectangle with concentric shells using the top legend as a cyclic color sequence.

    Input structure:
    - A grey legend band occupies the top rows and contains single-cell color entries ordered from left to right.
    - Below the band is a black main grid containing several separated one-pixel-thick hollow rectangle frames.
    - Every frame border color appears exactly once in the legend cycle, and every frame interior is black.

    Rule:
    - Remove the grey legend band and return only the main grid.
    - Keep each frame border color as shell depth 0.
    - For cells at Chebyshev distance d from that frame border, paint the legend color d steps after the frame's own border color, wrapping around the legend cycle.

    Key insight:
    - The legend supplies a cyclic order, but each rectangle chooses its own phase from its border color rather than starting at the leftmost legend entry.
    """
    g = np.asarray(grid, dtype=int)
    legend_h, cycle = _parse_legend(g)
    base = g[legend_h:, :].copy()
    frames = _find_hollow_frames(base, cycle)
    return _paint_cycle_shells(base.shape, frames, cycle)


def _sample_spaced_columns(width: int, n: int) -> list[int] | None:
    usable = list(range(1, width - 1))
    if len(usable) < n:
        return None
    for _ in range(100):
        cols = sorted(random.sample(usable, n))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    cols = np.linspace(1, width - 2, n, dtype=int).tolist()
    if len(set(cols)) == n and all(b - a >= 2 for a, b in zip(cols, cols[1:])):
        return cols
    return None


def _render_legend(width: int, cycle: tuple[int, ...]) -> np.ndarray | None:
    cols = _sample_spaced_columns(width, len(cycle))
    if cols is None:
        return None
    legend = np.full((LEGEND_H, width), LEGEND_BG, dtype=int)
    for c, color in zip(cols, cycle):
        legend[LEGEND_H // 2, c] = color
    return legend


def _draw_hollow_frame(g: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    g[r0 : r1 + 1, c0] = color
    g[r0 : r1 + 1, c1] = color
    g[r0, c0 : c1 + 1] = color
    g[r1, c0 : c1 + 1] = color


def _place_frames(
    h: int,
    w: int,
    dims: list[tuple[int, int]],
    colors: list[int],
) -> list[tuple[int, int, int, int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    items = sorted(zip(dims, colors), key=lambda item: item[0][0] * item[0][1], reverse=True)
    frames: list[tuple[int, int, int, int, int]] = []

    for (fh, fw), color in items:
        if fh > h - 2 or fw > w - 2:
            return None
        placed = False
        for _ in range(250):
            r0 = random.randint(1, h - fh - 1)
            c0 = random.randint(1, w - fw - 1)
            r1, c1 = r0 + fh - 1, c0 + fw - 1
            pr0, pc0 = max(0, r0 - 1), max(0, c0 - 1)
            pr1, pc1 = min(h - 1, r1 + 1), min(w - 1, c1 + 1)
            if occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                continue
            occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
            frames.append((r0, c0, r1, c1, color))
            placed = True
            break
        if not placed:
            return None

    return frames


def _sample_dimensions(main_h: int, main_w: int, cycle_len: int, count: int) -> list[tuple[int, int]] | None:
    deep_min = 2 * cycle_len + 1
    deep_h_max = min(13, main_h - 2)
    deep_w_max = min(14, main_w - 2)
    if deep_min > deep_h_max or deep_min > deep_w_max:
        return None

    dims = [(random.randint(deep_min, deep_h_max), random.randint(deep_min, deep_w_max))]
    for _ in range(count - 1):
        fh = random.randint(4, min(9, main_h - 2))
        fw = random.randint(4, min(10, main_w - 2))
        if fh == fw and random.random() < 0.7:
            if fw < min(10, main_w - 2):
                fw += 1
            elif fw > 4:
                fw -= 1
        dims.append((fh, fw))
    return dims


def _sample_frame_colors(cycle: tuple[int, ...], count: int) -> list[int]:
    colors = [cycle[0], random.choice(cycle[1:])]
    while len(colors) < count:
        colors.append(random.choice(cycle))
    random.shuffle(colors)
    return colors


def _render_main(
    shape: tuple[int, int],
    frames: list[tuple[int, int, int, int, int]],
) -> np.ndarray:
    g = np.full(shape, BLACK, dtype=int)
    for r0, c0, r1, c1, color in frames:
        _draw_hollow_frame(g, (r0, c0, r1, c1), color)
    return g


def _wrong_global_interior(
    shape: tuple[int, int],
    frames: list[tuple[int, int, int, int, int]],
    cycle: tuple[int, ...],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for r0, c0, r1, c1, color in frames:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                depth = min(r - r0, r1 - r, c - c0, c1 - c)
                out[r, c] = color if depth == 0 else cycle[depth % len(cycle)]
    return out


def _flat_one_step_interior(
    shape: tuple[int, int],
    frames: list[tuple[int, int, int, int, int]],
    cycle: tuple[int, ...],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    index = {color: i for i, color in enumerate(cycle)}
    for r0, c0, r1, c1, color in frames:
        fill = cycle[(index[color] + 1) % len(cycle)]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                depth = min(r - r0, r1 - r, c - c0, c1 - c)
                out[r, c] = color if depth == 0 else fill
    return out


def _gen_attempt() -> np.ndarray | None:
    cycle_len = random.choice([3, 3, 4, 4, 5])
    cycle = tuple(random.sample(PALETTE, cycle_len))
    main_h = random.randint(max(18, 2 * cycle_len + 5), 27)
    main_w = random.randint(max(18, 2 * cycle_len + 5), 30)
    frame_count = random.choice([2, 3, 3, 4])

    dims = _sample_dimensions(main_h, main_w, cycle_len, frame_count)
    if dims is None:
        return None
    colors = _sample_frame_colors(cycle, frame_count)
    frames = _place_frames(main_h, main_w, dims, colors)
    if frames is None:
        return None

    max_depths = [min(r1 - r0, c1 - c0) // 2 for r0, c0, r1, c1, _ in frames]
    if max(max_depths) < cycle_len:
        return None
    if len({color for *_box, color in frames}) < 2:
        return None
    if all(color == cycle[0] for *_box, color in frames):
        return None

    base = _render_main((main_h, main_w), frames)
    expected = _paint_cycle_shells((main_h, main_w), frames, cycle)
    if np.array_equal(expected, _wrong_global_interior((main_h, main_w), frames, cycle)):
        return None
    if np.array_equal(expected, _flat_one_step_interior((main_h, main_w), frames, cycle)):
        return None
    if np.array_equal(expected, _paint_cycle_shells((main_h, main_w), frames, tuple(reversed(cycle)))):
        return None

    legend = _render_legend(main_w, cycle)
    if legend is None:
        return None
    inp = np.vstack([legend, base])

    try:
        detected_h, detected_cycle = _parse_legend(inp)
        detected_frames = _find_hollow_frames(inp[detected_h:, :], detected_cycle)
        actual = main(inp)
    except Exception:
        return None

    if detected_cycle != cycle or len(detected_frames) != len(frames):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for fable/2/039/1")
