from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import colors_present, find_components


BG = 0
LOWER_HEIGHT = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
RNG = np.random.default_rng()
MAX_ATTEMPTS = 500


def _desc_support(shape: tuple[int, int], diff: int) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    out: list[tuple[int, int]] = []
    for c in range(w):
        r = c + int(diff)
        if 0 <= r < h:
            out.append((int(r), int(c)))
    return out


def _asc_support(shape: tuple[int, int], total: int) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    out: list[tuple[int, int]] = []
    for c in range(w):
        r = int(total) - c
        if 0 <= r < h:
            out.append((int(r), int(c)))
    return out


def _infer_line_support(coords: np.ndarray, shape: tuple[int, int]) -> set[tuple[int, int]]:
    coords = np.asarray(coords, dtype=int)
    assert coords.ndim == 2 and coords.shape[1] == 2
    assert coords.shape[0] >= 3, "each line must expose at least three visible pixels"

    rows = {int(r) for r in coords[:, 0]}
    cols = {int(c) for c in coords[:, 1]}
    diffs = {int(r - c) for r, c in coords}
    sums = {int(r + c) for r, c in coords}

    kinds: list[tuple[str, int]] = []
    if len(rows) == 1:
        kinds.append(("h", next(iter(rows))))
    if len(cols) == 1:
        kinds.append(("v", next(iter(cols))))
    if len(diffs) == 1:
        kinds.append(("d", next(iter(diffs))))
    if len(sums) == 1:
        kinds.append(("a", next(iter(sums))))
    assert len(kinds) == 1, "visible pixels must determine a unique line family"

    kind, param = kinds[0]
    h, w = map(int, shape)
    if kind == "h":
        return {(int(param), c) for c in range(w)}
    if kind == "v":
        return {(r, int(param)) for r in range(h)}
    if kind == "d":
        return set(_desc_support(shape, int(param)))
    return set(_asc_support(shape, int(param)))


def _find_separator_row(grid: np.ndarray) -> int:
    h = int(grid.shape[0])
    candidates = [
        r
        for r in range(1, h - 1)
        if np.all(grid[r] == BG) and np.any(grid[:r] != BG) and np.any(grid[r + 1 :] != BG)
    ]
    assert len(candidates) == 1, "expected exactly one black separator row"
    return int(candidates[0])


def _extract_bars(lower: np.ndarray) -> list[dict[str, int]]:
    comps = find_components(lower, bg=BG, connectivity=4, monochromatic=True)
    assert len(comps) == 4, "expected four monochrome bars"

    lower_h = int(lower.shape[0])
    bars: list[dict[str, int]] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        color = int(comp["color"])
        area = int(comp["area"])
        assert c0 == c1, "bars must be one cell wide"
        assert r1 == lower_h - 1, "bars must be bottom-aligned"
        assert area == r1 - r0 + 1, "bars must be solid vertical runs"
        bars.append({"color": color, "col": c0, "height": area})

    bars.sort(key=lambda bar: bar["col"])
    assert len({bar["color"] for bar in bars}) == 4, "bar colors must be unique"
    return bars


def _recover_depth_chain(upper: np.ndarray, colors: list[int]) -> list[int]:
    supports = {}
    for color in colors:
        pts = np.argwhere(upper == int(color))
        assert pts.size > 0, f"line color {color} must appear in upper area"
        supports[int(color)] = _infer_line_support(pts, upper.shape)

    outs = {int(color): set() for color in colors}
    indeg = {int(color): 0 for color in colors}
    for a, b in combinations(colors, 2):
        inter = supports[int(a)] & supports[int(b)]
        assert len(inter) == 1, "each pair of reconstructed lines must meet exactly once"
        cell = next(iter(inter))
        visible = int(upper[cell])
        assert visible in {int(a), int(b)}, "crossing pixel must show one of the two line colors"
        low, high = (int(b), int(a)) if visible == int(a) else (int(a), int(b))
        if high not in outs[low]:
            outs[low].add(high)
            indeg[high] += 1

    remaining = set(map(int, colors))
    order: list[int] = []
    while remaining:
        roots = sorted(color for color in remaining if indeg[color] == 0)
        assert len(roots) == 1, "over/under constraints must yield one unique depth chain"
        cur = int(roots[0])
        remaining.remove(cur)
        order.append(cur)
        for nxt in outs[cur]:
            indeg[nxt] -= 1

    assert len(order) == 4
    return order


def _render_bars(
    out: np.ndarray,
    sep: int,
    bars: list[dict[str, int]],
    heights_by_color: dict[int, int],
) -> None:
    out[sep + 1 :, :] = BG
    base = int(out.shape[0])
    for bar in bars:
        color = int(bar["color"])
        col = int(bar["col"])
        height = int(heights_by_color[color])
        assert 1 <= height <= LOWER_HEIGHT
        out[base - height : base, col] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Use the over/under order of four crossing lines as a cyclic key that permutes bar heights by color.

    Input structure:
    - Black background with one full black separator row splitting the grid into an upper line area and a lower bar area.
    - The upper area contains exactly four single-pixel straight lines in four distinct colors; each line is horizontal, vertical, or a 45-degree diagonal, and crossings show only the upper line's color.
    - The lower area contains four bottom-aligned one-cell-wide bars in the same four colors, with fixed columns and pairwise distinct heights.

    Rule:
    - Recover each line's full support from its visible collinear pixels, then read every crossing as an above/below constraint between the two involved colors.
    - Topologically order the colors from deepest to highest to obtain the unique depth chain c0 -> c1 -> c2 -> c3.
    - Interpret that chain as the cycle c0 -> c1 -> c2 -> c3 -> c0, and replace each bar's height with the original height of the next color in the cycle while keeping each bar's own color and column fixed.
    - Leave the upper line drawing and the separator row unchanged.

    Key insight:
    - The crossing colors do not directly change the picture above; instead they encode a unique depth order, and that hidden order is the only key that tells which other color's bar height each column must inherit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator_row(g)
    upper = g[:sep, :]
    lower = g[sep + 1 :, :]
    assert lower.shape[0] == LOWER_HEIGHT, "generator and solver both expect five lower rows"

    bars = _extract_bars(lower)
    colors = [int(bar["color"]) for bar in bars]
    assert colors_present(upper, exclude=[BG]) == sorted(colors)
    assert colors_present(lower, exclude=[BG]) == sorted(colors)
    chain = _recover_depth_chain(upper, colors)

    next_color = {int(chain[i]): int(chain[(i + 1) % len(chain)]) for i in range(len(chain))}
    original_heights = {int(bar["color"]): int(bar["height"]) for bar in bars}
    remapped_heights = {
        color: int(original_heights[next_color[color]])
        for color in map(int, colors)
    }

    out = g.copy()
    _render_bars(out, sep, bars, remapped_heights)
    return out


def _sample_bar_columns(width: int) -> list[int] | None:
    choices = np.arange(1, width - 1, dtype=int)
    if choices.size < 4:
        return None
    for _ in range(100):
        cols = sorted(int(x) for x in RNG.choice(choices, size=4, replace=False))
        if min(cols[i + 1] - cols[i] for i in range(3)) >= 2:
            return cols
    return None


def _sample_line_layout(upper_h: int, width: int) -> dict[str, int] | None:
    shape = (int(upper_h), int(width))
    for _ in range(300):
        rh = int(RNG.integers(2, upper_h - 2))
        cv = int(RNG.integers(2, width - 2))
        diff = int(RNG.integers(-(width - 3), upper_h - 2))
        total = int(RNG.integers(3, upper_h + width - 4))

        desc = _desc_support(shape, diff)
        asc = _asc_support(shape, total)
        if len(desc) < 7 or len(asc) < 7:
            continue
        if (diff + total) % 2 != 0:
            continue

        pts = {
            "hv": (rh, cv),
            "hd": (rh, rh - diff),
            "ha": (rh, total - rh),
            "vd": (cv + diff, cv),
            "va": (total - cv, cv),
            "da": ((diff + total) // 2, (total - diff) // 2),
        }
        if any(not (1 <= r < upper_h - 1 and 1 <= c < width - 1) for r, c in pts.values()):
            continue
        if len(set(pts.values())) != 6:
            continue
        return {"h": rh, "v": cv, "d": diff, "a": total}
    return None


def _line_cells(shape: tuple[int, int], kind: str, param: int) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    if kind == "h":
        return [(int(param), c) for c in range(w)]
    if kind == "v":
        return [(r, int(param)) for r in range(h)]
    if kind == "d":
        return _desc_support(shape, int(param))
    return _asc_support(shape, int(param))


def _cyclic_orders(items: list[int]) -> list[tuple[int, ...]]:
    n = len(items)
    return [tuple(items[i:] + items[:i]) for i in range(n)]


def _is_shortcut_bar_order(slot_colors: list[int], chain: list[int]) -> bool:
    seq = tuple(map(int, slot_colors))
    chain = list(map(int, chain))
    rev = list(reversed(chain))
    return seq in _cyclic_orders(chain) or seq in _cyclic_orders(rev)


def _make_expected_output(
    inp: np.ndarray,
    sep: int,
    bars: list[dict[str, int]],
    chain: list[int],
) -> np.ndarray:
    out = inp.copy()
    next_color = {int(chain[i]): int(chain[(i + 1) % len(chain)]) for i in range(len(chain))}
    original_heights = {int(bar["color"]): int(bar["height"]) for bar in bars}
    new_heights = {color: int(original_heights[next_color[color]]) for color in original_heights}
    _render_bars(out, sep, bars, new_heights)
    return out


def _slot_rotated_output(inp: np.ndarray, sep: int, bars: list[dict[str, int]], shift: int) -> np.ndarray:
    out = inp.copy()
    heights = [int(bar["height"]) for bar in bars]
    rotated = heights[int(shift) :] + heights[: int(shift)]
    _render_bars(
        out,
        sep,
        bars,
        {int(bar["color"]): int(rotated[i]) for i, bar in enumerate(bars)},
    )
    return out


def _gen_attempt() -> np.ndarray | None:
    upper_h = int(RNG.integers(10, 15))
    width = int(RNG.integers(17, 23))
    if upper_h + 1 + LOWER_HEIGHT > 30 or width > 30:
        return None

    cols = _sample_bar_columns(width)
    if cols is None:
        return None
    layout = _sample_line_layout(upper_h, width)
    if layout is None:
        return None

    colors = [int(x) for x in RNG.choice(np.array(COLOR_POOL, dtype=int), size=4, replace=False)]
    bars_left_to_right = [int(x) for x in RNG.permutation(colors)]
    depth_chain = [int(x) for x in RNG.permutation(colors)]
    if _is_shortcut_bar_order(bars_left_to_right, depth_chain):
        return None

    orientation_for_color = dict(zip(colors, ["h", "v", "d", "a"]))
    param_for_kind = {"h": layout["h"], "v": layout["v"], "d": layout["d"], "a": layout["a"]}
    heights = dict(zip(colors, RNG.permutation(np.array([2, 3, 4, 5], dtype=int)).tolist()))

    h = upper_h + 1 + LOWER_HEIGHT
    inp = np.zeros((h, width), dtype=int)
    upper_shape = (upper_h, width)

    for color in depth_chain:
        kind = orientation_for_color[int(color)]
        for r, c in _line_cells(upper_shape, kind, int(param_for_kind[kind])):
            inp[r, c] = int(color)

    sep = upper_h
    for color, col in zip(bars_left_to_right, cols):
        height = int(heights[int(color)])
        inp[h - height : h, int(col)] = int(color)

    try:
        bars = _extract_bars(inp[sep + 1 :, :])
        recovered_chain = _recover_depth_chain(inp[:sep, :], [int(bar["color"]) for bar in bars])
    except Exception:
        return None

    if recovered_chain != depth_chain:
        return None

    expected = _make_expected_output(inp, sep, bars, depth_chain)
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    # Reject examples that collapse to a simple left/right slot rotation.
    if np.array_equal(expected, _slot_rotated_output(inp, sep, bars, 1)):
        return None
    if np.array_equal(expected, _slot_rotated_output(inp, sep, bars, -1)):
        return None
    if np.array_equal(expected, inp):
        return None

    # Also reject the reverse depth cycle to keep the chain direction semantically necessary.
    reverse_expected = _make_expected_output(inp, sep, bars, list(reversed(depth_chain)))
    if np.array_equal(expected, reverse_expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid crossing-order bar permutation task")
