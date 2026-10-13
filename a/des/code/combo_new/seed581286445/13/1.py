from __future__ import annotations

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _frame_from_component(coords: np.ndarray, color: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return None

    rel = coords - np.array([r0, c0], dtype=int)
    border = (
        (rel[:, 0] == 0)
        | (rel[:, 0] == h - 1)
        | (rel[:, 1] == 0)
        | (rel[:, 1] == w - 1)
    )
    if not bool(np.all(border)):
        return None

    mask = np.zeros((h, w), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    if not bool(mask[0, :].all() and mask[-1, :].all() and mask[:, 0].all() and mask[:, -1].all()):
        return None

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "inner_h": h - 2,
        "inner_w": w - 2,
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            frame = _frame_from_component(comp, color)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda item: tuple(map(int, item["bbox"])))
    assert frames, "expected at least one rectangular frame"
    return frames


def _fill_window(count: int, inner_h: int, inner_w: int) -> tuple[int, int, int, int]:
    if count == 1:
        return 0, min(1, inner_h), 0, inner_w
    if count == 2:
        return 0, inner_h, 0, min(2, inner_w)
    if count == 3:
        return max(0, inner_h - 3), inner_h, 0, inner_w
    if count == 4:
        return 0, inner_h, max(0, inner_w - 4), inner_w
    raise AssertionError(f"unsupported fragment count {count}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each rectangular frame's internal fragment count to choose a directional fill stripe.

    Input structure:
    - The most common color is background.
    - Several non-overlapping 1-pixel-thick hollow rectangular frames appear in different colors.
    - Inside each frame are 1 to 4 small monochrome 4-connected fragments; within one frame, all fragments share one color different from the frame color.

    Rule:
    - For each frame, count the number N of connected fragments inside its interior.
    - Use the fragment color to fill only background cells in a clipped stripe inside that frame: N=1 fills the top interior row, N=2 fills the left two interior columns, N=3 fills the bottom three interior rows, and N=4 fills the right four interior columns.
    - Keep the original frame pixels and fragment pixels unchanged.

    Key insight:
    - The fragments are grouped by geometric containment inside each frame, and their count simultaneously encodes both fill direction and fill thickness.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    out = g.copy()
    for frame in _find_frames(g, bg):
        r0, c0, r1, c1 = map(int, frame["bbox"])
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        fragment_mask = inner != bg
        fragments = _components(fragment_mask)
        count = len(fragments)
        assert 1 <= count <= 4, "each frame must contain between 1 and 4 fragments"
        sizes = [int(comp.shape[0]) for comp in fragments]
        assert all(2 <= size <= 4 for size in sizes), "fragment sizes must stay within 2..4"

        fragment_colors = np.unique(inner[fragment_mask])
        assert fragment_colors.size == 1, "all fragments inside one frame must share one color"
        fill_color = int(fragment_colors[0])

        rr0, rr1, cc0, cc1 = _fill_window(count, inner.shape[0], inner.shape[1])
        view = out[r0 + 1 : r1, c0 + 1 : c1]
        stripe = np.zeros_like(view, dtype=bool)
        stripe[rr0:rr1, cc0:cc1] = True
        view[stripe & (view == bg)] = fill_color

    return out


_RNG = np.random.default_rng()
_GEN_RETRIES = 400
_COUNTS = (1, 2, 3, 4)

_FRAGMENT_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((1, 0), (1, 1), (1, 2), (0, 1)),
    ((0, 0), (1, 0), (2, 0), (1, 1)),
    ((0, 1), (1, 1), (2, 1), (1, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
)


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[int(_RNG.integers(0, len(seq)))]


def _shape_bbox(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [cell[0] for cell in shape]
    cols = [cell[1] for cell in shape]
    return max(rows) + 1, max(cols) + 1


def _sample_fragment(slot_h: int, slot_w: int) -> list[tuple[int, int]]:
    candidates: list[list[tuple[int, int]]] = []
    for shape in _FRAGMENT_SHAPES:
        sh, sw = _shape_bbox(shape)
        if sh > slot_h or sw > slot_w:
            continue
        for off_r in range(slot_h - sh + 1):
            for off_c in range(slot_w - sw + 1):
                cells = [(off_r + r, off_c + c) for r, c in shape]
                candidates.append(cells)
    assert candidates, "every slot must admit at least one fragment shape"
    return _choice(candidates)


def _pick_fragment_colors(bg: int, frame_colors: list[int]) -> list[int]:
    result = []
    for idx, frame_color in enumerate(frame_colors):
        other_frame_colors = [color for j, color in enumerate(frame_colors) if j != idx]
        if other_frame_colors and bool(_RNG.integers(0, 2)):
            pool = [color for color in other_frame_colors if color != frame_color]
        else:
            pool = [color for color in range(10) if color not in (bg, frame_color)]
        result.append(int(_choice(pool)))
    return result


def _build_frame_spec(count: int, frame_color: int, fragment_color: int) -> dict:
    top = _randint(0, 1)
    bottom = _randint(0, 1)
    left = _randint(0, 1)
    right = _randint(0, 1)
    row_heights = [_randint(2, 3), _randint(2, 3)]
    col_widths = [_randint(2, 3), _randint(2, 3)]
    row_gap = _randint(1, 2)
    col_gap = _randint(1, 2)

    inner_h = top + row_heights[0] + row_gap + row_heights[1] + bottom
    inner_w = left + col_widths[0] + col_gap + col_widths[1] + right
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    row_starts = [top, top + row_heights[0] + row_gap]
    col_starts = [left, left + col_widths[0] + col_gap]
    slots = [
        (row_starts[0], row_starts[0] + row_heights[0], col_starts[0], col_starts[0] + col_widths[0]),
        (row_starts[0], row_starts[0] + row_heights[0], col_starts[1], col_starts[1] + col_widths[1]),
        (row_starts[1], row_starts[1] + row_heights[1], col_starts[0], col_starts[0] + col_widths[0]),
        (row_starts[1], row_starts[1] + row_heights[1], col_starts[1], col_starts[1] + col_widths[1]),
    ]

    active = list(map(int, _RNG.choice(4, size=count, replace=False)))
    fragments: list[list[tuple[int, int]]] = []
    for idx in active:
        sr0, sr1, sc0, sc1 = slots[idx]
        local_cells = _sample_fragment(sr1 - sr0, sc1 - sc0)
        fragments.append([(sr0 + r, sc0 + c) for r, c in local_cells])

    return {
        "count": int(count),
        "frame_color": int(frame_color),
        "fragment_color": int(fragment_color),
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "outer_h": int(outer_h),
        "outer_w": int(outer_w),
        "fragments": fragments,
    }


def _layout_specs(specs: list[dict]) -> tuple[int, int] | None:
    ordered = list(specs)
    _RNG.shuffle(ordered)

    row_heights = [
        max(int(ordered[0]["outer_h"]), int(ordered[1]["outer_h"])) + _randint(0, 1),
        max(int(ordered[2]["outer_h"]), int(ordered[3]["outer_h"])) + _randint(0, 1),
    ]
    col_widths = [
        max(int(ordered[0]["outer_w"]), int(ordered[2]["outer_w"])) + _randint(0, 1),
        max(int(ordered[1]["outer_w"]), int(ordered[3]["outer_w"])) + _randint(0, 1),
    ]
    top_pad = _randint(1, 2)
    bottom_pad = _randint(1, 2)
    left_pad = _randint(1, 2)
    right_pad = _randint(1, 2)
    row_gap = _randint(1, 2)
    col_gap = _randint(1, 2)

    height = top_pad + row_heights[0] + row_gap + row_heights[1] + bottom_pad
    width = left_pad + col_widths[0] + col_gap + col_widths[1] + right_pad
    if height > 30 or width > 30:
        return None

    row_starts = [top_pad, top_pad + row_heights[0] + row_gap]
    col_starts = [left_pad, left_pad + col_widths[0] + col_gap]
    for idx, spec in enumerate(ordered):
        row = idx // 2
        col = idx % 2
        slack_r = row_heights[row] - int(spec["outer_h"])
        slack_c = col_widths[col] - int(spec["outer_w"])
        spec["r0"] = row_starts[row] + _randint(0, slack_r)
        spec["c0"] = col_starts[col] + _randint(0, slack_c)

    specs[:] = ordered
    return int(height), int(width)


def _render_input(height: int, width: int, bg: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((height, width), int(bg), dtype=int)
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        frame_color = int(spec["frame_color"])
        fragment_color = int(spec["fragment_color"])

        grid[r0, c0 : c0 + outer_w] = frame_color
        grid[r0 + outer_h - 1, c0 : c0 + outer_w] = frame_color
        grid[r0 : r0 + outer_h, c0] = frame_color
        grid[r0 : r0 + outer_h, c0 + outer_w - 1] = frame_color

        for fragment in spec["fragments"]:
            for rr, cc in fragment:
                grid[r0 + 1 + int(rr), c0 + 1 + int(cc)] = fragment_color
    return grid


def _render_expected(inp: np.ndarray, bg: int, specs: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        fill_color = int(spec["fragment_color"])
        rr0, rr1, cc0, cc1 = _fill_window(int(spec["count"]), inner_h, inner_w)
        view = out[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w]
        stripe = np.zeros((inner_h, inner_w), dtype=bool)
        stripe[rr0:rr1, cc0:cc1] = True
        view[stripe & (view == bg)] = fill_color
    return out


def _valid_background(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    return winners.size == 1 and int(winners[0]) == int(bg)


def _gen_attempt() -> np.ndarray | None:
    bg = _randint(0, 9)
    frame_colors = list(map(int, _RNG.choice([c for c in range(10) if c != bg], size=4, replace=False)))
    fragment_colors = _pick_fragment_colors(bg, frame_colors)

    specs = [
        _build_frame_spec(count, frame_color, fragment_color)
        for count, frame_color, fragment_color in zip(_COUNTS, frame_colors, fragment_colors, strict=False)
    ]

    shape = _layout_specs(specs)
    if shape is None:
        return None
    height, width = shape

    grid = _render_input(height, width, bg, specs)
    if not _valid_background(grid, bg):
        return None

    expected = _render_expected(grid, bg, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
