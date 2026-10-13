from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


# Visible warm-to-cool order for duplicate-size tie-breaking.
_SPECTRUM_ORDER = (2, 7, 4, 3, 8, 1, 6, 9)
_SPECTRUM_RANK = {color: i for i, color in enumerate(_SPECTRUM_ORDER)}

# Generator palette and bounds.
_BG = 0
_RULER = 5
_GRID_MAX = 28
_GEN_RETRIES = 400
_PACK_RETRIES = 80
_PLACE_ATTEMPTS = 120
_LAYOUT_OPTIONS = ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2))
_HEIGHT_VALUES = (2, 3, 4, 5, 6)
_WIDTH_VALUES = (2, 3, 4, 5, 6, 7)
_FRAGMENT_COLORS = tuple(color for color in _SPECTRUM_ORDER if color != _RULER)
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _color_rank(color: int) -> tuple[int, int]:
    return (_SPECTRUM_RANK.get(int(color), len(_SPECTRUM_ORDER)), int(color))


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _ruler_color(grid: np.ndarray, bg: int) -> int:
    edge = np.concatenate((grid[0, :], grid[:, 0]))
    colors = [int(v) for v in edge if int(v) != bg]
    assert colors, "expected grey ruler markers on the top row and left column"
    distinct = sorted(set(colors))
    assert len(distinct) == 1, "expected a single ruler color on the edges"
    return distinct[0]


def _segments_from_markers(size: int, markers: list[int]) -> list[tuple[int, int]]:
    prev = 1
    segments: list[tuple[int, int]] = []
    for line in sorted(markers):
        assert 1 <= int(line) < int(size), "grid lines must lie inside the interior"
        assert prev < line, "markers must leave at least one cell for every slot"
        segments.append((int(prev), int(line - prev)))
        prev = int(line + 1)
    assert prev < size, "markers must leave a final slot below and to the right"
    segments.append((int(prev), int(size - prev)))
    assert segments
    return segments


def _slot_specs(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]], list[int], list[int]]:
    h, w = grid.shape
    bg = _mode_color(grid)
    ruler = _ruler_color(grid, bg)
    row_lines = [int(r) for r in range(1, h) if int(grid[r, 0]) == ruler]
    col_lines = [int(c) for c in range(1, w) if int(grid[0, c]) == ruler]
    assert row_lines and col_lines, "expected at least one horizontal and one vertical divider"
    row_segments = _segments_from_markers(h, row_lines)
    col_segments = _segments_from_markers(w, col_lines)
    slots = [
        (r0, c0, sh, sw)
        for r0, sh in row_segments
        for c0, sw in col_segments
    ]
    return ruler, slots, row_lines, col_lines


def _extract_fragments(grid: np.ndarray, bg: int, ruler: int) -> list[dict[str, int]]:
    fragments: list[dict[str, int]] = []
    colors = sorted(
        int(v)
        for v in np.unique(grid)
        if int(v) not in (bg, ruler)
    )
    for color in colors:
        for comp in _components4(grid == int(color)):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0 = min(rows)
            c0 = min(cols)
            r1 = max(rows)
            c1 = max(cols)
            hh = int(r1 - r0 + 1)
            ww = int(c1 - c0 + 1)
            assert len(comp) == hh * ww, "every fragment must be a solid rectangle"
            fragments.append({"color": int(color), "r0": int(r0), "c0": int(c0), "h": hh, "w": ww})
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered solid rectangles into the hidden slot grid encoded by edge rulers.

    Input structure:
    - black background
    - grey pixels on row 0 mark vertical divider columns
    - grey pixels on column 0 mark horizontal divider rows
    - inside the grid are solid monochrome rectangular fragments, one per slot

    Rule:
    - decode the hidden row/column dividers from the edge rulers
    - build every slot between consecutive dividers and the outer border
    - match each fragment to a slot with the same rectangle shape
    - if several slots share the same shape, assign fragments in visible spectrum order
      to those same-shape slots in row-major order
    - output the grey ruler row/column, full grey divider lines, and the fragments packed
      into their matched slots
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    bg = _mode_color(g)
    ruler, slots, row_lines, col_lines = _slot_specs(g)
    fragments = _extract_fragments(g, bg, ruler)
    assert len(fragments) == len(slots), "fragment count must match slot count"

    slots_by_size: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    for slot in slots:
        slots_by_size[(int(slot[2]), int(slot[3]))].append(slot)

    frags_by_size: dict[tuple[int, int], list[dict[str, int]]] = defaultdict(list)
    for frag in fragments:
        frags_by_size[(int(frag["h"]), int(frag["w"]))].append(frag)

    assert set(slots_by_size) == set(frags_by_size), "fragment sizes must match slot sizes exactly"

    assignments: list[tuple[int, int, int, int, int]] = []
    for size, size_slots in slots_by_size.items():
        size_frags = frags_by_size[size]
        assert len(size_slots) == len(size_frags), "each slot size group needs the same number of fragments"
        ordered_slots = sorted(size_slots, key=lambda slot: (int(slot[0]), int(slot[1])))
        ordered_frags = sorted(size_frags, key=lambda frag: _color_rank(int(frag["color"])))
        for slot, frag in zip(ordered_slots, ordered_frags):
            assignments.append((int(slot[0]), int(slot[1]), int(slot[2]), int(slot[3]), int(frag["color"])))

    out = np.full((h, w), int(bg), dtype=int)
    out[0, :] = g[0, :]
    out[:, 0] = g[:, 0]
    for r in row_lines:
        out[int(r), 1:] = int(ruler)
    for c in col_lines:
        out[1:, int(c)] = int(ruler)
    for r0, c0, hh, ww, color in assignments:
        out[r0 : r0 + hh, c0 : c0 + ww] = int(color)
    return out


def _sample_axis_lengths(
    count: int,
    values: tuple[int, ...],
    require_duplicate: bool,
) -> list[int] | None:
    if require_duplicate:
        if count < 2:
            return None
        for _ in range(_PLACE_ATTEMPTS):
            picks = [random.choice(values) for _ in range(count)]
            if len(set(picks)) < len(picks):
                return picks
        return None
    if count > len(values):
        return None
    picks = random.sample(list(values), count)
    random.shuffle(picks)
    return picks


def _sample_layout() -> tuple[int, int, bool] | None:
    for _ in range(_PLACE_ATTEMPTS):
        n_rows, n_cols = random.choice(_LAYOUT_OPTIONS)
        duplicate_mode = bool(random.getrandbits(1))
        repeat_rows = duplicate_mode and bool(random.getrandbits(1))
        heights = _sample_axis_lengths(n_rows, _HEIGHT_VALUES, require_duplicate=repeat_rows)
        widths = _sample_axis_lengths(n_cols, _WIDTH_VALUES, require_duplicate=duplicate_mode and not repeat_rows)
        if heights is None or widths is None:
            continue
        h = 1 + sum(heights) + (n_rows - 1)
        w = 1 + sum(widths) + (n_cols - 1)
        if h <= _GRID_MAX and w <= _GRID_MAX:
            return n_rows, n_cols, duplicate_mode
    return None


def _build_segments(lengths: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    segments: list[tuple[int, int]] = []
    markers: list[int] = []
    pos = 1
    for i, span in enumerate(lengths):
        segments.append((int(pos), int(span)))
        pos += int(span)
        if i != len(lengths) - 1:
            markers.append(int(pos))
            pos += 1
    return segments, markers


def _build_slots_from_lengths(heights: list[int], widths: list[int]) -> tuple[int, int, list[tuple[int, int, int, int]], list[int], list[int]]:
    row_segments, row_lines = _build_segments(heights)
    col_segments, col_lines = _build_segments(widths)
    grid_h = 1 + sum(heights) + len(row_lines)
    grid_w = 1 + sum(widths) + len(col_lines)
    slots = [(r0, c0, hh, ww) for r0, hh in row_segments for c0, ww in col_segments]
    return grid_h, grid_w, slots, row_lines, col_lines


def _assign_colors(slots: list[tuple[int, int, int, int]]) -> list[dict[str, int]]:
    assert len(slots) <= len(_FRAGMENT_COLORS)
    color_pool = random.sample(list(_FRAGMENT_COLORS), len(slots))
    random.shuffle(color_pool)

    slots_by_size: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    for slot in slots:
        slots_by_size[(int(slot[2]), int(slot[3]))].append(slot)

    color_index = 0
    fragments: list[dict[str, int]] = []
    size_keys = list(slots_by_size)
    random.shuffle(size_keys)
    for size in size_keys:
        size_slots = sorted(slots_by_size[size], key=lambda slot: (int(slot[0]), int(slot[1])))
        picked = color_pool[color_index : color_index + len(size_slots)]
        color_index += len(size_slots)
        ordered_colors = sorted(picked, key=_color_rank)
        for slot, color in zip(size_slots, ordered_colors):
            fragments.append(
                {
                    "color": int(color),
                    "target_r": int(slot[0]),
                    "target_c": int(slot[1]),
                    "h": int(slot[2]),
                    "w": int(slot[3]),
                }
            )
    random.shuffle(fragments)
    return fragments


def _line_overlap_score(top: int, left: int, hh: int, ww: int, row_lines: list[int], col_lines: list[int]) -> int:
    rows = range(top, top + hh)
    cols = range(left, left + ww)
    row_hits = sum(1 for line in row_lines if line in rows) * ww
    col_hits = sum(1 for line in col_lines if line in cols) * hh
    return int(row_hits + col_hits)


def _pack_fragments(
    grid_h: int,
    grid_w: int,
    row_lines: list[int],
    col_lines: list[int],
    fragments: list[dict[str, int]],
) -> list[dict[str, int]] | None:
    forbidden_slot_starts: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for frag in fragments:
        forbidden_slot_starts[(int(frag["h"]), int(frag["w"]))].add((int(frag["target_r"]), int(frag["target_c"])))

    order = sorted(
        fragments,
        key=lambda frag: (-(int(frag["h"]) * int(frag["w"])), random.random()),
    )

    for _ in range(_PACK_RETRIES):
        occupied = np.zeros((grid_h, grid_w), dtype=bool)
        placed: list[dict[str, int]] = []
        ok = True
        for frag in order:
            hh = int(frag["h"])
            ww = int(frag["w"])
            candidates: list[tuple[int, int, int, int]] = []
            for r0 in range(1, grid_h - hh + 1):
                for c0 in range(1, grid_w - ww + 1):
                    if (r0, c0) in forbidden_slot_starts[(hh, ww)]:
                        continue
                    if bool(occupied[r0 : r0 + hh, c0 : c0 + ww].any()):
                        continue
                    overlap = _line_overlap_score(r0, c0, hh, ww, row_lines, col_lines)
                    distance = abs(r0 - int(frag["target_r"])) + abs(c0 - int(frag["target_c"]))
                    candidates.append((int(overlap), int(distance), int(r0), int(c0)))
            if not candidates:
                ok = False
                break
            candidates.sort(reverse=True)
            window = candidates[: min(12, len(candidates))]
            _, _, r0, c0 = random.choice(window)
            occupied[r0 : r0 + hh, c0 : c0 + ww] = True
            placed.append(
                {
                    "color": int(frag["color"]),
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": hh,
                    "w": ww,
                    "target_r": int(frag["target_r"]),
                    "target_c": int(frag["target_c"]),
                }
            )
        if ok:
            return placed
    return None


def _render_input(
    grid_h: int,
    grid_w: int,
    row_lines: list[int],
    col_lines: list[int],
    placed: list[dict[str, int]],
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), _BG, dtype=int)
    for r in row_lines:
        grid[int(r), 0] = _RULER
    for c in col_lines:
        grid[0, int(c)] = _RULER
    for frag in placed:
        r0 = int(frag["r0"])
        c0 = int(frag["c0"])
        grid[r0 : r0 + int(frag["h"]), c0 : c0 + int(frag["w"])] = int(frag["color"])
    return grid


def _render_output(
    grid_h: int,
    grid_w: int,
    row_lines: list[int],
    col_lines: list[int],
    fragments: list[dict[str, int]],
) -> np.ndarray:
    out = np.full((grid_h, grid_w), _BG, dtype=int)
    for r in row_lines:
        out[int(r), :] = _RULER
    for c in col_lines:
        out[:, int(c)] = _RULER
    out[0, 0] = _BG
    for frag in fragments:
        r0 = int(frag["target_r"])
        c0 = int(frag["target_c"])
        out[r0 : r0 + int(frag["h"]), c0 : c0 + int(frag["w"])] = int(frag["color"])
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    sampled = _sample_layout()
    if sampled is None:
        return None
    n_rows, n_cols, duplicate_mode = sampled

    repeat_rows = duplicate_mode and bool(random.getrandbits(1))
    heights = _sample_axis_lengths(n_rows, _HEIGHT_VALUES, require_duplicate=repeat_rows)
    widths = _sample_axis_lengths(n_cols, _WIDTH_VALUES, require_duplicate=duplicate_mode and not repeat_rows)
    if heights is None or widths is None:
        return None

    grid_h, grid_w, slots, row_lines, col_lines = _build_slots_from_lengths(heights, widths)
    if grid_h > _GRID_MAX or grid_w > _GRID_MAX:
        return None
    if len(slots) > len(_FRAGMENT_COLORS):
        return None

    fragments = _assign_colors(slots)
    placed = _pack_fragments(grid_h, grid_w, row_lines, col_lines, fragments)
    if placed is None:
        return None

    inp = _render_input(grid_h, grid_w, row_lines, col_lines, placed)
    out = _render_output(grid_h, grid_w, row_lines, col_lines, fragments)
    if _mode_color(inp) != _BG:
        return None

    if duplicate_mode:
        if len({(frag["h"], frag["w"]) for frag in fragments}) == len(fragments):
            return None
    else:
        if len({(frag["h"], frag["w"]) for frag in fragments}) != len(fragments):
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError(f"failed to generate a valid puzzle after {_GEN_RETRIES} attempts")
