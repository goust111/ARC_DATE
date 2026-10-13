from __future__ import annotations

import hashlib
import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]  # (r0, c0, r1, c1), inclusive

BACKGROUND = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400

_GEN_COUNTER = 0
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    bg = int(bg)

    out: list[dict] = []
    if monochromatic:
        for color in sorted(set(map(int, g.ravel())) - {bg}):
            for coords in components_mask(g == color, connectivity=connectivity):
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
        return out

    for coords in components_mask(g != bg, connectivity=connectivity):
        out.append(
            {
                "color": None,
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return out


def _hash_grid(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _read_nested_layer_colors(crop: np.ndarray) -> list[int]:
    g = np.asarray(crop, dtype=int)
    assert g.ndim == 2 and g.size > 0
    colors: list[int] = []
    cur = g
    while cur.size:
        color = int(cur[0, 0])
        assert np.all(cur[0, :] == color)
        assert np.all(cur[-1, :] == color)
        assert np.all(cur[:, 0] == color)
        assert np.all(cur[:, -1] == color)
        colors.append(color)
        if cur.shape[0] <= 2 or cur.shape[1] <= 2:
            assert np.all(cur == color)
            break
        inner = cur[1:-1, 1:-1]
        if inner.size == 0:
            break
        assert int(inner[0, 0]) != color
        cur = inner
    assert len(colors) == len(set(colors))
    return colors


def _find_dictionary_rectangle(g: np.ndarray, bg: int) -> tuple[dict, list[int], np.ndarray]:
    candidates: list[tuple[dict, list[int]]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=False):
        coords = comp["coords"]
        colors = sorted(set(map(int, g[coords[:, 0], coords[:, 1]])))
        if len(colors) <= 1:
            continue
        box = comp["bbox"]
        crop = crop_bbox(g, box)
        if np.any(crop == bg):
            continue
        if int(comp["area"]) != int(crop.size):
            continue
        try:
            layer_colors = _read_nested_layer_colors(crop)
        except AssertionError:
            continue
        if len(layer_colors) < 2:
            continue
        candidates.append((comp, layer_colors))

    assert len(candidates) == 1, "expected exactly one nested rectangle dictionary"
    rect_comp, layer_colors = candidates[0]
    rect_mask = np.zeros_like(g, dtype=bool)
    coords = rect_comp["coords"]
    rect_mask[coords[:, 0], coords[:, 1]] = True
    return rect_comp, layer_colors, rect_mask


def _extract_segments(g: np.ndarray, bg: int, rect_mask: np.ndarray) -> list[dict]:
    segments: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        if bool(rect_mask[coords[:, 0], coords[:, 1]].all()):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        assert r0 == r1, "moving objects must be one-cell-high segments"
        assert int(comp["area"]) == c1 - c0 + 1, "segments must be contiguous horizontal bars"
        segments.append(
            {
                "color": int(comp["color"]),
                "row": int(r0),
                "c0": int(c0),
                "c1": int(c1),
                "length": int(comp["area"]),
            }
        )
    assert segments, "expected at least one segment"
    return segments


def _ordered_segments_from_grid(g: np.ndarray) -> tuple[list[int], list[dict], np.ndarray]:
    bg = get_bg(g)
    _rect_comp, layer_colors, rect_mask = _find_dictionary_rectangle(g, bg)
    segments = _extract_segments(g, bg, rect_mask)
    layer_rank = {color: i for i, color in enumerate(layer_colors)}

    segment_colors = [seg["color"] for seg in segments]
    segment_lengths = [seg["length"] for seg in segments]
    assert len(segment_colors) == len(set(segment_colors)), "segment colors must be unique"
    assert len(segment_lengths) == len(set(segment_lengths)), "segment lengths must be unique"
    assert set(segment_colors).issubset(layer_rank), "every segment color must appear in the rectangle dictionary"

    ordered = sorted(segments, key=lambda seg: layer_rank[seg["color"]])
    return layer_colors, ordered, rect_mask


def _stack_segments(
    base: np.ndarray,
    ordered_segments: list[dict],
    *,
    bg: int,
) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    h, w = out.shape
    assert len(ordered_segments) <= h
    for idx, seg in enumerate(ordered_segments):
        row = h - 1 - idx
        length = int(seg["length"])
        color = int(seg["color"])
        assert np.all(out[row, w - length : w] == bg), "stack destination must stay off the rectangle"
        out[row, w - length : w] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-stack the scattered horizontal segments by reading their color order from the nested rectangle dictionary.

    Input structure:
    - A dominant black background contains one 4-connected multicolor nested rectangle made of concentric rectangular layers.
    - Separate monochrome foreground objects are one-cell-high horizontal segments whose colors are taken from some rectangle layers.
    - Segment colors are unique, and each segment stays disconnected from the rectangle and from the other segments.

    Rule:
    - Keep the nested rectangle fixed in place and erase every scattered segment from its original row.
    - Read the rectangle's layer colors from the outer border inward.
    - Following that outer-to-inner color order, place the matching segments from the bottom row upward, one per row, each right-aligned to the grid's right edge.

    Key insight:
    - Segment length is only a decoy; the actual ordering key is the cross-object color dictionary encoded by the rectangle's concentric layer sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    _layer_colors, ordered_segments, _rect_mask = _ordered_segments_from_grid(g)

    cleared = g.copy()
    for seg in ordered_segments:
        cleared[seg["row"], seg["c0"] : seg["c1"] + 1] = bg
    return _stack_segments(cleared, ordered_segments, bg=bg)


def _paint_nested_rectangle(
    grid: np.ndarray,
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
    layer_colors: list[int],
) -> None:
    for depth, color in enumerate(layer_colors):
        rr0 = r0 + depth
        cc0 = c0 + depth
        rr1 = r0 + outer_h - depth
        cc1 = c0 + outer_w - depth
        grid[rr0:rr1, cc0:cc1] = int(color)


def _non_monotonic(values: list[int]) -> bool:
    return values != sorted(values) and values != sorted(values, reverse=True)


def _pick_lengths(segment_count: int, width: int, force_mismatch: bool) -> list[int]:
    max_len = min(width - 2, 12)
    assert max_len - 1 >= segment_count
    pool = list(range(2, max_len + 1))
    for _ in range(200):
        lengths = random.sample(pool, segment_count)
        if force_mismatch and _non_monotonic(lengths):
            return lengths
        if not force_mismatch:
            return sorted(lengths)
    raise RuntimeError("failed to sample suitable segment lengths")


def _pick_scatter_order(target_colors: list[int]) -> list[int]:
    colors = target_colors[:]
    for _ in range(200):
        random.shuffle(colors)
        if colors != target_colors and colors != target_colors[::-1]:
            return colors[:]
    raise RuntimeError("failed to find a misleading scatter order")


def _expected_output(
    rectangle_only: np.ndarray,
    target_colors: list[int],
    color_to_length: dict[int, int],
) -> np.ndarray:
    ordered_segments = [{"color": color, "length": color_to_length[color]} for color in target_colors]
    return _stack_segments(rectangle_only, ordered_segments, bg=BACKGROUND)


def _wrong_length_output(
    rectangle_only: np.ndarray,
    colors: list[int],
    color_to_length: dict[int, int],
    reverse: bool = False,
) -> np.ndarray:
    order = sorted(colors, key=lambda color: color_to_length[color], reverse=reverse)
    ordered_segments = [{"color": color, "length": color_to_length[color]} for color in order]
    return _stack_segments(rectangle_only, ordered_segments, bg=BACKGROUND)


def _scenario_for_call(call_index: int) -> dict[str, bool]:
    mode = call_index % 4
    return {
        "all_layers": mode in (0, 2),
        "length_alignment": mode == 0,
    }


def _generate_attempt(scenario: dict[str, bool]) -> np.ndarray | None:
    all_layers = scenario["all_layers"]
    force_length_mismatch = not scenario["length_alignment"]

    if all_layers:
        layer_count = random.randint(3, 6)
        segment_count = layer_count
    else:
        layer_count = random.randint(4, 6)
        segment_count = random.randint(3, layer_count - 1)

    inner_h = random.randint(2, 4)
    inner_w = random.randint(3, 6)
    outer_h = inner_h + 2 * (layer_count - 1)
    outer_w = inner_w + 2 * (layer_count - 1)

    h_min = max(outer_h + 2 * segment_count + 3, outer_h + 6)
    h_max = min(30, h_min + 4)
    w_min = max(outer_w + 4, 16)
    w_max = min(30, w_min + 8)
    if h_min > 30 or w_min > 30:
        return None

    h = random.randint(h_min, h_max)
    w = random.randint(w_min, w_max)

    layer_colors = random.sample(list(PALETTE), layer_count)
    target_indices = list(range(layer_count)) if all_layers else sorted(random.sample(range(layer_count), segment_count))
    target_colors = [layer_colors[idx] for idx in target_indices]

    target_lengths = _pick_lengths(segment_count, w, force_length_mismatch)
    color_to_length = dict(zip(target_colors, target_lengths))

    rect_r0_max = h - outer_h - segment_count - 1
    if rect_r0_max < 1:
        return None
    rect_r0 = random.randint(1, rect_r0_max)
    rect_c0 = random.randint(1, w - outer_w - 1)
    rect_r1 = rect_r0 + outer_h - 1

    rectangle_only = np.full((h, w), BACKGROUND, dtype=int)
    _paint_nested_rectangle(rectangle_only, rect_r0, rect_c0, outer_h, outer_w, layer_colors)

    candidate_rows = [
        r
        for r in range(h - segment_count)
        if r < rect_r0 - 1 or r > rect_r1 + 1
    ]
    if len(candidate_rows) < segment_count:
        return None

    scatter_rows = sorted(random.sample(candidate_rows, segment_count))
    scatter_order = _pick_scatter_order(target_colors)

    inp = rectangle_only.copy()
    for row, color in zip(scatter_rows, scatter_order):
        length = color_to_length[color]
        c0 = random.randint(0, w - length)
        inp[row, c0 : c0 + length] = color

    if get_bg(inp) != BACKGROUND:
        return None

    expected = _expected_output(rectangle_only, target_colors, color_to_length)
    wrong_row = _expected_output(rectangle_only, scatter_order, color_to_length)
    if np.array_equal(expected, wrong_row):
        return None

    if force_length_mismatch:
        asc = _wrong_length_output(rectangle_only, target_colors, color_to_length, reverse=False)
        desc = _wrong_length_output(rectangle_only, target_colors, color_to_length, reverse=True)
        if np.array_equal(expected, asc) or np.array_equal(expected, desc):
            return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    try:
        layer_colors_read, ordered_segments, _rect_mask = _ordered_segments_from_grid(inp)
    except Exception:
        return None
    if [seg["color"] for seg in ordered_segments] != target_colors:
        return None
    if layer_colors_read != layer_colors:
        return None

    out_sig = _hash_grid(expected)
    if out_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(out_sig)
    return inp


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    call_index = _GEN_COUNTER
    _GEN_COUNTER += 1
    scenario = _scenario_for_call(call_index)
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(scenario)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-rectangle dictionary task")
