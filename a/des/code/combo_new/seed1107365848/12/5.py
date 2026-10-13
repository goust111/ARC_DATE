import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 400
MIN_SEG_LEN = 5
GRID_H_RANGE = (19, 27)
GRID_W_RANGE = (19, 27)
SEGMENT_COUNT_RANGE = (4, 7)

BASE_SHAPES = [
    np.array([[1], [1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1], [1], [1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _perimeter_positions(h: int, w: int) -> list[tuple[int, int, str]]:
    seq: list[tuple[int, int, str]] = []
    seq.extend((0, c, "top") for c in range(w))
    seq.extend((r, w - 1, "right") for r in range(1, h))
    seq.extend((h - 1, c, "bottom") for c in range(w - 2, -1, -1))
    seq.extend((r, 0, "left") for r in range(h - 2, 0, -1))
    return seq


def _border_segments(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seq = _perimeter_positions(h, w)
    colors = [int(grid[r, c]) for r, c, _ in seq]
    runs: list[dict] = []
    start = 0
    for idx in range(1, len(seq) + 1):
        if idx < len(seq) and colors[idx] == colors[start]:
            continue
        runs.append({"color": colors[start], "start": start, "end": idx})
        start = idx
    wrap_start = None
    wrap_end = None
    if len(runs) > 1 and runs[0]["color"] == runs[-1]["color"]:
        wrap_start = int(runs[-1]["start"])
        wrap_end = int(runs[0]["end"])
        runs[0]["start"] = wrap_start
        runs[0]["end"] = wrap_end
        runs.pop()

    segments = []
    for run in runs:
        if wrap_start is not None and run["start"] == wrap_start and run["end"] == wrap_end:
            cells = seq[run["start"] :] + seq[: run["end"]]
        else:
            cells = seq[run["start"] : run["end"]]
        mid = cells[(len(cells) - 1) // 2]
        segments.append(
            {
                "color": int(run["color"]),
                "cells": cells,
                "midpoint": (int(mid[0]), int(mid[1])),
                "side": str(mid[2]),
            }
        )
    return segments


def _components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append({"color": color, "coords": arr, "bbox": _bbox(arr)})
    return comps


def _translated_coords(
    coords: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
    midpoint: tuple[int, int],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    h, w = grid_shape
    mid_r, mid_c = midpoint

    if side == "top":
        dr = 1 - r0
        dc2 = 2 * mid_c - (c0 + c1)
        assert dc2 % 2 == 0, "horizontal bbox center must align to an integer column"
        dc = dc2 // 2
    elif side == "bottom":
        dr = (h - 2) - r1
        dc2 = 2 * mid_c - (c0 + c1)
        assert dc2 % 2 == 0, "horizontal bbox center must align to an integer column"
        dc = dc2 // 2
    elif side == "left":
        dc = 1 - c0
        dr2 = 2 * mid_r - (r0 + r1)
        assert dr2 % 2 == 0, "vertical bbox center must align to an integer row"
        dr = dr2 // 2
    else:
        assert side == "right"
        dc = (w - 2) - c1
        dr2 = 2 * mid_r - (r0 + r1)
        assert dr2 % 2 == 0, "vertical bbox center must align to an integer row"
        dr = dr2 // 2

    shifted = coords + np.array([dr, dc], dtype=int)
    sr0, sc0, sr1, sc1 = _bbox(shifted)
    if side == "top":
        assert sr0 == 1
    elif side == "bottom":
        assert sr1 == h - 2
    elif side == "left":
        assert sc0 == 1
    else:
        assert sc1 == w - 2
    assert 0 <= sr0 <= sr1 < h and 0 <= sc0 <= sc1 < w
    return shifted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Magnetically pull each inner fragment to the matching border segment and slide it to that segment's midpoint.

    Input structure:
    - The background is black.
    - A one-pixel border covers the full perimeter, but its color changes across contiguous border segments.
    - Inside the border there is exactly one small monochrome 4-connected fragment for each border-segment color.

    Rule:
    - For each inner fragment, find the unique border segment with the same color.
    - Take that segment's midpoint pixel in clockwise border order, using the earlier middle pixel for even-length segments.
    - Translate the fragment perpendicular to that side until it touches the border from the inside.
    - Then slide it along the border so the fragment's bounding-box center aligns with the segment midpoint; clear its old position and keep the border unchanged.

    Key insight:
    - The colored border is not decoration: its segmented geometry tells both where each fragment goes and which side's normal/tangent directions to use for the two-step translation.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and h >= 5 and w >= 5, "expected a bordered 2D grid"

    segments = _border_segments(g)
    seg_by_color = {int(seg["color"]): seg for seg in segments}
    assert len(seg_by_color) == len(segments), "border colors must define unique segments"

    comps = _components(g, bg=BG)
    comp_by_color = {int(comp["color"]): comp for comp in comps}
    assert len(comp_by_color) == len(comps), "inner fragment colors must be unique"
    assert set(comp_by_color) == set(seg_by_color), "inner fragments must match border segment colors exactly"

    out = g.copy()
    for comp in comps:
        out[comp["coords"][:, 0], comp["coords"][:, 1]] = BG

    occupied = out != BG
    for color, comp in comp_by_color.items():
        seg = seg_by_color[color]
        shifted = _translated_coords(
            comp["coords"],
            comp["bbox"],
            seg["side"],
            seg["midpoint"],
            g.shape,
        )
        assert not np.any(occupied[shifted[:, 0], shifted[:, 1]]), "translated fragments must not overlap"
        out[shifted[:, 0], shifted[:, 1]] = int(color)
        occupied[shifted[:, 0], shifted[:, 1]] = True
    return out


def _bounded_partition(total: int, parts: int, min_part: int) -> list[int]:
    values = []
    remaining = total
    for idx in range(parts):
        slots = parts - idx - 1
        if slots == 0:
            values.append(remaining)
            break
        lo = min_part
        hi = remaining - min_part * slots
        values.append(random.randint(lo, hi))
        remaining -= values[-1]
    random.shuffle(values)
    return values


def _side_sequences(h: int, w: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "top": [(0, c) for c in range(w)],
        "right": [(r, w - 1) for r in range(1, h)],
        "bottom": [(h - 1, c) for c in range(w - 2, -1, -1)],
        "left": [(r, 0) for r in range(h - 2, 0, -1)],
    }


def _choose_side_counts(total_segments: int) -> dict[str, int]:
    sides = ["top", "right", "bottom", "left"]
    counts = {side: 1 for side in sides}
    for _ in range(total_segments - 4):
        counts[random.choice(sides)] += 1
    return counts


def _shape_for_side(side: str, seg_len: int) -> np.ndarray | None:
    candidates = []
    for base in BASE_SHAPES:
        mask = np.rot90(base) if side in {"left", "right"} else base.copy()
        for flip_ud in (False, True):
            for flip_lr in (False, True):
                variant = mask
                if flip_ud:
                    variant = np.flipud(variant)
                if flip_lr:
                    variant = np.fliplr(variant)
                tangent = variant.shape[1] if side in {"top", "bottom"} else variant.shape[0]
                if tangent <= seg_len - 2:
                    candidates.append(np.array(variant, copy=True))
    if not candidates:
        return None
    return random.choice(candidates)


def _mask_coords(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return np.argwhere(mask) + np.array([r0, c0], dtype=int)


def _touches_occupied(coords: np.ndarray, occupied: np.ndarray, radius: int = 1) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        for rr in range(max(0, int(r) - radius), min(h, int(r) + radius + 1)):
            for cc in range(max(0, int(c) - radius), min(w, int(c) + radius + 1)):
                if occupied[rr, cc]:
                    return True
    return False


def _segment_layout(h: int, w: int, colors: list[int]) -> list[dict] | None:
    total_segments = len(colors)
    counts = _choose_side_counts(total_segments)
    side_seqs = _side_sequences(h, w)
    segs: list[dict] = []
    color_idx = 0
    for side in ["top", "right", "bottom", "left"]:
        seq = side_seqs[side]
        if len(seq) < MIN_SEG_LEN * counts[side]:
            return None
        lens = _bounded_partition(len(seq), counts[side], MIN_SEG_LEN)
        pos = 0
        for length in lens:
            cells = [(r, c, side) for r, c in seq[pos : pos + length]]
            mid = cells[(length - 1) // 2]
            segs.append(
                {
                    "color": int(colors[color_idx]),
                    "side": side,
                    "cells": cells,
                    "midpoint": (int(mid[0]), int(mid[1])),
                    "length": int(length),
                }
            )
            pos += length
            color_idx += 1
    return segs if color_idx == total_segments else None


def _final_coords(mask: np.ndarray, seg: dict, grid_shape: tuple[int, int]) -> np.ndarray:
    h, w = grid_shape
    mh, mw = mask.shape
    mid_r, mid_c = seg["midpoint"]
    if seg["side"] == "top":
        r0 = 1
        c0 = mid_c - mw // 2
    elif seg["side"] == "bottom":
        r0 = h - 1 - mh
        c0 = mid_c - mw // 2
    elif seg["side"] == "left":
        r0 = mid_r - mh // 2
        c0 = 1
    else:
        r0 = mid_r - mh // 2
        c0 = w - 1 - mw
    return _mask_coords(mask, int(r0), int(c0))


def _build_output(h: int, w: int, segments: list[dict]) -> tuple[np.ndarray, dict[int, dict]] | None:
    out = np.full((h, w), BG, dtype=int)
    for seg in segments:
        for r, c, _ in seg["cells"]:
            out[r, c] = int(seg["color"])

    occupied = np.zeros((h, w), dtype=bool)
    placements: dict[int, dict] = {}
    for seg in random.sample(segments, len(segments)):
        mask = _shape_for_side(seg["side"], seg["length"])
        if mask is None:
            return None
        coords = _final_coords(mask, seg, (h, w))
        if np.any(out[coords[:, 0], coords[:, 1]] != BG):
            return None
        if _touches_occupied(coords, occupied, radius=1):
            return None
        out[coords[:, 0], coords[:, 1]] = int(seg["color"])
        occupied[coords[:, 0], coords[:, 1]] = True
        placements[int(seg["color"])] = {"mask": mask, "coords": coords, "segment": seg}
    return out, placements


def _candidate_anchor_ranges(mask: np.ndarray, grid_shape: tuple[int, int]) -> tuple[range, range]:
    h, w = grid_shape
    mh, mw = mask.shape
    return range(2, h - mh - 1), range(2, w - mw - 1)


def _misaligned_from_target(
    anchor: tuple[int, int],
    mask: np.ndarray,
    side: str,
    midpoint: tuple[int, int],
    grid_shape: tuple[int, int],
) -> bool:
    h, w = grid_shape
    r0, c0 = anchor
    mh, mw = mask.shape
    r1, c1 = r0 + mh - 1, c0 + mw - 1
    mid_r, mid_c = midpoint
    if side == "top":
        return r0 >= 3 and (c0 + c1) != 2 * mid_c
    if side == "bottom":
        return r1 <= h - 4 and (c0 + c1) != 2 * mid_c
    if side == "left":
        return c0 >= 3 and (r0 + r1) != 2 * mid_r
    return c1 <= w - 4 and (r0 + r1) != 2 * mid_r


def _place_inputs(h: int, w: int, segments: list[dict], placements: dict[int, dict]) -> np.ndarray | None:
    inp = np.full((h, w), BG, dtype=int)
    for seg in segments:
        for r, c, _ in seg["cells"]:
            inp[r, c] = int(seg["color"])

    occupied = np.zeros((h, w), dtype=bool)
    colors = list(placements)
    colors.sort(key=lambda color: int(placements[color]["mask"].sum()), reverse=True)
    for color in colors:
        mask = placements[color]["mask"]
        seg = placements[color]["segment"]
        row_range, col_range = _candidate_anchor_ranges(mask, (h, w))
        anchors = [(r0, c0) for r0 in row_range for c0 in col_range]
        random.shuffle(anchors)
        final_coords = placements[color]["coords"]
        for anchor in anchors:
            if not _misaligned_from_target(anchor, mask, seg["side"], seg["midpoint"], (h, w)):
                continue
            coords = _mask_coords(mask, anchor[0], anchor[1])
            if np.array_equal(coords[np.lexsort((coords[:, 1], coords[:, 0]))],
                              final_coords[np.lexsort((final_coords[:, 1], final_coords[:, 0]))]):
                continue
            if _touches_occupied(coords, occupied, radius=1):
                continue
            if np.any(inp[coords[:, 0], coords[:, 1]] != BG):
                continue
            inp[coords[:, 0], coords[:, 1]] = int(color)
            occupied[coords[:, 0], coords[:, 1]] = True
            break
        else:
            return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    total_segments = random.randint(*SEGMENT_COUNT_RANGE)
    colors = random.sample(PALETTE, total_segments)

    segments = _segment_layout(h, w, colors)
    if segments is None:
        return None

    built = _build_output(h, w, segments)
    if built is None:
        return None
    expected, placements = built

    inp = _place_inputs(h, w, segments, placements)
    if inp is None:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
