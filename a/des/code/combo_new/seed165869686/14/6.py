from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    expected_area = 2 * h + 2 * w - 4
    if int(coords.shape[0]) != expected_area:
        return None
    pts = {(int(r), int(c)) for r, c in coords}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            present = (r, c) in pts
            if on_border != present:
                return None
    return r0, c0, r1, c1


def _edge_bar_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    cands: list[dict] = []
    lines = [
        ("top", grid[0, :]),
        ("bottom", grid[h - 1, :]),
        ("left", grid[:, 0]),
        ("right", grid[:, w - 1]),
    ]
    for side, line in lines:
        non_bg = np.flatnonzero(line != bg)
        if non_bg.size < 2:
            continue
        start = int(non_bg[0])
        end = int(non_bg[-1]) + 1
        if np.any(line[start:end] == bg):
            continue
        if np.any(line[:start] != bg) or np.any(line[end:] != bg):
            continue
        colors = [int(v) for v in line[start:end].tolist()]
        if len(colors) != len(set(colors)):
            continue
        cands.append(
            {
                "side": side,
                "orientation": "horizontal" if side in {"top", "bottom"} else "vertical",
                "start": start,
                "end": end,
                "colors": colors,
            }
        )
    return cands


def _remove_bar(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return grid[1:, :]
    if side == "bottom":
        return grid[:-1, :]
    if side == "left":
        return grid[:, 1:]
    if side == "right":
        return grid[:, :-1]
    raise ValueError(side)


def _find_frame_groups(canvas: np.ndarray, bg: int) -> list[dict]:
    groups: list[dict] = []
    for color in (int(v) for v in np.unique(canvas) if int(v) != bg):
        valid_frames: list[dict] = []
        for coords in _components(canvas == color):
            bbox = _frame_bbox(coords)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            interior = canvas[r0 + 1 : r1, c0 + 1 : c1]
            markers = np.argwhere(interior != bg)
            if markers.shape[0] != 1:
                continue
            mr, mc = map(int, markers[0])
            marker_color = int(interior[mr, mc])
            if marker_color == color:
                continue
            valid_frames.append(
                {
                    "bbox": bbox,
                    "marker_color": marker_color,
                    "patch": canvas[r0 : r1 + 1, c0 : c1 + 1].copy(),
                }
            )
        by_size: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for frame in valid_frames:
            r0, c0, r1, c1 = frame["bbox"]
            by_size[(int(r1 - r0 + 1), int(c1 - c0 + 1))].append(frame)
        for size, frames in by_size.items():
            groups.append({"border_color": color, "size": size, "frames": frames})
    return groups


def _extract_layout(grid: np.ndarray) -> tuple[dict, np.ndarray, dict[int, np.ndarray]]:
    bg = _get_bg(grid)
    best = None
    for bar in _edge_bar_candidates(grid, bg):
        canvas = _remove_bar(grid, bar["side"])
        for group in _find_frame_groups(canvas, bg):
            frames = group["frames"]
            if len(frames) != len(bar["colors"]):
                continue
            marker_colors = [int(frame["marker_color"]) for frame in frames]
            if len(marker_colors) != len(set(marker_colors)):
                continue
            if set(marker_colors) != set(bar["colors"]):
                continue
            score = (len(frames), group["size"][0] * group["size"][1])
            if best is None or score > best[0]:
                best = (score, bar, canvas, {int(frame["marker_color"]): frame["patch"] for frame in frames})
    assert best is not None, "expected one edge sort bar and one matching frame group"
    _score, bar, canvas, patches = best
    return bar, canvas, patches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder framed shapes by reading the color order from a separate edge strip.

    Input structure:
    - The background is black or the dominant color.
    - One outer edge contains a one-cell-wide sort strip: a contiguous row or column
      of distinct non-background colors, with any remaining cells on that edge left as background.
    - The remaining canvas contains several same-sized hollow rectangular frames.
      Each frame has a single-pixel colored marker somewhere inside it, and those
      marker colors are exactly the colors listed on the sort strip.

    Rule:
    - Read the strip order left-to-right for a horizontal strip or top-to-bottom for
      a vertical strip.
    - Match each frame to the strip entry with the same interior marker color.
    - Remove the strip, then place the full frame patches tightly from the canvas
      origin in the strip's direction: horizontal strips produce a left-to-right row,
      and vertical strips produce a top-to-bottom column.

    Key insight:
    - The edge strip encodes only an ordering, while the true identity of each movable
      shape comes from the tiny interior marker rather than from the shared frame color.
    """
    g = np.asarray(grid, dtype=int)
    bar, canvas, patches = _extract_layout(g)
    bg = _get_bg(g)

    out = np.full(canvas.shape, bg, dtype=int)
    cursor = 0
    if bar["orientation"] == "horizontal":
        for color in bar["colors"]:
            patch = patches[int(color)]
            ph, pw = patch.shape
            assert cursor + pw <= out.shape[1], "ordered frames must fit horizontally"
            out[:ph, cursor : cursor + pw] = patch
            cursor += pw
    else:
        for color in bar["colors"]:
            patch = patches[int(color)]
            ph, pw = patch.shape
            assert cursor + ph <= out.shape[0], "ordered frames must fit vertically"
            out[cursor : cursor + ph, :pw] = patch
            cursor += ph
    return out


# --------------- generator ---------------

_BG = 0
_FRAME = 5
_MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_COUNT_RANGE = (2, 5)
_SHAPE_SIZE_RANGE = (3, 6)
_EXTRA_SLACK_MAX = 4
_GEN_ATTEMPTS = 400


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _make_patch(rng: np.random.Generator, height: int, width: int, marker_color: int) -> np.ndarray:
    patch = np.full((height, width), _BG, dtype=int)
    patch[0, :] = _FRAME
    patch[height - 1, :] = _FRAME
    patch[:, 0] = _FRAME
    patch[:, width - 1] = _FRAME
    mr = _randint(rng, 1, height - 2)
    mc = _randint(rng, 1, width - 2)
    patch[mr, mc] = int(marker_color)
    return patch


def _slot_positions(
    rng: np.random.Generator,
    main_h: int,
    main_w: int,
    patch_h: int,
    patch_w: int,
    count: int,
    orientation: str,
) -> list[tuple[int, int]] | None:
    gap = _randint(rng, 1, 2)
    if orientation == "horizontal":
        row_choices = [1, 2] if count == 2 else [2, 3]
        slot_rows = int(rng.choice(row_choices))
        slot_cols = int(math.ceil(count / slot_rows))
        slot_h = slot_rows * patch_h + (slot_rows - 1) * gap
        slot_w = slot_cols * patch_w + (slot_cols - 1) * gap
        if slot_h + 1 > main_h or slot_w > main_w:
            return None
        r0 = _randint(rng, 1, main_h - slot_h)
        c0 = _randint(rng, 0, main_w - slot_w)
    else:
        col_choices = [1, 2] if count == 2 else [2, 3]
        slot_cols = int(rng.choice(col_choices))
        slot_rows = int(math.ceil(count / slot_cols))
        slot_h = slot_rows * patch_h + (slot_rows - 1) * gap
        slot_w = slot_cols * patch_w + (slot_cols - 1) * gap
        if slot_h > main_h or slot_w + 1 > main_w:
            return None
        r0 = _randint(rng, 0, main_h - slot_h)
        c0 = _randint(rng, 1, main_w - slot_w)

    slots = []
    for sr in range(slot_rows):
        for sc in range(slot_cols):
            slots.append((r0 + sr * (patch_h + gap), c0 + sc * (patch_w + gap)))
    if len(slots) < count:
        return None
    perm = rng.permutation(len(slots))[:count]
    return [slots[int(i)] for i in perm]


def _build_output(main_h: int, main_w: int, patches: dict[int, np.ndarray], order: list[int], orientation: str) -> np.ndarray:
    out = np.full((main_h, main_w), _BG, dtype=int)
    cursor = 0
    if orientation == "horizontal":
        for color in order:
            patch = patches[int(color)]
            ph, pw = patch.shape
            out[:ph, cursor : cursor + pw] = patch
            cursor += pw
    else:
        for color in order:
            patch = patches[int(color)]
            ph, pw = patch.shape
            out[cursor : cursor + ph, :pw] = patch
            cursor += ph
    return out


def _attempt_case(rng: np.random.Generator) -> np.ndarray | None:
    orientation = str(rng.choice(["horizontal", "vertical"]))
    count = _randint(rng, *_COUNT_RANGE)
    patch_h = _randint(rng, *_SHAPE_SIZE_RANGE)
    patch_w = _randint(rng, *_SHAPE_SIZE_RANGE)
    marker_colors = [int(v) for v in rng.choice(_MARKER_COLORS, size=count, replace=False).tolist()]
    order = [int(v) for v in rng.permutation(marker_colors).tolist()]

    if orientation == "horizontal":
        main_w_base = count * patch_w
        main_h_base = max(2 * patch_h + 1, patch_h + 2)
        if main_w_base > 30 or main_h_base > 29:
            return None
        main_w = main_w_base + _randint(rng, 0, min(_EXTRA_SLACK_MAX, 30 - main_w_base))
        main_h = main_h_base + _randint(rng, 0, min(_EXTRA_SLACK_MAX, 29 - main_h_base))
        total_h, total_w = main_h + 1, main_w
    else:
        main_h_base = count * patch_h
        main_w_base = max(2 * patch_w + 1, patch_w + 2)
        if main_h_base > 30 or main_w_base > 29:
            return None
        main_h = main_h_base + _randint(rng, 0, min(_EXTRA_SLACK_MAX, 30 - main_h_base))
        main_w = main_w_base + _randint(rng, 0, min(_EXTRA_SLACK_MAX, 29 - main_w_base))
        total_h, total_w = main_h, main_w + 1

    positions = _slot_positions(rng, main_h, main_w, patch_h, patch_w, count, orientation)
    if positions is None:
        return None

    patches = {color: _make_patch(rng, patch_h, patch_w, color) for color in marker_colors}
    canvas = np.full((main_h, main_w), _BG, dtype=int)
    color_order = [int(v) for v in rng.permutation(marker_colors).tolist()]
    for color, (r0, c0) in zip(color_order, positions, strict=True):
        patch = patches[int(color)]
        ph, pw = patch.shape
        canvas[r0 : r0 + ph, c0 : c0 + pw] = patch

    expected = _build_output(main_h, main_w, patches, order, orientation)
    if np.array_equal(canvas, expected):
        return None

    grid = np.full((total_h, total_w), _BG, dtype=int)
    if orientation == "horizontal":
        grid[1:, :] = canvas
        start = _randint(rng, 0, main_w - count)
        grid[0, start : start + count] = np.array(order, dtype=int)
    else:
        grid[:, 1:] = canvas
        start = _randint(rng, 0, main_h - count)
        grid[start : start + count, 0] = np.array(order, dtype=int)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_ATTEMPTS):
        grid = _attempt_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reorder-by-marker ARC input")
