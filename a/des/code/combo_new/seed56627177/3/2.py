from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
ORANGE = 7
MAROON = 9
PALETTE = (1, 2, 3, 4, 5, 6, 8)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _color_components(grid: np.ndarray, forbidden: set[int]) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in forbidden or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
            patch[coords[:, 0] - r0, coords[:, 1] - c0] = color
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "patch": patch,
                    "area": int(len(cells)),
                    "anchor": (int(r0), int(c0)),
                }
            )
    return comps


def _fragment_key(fragment: dict[str, object]) -> tuple[int, int, int, int]:
    patch = np.asarray(fragment["patch"])
    ar, ac = fragment["anchor"]
    return (-int(fragment["area"]), int(patch.shape[0]), int(ar), int(ac))


def _extract_fragments(rect: np.ndarray) -> list[dict[str, object]]:
    maroon = np.argwhere(rect == MAROON)
    assert maroon.size > 0, "each rectangle must contain a maroon divider"
    rows = {int(r) for r, _ in maroon}
    cols = {int(c) for _, c in maroon}
    assert len(rows) == 1 or len(cols) == 1, "divider must be a straight row or column"

    fragments = _color_components(rect, {BLACK, MAROON})
    assert 2 <= len(fragments) <= 3, "expected two or three fragments"
    fragments.sort(key=_fragment_key)
    return fragments


def _stack_patch(out: np.ndarray, patch: np.ndarray, bottom: int, center_col: int) -> int:
    h, w = patch.shape
    top = int(bottom) - h + 1
    left = int(center_col) - w // 2
    assert top >= 0 and left >= 0
    assert top + h <= out.shape[0] and left + w <= out.shape[1]
    mask = patch != BLACK
    view = out[top : top + h, left : left + w]
    assert not (view[mask] != BLACK).any(), "stacked fragments must not overlap"
    view[mask] = patch[mask]
    return top - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each maroon-divided multicolor rectangle into fragments and restack them above the nearest bottom marker.

    Input structure:
    - The background is black.
    - Several filled rectangular regions appear above the bottom row; each contains
      a single-pixel maroon divider line and two or three monochrome color fragments.
    - The bottom row contains isolated orange marker pixels that indicate the target
      stacking columns.

    Rule:
    - Treat each non-marker foreground rectangle as one source object.
    - Inside each rectangle, discard the maroon divider pixels and extract the
      remaining same-color 4-connected fragments.
    - Sort the fragments from largest area to smallest, erase the original rectangle,
      then stack those fragments above the horizontally nearest orange marker.
    - Every fragment keeps its original shape, each fragment is centered on the
      marker column, and each next fragment is placed directly above the previous one.

    Key insight:
    - The maroon pixels are cutting instructions rather than content: solve the
      puzzle by decomposing each rectangle into colored fragments, ranking them by
      area, and using the bottom orange markers as destination anchors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    marker_cols = [int(c) for c in np.where(g[h - 1] == ORANGE)[0]]
    assert marker_cols, "expected at least one bottom-row orange marker"

    source_mask = g != BLACK
    source_mask[h - 1, marker_cols] = False
    rectangles = _mask_components(source_mask)
    assert rectangles, "expected at least one rectangle"

    out = np.full_like(g, BLACK)
    out[h - 1, marker_cols] = ORANGE

    for coords in rectangles:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        rect = g[r0 : r1 + 1, c0 : c1 + 1]
        fragments = _extract_fragments(rect)

        center = (int(c0) + int(c1)) / 2.0
        marker_col = min(marker_cols, key=lambda mc: (abs(mc - center), mc))

        bottom = h - 2
        for fragment in fragments:
            bottom = _stack_patch(out, np.asarray(fragment["patch"]), bottom, marker_col)

    return out


def _random_gaps(total: int, parts: int) -> list[int]:
    gaps = [0] * parts
    for _ in range(total):
        gaps[random.randrange(parts)] += 1
    return gaps


def _make_two_piece(orientation: str, colors: list[int]) -> dict[str, object] | None:
    for _ in range(100):
        if orientation == "v":
            height = random.randint(3, 6)
            left_w = random.randint(1, 3)
            right_w = random.randint(1, 3)
            if left_w == right_w:
                continue
            patch = np.full((height, left_w + 1 + right_w), BLACK, dtype=int)
            patch[:, :left_w] = colors[0]
            patch[:, left_w] = MAROON
            patch[:, left_w + 1 :] = colors[1]
            fragments = [
                {"patch": np.full((height, left_w), colors[0], dtype=int), "area": height * left_w, "anchor": (0, 0)},
                {"patch": np.full((height, right_w), colors[1], dtype=int), "area": height * right_w, "anchor": (0, left_w + 1)},
            ]
            return {"patch": patch, "fragments": fragments, "orientation": orientation, "piece_count": 2}

        width = random.randint(3, 6)
        top_h = random.randint(1, 3)
        bottom_h = random.randint(1, 3)
        if top_h == bottom_h:
            continue
        patch = np.full((top_h + 1 + bottom_h, width), BLACK, dtype=int)
        patch[:top_h, :] = colors[0]
        patch[top_h, :] = MAROON
        patch[top_h + 1 :, :] = colors[1]
        fragments = [
            {"patch": np.full((top_h, width), colors[0], dtype=int), "area": top_h * width, "anchor": (0, 0)},
            {"patch": np.full((bottom_h, width), colors[1], dtype=int), "area": bottom_h * width, "anchor": (top_h + 1, 0)},
        ]
        return {"patch": patch, "fragments": fragments, "orientation": orientation, "piece_count": 2}
    return None


def _make_three_piece(orientation: str, colors: list[int]) -> dict[str, object] | None:
    for _ in range(200):
        if orientation == "v":
            height = random.randint(4, 6)
            left_w = random.randint(1, 3)
            right_w = random.randint(1, 3)
            split_row = random.randint(1, height - 1)
            split_left = random.choice([False, True])

            if split_left:
                areas = [split_row * left_w, (height - split_row) * left_w, height * right_w]
                if len(set(areas)) != 3:
                    continue
                patch = np.full((height, left_w + 1 + right_w), BLACK, dtype=int)
                patch[:split_row, :left_w] = colors[0]
                patch[split_row:, :left_w] = colors[1]
                patch[:, left_w] = MAROON
                patch[:, left_w + 1 :] = colors[2]
                fragments = [
                    {"patch": np.full((split_row, left_w), colors[0], dtype=int), "area": areas[0], "anchor": (0, 0)},
                    {
                        "patch": np.full((height - split_row, left_w), colors[1], dtype=int),
                        "area": areas[1],
                        "anchor": (split_row, 0),
                    },
                    {
                        "patch": np.full((height, right_w), colors[2], dtype=int),
                        "area": areas[2],
                        "anchor": (0, left_w + 1),
                    },
                ]
            else:
                areas = [height * left_w, split_row * right_w, (height - split_row) * right_w]
                if len(set(areas)) != 3:
                    continue
                patch = np.full((height, left_w + 1 + right_w), BLACK, dtype=int)
                patch[:, :left_w] = colors[0]
                patch[:, left_w] = MAROON
                patch[:split_row, left_w + 1 :] = colors[1]
                patch[split_row:, left_w + 1 :] = colors[2]
                fragments = [
                    {"patch": np.full((height, left_w), colors[0], dtype=int), "area": areas[0], "anchor": (0, 0)},
                    {
                        "patch": np.full((split_row, right_w), colors[1], dtype=int),
                        "area": areas[1],
                        "anchor": (0, left_w + 1),
                    },
                    {
                        "patch": np.full((height - split_row, right_w), colors[2], dtype=int),
                        "area": areas[2],
                        "anchor": (split_row, left_w + 1),
                    },
                ]
            return {"patch": patch, "fragments": fragments, "orientation": orientation, "piece_count": 3}

        width = random.randint(4, 7)
        top_h = random.randint(1, 3)
        bottom_h = random.randint(1, 3)
        split_col = random.randint(1, width - 1)
        split_top = random.choice([False, True])

        if split_top:
            areas = [top_h * split_col, top_h * (width - split_col), bottom_h * width]
            if len(set(areas)) != 3:
                continue
            patch = np.full((top_h + 1 + bottom_h, width), BLACK, dtype=int)
            patch[:top_h, :split_col] = colors[0]
            patch[:top_h, split_col:] = colors[1]
            patch[top_h, :] = MAROON
            patch[top_h + 1 :, :] = colors[2]
            fragments = [
                {"patch": np.full((top_h, split_col), colors[0], dtype=int), "area": areas[0], "anchor": (0, 0)},
                {
                    "patch": np.full((top_h, width - split_col), colors[1], dtype=int),
                    "area": areas[1],
                    "anchor": (0, split_col),
                },
                {
                    "patch": np.full((bottom_h, width), colors[2], dtype=int),
                    "area": areas[2],
                    "anchor": (top_h + 1, 0),
                },
            ]
        else:
            areas = [top_h * width, bottom_h * split_col, bottom_h * (width - split_col)]
            if len(set(areas)) != 3:
                continue
            patch = np.full((top_h + 1 + bottom_h, width), BLACK, dtype=int)
            patch[:top_h, :] = colors[0]
            patch[top_h, :] = MAROON
            patch[top_h + 1 :, :split_col] = colors[1]
            patch[top_h + 1 :, split_col:] = colors[2]
            fragments = [
                {"patch": np.full((top_h, width), colors[0], dtype=int), "area": areas[0], "anchor": (0, 0)},
                {
                    "patch": np.full((bottom_h, split_col), colors[1], dtype=int),
                    "area": areas[1],
                    "anchor": (top_h + 1, 0),
                },
                {
                    "patch": np.full((bottom_h, width - split_col), colors[2], dtype=int),
                    "area": areas[2],
                    "anchor": (top_h + 1, split_col),
                },
            ]
        return {"patch": patch, "fragments": fragments, "orientation": orientation, "piece_count": 3}
    return None


def _make_object(orientation: str, piece_count: int) -> dict[str, object] | None:
    colors = random.sample(PALETTE, piece_count)
    if piece_count == 2:
        return _make_two_piece(orientation, colors)
    return _make_three_piece(orientation, colors)


def _branch_values(n: int, values: tuple[object, object]) -> list[object]:
    chosen = [values[0], values[1]]
    while len(chosen) < n:
        chosen.append(random.choice(values))
    random.shuffle(chosen)
    return chosen


def _sorted_fragments_from_spec(obj: dict[str, object]) -> list[dict[str, object]]:
    fragments = [dict(fragment) for fragment in obj["fragments"]]
    fragments.sort(key=_fragment_key)
    return fragments


def _render_expected(shape: tuple[int, int], marker_cols: list[int], objects: list[dict[str, object]]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BLACK, dtype=int)
    out[h - 1, marker_cols] = ORANGE
    for marker_col, obj in zip(marker_cols, objects):
        bottom = h - 2
        for fragment in _sorted_fragments_from_spec(obj):
            patch = np.asarray(fragment["patch"])
            fh, fw = patch.shape
            top = bottom - fh + 1
            left = marker_col - fw // 2
            if top < 0 or left < 0 or top + fh > h or left + fw > w:
                raise ValueError("stack does not fit")
            mask = patch != BLACK
            view = out[top : top + fh, left : left + fw]
            if bool((view[mask] != BLACK).any()):
                raise ValueError("stack overlap")
            view[mask] = patch[mask]
            bottom = top - 1
    return out


def _layout_input(objects: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray] | None:
    lane_widths = []
    max_stack = 0
    max_rect_h = 0
    for obj in objects:
        rect = np.asarray(obj["patch"])
        fragments = _sorted_fragments_from_spec(obj)
        stack_height = sum(np.asarray(fragment["patch"]).shape[0] for fragment in fragments)
        stack_width = max(np.asarray(fragment["patch"]).shape[1] for fragment in fragments)
        lane_widths.append(max(rect.shape[1], stack_width) + 2)
        max_stack = max(max_stack, stack_height)
        max_rect_h = max(max_rect_h, rect.shape[0])

    min_h = max(max_stack + 1, max_rect_h + 4, 14)
    if min_h > 30:
        return None
    h = random.randint(min_h, min(30, min_h + 6))

    base_w = sum(lane_widths)
    if base_w > 30:
        return None
    slack = random.randint(0, min(30 - base_w, 5))
    gaps = _random_gaps(slack, len(objects) + 1)
    w = base_w + slack

    markers: list[int] = []
    lane_starts: list[int] = []
    cursor = gaps[0]
    for lane_w, gap_after in zip(lane_widths, gaps[1:]):
        lane_starts.append(cursor)
        markers.append(cursor + lane_w // 2)
        cursor += lane_w + gap_after

    inp = np.full((h, w), BLACK, dtype=int)
    inp[h - 1, markers] = ORANGE

    for lane_start, lane_w, obj in zip(lane_starts, lane_widths, objects):
        patch = np.asarray(obj["patch"])
        rect_h, rect_w = patch.shape
        row_max = h - rect_h - 3
        if row_max < 1:
            return None
        top = random.randint(1, row_max)
        col_lo = lane_start + 1
        col_hi = lane_start + lane_w - rect_w - 1
        if col_lo > col_hi:
            return None
        left = random.randint(col_lo, col_hi)
        view = inp[top : top + rect_h, left : left + rect_w]
        if bool((view != BLACK).any()):
            return None
        view[:] = patch

    expected = _render_expected((h, w), markers, objects)
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(2, 3)
    orientations = _branch_values(object_count, ("v", "h"))
    piece_counts = _branch_values(object_count, (2, 3))

    objects: list[dict[str, object]] = []
    for orientation, piece_count in zip(orientations, piece_counts):
        obj = _make_object(str(orientation), int(piece_count))
        if obj is None:
            return None
        objects.append(obj)

    layout = _layout_input(objects)
    if layout is None:
        return None
    inp, expected = layout

    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
