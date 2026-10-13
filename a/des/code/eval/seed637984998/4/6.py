from __future__ import annotations

import random
from typing import Any

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _marker_top_left(seg: dict[str, Any], h: int, w: int) -> tuple[int, int]:
    if seg["orientation"] == "h":
        row = int(seg["row"])
        if seg["attach"] == "right":
            return row - 1, 0
        return row - 1, w - 3
    col = int(seg["col"])
    if seg["attach"] == "down":
        return 0, col - 1
    return h - 3, col - 1


def _marker_patch(color: int, level: int, attach: str) -> np.ndarray:
    patch = np.full((3, 3), int(color), dtype=int)
    dots = int(level - 1)
    if dots <= 0:
        return patch
    if dots == 1:
        cols = [1]
    elif dots == 2:
        if attach == "right":
            cols = [0, 1]
        elif attach == "left":
            cols = [1, 2]
        else:
            cols = random.choice(([0, 1], [1, 2]))
    else:
        raise ValueError("levels above 3 are not generated")
    patch[1, cols] = BG
    return patch


def _draw_segment(grid: np.ndarray, seg: dict[str, Any]) -> None:
    color = int(seg["color"])
    if seg["orientation"] == "h":
        grid[int(seg["row"]), int(seg["c0"]) : int(seg["c1"]) + 1] = color
    else:
        grid[int(seg["r0"]) : int(seg["r1"]) + 1, int(seg["col"])] = color


def _render_input(grid_shape: tuple[int, int], segments: list[dict[str, Any]], order: list[int]) -> np.ndarray:
    h, w = grid_shape
    grid = np.full((h, w), BG, dtype=int)
    for idx in order:
        _draw_segment(grid, segments[idx])
    for seg in segments:
        r0, c0 = _marker_top_left(seg, h, w)
        patch = _marker_patch(int(seg["color"]), int(seg["level"]), str(seg["attach"]))
        grid[r0 : r0 + 3, c0 : c0 + 3] = patch
    return grid


def _render_output(grid_shape: tuple[int, int], segments: list[dict[str, Any]]) -> np.ndarray:
    h, w = grid_shape
    out = np.full((h, w), BG, dtype=int)
    for seg in sorted(segments, key=lambda item: int(item["level"])):
        _draw_segment(out, seg)
    return out


def _scan_markers(grid: np.ndarray) -> list[dict[str, Any]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    found: dict[int, dict[str, Any]] = {}

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            win = g[r0 : r0 + 3, c0 : c0 + 3]
            colors = set(map(int, np.unique(win)))
            colors.discard(BG)
            if len(colors) != 1:
                continue
            color = int(next(iter(colors)))
            if not bool((win[0, :] == color).all()) or not bool((win[2, :] == color).all()):
                continue
            mid = win[1, :]
            if not bool(((mid == color) | (mid == BG)).all()):
                continue

            attach_points = []
            candidates = {
                "left": (r0 + 1, c0 - 1),
                "right": (r0 + 1, c0 + 3),
                "up": (r0 - 1, c0 + 1),
                "down": (r0 + 3, c0 + 1),
            }
            for name, (rr, cc) in candidates.items():
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == color:
                    attach_points.append((name, rr, cc))
            if len(attach_points) != 1:
                continue

            perimeter_ok = True
            allowed = {(attach_points[0][1], attach_points[0][2])}
            for rr in range(r0 - 1, r0 + 4):
                for cc in range(c0 - 1, c0 + 4):
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if r0 <= rr <= r0 + 2 and c0 <= cc <= c0 + 2:
                        continue
                    if int(g[rr, cc]) == color and (rr, cc) not in allowed:
                        perimeter_ok = False
                        break
                if not perimeter_ok:
                    break
            if not perimeter_ok:
                continue

            side, rr, cc = attach_points[0]
            marker = {
                "color": color,
                "level": 1 + int(np.sum(mid == BG)),
                "attach": side,
                "endpoint": (int(rr), int(cc)),
                "bbox": (int(r0), int(c0), int(r0 + 2), int(c0 + 2)),
            }
            if color in found:
                prev = found[color]
                if prev != marker:
                    raise AssertionError(f"ambiguous marker for color {color}")
            found[color] = marker

    return list(found.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder straight line overlaps according to endpoint-adjacent 3x3 level markers.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    markers = _scan_markers(g)
    assert markers, "no markers found"

    non_bg_colors = {int(v) for v in np.unique(g) if int(v) != BG}
    marker_colors = {int(item["color"]) for item in markers}
    assert non_bg_colors == marker_colors, "every foreground color must have exactly one marker"

    segments: list[dict[str, Any]] = []
    for marker in markers:
        color = int(marker["color"])
        side = str(marker["attach"])
        er, ec = marker["endpoint"]
        mr0, mc0, mr1, mc1 = marker["bbox"]

        if side in {"left", "right"}:
            visible_cols = [
                c
                for c in range(w)
                if int(g[er, c]) == color and not (mr0 <= er <= mr1 and mc0 <= c <= mc1)
            ]
            assert visible_cols, f"no visible line cells for color {color}"
            far = min(visible_cols) if side == "left" else max(visible_cols)
            segments.append(
                {
                    "color": color,
                    "level": int(marker["level"]),
                    "orientation": "h",
                    "row": int(er),
                    "c0": int(min(ec, far)),
                    "c1": int(max(ec, far)),
                }
            )
        else:
            visible_rows = [
                r
                for r in range(h)
                if int(g[r, ec]) == color and not (mr0 <= r <= mr1 and mc0 <= ec <= mc1)
            ]
            assert visible_rows, f"no visible line cells for color {color}"
            far = min(visible_rows) if side == "up" else max(visible_rows)
            segments.append(
                {
                    "color": color,
                    "level": int(marker["level"]),
                    "orientation": "v",
                    "col": int(ec),
                    "r0": int(min(er, far)),
                    "r1": int(max(er, far)),
                }
            )

    out = np.full((h, w), BG, dtype=int)
    for seg in sorted(segments, key=lambda item: int(item["level"])):
        _draw_segment(out, seg)
    return out


def _choose_spaced_positions(count: int, low: int, high: int, gap: int) -> list[int] | None:
    values = list(range(low, high + 1))
    if count == 0:
        return []
    for _ in range(200):
        picks = sorted(random.sample(values, count))
        if all(b - a >= gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _crossing_order_changes(segments: list[dict[str, Any]], input_order: list[int], target_order: list[int]) -> bool:
    input_rank = {idx: pos for pos, idx in enumerate(input_order)}
    target_rank = {idx: pos for pos, idx in enumerate(target_order)}
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if segments[i]["orientation"] == segments[j]["orientation"]:
                continue
            input_above = int(input_rank[i]) > int(input_rank[j])
            target_above = int(target_rank[i]) > int(target_rank[j])
            if input_above != target_above:
                return True
    return False


def _build_segments(h: int, w: int) -> tuple[list[dict[str, Any]], list[int]] | None:
    n = random.choice([2, 3])
    if n == 2:
        nh, nv = 1, 1
    else:
        nh, nv = random.choice([(2, 1), (1, 2)])

    rows = _choose_spaced_positions(nh, 4, h - 5, gap=3)
    cols = _choose_spaced_positions(nv, 4, w - 5, gap=3)
    if rows is None or cols is None:
        return None

    colors = random.sample(PALETTE, n)
    segments: list[dict[str, Any]] = []

    min_col = min(cols)
    max_col = max(cols)
    for idx, row in enumerate(rows):
        attach = random.choice(["left", "right"])
        if attach == "right":
            end = random.randint(max_col + 1, w - 4)
            segments.append(
                {
                    "color": colors[idx],
                    "orientation": "h",
                    "attach": "right",
                    "row": int(row),
                    "c0": 3,
                    "c1": int(end),
                }
            )
        else:
            start = random.randint(3, min_col - 1)
            segments.append(
                {
                    "color": colors[idx],
                    "orientation": "h",
                    "attach": "left",
                    "row": int(row),
                    "c0": int(start),
                    "c1": w - 4,
                }
            )

    min_row = min(rows)
    max_row = max(rows)
    for idx, col in enumerate(cols):
        attach = random.choice(["up", "down"])
        if attach == "down":
            end = random.randint(max_row + 1, h - 4)
            segments.append(
                {
                    "color": colors[nh + idx],
                    "orientation": "v",
                    "attach": "down",
                    "col": int(col),
                    "r0": 3,
                    "r1": int(end),
                }
            )
        else:
            start = random.randint(3, min_row - 1)
            segments.append(
                {
                    "color": colors[nh + idx],
                    "orientation": "v",
                    "attach": "up",
                    "col": int(col),
                    "r0": int(start),
                    "r1": h - 4,
                }
            )

    target_order = random.sample(list(range(n)), n)
    for level, seg_idx in enumerate(target_order, start=1):
        segments[seg_idx]["level"] = int(level)

    for _ in range(60):
        input_order = random.sample(list(range(n)), n)
        if _crossing_order_changes(segments, input_order, target_order):
            return segments, input_order
    return None


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(15, 23)
    w = random.randint(15, 23)
    built = _build_segments(h, w)
    if built is None:
        return None

    segments, input_order = built

    inp = _render_input((h, w), segments, input_order)
    expected = _render_output((h, w), segments)

    detected = _scan_markers(inp)
    if len(detected) != len(segments):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            inp, _expected = result
            return inp
    raise RuntimeError("generate_input failed after 400 attempts")
