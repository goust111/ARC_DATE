from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 500


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _runs_from_positions(positions: list[int]) -> list[tuple[int, int]]:
    if not positions:
        return []
    runs: list[tuple[int, int]] = []
    start = positions[0]
    prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        runs.append((start, prev))
        start = pos
        prev = pos
    runs.append((start, prev))
    return runs


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _find_frame(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int]:
    h, w = grid.shape
    best_bbox: tuple[int, int, int, int] | None = None
    best_color = -1
    best_score: tuple[int, int, int] | None = None

    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            continue
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        if np.any((pts[:, 0] != r0) & (pts[:, 0] != r1) & (pts[:, 1] != c0) & (pts[:, 1] != c1)):
            continue

        top = grid[r0, c0 : c1 + 1]
        bottom = grid[r1, c0 : c1 + 1]
        left = grid[r0 : r1 + 1, c0]
        right = grid[r0 : r1 + 1, c1]
        if not (
            np.any(top == color)
            and np.any(bottom == color)
            and np.any(left == color)
            and np.any(right == color)
        ):
            continue

        perimeter = np.concatenate([top, bottom, left[1:-1], right[1:-1]])
        if any(int(v) not in (bg, color) for v in perimeter):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
            continue

        gap_counts = (
            int(np.count_nonzero(top[1:-1] == bg)),
            int(np.count_nonzero(right[1:-1] == bg)),
            int(np.count_nonzero(bottom[1:-1] == bg)),
            int(np.count_nonzero(left[1:-1] == bg)),
        )
        if sum(gap_counts) == 0:
            continue

        score = ((r1 - r0 + 1) * (c1 - c0 + 1), sum(gap_counts), int((grid == color).sum()))
        if best_score is None or score > best_score:
            best_score = score
            best_bbox = (r0, c0, r1, c1)
            best_color = color

    assert best_bbox is not None, "failed to locate the rectangular frame"
    return best_bbox, best_color


def _find_palette(
    grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]
) -> tuple[int, list[dict[str, int]]]:
    _, _, fr1, _ = frame_bbox
    h, _ = grid.shape
    best_row = -1
    best_segments: list[dict[str, int]] | None = None
    best_score: tuple[int, int, int] | None = None

    for r in range(fr1 + 1, h - 1):
        mask0 = grid[r] != bg
        mask1 = grid[r + 1] != bg
        if not np.any(mask0) or not np.array_equal(mask0, mask1):
            continue
        cols = np.flatnonzero(mask0)
        if cols.size == 0:
            continue
        start = int(cols[0])
        end = int(cols[-1])
        if end - start + 1 != int(cols.size):
            continue

        segments: list[dict[str, int]] = []
        c = start
        ok = True
        while c <= end:
            color = int(grid[r, c])
            if color == bg or int(grid[r + 1, c]) != color:
                ok = False
                break
            c2 = c
            while c2 <= end and int(grid[r, c2]) == color and int(grid[r + 1, c2]) == color:
                c2 += 1
            segments.append({"c0": c, "c1": c2 - 1, "color": color})
            c = c2
        if not ok or len(segments) < 2:
            continue
        if len({seg["color"] for seg in segments}) != len(segments):
            continue

        score = (r, end - start + 1, len(segments))
        if best_score is None or score > best_score:
            best_score = score
            best_row = r
            best_segments = segments

    assert best_segments is not None, "failed to locate the bottom palette strip"
    return best_row, best_segments


def _find_pieces(
    grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = frame_bbox
    submask = grid[r0 + 1 : r1, c0 + 1 : c1] != bg
    pieces: list[dict[str, object]] = []
    for comp in _components(submask):
        coords = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in comp]
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each interior fragment must be monochrome"
        pieces.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "color": next(iter(colors)),
            }
        )
    assert pieces, "expected at least one interior fragment"
    return pieces


def _clockwise_gaps(
    grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]
) -> list[dict[str, int]]:
    r0, c0, r1, c1 = frame_bbox
    gaps: list[dict[str, int]] = []

    top_runs = _runs_from_positions([c for c in range(c0 + 1, c1) if int(grid[r0, c]) == bg])
    right_runs = _runs_from_positions([r for r in range(r0 + 1, r1) if int(grid[r, c1]) == bg])
    bottom_runs = _runs_from_positions([c for c in range(c0 + 1, c1) if int(grid[r1, c]) == bg])
    left_runs = _runs_from_positions([r for r in range(r0 + 1, r1) if int(grid[r, c0]) == bg])

    for lo, hi in top_runs:
        gaps.append({"side": 0, "lo": lo, "hi": hi})
    for lo, hi in right_runs:
        gaps.append({"side": 1, "lo": lo, "hi": hi})
    for lo, hi in reversed(bottom_runs):
        gaps.append({"side": 2, "lo": lo, "hi": hi})
    for lo, hi in reversed(left_runs):
        gaps.append({"side": 3, "lo": lo, "hi": hi})

    assert gaps, "frame must have at least one gap"
    return gaps


def _piece_order_from_gaps(
    pieces: list[dict[str, object]],
    gaps: list[dict[str, int]],
    frame_bbox: tuple[int, int, int, int],
) -> list[int]:
    r0, c0, r1, c1 = frame_bbox
    order: list[int] = []

    for gap in gaps:
        side = int(gap["side"])
        lo = int(gap["lo"])
        hi = int(gap["hi"])
        matches: list[tuple[int, int]] = []
        for idx, piece in enumerate(pieces):
            pr0, pc0, pr1, pc1 = piece["bbox"]  # type: ignore[misc]
            if side in (0, 2):
                if not _interval_overlap(lo, hi, int(pc0), int(pc1)):
                    continue
                dist = int(pr0) - (r0 + 1) if side == 0 else (r1 - 1) - int(pr1)
            else:
                if not _interval_overlap(lo, hi, int(pr0), int(pr1)):
                    continue
                dist = int(pc0) - (c0 + 1) if side == 3 else (c1 - 1) - int(pc1)
            matches.append((dist, idx))

        assert matches, "every gap must align with a fragment"
        matches.sort()
        best_dist = matches[0][0]
        best = [idx for dist, idx in matches if dist == best_dist]
        assert len(best) == 1, "gap-to-fragment match must be unique"
        order.append(best[0])

    assert len(order) == len(pieces), "gap count must match fragment count"
    assert len(set(order)) == len(order), "each fragment must be matched exactly once"
    return order


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    frame_bbox, frame_color = _find_frame(g, bg)
    palette_row, palette = _find_palette(g, bg, frame_bbox)
    pieces = _find_pieces(g, bg, frame_bbox)
    gaps = _clockwise_gaps(g, bg, frame_bbox)
    order = _piece_order_from_gaps(pieces, gaps, frame_bbox)
    assert len(palette) == len(order), "palette block count must match fragment count"
    return {
        "bg": bg,
        "frame_bbox": frame_bbox,
        "frame_color": frame_color,
        "palette_row": palette_row,
        "palette": palette,
        "pieces": pieces,
        "gaps": gaps,
        "order": order,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor interior fragments by pairing them with a bottom palette strip,
    where the pairing order is determined by the frame's clockwise border gaps.
    """
    g = np.asarray(grid, dtype=int)
    scene = _extract_scene(g)
    pieces = scene["pieces"]  # type: ignore[assignment]
    palette = scene["palette"]  # type: ignore[assignment]
    order = scene["order"]  # type: ignore[assignment]

    out = g.copy()
    for piece_idx, block in zip(order, palette):
        color = int(block["color"])
        coords = pieces[piece_idx]["coords"]  # type: ignore[index]
        for r, c in coords:
            out[int(r), int(c)] = color
    return out


def _positive_split(total: int, parts: int, rng: random.Random) -> list[int]:
    values = [1] * parts
    for _ in range(total - parts):
        values[rng.randrange(parts)] += 1
    return values


def _layout_sequence(keys: list[int], sizes: list[int], rng: random.Random) -> tuple[dict[int, tuple[int, int]], int]:
    items = list(zip(keys, sizes))
    rng.shuffle(items)
    cursor = rng.randint(0, 2)
    slots: dict[int, tuple[int, int]] = {}
    for idx, (key, size) in enumerate(items):
        slots[key] = (cursor, cursor + size - 1)
        cursor += size
        if idx != len(items) - 1:
            cursor += rng.randint(1, 2)
    span = cursor + rng.randint(0, 2)
    return slots, span


def _draw_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0 : r0 + h, c0 : c0 + w] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    piece_total = rng.randint(4, 6)
    counts = _positive_split(piece_total, 4, rng)
    side_names = ["top", "right", "bottom", "left"]

    colors = list(range(10))
    rng.shuffle(colors)
    bg = colors.pop()
    frame_color = colors.pop()
    piece_colors = colors[:piece_total]
    rng.shuffle(piece_colors)

    pieces: list[dict[str, int | str]] = []
    next_id = 0
    color_idx = 0
    for side, count in zip(side_names, counts):
        for _ in range(count):
            pieces.append(
                {
                    "id": next_id,
                    "side": side,
                    "h": rng.randint(2, 4),
                    "w": rng.randint(2, 4),
                    "color": piece_colors[color_idx],
                }
            )
            next_id += 1
            color_idx += 1

    top_pieces = [p for p in pieces if p["side"] == "top"]
    right_pieces = [p for p in pieces if p["side"] == "right"]
    bottom_pieces = [p for p in pieces if p["side"] == "bottom"]
    left_pieces = [p for p in pieces if p["side"] == "left"]

    tb_keys = [int(p["id"]) for p in top_pieces + bottom_pieces]
    tb_sizes = [int(p["w"]) for p in top_pieces + bottom_pieces]
    col_slots, center_w = _layout_sequence(tb_keys, tb_sizes, rng)

    lr_keys = [int(p["id"]) for p in left_pieces + right_pieces]
    lr_sizes = [int(p["h"]) for p in left_pieces + right_pieces]
    row_slots, middle_h = _layout_sequence(lr_keys, lr_sizes, rng)

    top_h = max(int(p["h"]) for p in top_pieces) + rng.randint(0, 2)
    bottom_h = max(int(p["h"]) for p in bottom_pieces) + rng.randint(0, 2)
    left_w = max(int(p["w"]) for p in left_pieces) + rng.randint(0, 2)
    right_w = max(int(p["w"]) for p in right_pieces) + rng.randint(0, 2)

    sep_lc = rng.randint(1, 2)
    sep_cr = rng.randint(1, 2)
    sep_tm = rng.randint(1, 2)
    sep_mb = rng.randint(1, 2)

    center_c0 = left_w + sep_lc
    right_c0 = center_c0 + center_w + sep_cr
    middle_r0 = top_h + sep_tm
    bottom_r0 = middle_r0 + middle_h + sep_mb

    for piece in pieces:
        pid = int(piece["id"])
        side = str(piece["side"])
        h = int(piece["h"])
        w = int(piece["w"])
        if side == "top":
            slot_c0, _ = col_slots[pid]
            piece["rel_r"] = rng.randint(0, top_h - h)
            piece["rel_c"] = center_c0 + slot_c0
            piece["gap_axis"] = int(piece["rel_c"]) + rng.randint(0, w - 1)
        elif side == "bottom":
            slot_c0, _ = col_slots[pid]
            piece["rel_r"] = bottom_r0 + rng.randint(0, bottom_h - h)
            piece["rel_c"] = center_c0 + slot_c0
            piece["gap_axis"] = int(piece["rel_c"]) + rng.randint(0, w - 1)
        elif side == "left":
            slot_r0, _ = row_slots[pid]
            piece["rel_r"] = middle_r0 + slot_r0
            piece["rel_c"] = rng.randint(0, left_w - w)
            piece["gap_axis"] = int(piece["rel_r"]) + rng.randint(0, h - 1)
        else:
            slot_r0, _ = row_slots[pid]
            piece["rel_r"] = middle_r0 + slot_r0
            piece["rel_c"] = right_c0 + rng.randint(0, right_w - w)
            piece["gap_axis"] = int(piece["rel_r"]) + rng.randint(0, h - 1)

    interior_h = top_h + sep_tm + middle_h + sep_mb + bottom_h
    interior_w = left_w + sep_lc + center_w + sep_cr + right_w
    frame_h = interior_h + 2
    frame_w = interior_w + 2

    top_order = sorted(top_pieces, key=lambda p: int(p["gap_axis"]))
    right_order = sorted(right_pieces, key=lambda p: int(p["gap_axis"]))
    bottom_order = sorted(bottom_pieces, key=lambda p: int(p["gap_axis"]), reverse=True)
    left_order = sorted(left_pieces, key=lambda p: int(p["gap_axis"]), reverse=True)
    ordered_pieces = top_order + right_order + bottom_order + left_order

    ordered_colors = [int(p["color"]) for p in ordered_pieces]
    palette_colors = ordered_colors[:]
    while palette_colors == ordered_colors:
        rng.shuffle(palette_colors)

    palette_width = sum(int(p["w"]) for p in ordered_pieces)
    margin_top = rng.randint(1, 3)
    gap_to_palette = rng.randint(1, 3)
    margin_bottom = rng.randint(1, 2)
    grid_h = margin_top + frame_h + gap_to_palette + 2 + margin_bottom
    grid_w = max(frame_w, palette_width) + rng.randint(4, 8)

    frame_r0 = margin_top
    frame_c0 = rng.randint(1, grid_w - frame_w - 1)
    palette_r0 = frame_r0 + frame_h + gap_to_palette
    palette_c0 = rng.randint(1, grid_w - palette_width - 1)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    fr0, fc0 = frame_r0, frame_c0
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1

    grid[fr0, fc0 : fc1 + 1] = frame_color
    grid[fr1, fc0 : fc1 + 1] = frame_color
    grid[fr0 : fr1 + 1, fc0] = frame_color
    grid[fr0 : fr1 + 1, fc1] = frame_color

    for piece in pieces:
        gr0 = fr0 + 1 + int(piece["rel_r"])
        gc0 = fc0 + 1 + int(piece["rel_c"])
        _draw_rect(grid, gr0, gc0, int(piece["h"]), int(piece["w"]), int(piece["color"]))
        piece["gr0"] = gr0
        piece["gc0"] = gc0
        piece["gr1"] = gr0 + int(piece["h"]) - 1
        piece["gc1"] = gc0 + int(piece["w"]) - 1

    for piece in pieces:
        side = str(piece["side"])
        axis = int(piece["gap_axis"])
        if side == "top":
            grid[fr0, fc0 + 1 + axis] = bg
        elif side == "right":
            grid[fr0 + 1 + axis, fc1] = bg
        elif side == "bottom":
            grid[fr1, fc0 + 1 + axis] = bg
        else:
            grid[fr0 + 1 + axis, fc0] = bg

    cursor = palette_c0
    for piece, pal_color in zip(ordered_pieces, palette_colors):
        width = int(piece["w"])
        grid[palette_r0 : palette_r0 + 2, cursor : cursor + width] = pal_color
        cursor += width

    try:
        scene = _extract_scene(grid)
        if len(scene["pieces"]) != piece_total:  # type: ignore[arg-type]
            return None
        if len(scene["palette"]) != piece_total:  # type: ignore[arg-type]
            return None
        if {int(gap["side"]) for gap in scene["gaps"]} != {0, 1, 2, 3}:  # type: ignore[index]
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if _bg_color(grid) != bg:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
