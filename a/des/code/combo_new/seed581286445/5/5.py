import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    out: list[tuple[int, int]] = []
    start = prev = indices[0]
    for value in indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        out.append((start, prev - start + 1))
        start = prev = value
    out.append((start, prev - start + 1))
    return out


def _piece_info(mask: np.ndarray) -> dict[str, object]:
    rows, cols = np.nonzero(mask)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    return {
        "mask": cropped,
        "bbox": (r0, c0, r1, c1),
        "area": int(cropped.sum()),
    }


def _horizontal_descriptor(mask: np.ndarray) -> dict[str, object] | None:
    h, w = mask.shape
    leftmost: list[int] = []
    rightmost: list[int] = []
    for r in range(h):
        cols = np.flatnonzero(mask[r])
        if cols.size == 0 or int(cols[-1] - cols[0] + 1) != int(cols.size):
            return None
        leftmost.append(int(cols[0]))
        rightmost.append(int(cols[-1]))

    left_vals = sorted(set(leftmost))
    if left_vals == [0]:
        left_notch = None
    elif left_vals == [0, 1]:
        notch_rows = [r for r, value in enumerate(leftmost) if value == 1]
        notch_runs = _runs(notch_rows)
        if len(notch_runs) != 1:
            return None
        left_notch = notch_runs[0]
    else:
        return None

    right_vals = sorted(set(rightmost))
    if right_vals == [w - 1]:
        right_out = None
        base_w = w
    elif w >= 2 and right_vals == [w - 2, w - 1]:
        out_rows = [r for r, value in enumerate(rightmost) if value == w - 1]
        out_runs = _runs(out_rows)
        if len(out_runs) != 1:
            return None
        right_out = out_runs[0]
        base_w = w - 1
    else:
        return None

    return {
        "base_h": h,
        "base_w": base_w,
        "left_notch": left_notch,
        "right_out": right_out,
    }


def _vertical_descriptor(mask: np.ndarray) -> dict[str, object] | None:
    h, w = mask.shape
    topmost: list[int] = []
    bottommost: list[int] = []
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        if rows.size == 0 or int(rows[-1] - rows[0] + 1) != int(rows.size):
            return None
        topmost.append(int(rows[0]))
        bottommost.append(int(rows[-1]))

    top_vals = sorted(set(topmost))
    if top_vals == [0]:
        top_notch = None
    elif top_vals == [0, 1]:
        notch_cols = [c for c, value in enumerate(topmost) if value == 1]
        notch_runs = _runs(notch_cols)
        if len(notch_runs) != 1:
            return None
        top_notch = notch_runs[0]
    else:
        return None

    bottom_vals = sorted(set(bottommost))
    if bottom_vals == [h - 1]:
        bottom_out = None
        base_h = h
    elif h >= 2 and bottom_vals == [h - 2, h - 1]:
        out_cols = [c for c, value in enumerate(bottommost) if value == h - 1]
        out_runs = _runs(out_cols)
        if len(out_runs) != 1:
            return None
        bottom_out = out_runs[0]
        base_h = h - 1
    else:
        return None

    return {
        "base_h": base_h,
        "base_w": w,
        "top_notch": top_notch,
        "bottom_out": bottom_out,
    }


def _find_divider(grid: np.ndarray) -> int:
    divider_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    if len(divider_cols) != 1:
        raise ValueError("expected exactly one full-height grey divider")
    return divider_cols[0]


def _find_frame_bbox(grid: np.ndarray, divider_c: int) -> tuple[int, int, int, int]:
    grey_pts = np.argwhere((grid == GREY) & (np.arange(grid.shape[1])[None, :] > divider_c))
    if grey_pts.size == 0:
        raise ValueError("missing frame")
    r0, c0 = grey_pts.min(axis=0)
    r1, c1 = grey_pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_pieces(grid: np.ndarray, divider_c: int) -> list[dict[str, object]]:
    left = grid[:, :divider_c]
    comps = _components((left != BG) & (left != GREY))
    pieces: list[dict[str, object]] = []
    for comp in comps:
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[int(r - r0), int(c - c0)] = True
        colors = np.unique(left[rows, cols])
        if colors.size != 1:
            raise ValueError("pieces must be monochrome")
        pieces.append(
            {
                "color": int(colors[0]),
                "coords": comp.copy(),
                "mask": mask,
                "area": int(mask.sum()),
            }
        )
    if not pieces:
        raise ValueError("missing pieces")
    return pieces


def _assemble_horizontal(
    grid: np.ndarray,
    pieces: list[dict[str, object]],
    frame_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    descs: list[dict[str, object]] = []
    for piece in pieces:
        desc = _horizontal_descriptor(piece["mask"])
        if desc is None:
            raise ValueError("not a horizontal chain")
        descs.append(desc)

    inner_h = frame_bbox[2] - frame_bbox[0] - 1
    inner_w = frame_bbox[3] - frame_bbox[1] - 1
    if not descs or any(int(desc["base_h"]) != inner_h for desc in descs):
        raise ValueError("frame height mismatch")
    if sum(int(desc["base_w"]) for desc in descs) != inner_w:
        raise ValueError("frame width mismatch")

    areas = [int(piece["area"]) for piece in pieces]
    max_area = max(areas)
    root_candidates = [idx for idx, area in enumerate(areas) if area == max_area]
    if len(root_candidates) != 1:
        raise ValueError("largest piece must be unique")
    root = root_candidates[0]
    if descs[root]["left_notch"] is not None:
        raise ValueError("root cannot have an incoming left notch")

    left_map: dict[tuple[int, int], int] = {}
    for idx, desc in enumerate(descs):
        sig = desc["left_notch"]
        if sig is None:
            continue
        if sig in left_map:
            raise ValueError("left-notch signatures must be unique")
        left_map[sig] = idx

    order: list[int] = []
    used: set[int] = set()
    current = root
    while True:
        if current in used:
            raise ValueError("cycle in piece order")
        used.add(current)
        order.append(current)
        sig = descs[current]["right_out"]
        if sig is None:
            break
        if sig not in left_map:
            raise ValueError("missing matching child")
        current = left_map[sig]

    if len(order) != len(pieces):
        raise ValueError("horizontal chain is incomplete")

    out = np.array(grid, copy=True)
    for piece in pieces:
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    ir0, ic0 = frame_bbox[0] + 1, frame_bbox[1] + 1
    out[ir0 : frame_bbox[2], ic0 : frame_bbox[3]] = BG
    cursor = ic0
    for idx in order:
        color = int(pieces[idx]["color"])
        width = int(descs[idx]["base_w"])
        out[ir0 : frame_bbox[2], cursor : cursor + width] = color
        cursor += width
    if cursor != frame_bbox[3]:
        raise ValueError("horizontal packing did not fill the frame")
    return out


def _assemble_vertical(
    grid: np.ndarray,
    pieces: list[dict[str, object]],
    frame_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    descs: list[dict[str, object]] = []
    for piece in pieces:
        desc = _vertical_descriptor(piece["mask"])
        if desc is None:
            raise ValueError("not a vertical chain")
        descs.append(desc)

    inner_h = frame_bbox[2] - frame_bbox[0] - 1
    inner_w = frame_bbox[3] - frame_bbox[1] - 1
    if not descs or any(int(desc["base_w"]) != inner_w for desc in descs):
        raise ValueError("frame width mismatch")
    if sum(int(desc["base_h"]) for desc in descs) != inner_h:
        raise ValueError("frame height mismatch")

    areas = [int(piece["area"]) for piece in pieces]
    max_area = max(areas)
    root_candidates = [idx for idx, area in enumerate(areas) if area == max_area]
    if len(root_candidates) != 1:
        raise ValueError("largest piece must be unique")
    root = root_candidates[0]
    if descs[root]["top_notch"] is not None:
        raise ValueError("root cannot have an incoming top notch")

    top_map: dict[tuple[int, int], int] = {}
    for idx, desc in enumerate(descs):
        sig = desc["top_notch"]
        if sig is None:
            continue
        if sig in top_map:
            raise ValueError("top-notch signatures must be unique")
        top_map[sig] = idx

    order: list[int] = []
    used: set[int] = set()
    current = root
    while True:
        if current in used:
            raise ValueError("cycle in piece order")
        used.add(current)
        order.append(current)
        sig = descs[current]["bottom_out"]
        if sig is None:
            break
        if sig not in top_map:
            raise ValueError("missing matching child")
        current = top_map[sig]

    if len(order) != len(pieces):
        raise ValueError("vertical chain is incomplete")

    out = np.array(grid, copy=True)
    for piece in pieces:
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    ir0, ic0 = frame_bbox[0] + 1, frame_bbox[1] + 1
    out[ir0 : frame_bbox[2], ic0 : frame_bbox[3]] = BG
    cursor = ir0
    for idx in order:
        color = int(pieces[idx]["color"])
        height = int(descs[idx]["base_h"])
        out[cursor : cursor + height, ic0 : frame_bbox[3]] = color
        cursor += height
    if cursor != frame_bbox[2]:
        raise ValueError("vertical packing did not fill the frame")
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble single-color interlocking puzzle fragments into the empty frame on the right.

    Input structure:
    - The background is Black, and a full-height Grey divider splits the grid into a left scatter area and a right target area.
    - To the right of the divider there is one empty Grey rectangular frame whose interior is Black.
    - To the left of the divider there are several monochrome 4-connected fragments, each with a unique color.
    - Every fragment belongs to one chain: in horizontal cases a piece may have a left-edge notch and/or a right-edge protrusion; in vertical cases it may have a top-edge notch and/or a bottom-edge protrusion.

    Rule:
    - Find the largest fragment and treat it as the first placed piece.
    - Match every protrusion to the unique complementary notch with the same run shape, following the chain from the largest piece outward.
    - Place the recovered pieces into the right frame from the top-left corner, either left-to-right for horizontal chains or top-to-bottom for vertical chains.
    - After assembling all pieces, clear the scattered fragments on the left back to Black while keeping the Grey divider and frame.

    Key insight:
    - The colors do not determine the order; the order comes from reading the complementary edge signatures and following the unique chain that starts at the largest fragment.
    """
    g = np.asarray(grid, dtype=int)
    divider_c = _find_divider(g)
    frame_bbox = _find_frame_bbox(g, divider_c)
    pieces = _extract_pieces(g, divider_c)

    frame_h = frame_bbox[2] - frame_bbox[0] - 1
    frame_w = frame_bbox[3] - frame_bbox[1] - 1
    orientations = ["horizontal", "vertical"] if frame_w > frame_h else ["vertical", "horizontal"]
    last_error: Exception | None = None
    for orientation in orientations:
        try:
            if orientation == "horizontal":
                return _assemble_horizontal(g, pieces, frame_bbox)
            return _assemble_vertical(g, pieces, frame_bbox)
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"failed to reconstruct puzzle: {last_error}")


def _signature_choices(limit: int) -> list[tuple[int, int]]:
    choices: list[tuple[int, int]] = []
    for start in range(1, limit - 1):
        for length in range(1, limit - start):
            if start + length < limit:
                choices.append((start, length))
    return choices


def _build_horizontal_masks(widths: list[int], height: int, sigs: list[tuple[int, int]]) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    n = len(widths)
    for idx, width in enumerate(widths):
        has_right = idx < n - 1
        mask = np.zeros((height, width + (1 if has_right else 0)), dtype=bool)
        mask[:, :width] = True
        if idx > 0:
            start, length = sigs[idx - 1]
            mask[start : start + length, 0] = False
        if has_right:
            start, length = sigs[idx]
            mask[start : start + length, width] = True
        masks.append(mask)
    return masks


def _build_vertical_masks(heights: list[int], width: int, sigs: list[tuple[int, int]]) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    n = len(heights)
    for idx, height in enumerate(heights):
        has_bottom = idx < n - 1
        mask = np.zeros((height + (1 if has_bottom else 0), width), dtype=bool)
        mask[:height, :] = True
        if idx > 0:
            start, length = sigs[idx - 1]
            mask[0, start : start + length] = False
        if has_bottom:
            start, length = sigs[idx]
            mask[height, start : start + length] = True
        masks.append(mask)
    return masks


def _place_masks(masks: list[np.ndarray], region_h: int, region_w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(masks)), key=lambda idx: int(masks[idx].sum()), reverse=True)
    occupied = np.zeros((region_h, region_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(masks)
    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        if mh + 2 > region_h or mw + 2 > region_w:
            return None
        pts = np.argwhere(mask)
        placed = False
        for _ in range(600):
            top = random.randint(1, region_h - mh - 1)
            left = random.randint(1, region_w - mw - 1)
            ok = True
            for pr, pc in pts:
                rr = int(top + pr)
                cc = int(left + pc)
                r0 = max(0, rr - 1)
                r1 = min(region_h, rr + 2)
                c0 = max(0, cc - 1)
                c1 = min(region_w, cc + 2)
                if occupied[r0:r1, c0:c1].any():
                    ok = False
                    break
            if not ok:
                continue
            for pr, pc in pts:
                occupied[int(top + pr), int(left + pc)] = True
            placements[idx] = (top, left)
            placed = True
            break
        if not placed:
            return None
    return [placement for placement in placements if placement is not None]


def _expected_output(
    grid: np.ndarray,
    masks: list[np.ndarray],
    placements: list[tuple[int, int]],
    colors: list[int],
    orientation: str,
    spans: list[int],
    frame_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for mask, (top, left) in zip(masks, placements):
        pts = np.argwhere(mask)
        for pr, pc in pts:
            out[int(top + pr), int(left + pc)] = BG
    ir0, ic0 = frame_bbox[0] + 1, frame_bbox[1] + 1
    out[ir0 : frame_bbox[2], ic0 : frame_bbox[3]] = BG
    if orientation == "horizontal":
        cursor = ic0
        for color, width in zip(colors, spans):
            out[ir0 : frame_bbox[2], cursor : cursor + width] = color
            cursor += width
    else:
        cursor = ir0
        for color, height in zip(colors, spans):
            out[cursor : cursor + height, ic0 : frame_bbox[3]] = color
            cursor += height
    return out


def _horizontal_attempt() -> np.ndarray | None:
    n = random.randint(3, 4)
    height = random.randint(5, 7)
    other_widths = [random.randint(3, 5) for _ in range(n - 1)]
    root_width = max(other_widths) + random.randint(2, 3)
    widths = [root_width] + other_widths
    total_width = sum(widths)
    if total_width >= 18 or total_width <= height + 1:
        return None

    choices = _signature_choices(height)
    if len(choices) < n - 1:
        return None
    sigs = random.sample(choices, n - 1)
    masks = _build_horizontal_masks(widths, height, sigs)
    areas = [int(mask.sum()) for mask in masks]
    if areas[0] != max(areas) or areas.count(areas[0]) != 1:
        return None

    colors = random.sample(list(PIECE_COLORS), n)
    frame_h, frame_w = height + 2, total_width + 2
    max_left_w = 30 - frame_w - 4
    if max_left_w < max(mask.shape[1] for mask in masks) + 3:
        return None
    left_w = random.randint(max(max(mask.shape[1] for mask in masks) + 3, 10), min(max_left_w, 14))
    grid_h = random.randint(max(frame_h + 2, height + 8), min(24, 30))
    frame_top = random.randint(1, grid_h - frame_h - 1)
    divider_c = left_w
    frame_left = divider_c + 2
    grid_w = frame_left + frame_w + random.randint(1, 2)
    if grid_w > 30:
        return None

    placements = _place_masks(masks, grid_h, left_w)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[:, divider_c] = GREY
    grid[frame_top, frame_left : frame_left + frame_w] = GREY
    grid[frame_top + frame_h - 1, frame_left : frame_left + frame_w] = GREY
    grid[frame_top : frame_top + frame_h, frame_left] = GREY
    grid[frame_top : frame_top + frame_h, frame_left + frame_w - 1] = GREY
    for color, mask, (top, left) in zip(colors, masks, placements):
        pts = np.argwhere(mask)
        for pr, pc in pts:
            grid[int(top + pr), int(left + pc)] = color

    frame_bbox = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)
    expected = _expected_output(grid, masks, placements, colors, "horizontal", widths, frame_bbox)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def _vertical_attempt() -> np.ndarray | None:
    n = random.randint(3, 4)
    width = random.randint(5, 7)
    other_heights = [random.randint(3, 5) for _ in range(n - 1)]
    root_height = max(other_heights) + random.randint(2, 3)
    heights = [root_height] + other_heights
    total_height = sum(heights)
    if total_height >= 18 or total_height <= width + 1:
        return None

    choices = _signature_choices(width)
    if len(choices) < n - 1:
        return None
    sigs = random.sample(choices, n - 1)
    masks = _build_vertical_masks(heights, width, sigs)
    areas = [int(mask.sum()) for mask in masks]
    if areas[0] != max(areas) or areas.count(areas[0]) != 1:
        return None

    colors = random.sample(list(PIECE_COLORS), n)
    frame_h, frame_w = total_height + 2, width + 2
    max_left_w = 30 - frame_w - 4
    if max_left_w < width + 4:
        return None
    left_w = random.randint(max(width + 4, 10), min(max_left_w, 15))
    grid_h = random.randint(max(frame_h + 2, total_height + 4), min(26, 30))
    if grid_h <= frame_h + 1:
        return None
    frame_top = random.randint(1, grid_h - frame_h - 1)
    divider_c = left_w
    frame_left = divider_c + 2
    grid_w = frame_left + frame_w + random.randint(1, 2)
    if grid_w > 30:
        return None

    placements = _place_masks(masks, grid_h, left_w)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[:, divider_c] = GREY
    grid[frame_top, frame_left : frame_left + frame_w] = GREY
    grid[frame_top + frame_h - 1, frame_left : frame_left + frame_w] = GREY
    grid[frame_top : frame_top + frame_h, frame_left] = GREY
    grid[frame_top : frame_top + frame_h, frame_left + frame_w - 1] = GREY
    for color, mask, (top, left) in zip(colors, masks, placements):
        pts = np.argwhere(mask)
        for pr, pc in pts:
            grid[int(top + pr), int(left + pc)] = color

    frame_bbox = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)
    expected = _expected_output(grid, masks, placements, colors, "vertical", heights, frame_bbox)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _horizontal_attempt() if random.random() < 0.5 else _vertical_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid interlocking-chain puzzle")
