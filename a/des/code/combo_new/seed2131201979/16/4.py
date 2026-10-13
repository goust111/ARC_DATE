from __future__ import annotations

import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _touches_bbox_side(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return (
        any(r == r0 for r, _ in coords)
        and any(r == r1 for r, _ in coords)
        and any(c == c0 for _, c in coords)
        and any(c == c1 for _, c in coords)
    )


def _enclosed_rectangles(subgrid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    h, w = subgrid.shape
    slots: list[tuple[int, int, int, int]] = []
    for coords in _connected_components(subgrid == bg):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if len(coords) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        slots.append((r0, c0, r1, c1))
    return slots


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    best: tuple[int, int, tuple[int, int, int, int]] | None = None
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _connected_components(grid == color):
            if len(coords) < 20:
                continue
            bbox = _bbox(coords)
            r0, c0, r1, c1 = bbox
            if r1 - r0 < 4 or c1 - c0 < 4:
                continue
            if not _touches_bbox_side(coords, bbox):
                continue
            subgrid = grid[r0 : r1 + 1, c0 : c1 + 1]
            slots = _enclosed_rectangles(subgrid, bg)
            if not slots:
                continue
            score = len(coords)
            if best is None or score > best[0]:
                best = (score, color, bbox)
    assert best is not None, "expected a unique frame template"
    return best[1], best[2]


def _marker_for_slot(
    subgrid: np.ndarray,
    frame_color: int,
    bg: int,
    slot_bbox: tuple[int, int, int, int],
) -> tuple[int, str]:
    r0, c0, r1, c1 = slot_bbox
    sides = {
        "top": [(r0 - 1, c) for c in range(c0, c1 + 1)],
        "bottom": [(r1 + 1, c) for c in range(c0, c1 + 1)],
        "left": [(r, c0 - 1) for r in range(r0, r1 + 1)],
        "right": [(r, c1 + 1) for r in range(r0, r1 + 1)],
    }
    found: list[tuple[int, str]] = []
    for side, cells in sides.items():
        colors = {
            int(subgrid[r, c])
            for r, c in cells
            if int(subgrid[r, c]) not in (bg, frame_color)
        }
        if colors:
            assert len(colors) == 1, "each slot side should expose at most one marker color"
            found.append((next(iter(colors)), side))
    assert len(found) == 1, "each slot should have exactly one colored marker"
    return found[0]


def _notch_side(patch: np.ndarray, bg: int) -> str:
    holes = np.argwhere(patch == bg)
    assert len(holes) == 1, "piece patch should contain exactly one notch"
    r, c = map(int, holes[0])
    h, w = patch.shape
    sides = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    assert len(sides) == 1, "the notch must lie on exactly one side"
    return sides[0]


def _is_piece_patch(patch: np.ndarray, color: int, bg: int) -> bool:
    if patch.shape[0] < 2 or patch.shape[1] < 2:
        return False
    values = set(map(int, np.unique(patch)))
    if values - {bg, color}:
        return False
    if int(np.count_nonzero(patch == bg)) != 1:
        return False
    if int(np.count_nonzero(patch == color)) != patch.size - 1:
        return False
    try:
        _notch_side(patch, bg)
    except AssertionError:
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place each scattered monochrome fragment into the matching frame slot after rotating its notch toward the slot marker.

    Input structure:
    - The background is the dominant color.
    - One large monochrome frame template contains several rectangular empty slots separated by frame-colored walls.
    - Each slot is surrounded by frame walls and has exactly one single-pixel colored marker on one side of its border.
    - Outside the template, each marker color also appears as one scattered monochrome rectangular fragment with one background notch on exactly one edge.

    Rule:
    - Match every fragment to the unique slot that has the same marker color.
    - Read the marker side of that slot (top, bottom, left, or right).
    - Rotate the fragment by 0, 90, 180, or 270 degrees so its notch faces the marker side and its rectangle fits inside the slot with a one-cell background margin.
    - Clear the fragment from its original location, keep the frame and markers unchanged, and place the rotated fragment centered inside the slot.

    Key insight:
    - The marker color identifies which slot a fragment belongs to, while the marker side determines the only valid rotation by forcing the fragment's missing edge cell to face that side.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frame_color, frame_bbox = _find_frame(g, bg)
    fr0, fc0, fr1, fc1 = frame_bbox
    frame_sub = g[fr0 : fr1 + 1, fc0 : fc1 + 1]

    slot_map: dict[int, dict[str, object]] = {}
    for local_slot in _enclosed_rectangles(frame_sub, bg):
        color, side = _marker_for_slot(frame_sub, frame_color, bg, local_slot)
        sr0, sc0, sr1, sc1 = local_slot
        assert color not in slot_map, "slot marker colors must be unique"
        slot_map[color] = {
            "side": side,
            "slot_bbox": (fr0 + sr0, fc0 + sc0, fr0 + sr1, fc0 + sc1),
        }
    assert slot_map, "expected at least one slot"

    pieces: dict[int, dict[str, object]] = {}
    for color in slot_map:
        components = _connected_components(g == color)
        candidates = []
        for coords in components:
            r0, c0, r1, c1 = _bbox(coords)
            inside_frame = fr0 <= r0 and r1 <= fr1 and fc0 <= c0 and c1 <= fc1
            if inside_frame:
                continue
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            if _is_piece_patch(patch, color, bg):
                candidates.append({"coords": coords, "patch": patch})
        assert len(candidates) == 1, "each slot color must have exactly one scattered fragment"
        pieces[color] = candidates[0]

    out = g.copy()
    for piece in pieces.values():
        for r, c in piece["coords"]:
            out[r, c] = bg

    for color, slot_info in slot_map.items():
        piece_patch = np.array(pieces[color]["patch"], dtype=int)
        side = str(slot_info["side"])
        sr0, sc0, sr1, sc1 = slot_info["slot_bbox"]
        slot_h = sr1 - sr0 + 1
        slot_w = sc1 - sc0 + 1

        candidates = []
        for k in range(4):
            rotated = np.rot90(piece_patch, k)
            if rotated.shape != (slot_h - 2, slot_w - 2):
                continue
            if _notch_side(rotated, bg) != side:
                continue
            candidates.append(rotated)
        assert len(candidates) == 1, "each fragment should have one valid rotation for its slot"
        rotated = candidates[0]
        target = out[sr0 + 1 : sr1, sc0 + 1 : sc1]
        target[:, :] = bg
        target[rotated != bg] = rotated[rotated != bg]

    return out


GEN_LAYOUTS = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]
GEN_SIDES = ("top", "bottom", "left", "right")
GEN_MIN_PIECE = 3
GEN_MAX_PIECE = 4
GEN_MAX_ATTEMPTS = 400


def _make_piece(height: int, width: int, color: int, bg: int, side: str) -> np.ndarray:
    patch = np.full((height, width), color, dtype=int)
    if side in ("top", "bottom"):
        notch_col = random.randint(1, width - 2)
        notch_row = 0 if side == "top" else height - 1
        patch[notch_row, notch_col] = bg
    else:
        notch_row = random.randint(1, height - 2)
        notch_col = 0 if side == "left" else width - 1
        patch[notch_row, notch_col] = bg
    return patch


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int, padding: int = 1) -> bool:
    h, w = patch.shape
    H, W = canvas.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for dr, dc in np.argwhere(patch != bg):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - padding)
        r2 = min(H, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(W, cc + padding + 1)
        if np.any(canvas[r1:r2, c1:c2] != bg):
            return False
    return True


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    target = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    target[mask] = patch[mask]


def _build_frame_and_solution(
    bg: int,
    frame_color: int,
    slot_colors: list[int],
    layout: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    rows, cols = layout
    piece_heights = [random.randint(GEN_MIN_PIECE, GEN_MAX_PIECE) for _ in range(rows)]
    piece_widths = [random.randint(GEN_MIN_PIECE, GEN_MAX_PIECE) for _ in range(cols)]
    inner_heights = [h + 2 for h in piece_heights]
    inner_widths = [w + 2 for w in piece_widths]

    frame_h = sum(inner_heights) + rows + 1
    frame_w = sum(inner_widths) + cols + 1
    empty = np.full((frame_h, frame_w), frame_color, dtype=int)
    solved = np.full((frame_h, frame_w), frame_color, dtype=int)
    pieces: list[np.ndarray] = []

    row_starts = []
    cursor = 1
    for inner_h in inner_heights:
        row_starts.append(cursor)
        cursor += inner_h + 1

    col_starts = []
    cursor = 1
    for inner_w in inner_widths:
        col_starts.append(cursor)
        cursor += inner_w + 1

    color_idx = 0
    for i, inner_h in enumerate(inner_heights):
        for j, inner_w in enumerate(inner_widths):
            r0 = row_starts[i]
            c0 = col_starts[j]
            empty[r0 : r0 + inner_h, c0 : c0 + inner_w] = bg
            solved[r0 : r0 + inner_h, c0 : c0 + inner_w] = bg

            color = slot_colors[color_idx]
            side = random.choice(GEN_SIDES)
            piece = _make_piece(piece_heights[i], piece_widths[j], color, bg, side)
            pieces.append(piece)

            pr0 = r0 + 1
            pc0 = c0 + 1
            _paste(solved, piece, pr0, pc0, bg)

            if side == "top":
                marker_c = random.randint(c0 + 1, c0 + inner_w - 2)
                empty[r0 - 1, marker_c] = color
                solved[r0 - 1, marker_c] = color
            elif side == "bottom":
                marker_c = random.randint(c0 + 1, c0 + inner_w - 2)
                empty[r0 + inner_h, marker_c] = color
                solved[r0 + inner_h, marker_c] = color
            elif side == "left":
                marker_r = random.randint(r0 + 1, r0 + inner_h - 2)
                empty[marker_r, c0 - 1] = color
                solved[marker_r, c0 - 1] = color
            else:
                marker_r = random.randint(r0 + 1, r0 + inner_h - 2)
                empty[marker_r, c0 + inner_w] = color
                solved[marker_r, c0 + inner_w] = color

            color_idx += 1

    return empty, solved, pieces


def _scatter_pieces(
    grid: np.ndarray,
    pieces: list[np.ndarray],
    bg: int,
    frame_bbox: tuple[int, int, int, int],
    horizontal: bool,
) -> bool:
    fr0, fc0, fr1, fc1 = frame_bbox
    H, W = grid.shape
    order = list(range(len(pieces)))
    random.shuffle(order)
    for idx in order:
        piece = np.rot90(pieces[idx], random.randrange(4))
        ph, pw = piece.shape
        candidates: list[tuple[int, int]] = []
        if horizontal:
            min_c = fc1 + 2
            if min_c + pw > W:
                return False
            for r0 in range(0, H - ph + 1):
                for c0 in range(min_c, W - pw + 1):
                    candidates.append((r0, c0))
        else:
            min_r = fr1 + 2
            if min_r + ph > H:
                return False
            for r0 in range(min_r, H - ph + 1):
                for c0 in range(0, W - pw + 1):
                    candidates.append((r0, c0))
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            if _can_place(grid, piece, r0, c0, bg, padding=1):
                _paste(grid, piece, r0, c0, bg)
                placed = True
                break
        if not placed:
            return False
    return True


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    layout = random.choice(GEN_LAYOUTS)
    slot_count = layout[0] * layout[1]
    palette = list(range(10))
    bg = random.choice(palette)
    palette.remove(bg)
    frame_color = random.choice(palette)
    palette.remove(frame_color)
    random.shuffle(palette)
    slot_colors = palette[:slot_count]
    horizontal = random.choice([False, True])

    empty_frame, solved_frame, solved_pieces = _build_frame_and_solution(bg, frame_color, slot_colors, layout)
    frame_h, frame_w = empty_frame.shape

    if horizontal:
        min_h = max(frame_h + 2, 16)
        max_h = 30
        min_w = frame_w + 8
        max_w = 30
    else:
        min_h = frame_h + 8
        max_h = 30
        min_w = max(frame_w + 2, 16)
        max_w = 30
    if min_h > max_h or min_w > max_w:
        return None

    H = random.randint(min_h, max_h)
    W = random.randint(min_w, max_w)
    inp = np.full((H, W), bg, dtype=int)
    out = np.full((H, W), bg, dtype=int)

    if horizontal:
        frame_r0 = random.randint(0, H - frame_h)
        max_frame_c0 = min(2, W - frame_w - 8)
        if max_frame_c0 < 0:
            return None
        frame_c0 = random.randint(0, max_frame_c0)
    else:
        max_frame_r0 = min(2, H - frame_h - 8)
        if max_frame_r0 < 0:
            return None
        frame_r0 = random.randint(0, max_frame_r0)
        frame_c0 = random.randint(0, W - frame_w)

    _paste(inp, empty_frame, frame_r0, frame_c0, bg)
    _paste(out, solved_frame, frame_r0, frame_c0, bg)
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    if not _scatter_pieces(inp, solved_pieces, bg, frame_bbox, horizontal):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        sample = _generate_attempt()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
