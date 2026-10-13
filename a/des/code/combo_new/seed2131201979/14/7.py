from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME_COLOR = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _find_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in colors or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            coord_array = np.array(coords, dtype=int)
            bbox = _bbox(coord_array)
            components.append(
                {
                    "color": color,
                    "coords": coord_array,
                    "bbox": bbox,
                    "mask": _component_mask(coord_array, bbox),
                }
            )
    return components


def _is_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 4 or w < 4:
        return False
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def _extract_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    frames = []
    for comp in _find_components(grid, {FRAME_COLOR}):
        if _is_frame(comp["mask"]):
            frames.append(comp["bbox"])
    assert frames, "expected one grey outline frame"
    return max(frames, key=lambda b: ((b[2] - b[0] + 1) * (b[3] - b[1] + 1), b[1]))


def _piece_info(mask: np.ndarray) -> dict:
    rows = mask.sum(axis=1)
    cols = mask.sum(axis=0)
    assert np.all(rows > 0) and np.all(cols > 0)
    left = np.argmax(mask, axis=1).astype(int)
    right = (mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)).astype(int)
    top = np.argmax(mask, axis=0).astype(int)
    bottom = (mask.shape[0] - 1 - np.argmax(mask[::-1, :], axis=0)).astype(int)
    return {
        "mask": mask,
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }


def _validate_horizontal(order: list[int], xs: list[int], infos: list[dict], out_h: int, out_w: int) -> list[tuple[int, int, int]] | None:
    occupied = np.zeros((out_h, out_w), dtype=bool)
    placements: list[tuple[int, int, int]] = []
    for idx, x in zip(order, xs):
        mask = infos[idx]["mask"]
        rr, cc = np.nonzero(mask)
        cc = cc + x
        if x < 0 or int(cc.max(initial=-1)) >= out_w:
            return None
        if np.any(occupied[rr, cc]):
            return None
        occupied[rr, cc] = True
        placements.append((idx, 0, x))
    if not occupied.all():
        return None
    return placements


def _validate_vertical(order: list[int], ys: list[int], infos: list[dict], out_h: int, out_w: int) -> list[tuple[int, int, int]] | None:
    occupied = np.zeros((out_h, out_w), dtype=bool)
    placements: list[tuple[int, int, int]] = []
    for idx, y in zip(order, ys):
        mask = infos[idx]["mask"]
        rr, cc = np.nonzero(mask)
        rr = rr + y
        if y < 0 or int(rr.max(initial=-1)) >= out_h:
            return None
        if np.any(occupied[rr, cc]):
            return None
        occupied[rr, cc] = True
        placements.append((idx, y, 0))
    if not occupied.all():
        return None
    return placements


def _solve_horizontal(pieces: list[dict], out_h: int, out_w: int) -> list[tuple[int, int, int]] | None:
    if not pieces or any(piece["mask"].shape[0] != out_h for piece in pieces):
        return None

    infos = [_piece_info(piece["mask"]) for piece in pieces]
    n = len(infos)
    offsets: list[list[int | None]] = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            delta = infos[i]["right"] - infos[j]["left"] + 1
            if np.all(delta == delta[0]):
                offsets[i][j] = int(delta[0])

    def dfs(order: list[int], xs: list[int], used: set[int]) -> list[tuple[int, int, int]] | None:
        if len(order) == n:
            return _validate_horizontal(order, xs, infos, out_h, out_w)
        cur = order[-1]
        for nxt in range(n):
            if nxt in used:
                continue
            delta = offsets[cur][nxt]
            if delta is None:
                continue
            next_x = xs[-1] + delta
            if next_x < 0:
                continue
            used.add(nxt)
            order.append(nxt)
            xs.append(next_x)
            solved = dfs(order, xs, used)
            if solved is not None:
                return solved
            xs.pop()
            order.pop()
            used.remove(nxt)
        return None

    for start in range(n):
        solved = dfs([start], [0], {start})
        if solved is not None:
            return solved
    return None


def _solve_vertical(pieces: list[dict], out_h: int, out_w: int) -> list[tuple[int, int, int]] | None:
    if not pieces or any(piece["mask"].shape[1] != out_w for piece in pieces):
        return None

    infos = [_piece_info(piece["mask"]) for piece in pieces]
    n = len(infos)
    offsets: list[list[int | None]] = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            delta = infos[i]["bottom"] - infos[j]["top"] + 1
            if np.all(delta == delta[0]):
                offsets[i][j] = int(delta[0])

    def dfs(order: list[int], ys: list[int], used: set[int]) -> list[tuple[int, int, int]] | None:
        if len(order) == n:
            return _validate_vertical(order, ys, infos, out_h, out_w)
        cur = order[-1]
        for nxt in range(n):
            if nxt in used:
                continue
            delta = offsets[cur][nxt]
            if delta is None:
                continue
            next_y = ys[-1] + delta
            if next_y < 0:
                continue
            used.add(nxt)
            order.append(nxt)
            ys.append(next_y)
            solved = dfs(order, ys, used)
            if solved is not None:
                return solved
            ys.pop()
            order.pop()
            used.remove(nxt)
        return None

    for start in range(n):
        solved = dfs([start], [0], {start})
        if solved is not None:
            return solved
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered jagged-edged rectangular fragments into the grey target frame.

    Input structure:
    - The background is black.
    - A single grey hollow rectangle on the right marks the target frame.
    - Several non-grey monochrome fragments sit elsewhere; each fragment is a rectangle-like piece whose shared borders have one-cell dents or tabs.

    Rule:
    - Detect the grey frame and measure its interior rectangle.
    - Group each colored fragment as one movable piece, then compare its jagged edge profile against the other pieces to recover the only gapless left-to-right or top-to-bottom assembly.
    - Clear the scattered source pieces and place the reconstructed full rectangle into the frame interior, aligned to the frame's top-left inner corner.

    Key insight:
    - The decisive information is the complementary silhouette of each piece boundary: matching dent/tab profiles determine adjacency, while the frame fixes the final rectangle size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_bbox = _extract_frame(g)
    fr0, fc0, fr1, fc1 = frame_bbox
    inner_h = fr1 - fr0 - 1
    inner_w = fc1 - fc0 - 1
    assert inner_h > 0 and inner_w > 0

    piece_colors = {int(color) for color in np.unique(g) if int(color) not in {BG, FRAME_COLOR}}
    pieces = _find_components(g, piece_colors)
    assert pieces, "expected colored puzzle pieces"

    placements = _solve_horizontal(pieces, inner_h, inner_w)
    if placements is None:
        placements = _solve_vertical(pieces, inner_h, inner_w)
    assert placements is not None, "failed to reconstruct puzzle pieces"

    out = g.copy()
    for piece in pieces:
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for idx, top, left in placements:
        piece = pieces[idx]
        mask = piece["mask"]
        rr, cc = np.nonzero(mask)
        out[fr0 + 1 + top + rr, fc0 + 1 + left + cc] = int(piece["color"])

    return out


def _random_partition(total: int, parts: int, minimum: int) -> list[int]:
    values = [minimum] * parts
    for _ in range(total - parts * minimum):
        values[random.randrange(parts)] += 1
    random.shuffle(values)
    return values


def _sample_teeth(length: int, used: set[tuple[int, ...]]) -> np.ndarray:
    interior = list(range(1, length - 1))
    for _ in range(120):
        mask = np.zeros(length, dtype=int)
        count = random.randint(1, min(3, len(interior)))
        starts = random.sample(interior, count)
        for start in starts:
            run = 1 if random.random() < 0.7 else 2
            for idx in range(start, min(length - 1, start + run)):
                mask[idx] = 1
        key = tuple(int(x) for x in mask)
        if np.any(mask) and key not in used:
            used.add(key)
            return mask
    raise RuntimeError("failed to create unique jagged seam")


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rr, cc = np.nonzero(mask)
    return mask[int(rr.min()) : int(rr.max()) + 1, int(cc.min()) : int(cc.max()) + 1]


def _build_horizontal_pieces(piece_count: int) -> tuple[list[np.ndarray], int, int]:
    final_h = random.randint(8, 12)
    final_w = random.randint(max(8, 2 * piece_count), 12)
    widths = _random_partition(final_w, piece_count, 2)
    cuts = np.cumsum(widths[:-1])
    sign = random.choice((-1, 1))
    used_masks: set[tuple[int, ...]] = set()

    boundaries = [np.zeros(final_h, dtype=int)]
    for cut in cuts:
        boundaries.append(np.full(final_h, int(cut), dtype=int) + sign * _sample_teeth(final_h, used_masks))
    boundaries.append(np.full(final_h, final_w, dtype=int))

    pieces = []
    for idx in range(piece_count):
        full_mask = np.zeros((final_h, final_w), dtype=bool)
        left = boundaries[idx]
        right = boundaries[idx + 1]
        for r in range(final_h):
            c0 = int(left[r])
            c1 = int(right[r])
            if c1 <= c0:
                raise RuntimeError("invalid horizontal seam layout")
            full_mask[r, c0:c1] = True
        pieces.append(full_mask)
    return pieces, final_h, final_w


def _build_vertical_pieces(piece_count: int) -> tuple[list[np.ndarray], int, int]:
    final_h = random.randint(max(8, 2 * piece_count), 12)
    final_w = random.randint(8, 12)
    heights = _random_partition(final_h, piece_count, 2)
    cuts = np.cumsum(heights[:-1])
    sign = random.choice((-1, 1))
    used_masks: set[tuple[int, ...]] = set()

    boundaries = [np.zeros(final_w, dtype=int)]
    for cut in cuts:
        boundaries.append(np.full(final_w, int(cut), dtype=int) + sign * _sample_teeth(final_w, used_masks))
    boundaries.append(np.full(final_w, final_h, dtype=int))

    pieces = []
    for idx in range(piece_count):
        full_mask = np.zeros((final_h, final_w), dtype=bool)
        top = boundaries[idx]
        bottom = boundaries[idx + 1]
        for c in range(final_w):
            r0 = int(top[c])
            r1 = int(bottom[c])
            if r1 <= r0:
                raise RuntimeError("invalid vertical seam layout")
            full_mask[r0:r1, c] = True
        pieces.append(full_mask)
    return pieces, final_h, final_w


def _scatter_pieces(piece_masks: list[np.ndarray], grid_h: int, left_zone_w: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((grid_h, left_zone_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(piece_masks)
    order = sorted(range(len(piece_masks)), key=lambda idx: int(piece_masks[idx].sum()), reverse=True)

    for idx in order:
        mask = piece_masks[idx]
        ph, pw = mask.shape
        if ph > grid_h or pw > left_zone_w:
            return None
        candidates = []
        for _ in range(200):
            top = random.randint(0, grid_h - ph)
            left = random.randint(0, left_zone_w - pw)
            r0 = max(0, top - 1)
            r1 = min(grid_h, top + ph + 1)
            c0 = max(0, left - 1)
            c1 = min(left_zone_w, left + pw + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            candidates.append((top, left))
            if len(candidates) >= 12:
                break
        if not candidates:
            return None
        top, left = random.choice(candidates)
        rr, cc = np.nonzero(mask)
        occupied[top + rr, left + cc] = True
        placements[idx] = (top, left)

    assert all(item is not None for item in placements)
    return [(int(top), int(left)) for top, left in placements if top is not None and left is not None]


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = FRAME_COLOR
    grid[bottom, left : right + 1] = FRAME_COLOR
    grid[top : bottom + 1, left] = FRAME_COLOR
    grid[top : bottom + 1, right] = FRAME_COLOR


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    piece_count = random.randint(3, 5)
    if random.random() < 0.5:
        full_pieces, inner_h, inner_w = _build_horizontal_pieces(piece_count)
    else:
        full_pieces, inner_h, inner_w = _build_vertical_pieces(piece_count)

    colors = random.sample(PIECE_COLORS, piece_count)
    cropped_pieces = [_crop_mask(mask) for mask in full_pieces]
    piece_max_h = max(mask.shape[0] for mask in cropped_pieces)
    piece_max_w = max(mask.shape[1] for mask in cropped_pieces)
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    gap = random.randint(2, 3)

    max_left_zone_w = 30 - gap - frame_w
    if max_left_zone_w < piece_max_w:
        raise RuntimeError("not enough horizontal room for scattered pieces")
    left_zone_w = random.randint(piece_max_w, max_left_zone_w)

    min_grid_h = max(frame_h, piece_max_h) + 2
    max_grid_h = min(30, min_grid_h + 10)
    if min_grid_h > max_grid_h:
        raise RuntimeError("not enough vertical room for scattered pieces")
    grid_h = random.randint(min_grid_h, max_grid_h)

    frame_left = left_zone_w + gap
    frame_top = random.randint(0, grid_h - frame_h)
    grid_w = frame_left + frame_w

    scattered = _scatter_pieces(cropped_pieces, grid_h, left_zone_w)
    if scattered is None:
        raise RuntimeError("failed to scatter pieces")

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)
    _draw_frame(inp, frame_top, frame_left, inner_h, inner_w)
    _draw_frame(expected, frame_top, frame_left, inner_h, inner_w)

    for color, mask, (top, left) in zip(colors, cropped_pieces, scattered):
        rr, cc = np.nonzero(mask)
        inp[top + rr, left + cc] = color

    for color, mask in zip(colors, full_pieces):
        rr, cc = np.nonzero(mask)
        expected[frame_top + 1 + rr, frame_left + 1 + cc] = color

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inp, expected = _build_case()
        except RuntimeError:
            continue
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
