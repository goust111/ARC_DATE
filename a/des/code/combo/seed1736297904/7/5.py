from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ["top", "right", "bottom", "left"]
SIDE_TO_ROT_K = {
    "top": 0,      # no rotation
    "right": 3,    # clockwise 90 == ccw 270
    "bottom": 2,   # 180
    "left": 1,     # ccw 90
}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _components(mask: np.ndarray, *, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in dirs:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_side_from_marker(marker: tuple[int, int], shape: tuple[int, int]) -> str | None:
    h, w = shape
    centers = {
        "top": (0, w // 2),
        "right": (h // 2, w - 1),
        "bottom": (h - 1, w // 2),
        "left": (h // 2, 0),
    }
    for side, pos in centers.items():
        if marker == pos:
            return side
    return None


def _extract_frame(g: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    if h < 3 or w < 3:
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.any(patch[1:-1, 1:-1] != BACKGROUND):
        return None
    if not np.all(patch[border] != BACKGROUND):
        return None

    border_patch = patch[border]
    if np.any((border_patch != GREY) & (border_patch == BACKGROUND)):
        return None
    marker_positions = np.argwhere(border & (patch != GREY))
    if marker_positions.shape[0] != 1:
        return None
    mr, mc = map(int, marker_positions[0])
    side = _frame_side_from_marker((mr, mc), (h, w))
    if side is None:
        return None
    marker_color = int(patch[mr, mc])
    if marker_color in (BACKGROUND, GREY):
        return None
    if np.any((patch != GREY) & border & ~((np.arange(h)[:, None] == mr) & (np.arange(w)[None, :] == mc))):
        return None

    return {
        "type": "frame",
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "inner_shape": (int(h - 2), int(w - 2)),
        "side": side,
        "rot_k": int(SIDE_TO_ROT_K[side]),
        "marker_color": int(marker_color),
    }


def _extract_piece(g: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    counts = Counter(int(g[r, c]) for r, c in coords)
    if len(counts) not in (2, 3):
        raise AssertionError("piece must use 2 or 3 colors")
    locator_colors = [color for color, count in counts.items() if count == 1]
    if len(locator_colors) != 1:
        raise AssertionError("piece must have exactly one singleton locator color")
    return {
        "type": "piece",
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "coords": [(int(r), int(c)) for r, c in coords],
        "patch": patch,
        "locator_color": int(locator_colors[0]),
    }


def _extract_scene(g: np.ndarray) -> tuple[list[dict], list[dict]]:
    comps = _components(g != BACKGROUND, connectivity=8)
    pieces: list[dict] = []
    frames: list[dict] = []
    for coords in comps:
        frame = _extract_frame(g, coords)
        if frame is not None:
            frames.append(frame)
        else:
            pieces.append(_extract_piece(g, coords))

    if not pieces or not frames or len(pieces) != len(frames):
        raise AssertionError("expected equal non-zero counts of pieces and frames")

    piece_colors = [int(p["locator_color"]) for p in pieces]
    frame_colors = [int(f["marker_color"]) for f in frames]
    if len(piece_colors) != len(set(piece_colors)):
        raise AssertionError("locator colors must be unique across pieces")
    if len(frame_colors) != len(set(frame_colors)):
        raise AssertionError("frame marker colors must be unique")
    if set(piece_colors) != set(frame_colors):
        raise AssertionError("pieces and frames must pair by locator color")
    return pieces, frames


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    pieces, frames = _extract_scene(g)
    piece_by_color = {int(piece["locator_color"]): piece for piece in pieces}

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[int(r), int(c)] = BACKGROUND

    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        out[r0, c0 : c1 + 1] = GREY
        out[r1, c0 : c1 + 1] = GREY
        out[r0 : r1 + 1, c0] = GREY
        out[r0 : r1 + 1, c1] = GREY

        piece = piece_by_color[int(frame["marker_color"])]
        rotated = np.rot90(piece["patch"], int(frame["rot_k"]))
        if tuple(map(int, rotated.shape)) != tuple(map(int, frame["inner_shape"])):
            raise AssertionError("rotated piece must match frame interior")
        out[r0 + 1 : r1, c0 + 1 : c1] = rotated

    return out


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    for _ in range(200):
        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        score = sum((nr + rr, nc + cc) in cells for rr, cc in DIR4)
                        frontier.append((score, nr, nc))
            if not frontier:
                break
            best = max(score for score, _, _ in frontier)
            candidates = [(nr, nc) for score, nr, nc in frontier if score == best]
            cells.add(random.choice(candidates))
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[int(r), int(c)] = True
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if int(mask.sum()) == h * w:
            continue
        return mask
    return None


def _grow_subset(mask: np.ndarray, *, forbidden: set[tuple[int, int]], size: int) -> set[tuple[int, int]] | None:
    available = [tuple(map(int, rc)) for rc in np.argwhere(mask) if tuple(map(int, rc)) not in forbidden]
    if len(available) < size:
        return None
    seed = random.choice(available)
    subset = {seed}
    while len(subset) < size:
        frontier = []
        for r, c in subset:
            for dr, dc in DIR4:
                nb = (int(r + dr), int(c + dc))
                if 0 <= nb[0] < mask.shape[0] and 0 <= nb[1] < mask.shape[1] and bool(mask[nb]) and nb not in forbidden and nb not in subset:
                    frontier.append(nb)
        if frontier:
            subset.add(random.choice(frontier))
            continue
        remaining = [rc for rc in available if rc not in subset]
        if not remaining:
            return None
        subset.add(random.choice(remaining))
    return subset


def _make_piece(locator_color: int, body_pool: list[int], *, prefer_rect: bool) -> np.ndarray | None:
    dims = [(3, 3), (3, 5), (5, 3), (5, 5)]
    if prefer_rect:
        dims = [(3, 5), (5, 3), (3, 3), (5, 5)]
    for _ in range(300):
        h, w = random.choice(dims)
        min_area = max(5, h + w - 1)
        max_area = max(min_area, int(np.floor(0.78 * h * w)))
        area = random.randint(min_area, max_area)
        mask = _random_connected_mask(h, w, area)
        if mask is None:
            continue

        patch = np.zeros((h, w), dtype=int)
        cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        locator_cell = random.choice(cells)

        n_colors = 2 if random.random() < 0.5 else 3
        body_colors = random.sample(body_pool, n_colors - 1)
        patch[mask] = int(body_colors[0])
        patch[locator_cell] = int(locator_color)

        if n_colors == 3:
            max_second = len(cells) - 3
            if max_second < 2:
                continue
            second_count = random.randint(2, max_second)
            second_cells = _grow_subset(mask, forbidden={locator_cell}, size=second_count)
            if second_cells is None:
                continue
            for r, c in second_cells:
                patch[int(r), int(c)] = int(body_colors[1])

        if any(np.array_equal(patch, np.rot90(patch, k)) for k in (1, 2, 3)):
            continue
        return patch
    return None


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_shape: tuple[int, int], side: str, marker_color: int) -> None:
    ih, iw = map(int, inner_shape)
    h = ih + 2
    w = iw + 2
    grid[top : top + h, left : left + w] = BACKGROUND
    grid[top, left : left + w] = GREY
    grid[top + h - 1, left : left + w] = GREY
    grid[top : top + h, left] = GREY
    grid[top : top + h, left + w - 1] = GREY
    marker_offsets = {
        "top": (0, w // 2),
        "right": (h // 2, w - 1),
        "bottom": (h - 1, w // 2),
        "left": (h // 2, 0),
    }
    mr, mc = marker_offsets[side]
    grid[int(top + mr), int(left + mc)] = int(marker_color)


def _try_build_grid(pieces: list[dict], *, layout: str, pieces_first: bool) -> np.ndarray | None:
    n = len(pieces)
    piece_order = random.sample(range(n), n)
    frame_order = random.sample(range(n), n)

    if layout == "horizontal":
        slot_sizes = [
            max(
                int(pieces[piece_order[i]]["patch"].shape[0]),
                int(pieces[frame_order[i]]["rotated_patch"].shape[0]) + 2,
            )
            for i in range(n)
        ]
        gap = random.randint(1, 2)
        top_margin = random.randint(1, 2)
        bottom_margin = random.randint(1, 2)
        left_margin = random.randint(1, 2)
        right_margin = random.randint(1, 2)
        piece_strip_w = max(int(p["patch"].shape[1]) for p in pieces) + random.randint(2, 4)
        frame_strip_w = max(int(p["rotated_patch"].shape[1]) for p in pieces) + 2 + random.randint(2, 4)
        if pieces_first:
            left_strip_w = piece_strip_w
            right_strip_w = frame_strip_w
        else:
            left_strip_w = frame_strip_w
            right_strip_w = piece_strip_w
        center_gap = random.randint(4, 7)
        h = top_margin + bottom_margin + sum(slot_sizes) + gap * (n - 1)
        w = left_margin + left_strip_w + center_gap + right_strip_w + right_margin
        if h > 30 or w > 30:
            return None
        grid = np.full((h, w), BACKGROUND, dtype=int)
        cur = top_margin
        zone_a_x = left_margin
        zone_b_x = left_margin + left_strip_w + center_gap
        if pieces_first:
            piece_zone_x, piece_zone_w = zone_a_x, left_strip_w
            frame_zone_x, frame_zone_w = zone_b_x, right_strip_w
        else:
            piece_zone_x, piece_zone_w = zone_b_x, right_strip_w
            frame_zone_x, frame_zone_w = zone_a_x, left_strip_w
        for i in range(n):
            piece = pieces[piece_order[i]]
            frame_piece = pieces[frame_order[i]]
            slot_h = int(slot_sizes[i])
            ph, pw = map(int, piece["patch"].shape)
            fh = int(frame_piece["rotated_patch"].shape[0]) + 2
            fw = int(frame_piece["rotated_patch"].shape[1]) + 2
            py = cur + random.randint(0, slot_h - ph)
            fy = cur + random.randint(0, slot_h - fh)
            px = piece_zone_x + random.randint(0, piece_zone_w - pw)
            fx = frame_zone_x + random.randint(0, frame_zone_w - fw)
            patch = piece["patch"]
            grid[py : py + ph, px : px + pw] = patch
            _draw_frame(grid, fy, fx, frame_piece["rotated_patch"].shape, frame_piece["side"], int(frame_piece["locator_color"]))
            cur += slot_h + gap
        return grid

    slot_sizes = [
        max(
            int(pieces[piece_order[i]]["patch"].shape[1]),
            int(pieces[frame_order[i]]["rotated_patch"].shape[1]) + 2,
        )
        for i in range(n)
    ]
    gap = random.randint(1, 2)
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    piece_strip_h = max(int(p["patch"].shape[0]) for p in pieces) + random.randint(2, 4)
    frame_strip_h = max(int(p["rotated_patch"].shape[0]) for p in pieces) + 2 + random.randint(2, 4)
    if pieces_first:
        top_strip_h = piece_strip_h
        bottom_strip_h = frame_strip_h
    else:
        top_strip_h = frame_strip_h
        bottom_strip_h = piece_strip_h
    center_gap = random.randint(4, 7)
    h = top_margin + top_strip_h + center_gap + bottom_strip_h + bottom_margin
    w = left_margin + right_margin + sum(slot_sizes) + gap * (n - 1)
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BACKGROUND, dtype=int)
    zone_a_y = top_margin
    zone_b_y = top_margin + top_strip_h + center_gap
    if pieces_first:
        piece_zone_y, piece_zone_h = zone_a_y, top_strip_h
        frame_zone_y, frame_zone_h = zone_b_y, bottom_strip_h
    else:
        piece_zone_y, piece_zone_h = zone_b_y, bottom_strip_h
        frame_zone_y, frame_zone_h = zone_a_y, top_strip_h
    cur = left_margin
    for i in range(n):
        piece = pieces[piece_order[i]]
        frame_piece = pieces[frame_order[i]]
        slot_w = int(slot_sizes[i])
        ph, pw = map(int, piece["patch"].shape)
        fh = int(frame_piece["rotated_patch"].shape[0]) + 2
        fw = int(frame_piece["rotated_patch"].shape[1]) + 2
        py = piece_zone_y + random.randint(0, piece_zone_h - ph)
        fy = frame_zone_y + random.randint(0, frame_zone_h - fh)
        px = cur + random.randint(0, slot_w - pw)
        fx = cur + random.randint(0, slot_w - fw)
        patch = piece["patch"]
        grid[py : py + ph, px : px + pw] = patch
        _draw_frame(grid, fy, fx, frame_piece["rotated_patch"].shape, frame_piece["side"], int(frame_piece["locator_color"]))
        cur += slot_w + gap
    return grid


def _generate_attempt() -> np.ndarray | None:
    n = random.randint(2, 4)
    locator_colors = random.sample(COLOR_POOL, n)
    body_pool = [color for color in COLOR_POOL if color not in locator_colors]
    side_cycle = random.sample(SIDES, len(SIDES))
    sides = side_cycle[:n] if n < 4 else side_cycle

    pieces: list[dict] = []
    need_rect = any(side in ("right", "left") for side in sides)
    got_rect_turn = False
    for i in range(n):
        side = sides[i]
        prefer_rect = side in ("right", "left") and (need_rect and not got_rect_turn)
        patch = _make_piece(int(locator_colors[i]), body_pool, prefer_rect=prefer_rect)
        if patch is None:
            return None
        rotated_patch = np.rot90(patch, SIDE_TO_ROT_K[side])
        if side in ("right", "left") and patch.shape[0] != patch.shape[1]:
            got_rect_turn = True
        pieces.append(
            {
                "locator_color": int(locator_colors[i]),
                "side": side,
                "patch": patch,
                "rotated_patch": rotated_patch,
            }
        )

    if need_rect and not got_rect_turn:
        return None

    layouts = [("horizontal", True), ("horizontal", False), ("vertical", True), ("vertical", False)]
    random.shuffle(layouts)
    grid = None
    for layout, pieces_first in layouts:
        grid = _try_build_grid(pieces, layout=layout, pieces_first=pieces_first)
        if grid is not None:
            break
    if grid is None:
        return None

    parsed_pieces, parsed_frames = _extract_scene(grid)
    if len(parsed_pieces) != n or len(parsed_frames) != n:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.all(out >= 0):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample after 300 attempts")
