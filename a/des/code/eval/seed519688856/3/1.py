from __future__ import annotations

import random

import numpy as np


BG = 0
ORANGE = 7
FRAME_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400
PLACE_RETRIES = 200


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _frame_gap_side(mask: np.ndarray) -> int | None:
    h, w = mask.shape
    if h != w or h < 5:
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if bool(mask[1:-1, 1:-1].any()):
        return None

    missing = border & ~mask
    if int(missing.sum()) != 2:
        return None
    if int((mask & ~border).sum()) != 0:
        return None

    pts = np.argwhere(missing)
    rows = sorted(int(v) for v in pts[:, 0])
    cols = sorted(int(v) for v in pts[:, 1])
    if rows[0] == rows[1] == 0 and cols[1] == cols[0] + 1 and 0 < cols[0] and cols[1] < w - 1:
        return 0
    if rows[0] == rows[1] == h - 1 and cols[1] == cols[0] + 1 and 0 < cols[0] and cols[1] < w - 1:
        return 2
    if cols[0] == cols[1] == 0 and rows[1] == rows[0] + 1 and 0 < rows[0] and rows[1] < h - 1:
        return 3
    if cols[0] == cols[1] == w - 1 and rows[1] == rows[0] + 1 and 0 < rows[0] and rows[1] < h - 1:
        return 1
    return None


def _piece_info(patch: np.ndarray, bg: int) -> dict | None:
    h, w = patch.shape
    if h != w or h < 3 or bool((patch == bg).any()):
        return None

    colors = set(map(int, np.unique(patch)))
    if ORANGE not in colors or len(colors) != 2:
        return None

    orange_pts = np.argwhere(patch == ORANGE)
    orange_count = int(orange_pts.shape[0])
    if orange_count not in (1, 2, 3):
        return None

    on_single_side = (
        bool(np.all(orange_pts[:, 0] == 0))
        or bool(np.all(orange_pts[:, 0] == h - 1))
        or bool(np.all(orange_pts[:, 1] == 0))
        or bool(np.all(orange_pts[:, 1] == w - 1))
    )
    if not on_single_side:
        return None

    body_color = int(next(color for color in colors if color != ORANGE))
    return {
        "body_color": body_color,
        "turns": orange_count,
        "patch": patch.copy(),
        "size": h,
    }


def _parse_scene(grid: np.ndarray) -> tuple[int, dict[int, dict], dict[int, dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames: dict[int, dict] = {}
    pieces: dict[int, dict] = {}

    for coords in _find_components(g != bg):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, np.unique(patch[patch != bg])))
        if not colors:
            continue

        if len(colors) == 1:
            color = int(next(iter(colors)))
            mask = patch == color
            gap_side = _frame_gap_side(mask)
            if gap_side is not None:
                if color in frames:
                    raise AssertionError("duplicate frame color")
                frames[color] = {
                    "bbox": (r0, c0, r1, c1),
                    "gap_side": gap_side,
                    "inner_size": int(patch.shape[0] - 2),
                }
                continue

        piece = _piece_info(patch, bg)
        if piece is not None:
            color = int(piece["body_color"])
            if color in pieces:
                raise AssertionError("duplicate piece color")
            piece["bbox"] = (r0, c0, r1, c1)
            pieces[color] = piece

    assert frames and pieces and set(frames) == set(pieces)
    inner_sizes = {int(frame["inner_size"]) for frame in frames.values()}
    assert len(inner_sizes) == 1
    expected_size = inner_sizes.pop()
    for piece in pieces.values():
        assert int(piece["size"]) == expected_size
    return bg, frames, pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored piece to the frame with the same color, rotate the piece
    clockwise by the number of orange markers on its edge, and place it inside.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, pieces = _parse_scene(g)
    out = g.copy()

    for color, piece in pieces.items():
        r0, c0, r1, c1 = piece["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for color, frame in frames.items():
        fr0, fc0, fr1, fc1 = frame["bbox"]
        rotated = np.rot90(pieces[color]["patch"], -int(pieces[color]["turns"]))
        out[fr0 + 1 : fr1, fc0 + 1 : fc1] = rotated

    return out


def _balanced_draw(values: tuple[int, ...], count: int, rng: random.Random) -> list[int]:
    seq = list(values) * ((count + len(values) - 1) // len(values))
    rng.shuffle(seq)
    return [int(v) for v in seq[:count]]


def _make_frame_patch(size: int, color: int, side: int, gap_start: int) -> np.ndarray:
    patch = np.full((size, size), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if side == 0:
        patch[0, gap_start : gap_start + 2] = BG
    elif side == 1:
        patch[gap_start : gap_start + 2, -1] = BG
    elif side == 2:
        patch[-1, gap_start : gap_start + 2] = BG
    else:
        patch[gap_start : gap_start + 2, 0] = BG
    return patch


def _marker_indices(size: int, count: int, rng: random.Random) -> list[int]:
    if count == 1:
        middle = list(range(1, size - 1))
        return [int(rng.choice(middle))]
    start = rng.randint(0, size - count)
    return list(range(start, start + count))


def _make_piece_patch(size: int, color: int, side: int, marker_count: int, rng: random.Random) -> np.ndarray:
    patch = np.full((size, size), color, dtype=int)
    idxs = _marker_indices(size, marker_count, rng)
    if side == 0:
        patch[0, idxs] = ORANGE
    elif side == 1:
        patch[idxs, -1] = ORANGE
    elif side == 2:
        patch[-1, idxs] = ORANGE
    else:
        patch[idxs, 0] = ORANGE
    return patch


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(occupied.shape[0], r + h + pad)
    cc1 = min(occupied.shape[1], c + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
    pad: int = 1,
) -> tuple[int, int] | None:
    gh, gw = grid.shape
    ph, pw = patch.shape
    for _ in range(PLACE_RETRIES):
        r = rng.randint(0, gh - ph)
        c = rng.randint(0, gw - pw)
        if not _can_place(occupied, r, c, ph, pw, pad):
            continue
        grid[r : r + ph, c : c + pw] = patch
        rr0 = max(0, r - pad)
        cc0 = max(0, c - pad)
        rr1 = min(gh, r + ph + pad)
        cc1 = min(gw, c + pw + pad)
        occupied[rr0:rr1, cc0:cc1] = True
        return int(r), int(c)
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    outer_size = int(rng.choice((5, 6, 7, 8)))
    max_frames = {5: 5, 6: 5, 7: 4, 8: 3}[outer_size]
    frame_count = int(rng.randint(2, max_frames))
    inner_size = int(outer_size - 2)

    area_need = frame_count * (outer_size + 2) ** 2 + frame_count * (inner_size + 2) ** 2
    min_dim = max(16, int(np.ceil(np.sqrt(area_need))) + 2)
    if min_dim > 30:
        return None

    height = int(rng.randint(min_dim, 30))
    width = int(rng.randint(min_dim, 30))
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    colors = rng.sample(FRAME_COLORS, frame_count)
    gap_sides = _balanced_draw((0, 1, 2, 3), frame_count, rng)
    marker_sides = _balanced_draw((0, 1, 2, 3), frame_count, rng)
    marker_counts = _balanced_draw((1, 2, 3), frame_count, rng)

    objects: list[tuple[int, np.ndarray]] = []
    for i in range(frame_count):
        gap_start = int(rng.randint(1, outer_size - 3))
        frame_patch = _make_frame_patch(outer_size, int(colors[i]), int(gap_sides[i]), gap_start)
        piece_patch = _make_piece_patch(
            inner_size,
            int(colors[i]),
            int(marker_sides[i]),
            int(marker_counts[i]),
            rng,
        )
        objects.append((outer_size * outer_size, frame_patch))
        objects.append((inner_size * inner_size, piece_patch))

    rng.shuffle(objects)
    objects.sort(key=lambda item: item[0], reverse=True)
    for _area, patch in objects:
        if _place_patch(grid, occupied, patch, rng) is None:
            return None

    try:
        _parse_scene(grid)
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
