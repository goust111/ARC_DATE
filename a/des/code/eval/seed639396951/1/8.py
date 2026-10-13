from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MARKER_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
PATTERN_MODES = ("diag", "anti", "rows", "cols")
LAYOUTS = ("v3", "h3", "grid4")
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_component(arr: np.ndarray, coords: list[tuple[int, int]], fill: int) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), fill, dtype=int)
    for r, c in coords:
        out[r - r0, c - c0] = int(arr[r, c])
    return out, (int(r0), int(c0))


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h = int(grid.shape[0])
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        row = grid[r]
        if int(row[0]) == bg:
            continue
        if np.all(row == row[0]) and np.any(grid[:r] != bg) and np.any(grid[r + 1 :] != bg):
            candidates.append((int(r), int(row[0])))
    assert len(candidates) == 1, "expected one full-width separator row"
    return candidates[0]


def _target_regions(frame: np.ndarray, wall: int, bg: int) -> dict[int, dict]:
    infos: dict[int, dict] = {}
    for coords in _components(frame != wall):
        values = {int(frame[r, c]) for r, c in coords}
        marker_colors = values - {int(bg)}
        assert len(marker_colors) == 1, "each target region must have one marker color"
        marker = int(next(iter(marker_colors)))
        marker_cells = [(r, c) for r, c in coords if int(frame[r, c]) == marker]
        assert len(marker_cells) == 1, "each target region must have one marker pixel"
        infos[marker] = {
            "coords": {(int(r), int(c)) for r, c in coords},
            "marker_pos": marker_cells[0],
        }
    return infos


def _piece_components(bottom: np.ndarray, bg: int, markers: set[int]) -> dict[int, np.ndarray]:
    pieces: dict[int, np.ndarray] = {}
    for coords in _components(bottom != bg):
        colors = {int(bottom[r, c]) for r, c in coords}
        marker_colors = colors & markers
        assert len(marker_colors) == 1, "each piece must contain exactly one known marker color"
        marker = int(next(iter(marker_colors)))
        piece, _ = _crop_component(bottom, coords, bg)
        pieces[marker] = piece
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each loose multi-color piece with the frame region sharing its marker color,
    rotate it so its shape matches the region, and use the marker pixel to choose the
    unique legal alignment inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    sep_row, wall = _find_separator(g, bg)

    top = g[:sep_row]
    assert np.any(top != bg), "frame area must contain non-background pixels"
    top_pts = np.argwhere(top != bg)
    r0 = int(top_pts[:, 0].min())
    c0 = int(top_pts[:, 1].min())
    r1 = int(top_pts[:, 0].max())
    c1 = int(top_pts[:, 1].max())
    frame = top[r0 : r1 + 1, c0 : c1 + 1].copy()

    targets = _target_regions(frame, wall, bg)
    pieces = _piece_components(g[sep_row + 1 :], bg, set(targets))
    assert set(pieces) == set(targets), "marker colors must pair pieces and regions one-to-one"

    out = frame.copy()
    for marker, target in targets.items():
        piece = pieces[marker]
        candidates: list[tuple[np.ndarray, int, int]] = []
        for rot in range(4):
            rotated = np.rot90(piece, rot)
            mask = rotated != bg
            marker_cells = np.argwhere(rotated == marker)
            assert marker_cells.shape[0] == 1, "piece must keep a unique marker cell under rotation"
            mr, mc = map(int, marker_cells[0])
            dr = int(target["marker_pos"][0] - mr)
            dc = int(target["marker_pos"][1] - mc)
            translated = {
                (int(rr + dr), int(cc + dc))
                for rr, cc in np.argwhere(mask)
            }
            if translated == target["coords"]:
                candidates.append((rotated, dr, dc))
        assert len(candidates) == 1, "rotation and marker alignment must determine a unique placement"
        rotated, dr, dc = candidates[0]
        for rr, cc in np.argwhere(rotated != bg):
            out[int(rr + dr), int(cc + dc)] = int(rotated[rr, cc])

    return out


def _make_vertical_wall(h: int, base_c: int, rng: random.Random, can_left: bool, can_right: bool) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    choices = [0]
    if can_left:
        choices.append(-1)
    if can_right:
        choices.append(1)
    direction = int(rng.choice(choices))
    start = end = None
    if direction != 0 and h >= 7:
        start = int(rng.randint(2, h - 3))
        end = int(rng.randint(start + 1, min(h - 1, start + 3)))

    c = int(base_c)
    for r in range(1, h + 1):
        cells.add((int(r), int(c)))
        if start is not None and r == start:
            cells.add((int(r), int(c + direction)))
            c += direction
        if end is not None and r == end:
            cells.add((int(r), int(c - direction)))
            c -= direction
    return cells


def _make_horizontal_wall(w: int, base_r: int, rng: random.Random, can_up: bool, can_down: bool) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    choices = [0]
    if can_up:
        choices.append(-1)
    if can_down:
        choices.append(1)
    direction = int(rng.choice(choices))
    start = end = None
    if direction != 0 and w >= 7:
        start = int(rng.randint(2, w - 3))
        end = int(rng.randint(start + 1, min(w - 1, start + 3)))

    r = int(base_r)
    for c in range(1, w + 1):
        cells.add((int(r), int(c)))
        if start is not None and c == start:
            cells.add((int(r + direction), int(c)))
            r += direction
        if end is not None and c == end:
            cells.add((int(r - direction), int(c)))
            r -= direction
    return cells


def _candidate_marker_cells(frame: np.ndarray, coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = frame.shape
    candidates: list[tuple[int, int]] = []
    for r, c in coords:
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(frame[nr, nc]) != GREY:
                continue
            if nr in (0, h - 1) or nc in (0, w - 1):
                continue
            candidates.append((int(r), int(c)))
            break
    return candidates


def _layout_frame(rng: random.Random) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    layout = str(rng.choice(LAYOUTS))

    for _ in range(120):
        if layout == "v3":
            inner_h = int(rng.randint(7, 11))
            inner_w = int(rng.randint(13, 17))
            c1 = int(rng.randint(4, inner_w - 8))
            c2 = int(rng.randint(c1 + 5, inner_w - 3))
            walls = []
            walls.append(_make_vertical_wall(inner_h, c1, rng, can_left=c1 >= 5, can_right=(c2 - c1) >= 6))
            walls.append(_make_vertical_wall(inner_h, c2, rng, can_left=(c2 - c1) >= 6, can_right=(inner_w - c2) >= 4))
            expected = 3
        elif layout == "h3":
            inner_h = int(rng.randint(13, 17))
            inner_w = int(rng.randint(7, 11))
            r1 = int(rng.randint(4, inner_h - 8))
            r2 = int(rng.randint(r1 + 5, inner_h - 3))
            walls = []
            walls.append(_make_horizontal_wall(inner_w, r1, rng, can_up=r1 >= 5, can_down=(r2 - r1) >= 6))
            walls.append(_make_horizontal_wall(inner_w, r2, rng, can_up=(r2 - r1) >= 6, can_down=(inner_h - r2) >= 4))
            expected = 3
        else:
            inner_h = int(rng.randint(9, 13))
            inner_w = int(rng.randint(9, 13))
            cut_r = int(rng.randint(4, inner_h - 3))
            cut_c = int(rng.randint(4, inner_w - 3))
            walls = []
            walls.append(_make_vertical_wall(inner_h, cut_c, rng, can_left=cut_c >= 5, can_right=(inner_w - cut_c) >= 4))
            walls.append(_make_horizontal_wall(inner_w, cut_r, rng, can_up=cut_r >= 5, can_down=(inner_h - cut_r) >= 4))
            expected = 4

        frame = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
        frame[0, :] = GREY
        frame[-1, :] = GREY
        frame[:, 0] = GREY
        frame[:, -1] = GREY
        for wall_cells in walls:
            for r, c in wall_cells:
                frame[r, c] = GREY

        regions = _components(frame == BG)
        if len(regions) != expected:
            continue
        if min(len(coords) for coords in regions) < 8:
            continue
        if any(not _candidate_marker_cells(frame, coords) for coords in regions):
            continue
        return frame, regions

    raise RuntimeError(f"failed to build layout {layout}")


def _color_piece(mask_piece: np.ndarray, marker_color: int, rng: random.Random, body_pool: list[int]) -> np.ndarray:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask_piece != GREY)]
    marker_cells = [(r, c) for r, c in coords if int(mask_piece[r, c]) == marker_color]
    assert len(marker_cells) == 1
    marker = marker_cells[0]
    body_count = min(len(body_pool), 3)
    palette_size = 2 if body_count == 2 else int(rng.choice([2, body_count]))
    body_colors = rng.sample(body_pool, palette_size)
    mode = str(rng.choice(PATTERN_MODES))
    bias = int(rng.randint(0, 3))

    out = np.full(mask_piece.shape, BG, dtype=int)
    for r, c in coords:
        if (r, c) == marker:
            out[r, c] = marker_color
            continue
        if mode == "diag":
            idx = (2 * r + c + bias) % palette_size
        elif mode == "anti":
            idx = (r + 2 * c + bias) % palette_size
        elif mode == "rows":
            idx = (r + bias) % palette_size
        else:
            idx = (c + bias) % palette_size
        out[r, c] = int(body_colors[idx])

    body_values = {int(out[r, c]) for r, c in coords if (r, c) != marker}
    if len(body_values) < 2:
        ordered = sorted((r, c) for r, c in coords if (r, c) != marker)
        for i, (r, c) in enumerate(ordered):
            out[r, c] = int(body_colors[i % palette_size])

    return out


def _place_pieces(pieces: list[np.ndarray], height: int, width: int, rng: random.Random) -> np.ndarray | None:
    canvas = np.full((height, width), BG, dtype=int)
    occ = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(pieces)), key=lambda i: int(np.sum(pieces[i] != BG)), reverse=True)

    for idx in order:
        piece = pieces[idx]
        ph, pw = piece.shape
        slots = [
            (r, c)
            for r in range(1, height - ph)
            for c in range(1, width - pw)
        ]
        rng.shuffle(slots)
        placed = False
        cells = [(int(r), int(c)) for r, c in np.argwhere(piece != BG)]
        for top, left in slots:
            ok = True
            for r, c in cells:
                gr = top + r
                gc = left + c
                if occ[max(0, gr - 1) : min(height, gr + 2), max(0, gc - 1) : min(width, gc + 2)].any():
                    ok = False
                    break
            if not ok:
                continue
            for r, c in cells:
                gr = top + r
                gc = left + c
                canvas[gr, gc] = int(piece[r, c])
                occ[gr, gc] = True
            placed = True
            break
        if not placed:
            return None
    return canvas


def _build_sample(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    frame_in, regions = _layout_frame(rng)
    region_order = sorted(regions, key=lambda coords: (min(r for r, _ in coords), min(c for _, c in coords)))
    piece_count = len(region_order)
    marker_colors = rng.sample(MARKER_POOL, piece_count)
    body_pool = [c for c in MARKER_POOL if c not in marker_colors]
    assert len(body_pool) >= 4

    frame_out = frame_in.copy()
    scattered: list[np.ndarray] = []

    for marker_color, coords in zip(marker_colors, region_order):
        marker_pos = rng.choice(_candidate_marker_cells(frame_in, coords))
        frame_in[marker_pos] = int(marker_color)

        region_mask, (r0, c0) = _crop_component(frame_in, coords, GREY)
        piece = _color_piece(region_mask, int(marker_color), rng, body_pool)

        for rr, cc in np.argwhere(region_mask != GREY):
            frame_out[r0 + rr, c0 + cc] = int(piece[rr, cc])

        rot = int(rng.randint(0, 3))
        scattered.append(np.rot90(piece, rot))

    max_piece_h = max(int(piece.shape[0]) for piece in scattered)
    max_piece_w = max(int(piece.shape[1]) for piece in scattered)
    total_piece_area = sum(int(np.sum(piece != BG)) for piece in scattered)

    frame_h, frame_w = frame_in.shape
    width = max(
        int(frame_w + rng.randint(2, 6)),
        int(2 * max_piece_w + 8),
        int(total_piece_area // 2 + 8),
    )
    bottom_h = max(
        int(max_piece_h * 2 + 5),
        int(total_piece_area // max(8, width - 2) + max_piece_h + 5),
    )
    bottom_h += int(rng.randint(0, 3))

    bottom = _place_pieces(scattered, bottom_h, width, rng)
    if bottom is None:
        raise RuntimeError("failed to place pieces")

    top_margin = int(rng.randint(1, 2))
    sep_row = int(top_margin + frame_h + rng.randint(1, 2))
    bottom_margin = int(rng.randint(1, 2))
    full_h = int(sep_row + 1 + bottom_h + bottom_margin)
    full_w = int(width)
    grid = np.full((full_h, full_w), BG, dtype=int)

    left_margin = int(rng.randint(1, max(1, full_w - frame_w - 1)))
    grid[top_margin : top_margin + frame_h, left_margin : left_margin + frame_w] = frame_in
    grid[sep_row, :] = GREY
    grid[sep_row + 1 : sep_row + 1 + bottom_h, :] = bottom

    return grid, frame_out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            inp, expected = _build_sample(rng)
        except (AssertionError, RuntimeError):
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate valid input")
