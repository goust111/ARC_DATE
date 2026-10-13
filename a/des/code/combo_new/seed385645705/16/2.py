from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "right", "bottom", "left")


def _most_common(values) -> int:
    counts = Counter(int(v) for v in values)
    assert counts, "expected at least one value"
    return int(counts.most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _touches_border(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _piece_marker_side(patch: np.ndarray) -> tuple[str, int, int]:
    body = _most_common(patch.ravel())
    edge_arrays = {
        "top": patch[0, :],
        "right": patch[:, -1],
        "bottom": patch[-1, :],
        "left": patch[:, 0],
    }
    candidates: list[tuple[str, int]] = []
    for side, arr in edge_arrays.items():
        vals = np.unique(arr)
        if len(vals) == 1 and int(vals[0]) != body:
            candidates.append((side, int(vals[0])))
    assert len(candidates) == 1, "each piece must have exactly one marked edge"
    side, color = candidates[0]
    return side, color, body


def _hole_marker_side(
    grid: np.ndarray, bbox: tuple[int, int, int, int], frame_color: int
) -> tuple[str, int]:
    r0, c0, r1, c1 = bbox
    assert r0 > 0 and c0 > 0 and r1 + 1 < grid.shape[0] and c1 + 1 < grid.shape[1]
    side_arrays = {
        "top": grid[r0 - 1, c0 : c1 + 1],
        "right": grid[r0 : r1 + 1, c1 + 1],
        "bottom": grid[r1 + 1, c0 : c1 + 1],
        "left": grid[r0 : r1 + 1, c0 - 1],
    }
    candidates: list[tuple[str, int]] = []
    for side, arr in side_arrays.items():
        vals = np.unique(arr)
        if len(vals) == 1 and int(vals[0]) != frame_color:
            candidates.append((side, int(vals[0])))
    assert len(candidates) == 1, "each hole must have exactly one uniformly marked side"
    return candidates[0]


def _make_piece(
    height: int, width: int, body_color: int, marker_color: int, side: str
) -> np.ndarray:
    patch = np.full((height, width), int(body_color), dtype=int)
    if side == "top":
        patch[0, :] = int(marker_color)
    elif side == "right":
        patch[:, -1] = int(marker_color)
    elif side == "bottom":
        patch[-1, :] = int(marker_color)
    elif side == "left":
        patch[:, 0] = int(marker_color)
    else:
        raise AssertionError(side)
    return patch


def _non_bg_component_info(grid: np.ndarray, bg: int) -> list[dict]:
    infos = []
    for coords in _components(grid != int(bg)):
        bb = _bbox(coords)
        infos.append({"coords": coords, "bbox": bb, "area": len(coords)})
    infos.sort(key=lambda item: item["area"], reverse=True)
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered marked rectangles into the uniquely matching frame holes.

    Input structure:
    - The background is the dominant azure-like color.
    - One large irregular foreground component acts as a maroon-like puzzle frame and surrounds several rectangular background holes.
    - Each hole is bordered by frame pixels, except that exactly one full side of the hole border is recolored to a distinct marker color.
    - Away from the frame, each remaining foreground component is a solid rectangle whose interior is one color and whose single marked edge uses one hole's marker color.

    Rule:
    - Treat the largest foreground component as the frame and find every enclosed rectangular background hole inside it.
    - For each scattered rectangle, match it to the hole with the same marker color and with compatible dimensions after a 0/90/180/270-degree rotation.
    - Choose the unique rotation whose marked edge ends up on the same side as the hole's marked border, place that rotated rectangle into the hole, and erase the original scattered rectangle back to background.

    Key insight: The marker color is the alignment key; size narrows the candidate hole, and the marked side fixes the only valid rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common(g.ravel())
    comps = _non_bg_component_info(g, bg)
    assert comps, "expected a frame component and at least one foreground object"

    frame = comps[0]
    frame_mask = np.zeros(g.shape, dtype=bool)
    for r, c in frame["coords"]:
        frame_mask[r, c] = True
    frame_color = _most_common(g[r, c] for r, c in frame["coords"])

    holes_by_marker: dict[int, dict] = {}
    for coords in _components(g == int(bg)):
        if _touches_border(coords, g.shape):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        expected_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert len(coords) == expected_area, "holes must be rectangular"
        side, marker_color = _hole_marker_side(g, (r0, c0, r1, c1), frame_color)
        holes_by_marker[marker_color] = {
            "bbox": (r0, c0, r1, c1),
            "shape": (r1 - r0 + 1, c1 - c0 + 1),
            "side": side,
        }

    pieces = comps[1:]
    assert pieces, "expected scattered pieces outside the frame"
    assert len(pieces) == len(holes_by_marker), "piece count must equal hole count"

    out = g.copy()
    for comp in pieces:
        r0, c0, r1, c1 = comp["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert np.all(patch != int(bg)), "pieces must be solid rectangles without background gaps"
        _side, marker_color, _body = _piece_marker_side(patch)
        assert marker_color in holes_by_marker, "piece marker color must match a hole marker"
        hole = holes_by_marker[marker_color]

        fits: list[np.ndarray] = []
        for k in range(4):
            rotated = np.rot90(patch, k)
            if rotated.shape != hole["shape"]:
                continue
            side, color, _ = _piece_marker_side(rotated)
            if color == marker_color and side == hole["side"]:
                fits.append(rotated)
        assert len(fits) == 1, "marker side must determine a unique rotation"

        for rr, cc in comp["coords"]:
            out[rr, cc] = int(bg)

        hr0, hc0, hr1, hc1 = hole["bbox"]
        target = out[hr0 : hr1 + 1, hc0 : hc1 + 1]
        assert np.all(target == int(bg)), "hole interior must be empty in the input"
        target[:, :] = fits[0]

    return out


GEN_BG = 8
GEN_FRAME = 9
GEN_COLOR_POOL = (0, 1, 2, 3, 4, 5, 6, 7)
GEN_GRID_MIN = 20
GEN_GRID_MAX = 30
GEN_HOLE_COUNT_CHOICES = (1, 2, 3, 4)
GEN_HOLE_SIZE_RANGE = (3, 5)
GEN_BOARD_RETRIES = 40
GEN_PLACEMENT_RETRIES = 250
GEN_MAX_ATTEMPTS = 400


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}
    bg_count = freq.get(int(bg), 0)
    return all(bg_count > count for color, count in freq.items() if color != int(bg))


def _can_place_patch(
    canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int, padding: int = 1
) -> bool:
    h, w = patch.shape
    H, W = canvas.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for dr, dc in np.argwhere(patch != int(bg)):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - padding)
        r2 = min(H, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(W, cc + padding + 1)
        if np.any(canvas[r1:r2, c1:c2] != int(bg)):
            return False
    return True


def _paint_hole_border(
    canvas: np.ndarray, r0: int, c0: int, height: int, width: int, side: str, marker: int
) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if side == "top":
        canvas[r0 - 1, c0 : c1 + 1] = int(marker)
    elif side == "right":
        canvas[r0 : r1 + 1, c1 + 1] = int(marker)
    elif side == "bottom":
        canvas[r1 + 1, c0 : c1 + 1] = int(marker)
    elif side == "left":
        canvas[r0 : r1 + 1, c0 - 1] = int(marker)
    else:
        raise AssertionError(side)


def _gen_local_board(
    rng: random.Random, hole_count: int
) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    for _ in range(GEN_BOARD_RETRIES):
        dims = [
            (
                rng.randint(GEN_HOLE_SIZE_RANGE[0], GEN_HOLE_SIZE_RANGE[1]),
                rng.randint(GEN_HOLE_SIZE_RANGE[0], GEN_HOLE_SIZE_RANGE[1]),
            )
            for _ in range(hole_count)
        ]

        board_h = rng.randint(max(11, 8 + 2 * hole_count), 18)
        board_w = rng.randint(max(11, 8 + 2 * hole_count), 18)

        colors = list(GEN_COLOR_POOL)
        rng.shuffle(colors)
        marker_colors = colors[:hole_count]
        body_colors = colors[hole_count : 2 * hole_count]
        if len(body_colors) < hole_count:
            continue

        inp = np.full((board_h, board_w), GEN_FRAME, dtype=int)
        out = inp.copy()
        occupied = np.zeros((board_h, board_w), dtype=bool)
        holes: list[dict] = []
        order = list(range(hole_count))
        rng.shuffle(order)

        ok = True
        for idx in order:
            hh, hw = dims[idx]
            positions = []
            for r0 in range(1, board_h - hh):
                for c0 in range(1, board_w - hw):
                    if occupied[r0 - 1 : r0 + hh + 1, c0 - 1 : c0 + hw + 1].any():
                        continue
                    positions.append((r0, c0))
            if not positions:
                ok = False
                break

            r0, c0 = positions[rng.randrange(len(positions))]
            side = rng.choice(SIDES)
            marker = marker_colors[idx]
            body = body_colors[idx]
            patch = _make_piece(hh, hw, body, marker, side)

            inp[r0 : r0 + hh, c0 : c0 + hw] = GEN_BG
            out[r0 : r0 + hh, c0 : c0 + hw] = patch
            _paint_hole_border(inp, r0, c0, hh, hw, side, marker)
            _paint_hole_border(out, r0, c0, hh, hw, side, marker)

            occupied[r0 - 1 : r0 + hh + 1, c0 - 1 : c0 + hw + 1] = True
            holes.append(
                {
                    "bbox": (r0, c0, r0 + hh - 1, c0 + hw - 1),
                    "patch": patch,
                    "marker": marker,
                    "side": side,
                }
            )

        if ok:
            return inp, out, holes
    return None


def _add_attachments(
    rng: random.Random,
    inp: np.ndarray,
    out: np.ndarray,
    board_r: int,
    board_c: int,
    board_h: int,
    board_w: int,
) -> None:
    H, W = inp.shape
    candidate_sides = []
    if board_r >= 1:
        candidate_sides.append("top")
    if board_r + board_h < H:
        candidate_sides.append("bottom")
    if board_c >= 1:
        candidate_sides.append("left")
    if board_c + board_w < W:
        candidate_sides.append("right")

    rng.shuffle(candidate_sides)
    attach_count = rng.randint(0, min(3, len(candidate_sides)))
    for side in candidate_sides[:attach_count]:
        if side in ("top", "bottom"):
            max_h = min(3, board_r if side == "top" else H - (board_r + board_h))
            if max_h <= 0:
                continue
            ah = rng.randint(1, max_h)
            aw = rng.randint(3, max(3, board_w - 1))
            if aw >= board_w:
                aw = board_w - 1
            if aw < 3:
                continue
            offset = rng.randint(0, board_w - aw)
            c0 = board_c + offset
            if side == "top":
                r0 = board_r - ah
            else:
                r0 = board_r + board_h
            region = (slice(r0, r0 + ah), slice(c0, c0 + aw))
        else:
            max_w = min(3, board_c if side == "left" else W - (board_c + board_w))
            if max_w <= 0:
                continue
            aw = rng.randint(1, max_w)
            ah = rng.randint(3, max(3, board_h - 1))
            if ah >= board_h:
                ah = board_h - 1
            if ah < 3:
                continue
            offset = rng.randint(0, board_h - ah)
            r0 = board_r + offset
            if side == "left":
                c0 = board_c - aw
            else:
                c0 = board_c + board_w
            region = (slice(r0, r0 + ah), slice(c0, c0 + aw))

        if np.any(inp[region] != GEN_BG):
            continue
        inp[region] = GEN_FRAME
        out[region] = GEN_FRAME


def _place_scattered_pieces(
    rng: random.Random, canvas: np.ndarray, pieces: list[np.ndarray]
) -> bool:
    order = list(range(len(pieces)))
    order.sort(key=lambda idx: pieces[idx].size, reverse=True)
    for idx in order:
        patch = pieces[idx]
        ph, pw = patch.shape
        positions = [(r0, c0) for r0 in range(canvas.shape[0] - ph + 1) for c0 in range(canvas.shape[1] - pw + 1)]
        rng.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if not _can_place_patch(canvas, patch, r0, c0, GEN_BG, padding=1):
                continue
            canvas[r0 : r0 + ph, c0 : c0 + pw] = patch
            placed = True
            break
        if not placed:
            return False
    return True


def _frame_is_unique_largest(grid: np.ndarray) -> bool:
    infos = _non_bg_component_info(grid, GEN_BG)
    if len(infos) < 2:
        return False
    top = int(infos[0]["area"])
    second = int(infos[1]["area"])
    return top > second


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    hole_count = rng.choice(GEN_HOLE_COUNT_CHOICES)
    board = _gen_local_board(rng, hole_count)
    if board is None:
        return None
    board_inp, board_out, holes = board
    board_h, board_w = board_inp.shape

    min_h = max(GEN_GRID_MIN, board_h + 4)
    min_w = max(GEN_GRID_MIN, board_w + 4)
    max_h = min(GEN_GRID_MAX, board_h + 10)
    max_w = min(GEN_GRID_MAX, board_w + 10)
    if min_h > max_h or min_w > max_w:
        return None

    H = rng.randint(min_h, max_h)
    W = rng.randint(min_w, max_w)
    board_r = rng.randint(1, H - board_h - 1)
    board_c = rng.randint(1, W - board_w - 1)

    inp = np.full((H, W), GEN_BG, dtype=int)
    out = inp.copy()
    inp[board_r : board_r + board_h, board_c : board_c + board_w] = board_inp
    out[board_r : board_r + board_h, board_c : board_c + board_w] = board_out

    _add_attachments(rng, inp, out, board_r, board_c, board_h, board_w)

    pieces = []
    for hole in holes:
        rotation = rng.randrange(4)
        pieces.append(np.rot90(hole["patch"], rotation))
    rng.shuffle(pieces)

    if not _place_scattered_pieces(rng, inp, pieces):
        return None
    if not _background_is_dominant(inp, GEN_BG):
        return None
    if not _frame_is_unique_largest(inp):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
