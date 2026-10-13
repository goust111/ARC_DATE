from __future__ import annotations

import math
import random

import numpy as np


BG = 0
MARKER = 2
PIECE_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_CANVAS = 30
GEN_ATTEMPTS = 400
GEN_GROUP_RANGE = (1, 3)
GEN_SIDE_RANGE = (4, 7)
GEN_MARGIN_RANGE = (0, 2)
GEN_PAD = 1
GEN_PARTITION_KINDS = ("grid", "vsplit", "hsplit")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
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


def _tight_patch(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def _marker_turns(marker_rc: tuple[int, int], shape: tuple[int, int]) -> int:
    r, c = marker_rc
    h, w = shape
    if (r, c) == (0, 0):
        return 0
    if (r, c) == (0, w - 1):
        return 1
    if (r, c) == (h - 1, w - 1):
        return 2
    if (r, c) == (h - 1, 0):
        return 3
    raise AssertionError("marker must sit at a bbox corner")


def _decode_piece(patch: np.ndarray) -> tuple[np.ndarray, int]:
    patch = np.asarray(patch, dtype=int)
    assert patch.shape[0] > 1 and patch.shape[1] > 1, "marker corners must be distinct"
    marker_pts = np.argwhere(patch == MARKER)
    assert marker_pts.shape == (1, 2), "each piece must contain exactly one red marker"
    marker_rc = (int(marker_pts[0, 0]), int(marker_pts[0, 1]))
    base_colors = {int(v) for v in np.unique(patch) if int(v) not in (BG, MARKER)}
    assert len(base_colors) == 1, "piece must be monochrome apart from the red marker"
    color = base_colors.pop()
    turns = _marker_turns(marker_rc, patch.shape)

    restored = patch.copy()
    restored[marker_rc] = color
    canonical = np.rot90(restored, -turns)
    return canonical, int(color)


def _assemble_square(pieces: list[np.ndarray], color: int) -> np.ndarray:
    masks = [piece != BG for piece in pieces]
    coords_by_piece = [
        [(int(r), int(c)) for r, c in np.argwhere(mask)]
        for mask in masks
    ]
    areas = [len(coords) for coords in coords_by_piece]
    total_area = sum(areas)
    side = math.isqrt(total_area)
    assert side * side == total_area, "group area must be a perfect square"

    order = sorted(
        range(len(pieces)),
        key=lambda idx: (-areas[idx], pieces[idx].shape[0] + pieces[idx].shape[1], idx),
    )
    board = np.zeros((side, side), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(pieces)
    used = [False] * len(pieces)

    def can_place(idx: int, r0: int, c0: int) -> bool:
        for pr, pc in coords_by_piece[idx]:
            rr = r0 + pr
            cc = c0 + pc
            if not (0 <= rr < side and 0 <= cc < side):
                return False
            if board[rr, cc]:
                return False
        return True

    def paint(idx: int, r0: int, c0: int, value: bool) -> None:
        for pr, pc in coords_by_piece[idx]:
            board[r0 + pr, c0 + pc] = value

    def first_empty() -> tuple[int, int] | None:
        pts = np.argwhere(~board)
        if pts.size == 0:
            return None
        return int(pts[0, 0]), int(pts[0, 1])

    def bt(depth: int) -> bool:
        if depth == len(order):
            return bool(board.all())

        anchor = first_empty()
        assert anchor is not None
        ar, ac = anchor

        for idx in order:
            if used[idx]:
                continue
            tested: set[tuple[int, int]] = set()
            for pr, pc in coords_by_piece[idx]:
                r0 = ar - pr
                c0 = ac - pc
                pos = (int(r0), int(c0))
                if pos in tested:
                    continue
                tested.add(pos)
                if not can_place(idx, r0, c0):
                    continue
                used[idx] = True
                placements[idx] = pos
                paint(idx, r0, c0, True)
                if bt(depth + 1):
                    return True
                paint(idx, r0, c0, False)
                placements[idx] = None
                used[idx] = False
        return False

    assert bt(0), "pieces must tile one square exactly"

    out = np.zeros((side, side), dtype=int)
    for idx, placement in enumerate(placements):
        assert placement is not None
        r0, c0 = placement
        for pr, pc in coords_by_piece[idx]:
            out[r0 + pr, c0 + pc] = color

    assert bool(np.all(out != BG))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Group marker-tagged same-color fragments, undo the encoded rotations, and reassemble each color set into a square.

    Input structure:
    - Black background with scattered 4-connected fragments, each fragment using one non-black color except for exactly one red marker pixel.
    - Fragments are grouped by their underlying non-red color, with exactly four fragments per color group.
    - The red marker lies on one corner of the fragment's tight bounding box.

    Rule:
    - Detect every fragment as one connected component of non-black cells.
    - For each fragment, read the red marker corner as a rotation code: top-left means no rotation, top-right means rotate clockwise 90 degrees, bottom-right means rotate 180 degrees, and bottom-left means rotate counterclockwise 90 degrees.
    - Restore the red marker to the fragment's base color, apply the decoded rotation, and assemble the four same-color fragments into the unique full square they tile.
    - Order the reconstructed squares from left to right by the leftmost input position of each color group, separating adjacent output squares by one black column.

    Key insight:
    - The red pixel is both a missing part of the fragment and a rotation instruction, so each group must be decoded before its four pieces can be packed back into a square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    groups: dict[int, list[dict[str, object]]] = {}
    for coords in _components(g != BG):
        patch = _tight_patch(g, coords)
        canonical, color = _decode_piece(patch)
        min_col = int(coords[:, 1].min())
        min_row = int(coords[:, 0].min())
        groups.setdefault(color, []).append(
            {
                "piece": canonical,
                "min_col": min_col,
                "min_row": min_row,
            }
        )

    assert groups, "expected at least one color group"

    ordered_groups = sorted(
        groups.values(),
        key=lambda items: (
            min(int(item["min_col"]) for item in items),
            min(int(item["min_row"]) for item in items),
        ),
    )

    squares: list[np.ndarray] = []
    for items in ordered_groups:
        assert len(items) == 4, "each color group must contain four fragments"
        color = int(np.max(items[0]["piece"]))  # piece colors are monochrome after decoding
        square = _assemble_square([item["piece"] for item in items], color)
        squares.append(square)

    side_lengths = {square.shape[0] for square in squares}
    assert len(side_lengths) == 1, "all reconstructed squares must share one size"

    out_h = squares[0].shape[0]
    out_w = sum(square.shape[1] for square in squares) + len(squares) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    cursor = 0
    for i, square in enumerate(squares):
        width = square.shape[1]
        out[:, cursor : cursor + width] = square
        cursor += width
        if i + 1 < len(squares):
            cursor += 1
    return out


def _make_rect_piece(height: int, width: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=int)


def _partition_rectangles(side: int) -> list[tuple[int, int]]:
    min_cut = 2
    max_cut = side - 2
    if max_cut < min_cut:
        raise RuntimeError("side is too small for non-degenerate pieces")

    for _ in range(100):
        kind = random.choice(GEN_PARTITION_KINDS)
        dims: list[tuple[int, int]]
        if kind == "grid":
            cut_r = random.randint(min_cut, max_cut)
            cut_c = random.randint(min_cut, max_cut)
            dims = [
                (cut_r, cut_c),
                (cut_r, side - cut_c),
                (side - cut_r, cut_c),
                (side - cut_r, side - cut_c),
            ]
        elif kind == "vsplit":
            cut_c = random.randint(min_cut, max_cut)
            cut_l = random.randint(min_cut, max_cut)
            cut_r = random.randint(min_cut, max_cut)
            dims = [
                (cut_l, cut_c),
                (side - cut_l, cut_c),
                (cut_r, side - cut_c),
                (side - cut_r, side - cut_c),
            ]
        else:
            cut_r = random.randint(min_cut, max_cut)
            cut_t = random.randint(min_cut, max_cut)
            cut_b = random.randint(min_cut, max_cut)
            dims = [
                (cut_r, cut_t),
                (cut_r, side - cut_t),
                (side - cut_r, cut_b),
                (side - cut_r, side - cut_b),
            ]

        if min(min(h, w) for h, w in dims) < 2:
            continue
        return [(int(h), int(w)) for h, w in dims]
    raise RuntimeError("failed to sample a non-degenerate partition")


def _encode_piece(piece: np.ndarray, turns: int) -> np.ndarray:
    encoded = np.rot90(piece, turns).copy()
    h, w = encoded.shape
    if turns == 0:
        encoded[0, 0] = MARKER
    elif turns == 1:
        encoded[0, w - 1] = MARKER
    elif turns == 2:
        encoded[h - 1, w - 1] = MARKER
    else:
        encoded[h - 1, 0] = MARKER
    return encoded


def _can_place_bbox(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0], r0 + h + pad)
    cc1 = min(blocked.shape[1], c0 + w + pad)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_bbox(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0], r0 + h + pad)
    cc1 = min(blocked.shape[1], c0 + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _random_spot(blocked: np.ndarray, h: int, w: int, pad: int) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    max_r = blocked.shape[0] - h
    max_c = blocked.shape[1] - w
    for r0 in range(max_r + 1):
        for c0 in range(max_c + 1):
            if _can_place_bbox(blocked, r0, c0, h, w, pad):
                candidates.append((int(r0), int(c0)))
    if not candidates:
        return None
    return random.choice(candidates)


def _crop_nonzero(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    top = random.randint(*GEN_MARGIN_RANGE)
    left = random.randint(*GEN_MARGIN_RANGE)
    bottom = random.randint(*GEN_MARGIN_RANGE)
    right = random.randint(*GEN_MARGIN_RANGE)
    rr0 = max(0, r0 - top)
    cc0 = max(0, c0 - left)
    rr1 = min(grid.shape[0] - 1, r1 + bottom)
    cc1 = min(grid.shape[1] - 1, c1 + right)
    return grid[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _build_group(color: int, side: int) -> list[np.ndarray]:
    dims = _partition_rectangles(side)
    turns = list(range(4))
    random.shuffle(turns)
    pieces: list[np.ndarray] = []
    for (h, w), turn in zip(dims, turns, strict=True):
        canonical = _make_rect_piece(h, w, color)
        pieces.append(_encode_piece(canonical, turn))
    return pieces


def _gen_attempt() -> np.ndarray | None:
    side = random.randint(*GEN_SIDE_RANGE)
    max_groups = min(GEN_GROUP_RANGE[1], (30 - 1) // (side + 1))
    group_count = random.randint(GEN_GROUP_RANGE[0], max_groups)
    colors = random.sample(PIECE_COLORS, group_count)
    group_color = {gid: color for gid, color in enumerate(colors)}

    pieces_to_place: list[dict[str, object]] = []
    for group_id, color in enumerate(colors):
        for piece in _build_group(color, side):
            pieces_to_place.append(
                {
                    "group_id": group_id,
                    "piece": piece,
                    "area": int(np.count_nonzero(piece)),
                }
            )

    random.shuffle(pieces_to_place)
    pieces_to_place.sort(key=lambda item: -int(item["area"]))

    canvas = np.zeros((GEN_CANVAS, GEN_CANVAS), dtype=int)
    blocked = np.zeros_like(canvas, dtype=bool)
    placed: list[dict[str, object]] = []

    for item in pieces_to_place:
        piece = np.asarray(item["piece"], dtype=int)
        spot = _random_spot(blocked, piece.shape[0], piece.shape[1], GEN_PAD)
        if spot is None:
            return None
        r0, c0 = spot
        canvas[r0 : r0 + piece.shape[0], c0 : c0 + piece.shape[1]] = piece
        _mark_bbox(blocked, r0, c0, piece.shape[0], piece.shape[1], GEN_PAD)
        placed.append(
            {
                "group_id": int(item["group_id"]),
                "r0": r0,
                "c0": c0,
                "piece": piece,
            }
        )

    by_group: dict[int, int] = {}
    for item in placed:
        gid = int(item["group_id"])
        c0 = int(item["c0"])
        by_group[gid] = min(c0, by_group.get(gid, c0))
    if len(set(by_group.values())) != group_count:
        return None

    cropped = _crop_nonzero(canvas)
    order = [gid for gid, _ in sorted(by_group.items(), key=lambda item: (item[1], item[0]))]
    expected = np.zeros((side, group_count * side + group_count - 1), dtype=int)
    cursor = 0
    for i, gid in enumerate(order):
        expected[:, cursor : cursor + side] = int(group_color[gid])
        cursor += side
        if i + 1 < len(order):
            cursor += 1
    if not np.array_equal(main(cropped), expected):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
