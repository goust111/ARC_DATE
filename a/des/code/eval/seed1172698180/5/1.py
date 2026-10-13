from __future__ import annotations

from collections import defaultdict

import numpy as np


DIRECTIONS = ("ul", "ur", "dl", "dr")
L_MASKS: dict[str, np.ndarray] = {
    "ul": np.array([[1, 1], [1, 0]], dtype=bool),
    "ur": np.array([[1, 1], [0, 1]], dtype=bool),
    "dl": np.array([[1, 0], [1, 1]], dtype=bool),
    "dr": np.array([[0, 1], [1, 1]], dtype=bool),
}

# --------------- solver ---------------


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if visited[r0, c0] or int(grid[r0, c0]) == bg:
                continue
            color = int(grid[r0, c0])
            stack = [(r0, c0)]
            visited[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                    "area": len(coords),
                }
            )
    return comps


def _classify_frame(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r - r0), int(c - c0)] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        return None

    corners = {
        "ul": (0, 0),
        "ur": (0, w - 1),
        "dl": (h - 1, 0),
        "dr": (h - 1, w - 1),
    }
    missing = [name for name, (rr, cc) in corners.items() if not mask[rr, cc]]
    if len(missing) != 1:
        return None

    expected = border.copy()
    miss_r, miss_c = corners[missing[0]]
    expected[miss_r, miss_c] = False
    if not np.array_equal(mask, expected):
        return None

    return {
        "color": int(comp["color"]),
        "bbox": tuple(map(int, comp["bbox"])),
        "missing": missing[0],
    }


def _classify_l_piece(comp: dict) -> dict | None:
    if int(comp["area"]) != 3:
        return None

    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if (h, w) != (2, 2):
        return None

    mask = np.zeros((2, 2), dtype=bool)
    rel_coords: list[tuple[int, int]] = []
    for r, c in comp["coords"]:
        rr = int(r - r0)
        cc = int(c - c0)
        mask[rr, cc] = True
        rel_coords.append((rr, cc))

    for direction, want in L_MASKS.items():
        if np.array_equal(mask, want):
            return {
                "color": int(comp["color"]),
                "bbox": tuple(map(int, comp["bbox"])),
                "coords": [(int(r), int(c)) for r, c in comp["coords"]],
                "rel_coords": sorted(rel_coords),
                "direction": direction,
            }
    return None


def _frame_corner_origin(bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    if direction == "ul":
        return r0, c0
    if direction == "ur":
        return r0, c1 - 1
    if direction == "dl":
        return r1 - 1, c0
    if direction == "dr":
        return r1 - 1, c1 - 1
    raise ValueError(direction)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular frame's missing corner with the matching L piece, then recolor the
    inserted piece to the frame color and erase the used source piece.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _components_by_color(g, bg)

    frames: list[dict] = []
    pieces: list[dict] = []
    for comp in comps:
        frame = _classify_frame(comp)
        if frame is not None:
            frames.append(frame)
            continue
        piece = _classify_l_piece(comp)
        if piece is not None:
            pieces.append(piece)

    out = g.copy()
    pieces_by_direction: dict[str, list[dict]] = defaultdict(list)
    for piece in sorted(pieces, key=lambda p: tuple(p["bbox"])):
        pieces_by_direction[piece["direction"]].append(piece)

    for frame in sorted(frames, key=lambda f: tuple(f["bbox"])):
        direction = frame["missing"]
        if not pieces_by_direction[direction]:
            continue
        piece = pieces_by_direction[direction].pop(0)
        for r, c in piece["coords"]:
            out[int(r), int(c)] = bg

        dst_r0, dst_c0 = _frame_corner_origin(frame["bbox"], direction)
        for dr, dc in piece["rel_coords"]:
            out[int(dst_r0 + dr), int(dst_c0 + dc)] = int(frame["color"])

    return out


# --------------- generator ---------------

GEN_RETRIES = 400
PLACEMENT_TRIES = 300
EDGE_BAND = 3
PAD = 1


def _frame_patch(color: int, bg: int, h: int, w: int, missing: str) -> np.ndarray:
    patch = np.full((h, w), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if missing == "ul":
        patch[0, 0] = bg
    elif missing == "ur":
        patch[0, -1] = bg
    elif missing == "dl":
        patch[-1, 0] = bg
    else:
        patch[-1, -1] = bg
    return patch


def _l_patch(color: int, bg: int, direction: str) -> np.ndarray:
    patch = np.full((2, 2), bg, dtype=int)
    patch[L_MASKS[direction]] = color
    return patch


def _reserve(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = PAD) -> None:
    hh, ww = blocked.shape
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(hh, int(r0 + h + pad))
    cc1 = min(ww, int(c0 + w + pad))
    blocked[rr0:rr1, cc0:cc1] = True


def _can_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    hh, ww = blocked.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    return not bool(np.any(blocked[r0 : r0 + h, c0 : c0 + w]))


def _paste_full(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _random_frame_position(
    rng: np.random.Generator,
    blocked: np.ndarray,
    h: int,
    w: int,
) -> tuple[int, int] | None:
    hh, ww = blocked.shape
    row_lo = EDGE_BAND
    row_hi = hh - EDGE_BAND - h
    col_lo = EDGE_BAND
    col_hi = ww - EDGE_BAND - w
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(PLACEMENT_TRIES):
        r0 = int(rng.integers(row_lo, row_hi + 1))
        c0 = int(rng.integers(col_lo, col_hi + 1))
        if _can_place(blocked, r0, c0, h, w):
            return r0, c0
    return None


def _random_edge_position(
    rng: np.random.Generator,
    blocked: np.ndarray,
    h: int,
    w: int,
) -> tuple[int, int] | None:
    hh, ww = blocked.shape
    sides = ["top", "bottom", "left", "right"]
    for _ in range(PLACEMENT_TRIES):
        side = sides[int(rng.integers(0, len(sides)))]
        if side == "top":
            row_lo, row_hi = 0, min(EDGE_BAND - 1, hh - h)
            col_lo, col_hi = 0, ww - w
        elif side == "bottom":
            row_lo, row_hi = max(0, hh - EDGE_BAND - h), hh - h
            col_lo, col_hi = 0, ww - w
        elif side == "left":
            row_lo, row_hi = 0, hh - h
            col_lo, col_hi = 0, min(EDGE_BAND - 1, ww - w)
        else:
            row_lo, row_hi = 0, hh - h
            col_lo, col_hi = max(0, ww - EDGE_BAND - w), ww - w

        if row_lo > row_hi or col_lo > col_hi:
            continue
        r0 = int(rng.integers(row_lo, row_hi + 1))
        c0 = int(rng.integers(col_lo, col_hi + 1))
        if _can_place(blocked, r0, c0, h, w):
            return r0, c0
    return None


def _expected_output(
    inp: np.ndarray,
    bg: int,
    frame_records: list[dict],
    piece_records: dict[str, dict],
) -> np.ndarray:
    out = inp.copy()
    for frame in frame_records:
        direction = frame["direction"]
        piece = piece_records[direction]
        for r, c in piece["coords"]:
            out[int(r), int(c)] = bg
        dst_r0, dst_c0 = _frame_corner_origin(frame["bbox"], direction)
        for dr, dc in piece["rel_coords"]:
            out[int(dst_r0 + dr), int(dst_c0 + dc)] = int(frame["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(1, 5))
    used_directions = list(rng.choice(np.array(DIRECTIONS), size=n_frames, replace=False))
    remaining = [d for d in DIRECTIONS if d not in used_directions]
    extra_count = int(rng.integers(0, len(remaining) + 1))
    extra_directions = list(rng.choice(np.array(remaining), size=extra_count, replace=False))
    piece_directions = used_directions + extra_directions
    rng.shuffle(piece_directions)

    min_side = 13 + 3 * n_frames
    h_grid = int(rng.integers(min_side, 31))
    w_grid = int(rng.integers(min_side, 31))
    grid = np.zeros((h_grid, w_grid), dtype=int)

    n_colors = 1 + n_frames + len(piece_directions)
    palette = [int(x) for x in rng.choice(np.arange(10), size=n_colors, replace=False)]
    bg = palette[0]
    grid.fill(bg)
    frame_colors = palette[1 : 1 + n_frames]
    piece_colors = palette[1 + n_frames :]

    size_caps = {
        1: (9, 10),
        2: (8, 9),
        3: (7, 8),
        4: (6, 7),
    }
    max_h, max_w = size_caps[n_frames]
    max_h = min(max_h, h_grid - 2 * EDGE_BAND)
    max_w = min(max_w, w_grid - 2 * EDGE_BAND)
    if max_h < 4 or max_w < 4:
        return None

    blocked = np.zeros_like(grid, dtype=bool)
    frame_records: list[dict] = []
    for color, direction in zip(frame_colors, used_directions):
        fh = int(rng.integers(4, max_h + 1))
        fw = int(rng.integers(4, max_w + 1))
        patch = _frame_patch(int(color), bg, fh, fw, direction)
        pos = _random_frame_position(rng, blocked, fh, fw)
        if pos is None:
            return None
        r0, c0 = pos
        _paste_full(grid, patch, r0, c0)
        _reserve(blocked, r0, c0, fh, fw)
        frame_records.append(
            {
                "bbox": (r0, c0, r0 + fh - 1, c0 + fw - 1),
                "color": int(color),
                "direction": str(direction),
            }
        )

    piece_records: dict[str, dict] = {}
    for color, direction in zip(piece_colors, piece_directions):
        patch = _l_patch(int(color), bg, str(direction))
        pos = _random_edge_position(rng, blocked, 2, 2)
        if pos is None:
            return None
        r0, c0 = pos
        _paste_full(grid, patch, r0, c0)
        _reserve(blocked, r0, c0, 2, 2)
        rel_coords = [tuple(map(int, rc)) for rc in np.argwhere(L_MASKS[str(direction)])]
        piece_records[str(direction)] = {
            "coords": [(r0 + dr, c0 + dc) for dr, dc in rel_coords],
            "rel_coords": rel_coords,
        }

    expected = _expected_output(grid, bg, frame_records, piece_records)
    values, counts = np.unique(grid, return_counts=True)
    if int(values[int(np.argmax(counts))]) != bg:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
