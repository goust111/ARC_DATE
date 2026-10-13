import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
FRAME_COLOR = 5
BG_CHOICES = [0, 1, 2, 3, 4, 6, 7, 8, 9]
OUTER_H_RANGE = (11, 17)
OUTER_W_RANGE = (11, 17)
MARGIN_RANGE = (3, 7)
PIECE_COUNT_RANGE = (4, 7)
MAX_PIECE_CELLS = 6
MAX_GENERATION_ATTEMPTS = 400


def _mode(values: np.ndarray) -> int:
    vals = np.asarray(values, dtype=int).ravel()
    counts = np.bincount(vals, minlength=10)
    return int(np.argmax(counts))


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
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    best = None
    best_score = None
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if len(coords) == 0:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        border_mask = (
            (coords[:, 0] == r0)
            | (coords[:, 0] == r1)
            | (coords[:, 1] == c0)
            | (coords[:, 1] == c1)
        )
        if not np.all(border_mask):
            continue
        if grid[r0 + 1 : r1, c0 + 1 : c1].size == 0:
            continue
        score = (h * w, len(coords))
        if best is None or score > best_score:
            best = (color, (r0, c0, r1, c1))
            best_score = score
    assert best is not None, "failed to identify the rectangular frame"
    return best


def _marker_side(r: int, c: int, frame_bbox: tuple[int, int, int, int]) -> str:
    fr0, fc0, fr1, fc1 = frame_bbox
    sides = []
    if r == fr0 and fc0 < c < fc1:
        sides.append("top")
    if r == fr1 and fc0 < c < fc1:
        sides.append("bottom")
    if c == fc0 and fr0 < r < fr1:
        sides.append("left")
    if c == fc1 and fr0 < r < fr1:
        sides.append("right")
    assert len(sides) == 1, "marker must lie on exactly one frame side"
    return sides[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each exterior colored fragment into the frame, matched by marker color.

    The marker's side chooses the insertion side. The fragment is aligned to the
    corresponding inner frame edge and centered on the marker row or column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _mode(g)
    frame_color, frame_bbox = _find_frame(g, bg)
    fr0, fc0, fr1, fc1 = frame_bbox

    jobs = []
    for color0 in sorted(np.unique(g)):
        color = int(color0)
        if color in (bg, frame_color):
            continue
        comps = _components(g == color)
        assert len(comps) == 2, "each color must form one marker and one fragment"

        marker = None
        fragment = None
        for comp in comps:
            if len(comp) == 1:
                r, c = map(int, comp[0])
                if (
                    (r == fr0 or r == fr1 or c == fc0 or c == fc1)
                    and fr0 <= r <= fr1
                    and fc0 <= c <= fc1
                ):
                    marker = (r, c)
                    continue
            inside_box = np.any(
                (fr0 <= comp[:, 0])
                & (comp[:, 0] <= fr1)
                & (fc0 <= comp[:, 1])
                & (comp[:, 1] <= fc1)
            )
            assert not inside_box, "fragments must be outside the frame"
            fragment = comp

        assert marker is not None and fragment is not None
        side = _marker_side(marker[0], marker[1], frame_bbox)

        sr0, sc0, sr1, sc1 = _bbox(fragment)
        height = sr1 - sr0 + 1
        width = sc1 - sc0 + 1
        rel_cells = [(int(r - sr0), int(c - sc0)) for r, c in fragment]

        if side == "top":
            dr0 = fr0 + 1
            dc0 = marker[1] - width // 2
        elif side == "bottom":
            dr0 = fr1 - height
            dc0 = marker[1] - width // 2
        elif side == "left":
            dr0 = marker[0] - height // 2
            dc0 = fc0 + 1
        else:
            dr0 = marker[0] - height // 2
            dc0 = fc1 - width

        assert fr0 + 1 <= dr0 <= fr1 - height + 1
        assert fc0 + 1 <= dc0 <= fc1 - width + 1
        jobs.append(
            {
                "color": color,
                "marker": marker,
                "fragment": fragment,
                "dest0": (dr0, dc0),
                "rel_cells": rel_cells,
            }
        )

    out = g.copy()
    for job in jobs:
        out[job["fragment"][:, 0], job["fragment"][:, 1]] = bg
        out[job["marker"]] = frame_color

    claimed: dict[tuple[int, int], int] = {}
    for job in jobs:
        dr0, dc0 = job["dest0"]
        for rr, cc in job["rel_cells"]:
            r = dr0 + rr
            c = dc0 + cc
            assert fr0 < r < fr1 and fc0 < c < fc1
            assert (r, c) not in claimed, "moved fragments must not overlap"
            claimed[(r, c)] = job["color"]

    for (r, c), color in claimed.items():
        out[r, c] = color
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    normalized = sorted((r - min_r, c - min_c) for r, c in cells)
    return np.array(normalized, dtype=int)


def _random_piece(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h <= 0 or max_w <= 0:
        return None
    max_cells = min(MAX_PIECE_CELLS, max_h * max_w)
    for _ in range(80):
        target = random.randint(1, max_cells)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))
        coords = _normalize_cells(cells)
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        if h <= max_h and w <= max_w:
            return coords
    return None


def _mark_halo(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    mask[nr, nc] = True


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _build_case() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    piece_colors = [c for c in range(10) if c not in (bg, FRAME_COLOR)]

    outer_h = random.randint(*OUTER_H_RANGE)
    outer_w = random.randint(*OUTER_W_RANGE)
    mt = random.randint(*MARGIN_RANGE)
    mb = random.randint(*MARGIN_RANGE)
    ml = random.randint(*MARGIN_RANGE)
    mr = random.randint(*MARGIN_RANGE)

    height = outer_h + mt + mb
    width = outer_w + ml + mr
    fr0, fc0 = mt, ml
    fr1, fc1 = fr0 + outer_h - 1, fc0 + outer_w - 1
    inner_h, inner_w = outer_h - 2, outer_w - 2

    piece_count = random.randint(*PIECE_COUNT_RANGE)
    if piece_count > len(piece_colors):
        return None

    sides = ["top", "bottom", "left", "right"]
    while len(sides) < piece_count:
        sides.append(random.choice(("top", "bottom", "left", "right")))
    random.shuffle(sides)
    colors = random.sample(piece_colors, piece_count)

    grid = np.full((height, width), bg, dtype=int)
    grid[fr0, fc0 : fc1 + 1] = FRAME_COLOR
    grid[fr1, fc0 : fc1 + 1] = FRAME_COLOR
    grid[fr0 : fr1 + 1, fc0] = FRAME_COLOR
    grid[fr0 : fr1 + 1, fc1] = FRAME_COLOR

    blocked_inner = np.zeros((inner_h, inner_w), dtype=bool)
    marker_cells: set[tuple[int, int]] = set()
    pieces = []

    for side, color in zip(sides, colors):
        placed = False
        for _ in range(200):
            if side in ("top", "bottom"):
                piece = _random_piece(min(4, inner_h), min(6, inner_w))
            else:
                piece = _random_piece(min(6, inner_h), min(4, inner_w))
            if piece is None:
                continue
            h = int(piece[:, 0].max()) + 1
            w = int(piece[:, 1].max()) + 1

            if side == "top":
                r0_in = 0
                c0_in = random.randint(0, inner_w - w)
                marker = (fr0, fc0 + 1 + c0_in + w // 2)
            elif side == "bottom":
                r0_in = inner_h - h
                c0_in = random.randint(0, inner_w - w)
                marker = (fr1, fc0 + 1 + c0_in + w // 2)
            elif side == "left":
                r0_in = random.randint(0, inner_h - h)
                c0_in = 0
                marker = (fr0 + 1 + r0_in + h // 2, fc0)
            else:
                r0_in = random.randint(0, inner_h - h)
                c0_in = inner_w - w
                marker = (fr0 + 1 + r0_in + h // 2, fc1)

            dest_cells = [(r0_in + int(r), c0_in + int(c)) for r, c in piece]
            if any(blocked_inner[r, c] for r, c in dest_cells):
                continue
            if marker in marker_cells:
                continue

            _mark_halo(blocked_inner, dest_cells)
            marker_cells.add(marker)
            pieces.append(
                {
                    "color": color,
                    "side": side,
                    "piece": piece,
                    "dest_in": (r0_in, c0_in),
                    "marker": marker,
                }
            )
            placed = True
            break
        if not placed:
            return None

    blocked_full = np.zeros((height, width), dtype=bool)
    frame_cells = (
        [(fr0, c) for c in range(fc0, fc1 + 1)]
        + [(fr1, c) for c in range(fc0, fc1 + 1)]
        + [(r, fc0) for r in range(fr0 + 1, fr1)]
        + [(r, fc1) for r in range(fr0 + 1, fr1)]
    )
    _mark_halo(blocked_full, frame_cells)

    for piece_info in pieces:
        piece = piece_info["piece"]
        h = int(piece[:, 0].max()) + 1
        w = int(piece[:, 1].max()) + 1
        placed = False
        for _ in range(300):
            sr0 = random.randint(0, height - h)
            sc0 = random.randint(0, width - w)
            sr1 = sr0 + h - 1
            sc1 = sc0 + w - 1
            separated = sr1 < fr0 or sr0 > fr1 or sc1 < fc0 or sc0 > fc1
            if not separated:
                continue
            source_cells = [(sr0 + int(r), sc0 + int(c)) for r, c in piece]
            if any(blocked_full[r, c] for r, c in source_cells):
                continue
            _mark_halo(blocked_full, source_cells)
            piece_info["source_cells"] = source_cells
            placed = True
            break
        if not placed:
            return None

    for piece_info in pieces:
        _paint_cells(grid, piece_info["source_cells"], piece_info["color"])
        grid[piece_info["marker"]] = piece_info["color"]

    expected = grid.copy()
    for piece_info in pieces:
        _paint_cells(expected, piece_info["source_cells"], bg)
        expected[piece_info["marker"]] = FRAME_COLOR
        r0_in, c0_in = piece_info["dest_in"]
        dest_cells = [
            (fr0 + 1 + r0_in + int(r), fc0 + 1 + c0_in + int(c))
            for r, c in piece_info["piece"]
        ]
        _paint_cells(expected, dest_cells, piece_info["color"])

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 4")
