from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _hole_components(patch: np.ndarray) -> list[list[tuple[int, int]]]:
    holes: list[list[tuple[int, int]]] = []
    for comp in _components(patch == BLACK):
        if all(0 < r < patch.shape[0] - 1 and 0 < c < patch.shape[1] - 1 for r, c in comp):
            holes.append(comp)
    return holes


def _find_rooms(grid: np.ndarray) -> list[dict]:
    rooms: list[dict] = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 - r0 >= 2 and c1 - c0 >= 2

        gap = None
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if r in (r0, r1) or c in (c0, c1):
                    if int(grid[r, c]) == GREY:
                        continue
                    assert int(grid[r, c]) == BLACK
                    assert gap is None
                    gap = (r, c)
                else:
                    assert int(grid[r, c]) != GREY
        assert gap is not None

        if gap[0] == r0:
            side = "top"
        elif gap[0] == r1:
            side = "bottom"
        elif gap[1] == c0:
            side = "left"
        else:
            assert gap[1] == c1
            side = "right"

        dots = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(grid[r, c]) == MAROON
        ]
        assert 1 <= len(dots) <= 4
        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "side": side,
                "dot_coords": dots,
                "dot_count": len(dots),
            }
        )

    assert 3 <= len(rooms) <= 4
    return rooms


def _find_pieces(grid: np.ndarray) -> list[dict]:
    pieces: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {BLACK, GREY, MAROON}):
        comps = _components(grid == color)
        assert len(comps) == 1
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        hole_comps = _hole_components(patch)
        assert hole_comps
        rel_cells = [(r - r0, c - c0) for r, c in coords]
        pieces.append(
            {
                "color": color,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
                "rel_cells": rel_cells,
                "holes": hole_comps,
                "hole_count": len(hole_comps),
            }
        )
    return pieces


def _placement(room: dict, piece: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = room["bbox"]
    gap_r, gap_c = room["gap"]
    ph, pw = piece["h"], piece["w"]
    side = room["side"]

    if side in {"top", "bottom"}:
        assert pw % 2 == 1
        dest_c = gap_c - pw // 2
        assert c0 + 1 <= dest_c <= c1 - pw
        dest_r = r1 - ph if side == "top" else r0 + 1
    else:
        assert ph % 2 == 1
        dest_r = gap_r - ph // 2
        assert r0 + 1 <= dest_r <= r1 - ph
        dest_c = c1 - pw if side == "left" else c0 + 1

    return dest_r, dest_c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each holed block to the room with the same maroon-dot count, slide it through the gap,
    then refill its holes with maroon and erase the room's original dots.
    """
    g = np.asarray(grid, dtype=int)
    rooms = _find_rooms(g)
    pieces = _find_pieces(g)

    room_by_count: dict[int, dict] = {}
    for room in rooms:
        assert room["dot_count"] not in room_by_count
        room_by_count[room["dot_count"]] = room

    piece_by_count: dict[int, dict] = {}
    for piece in pieces:
        assert piece["hole_count"] not in piece_by_count
        piece_by_count[piece["hole_count"]] = piece

    assert room_by_count.keys() == piece_by_count.keys()

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[r, c] = BLACK
    for room in rooms:
        for r, c in room["dot_coords"]:
            out[r, c] = BLACK

    for count, room in room_by_count.items():
        piece = piece_by_count[count]
        dest_r, dest_c = _placement(room, piece)
        for rr, cc in piece["rel_cells"]:
            out[dest_r + rr, dest_c + cc] = piece["color"]
        for hole in piece["holes"]:
            for rr, cc in hole:
                out[dest_r + rr, dest_c + cc] = MAROON

    return out


def _pick_spaced_cells(
    height: int,
    width: int,
    count: int,
    rng: random.Random,
    border: int,
) -> list[tuple[int, int]] | None:
    cells = [
        (r, c)
        for r in range(border, height - border)
        for c in range(border, width - border)
    ]
    rng.shuffle(cells)
    picked: list[tuple[int, int]] = []
    for r, c in cells:
        if all(abs(r - pr) + abs(c - pc) > 1 for pr, pc in picked):
            picked.append((r, c))
            if len(picked) == count:
                return picked
    return None


def _make_piece_spec(count: int, side: str, rng: random.Random) -> dict:
    for _ in range(200):
        if side in {"top", "bottom"}:
            width = rng.choice((3, 5, 7))
            height = rng.randint(3, 6)
        else:
            height = rng.choice((3, 5, 7))
            width = rng.randint(3, 6)
        holes = _pick_spaced_cells(height, width, count, rng, border=1)
        if holes is None:
            continue
        return {"h": height, "w": width, "holes": holes}
    raise RuntimeError("failed to build piece spec")


def _make_room_spec(piece: dict, count: int, side: str, rng: random.Random) -> dict:
    for _ in range(300):
        inner_h = piece["h"] + rng.randint(0, 3)
        inner_w = piece["w"] + rng.randint(0, 3)
        dots = _pick_spaced_cells(inner_h, inner_w, count, rng, border=0)
        if dots is None:
            continue

        if side in {"top", "bottom"}:
            lo = 1 + piece["w"] // 2
            hi = inner_w - piece["w"] // 2
            if lo > hi:
                continue
            gap_axis = rng.randint(lo, hi)
            gap_local = (0, gap_axis) if side == "top" else (inner_h + 1, gap_axis)
        else:
            lo = 1 + piece["h"] // 2
            hi = inner_h - piece["h"] // 2
            if lo > hi:
                continue
            gap_axis = rng.randint(lo, hi)
            gap_local = (gap_axis, 0) if side == "left" else (gap_axis, inner_w + 1)

        return {
            "inner_h": inner_h,
            "inner_w": inner_w,
            "bbox_h": inner_h + 2,
            "bbox_w": inner_w + 2,
            "gap_local": gap_local,
            "dots": dots,
        }
    raise RuntimeError("failed to build room spec")


def _try_place(occupied: np.ndarray, obj_h: int, obj_w: int, rng: random.Random) -> tuple[int, int] | None:
    grid_h, grid_w = occupied.shape
    if obj_h + 2 > grid_h or obj_w + 2 > grid_w:
        return None
    for _ in range(500):
        r0 = rng.randint(1, grid_h - obj_h - 1)
        c0 = rng.randint(1, grid_w - obj_w - 1)
        rs = slice(max(0, r0 - 1), min(grid_h, r0 + obj_h + 1))
        cs = slice(max(0, c0 - 1), min(grid_w, c0 + obj_w + 1))
        if occupied[rs, cs].any():
            continue
        occupied[rs, cs] = True
        return r0, c0
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_count = rng.choice((3, 4))
    counts = rng.sample((1, 2, 3, 4), room_count)
    colors = rng.sample(PIECE_COLORS, room_count)
    room_specs = []
    piece_specs = []
    total_area = 0

    for count, color in zip(counts, colors):
        side = rng.choice(("top", "bottom", "left", "right"))
        piece = _make_piece_spec(count, side, rng)
        room = _make_room_spec(piece, count, side, rng)
        room_specs.append({"count": count, "color": color, "side": side, **room})
        piece_specs.append({"count": count, "color": color, "side": side, **piece})
        total_area += room["bbox_h"] * room["bbox_w"] + piece["h"] * piece["w"]

    base = max(22, int(np.ceil(np.sqrt(total_area * 1.8))))
    grid_h = rng.randint(base, max(base, base + 4))
    grid_w = rng.randint(base, max(base, base + 4))
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    room_specs.sort(key=lambda spec: spec["bbox_h"] * spec["bbox_w"], reverse=True)
    piece_specs.sort(key=lambda spec: spec["h"] * spec["w"], reverse=True)

    for spec in room_specs:
        placed = _try_place(occupied, spec["bbox_h"], spec["bbox_w"], rng)
        if placed is None:
            return None
        spec["top_left"] = placed

    for spec in piece_specs:
        placed = _try_place(occupied, spec["h"], spec["w"], rng)
        if placed is None:
            return None
        spec["top_left"] = placed

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    for spec in room_specs:
        r0, c0 = spec["top_left"]
        r1 = r0 + spec["bbox_h"] - 1
        c1 = c0 + spec["bbox_w"] - 1
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        gr, gc = spec["gap_local"]
        grid[r0 + gr, c0 + gc] = BLACK
        for dr, dc in spec["dots"]:
            grid[r0 + 1 + dr, c0 + 1 + dc] = MAROON

    for spec in piece_specs:
        r0, c0 = spec["top_left"]
        grid[r0 : r0 + spec["h"], c0 : c0 + spec["w"]] = spec["color"]
        for hr, hc in spec["holes"]:
            grid[r0 + hr, c0 + hc] = BLACK

    # Reject accidental extra components or malformed geometry.
    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
