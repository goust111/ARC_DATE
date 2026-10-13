from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask


GREY = Color.GREY
BLACK = Color.BLACK
RANK_COLORS = [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE, Color.PINK]
SEED_COLOR_POOL = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

MIN_ROOM_H = 3
MIN_ROOM_W = 3
GEN_RETRIES = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _analyze_regions(grid: np.ndarray) -> list[dict]:
    regions = components_mask(grid != GREY, connectivity=4)
    out: list[dict] = []
    for coords in regions:
        vals = grid[coords[:, 0], coords[:, 1]]
        seed_idx = np.flatnonzero((vals != BLACK) & (vals != GREY))
        if seed_idx.size != 1:
            raise ValueError("every open region must contain exactly one seed")
        seed_pos = tuple(map(int, coords[int(seed_idx[0])]))
        seed_color = int(vals[int(seed_idx[0])])
        black_area = int(np.count_nonzero(vals == BLACK))
        out.append(
            {
                "coords": coords,
                "seed_pos": seed_pos,
                "seed_color": seed_color,
                "black_area": black_area,
            }
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each seed and its four wall-blocked rays by the size rank of the grey-walled region containing it.

    Input structure:
    - The grid contains black cells and a connected one-cell-wide grey wall network.
    - The grey walls partition the remaining cells into 4-connected regions.
    - Each region contains exactly one non-grey, non-black seed pixel of a distinct color.

    Rule:
    - Treat every non-grey cell as belonging to one open region, and count that region's black cells only.
    - Rank the regions by black area from smallest to largest.
    - Map ranks 1..6 to colors red, blue, green, yellow, orange, magenta.
    - Recolor the seed and extend that rank color from the seed in the four cardinal directions until a grey wall or the grid edge stops the ray.
    - Keep grey walls unchanged and leave every untouched black cell black.

    Key insight:
    - The seed's original color is irrelevant: the same grey wall geometry both defines the area ranking and blocks the rays, so each region's output color comes only from its size order.
    """
    g = np.asarray(grid, dtype=int)
    regions = _analyze_regions(g)
    if len(regions) > len(RANK_COLORS):
        raise ValueError("supports at most six regions")

    areas = [info["black_area"] for info in regions]
    if len(set(areas)) != len(areas):
        raise ValueError("region black areas must be unique")

    ordered = sorted(regions, key=lambda info: info["black_area"])
    color_by_seed: dict[tuple[int, int], int] = {}
    for rank, info in enumerate(ordered):
        color_by_seed[info["seed_pos"]] = int(RANK_COLORS[rank])

    out = g.copy()
    H, W = g.shape
    for info in regions:
        sr, sc = info["seed_pos"]
        paint = color_by_seed[(sr, sc)]
        out[sr, sc] = paint
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = sr + dr
            cc = sc + dc
            while 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != GREY:
                out[rr, cc] = paint
                rr += dr
                cc += dc
    return out


def _room_area(room: dict) -> int:
    return int((room["r1"] - room["r0"] + 1) * (room["c1"] - room["c0"] + 1))


def _room_h(room: dict) -> int:
    return int(room["r1"] - room["r0"] + 1)


def _room_w(room: dict) -> int:
    return int(room["c1"] - room["c0"] + 1)


def _can_split(room: dict, orientation: str, has_walls: bool) -> bool:
    if orientation == "V":
        if _room_w(room) < 2 * MIN_ROOM_W + 1:
            return False
        return (not has_walls) or bool(room["top_wall"] or room["bottom_wall"])
    if orientation == "H":
        if _room_h(room) < 2 * MIN_ROOM_H + 1:
            return False
        return (not has_walls) or bool(room["left_wall"] or room["right_wall"])
    raise ValueError("bad orientation")


def _split_room(room: dict, orientation: str, cut: int) -> tuple[list[tuple[int, int]], list[dict]]:
    r0, c0, r1, c1 = room["r0"], room["c0"], room["r1"], room["c1"]
    if orientation == "V":
        wall = [(r, cut) for r in range(r0, r1 + 1)]
        left = {
            "r0": r0,
            "c0": c0,
            "r1": r1,
            "c1": cut - 1,
            "top_wall": room["top_wall"],
            "bottom_wall": room["bottom_wall"],
            "left_wall": room["left_wall"],
            "right_wall": True,
        }
        right = {
            "r0": r0,
            "c0": cut + 1,
            "r1": r1,
            "c1": c1,
            "top_wall": room["top_wall"],
            "bottom_wall": room["bottom_wall"],
            "left_wall": True,
            "right_wall": room["right_wall"],
        }
        return wall, [left, right]

    wall = [(cut, c) for c in range(c0, c1 + 1)]
    top = {
        "r0": r0,
        "c0": c0,
        "r1": cut - 1,
        "c1": c1,
        "top_wall": room["top_wall"],
        "bottom_wall": True,
        "left_wall": room["left_wall"],
        "right_wall": room["right_wall"],
    }
    bottom = {
        "r0": cut + 1,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "top_wall": True,
        "bottom_wall": room["bottom_wall"],
        "left_wall": room["left_wall"],
        "right_wall": room["right_wall"],
    }
    return wall, [top, bottom]


def _pick_seed_colors(rng: np.random.Generator, region_count: int) -> list[int]:
    target_colors = RANK_COLORS[:region_count]
    for _ in range(100):
        colors = list(map(int, rng.choice(SEED_COLOR_POOL, size=region_count, replace=False)))
        rng.shuffle(colors)
        same = sum(colors[i] == target_colors[i] for i in range(region_count))
        if same <= 1 and any(color in (Color.TEAL, Color.MAROON) for color in colors):
            return colors
    raise RuntimeError("failed to sample seed colors")


def _place_seeds(grid: np.ndarray, rooms: list[dict], rng: np.random.Generator) -> None:
    ordered = sorted(rooms, key=_room_area)
    seed_colors = _pick_seed_colors(rng, len(rooms))
    for room, seed_color in zip(ordered, seed_colors):
        rows = np.arange(room["r0"] + 1, room["r1"])
        cols = np.arange(room["c0"] + 1, room["c1"])
        sr = int(rng.choice(rows))
        sc = int(rng.choice(cols))
        grid[sr, sc] = int(seed_color)


def _build_partition(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(12, 18))
    W = int(rng.integers(12, 18))
    target_regions = int(rng.integers(4, 7))

    rooms = [
        {
            "r0": 0,
            "c0": 0,
            "r1": H - 1,
            "c1": W - 1,
            "top_wall": False,
            "bottom_wall": False,
            "left_wall": False,
            "right_wall": False,
        }
    ]
    walls: set[tuple[int, int]] = set()
    used_orientations: list[str] = []

    while len(rooms) < target_regions:
        candidates: list[tuple[int, str]] = []
        has_walls = bool(walls)
        missing = {"H", "V"} - set(used_orientations)
        for idx, room in enumerate(rooms):
            for orientation in ("H", "V"):
                if missing and len(rooms) == target_regions - 1 and orientation not in missing:
                    continue
                if _can_split(room, orientation, has_walls):
                    candidates.append((idx, orientation))
        if not candidates:
            return None

        weights = np.array([_room_area(rooms[idx]) for idx, _orientation in candidates], dtype=float)
        weights /= weights.sum()
        pick = int(rng.choice(len(candidates), p=weights))
        room_idx, orientation = candidates[pick]
        room = rooms.pop(room_idx)

        if orientation == "V":
            cuts = list(range(room["c0"] + MIN_ROOM_W, room["c1"] - MIN_ROOM_W + 1))
        else:
            cuts = list(range(room["r0"] + MIN_ROOM_H, room["r1"] - MIN_ROOM_H + 1))
        if not cuts:
            return None
        cut = int(rng.choice(cuts))
        wall_cells, split_rooms = _split_room(room, orientation, cut)
        walls.update(wall_cells)
        rooms.extend(split_rooms)
        used_orientations.append(orientation)

    areas = [_room_area(room) - 1 for room in rooms]
    if len(set(areas)) != len(areas):
        return None
    if min(_room_h(room) for room in rooms) < MIN_ROOM_H or min(_room_w(room) for room in rooms) < MIN_ROOM_W:
        return None

    grid = np.full((H, W), BLACK, dtype=int)
    for r, c in walls:
        grid[int(r), int(c)] = GREY
    _place_seeds(grid, rooms, rng)
    return grid


def _validate_input(grid: np.ndarray) -> bool:
    H, W = grid.shape
    if H > 30 or W > 30:
        return False

    grey_components = components_mask(grid == GREY, connectivity=4)
    if len(grey_components) != 1:
        return False

    try:
        regions = _analyze_regions(grid)
    except ValueError:
        return False

    if not (4 <= len(regions) <= 6):
        return False

    areas = [info["black_area"] for info in regions]
    if len(set(areas)) != len(areas):
        return False

    seed_colors = [info["seed_color"] for info in regions]
    if len(set(seed_colors)) != len(seed_colors):
        return False

    for info in regions:
        sr, sc = info["seed_pos"]
        for nr, nc in _neighbors4(sr, sc):
            if not (0 <= nr < H and 0 <= nc < W):
                return False
            if int(grid[nr, nc]) != BLACK:
                return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    recolored = int(np.count_nonzero((grid == BLACK) & (out != BLACK)))
    if recolored < len(regions) * 4:
        return False
    if recolored > int(0.6 * np.count_nonzero(grid != GREY)):
        return False

    ordered = sorted(regions, key=lambda info: info["black_area"])
    expected = RANK_COLORS[: len(ordered)]
    if sum(int(info["seed_color"]) == int(color) for info, color in zip(ordered, expected)) > 1:
        return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _build_partition(rng)
        if grid is not None and _validate_input(grid):
            return grid
    raise RuntimeError("failed to generate valid wall-ranked ray puzzle")
