from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SIDES = ("top", "right", "bottom", "left")
SIDE_TO_VEC = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
OPPOSITE = {
    "top": "bottom",
    "right": "left",
    "bottom": "top",
    "left": "right",
}
GEN_RETRIES = 400
PLACE_RETRIES = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not bool(mask[r0, c0]) or bool(seen[r0, c0]):
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                comp.append((int(r), int(c)))
                for dr, dc in SIDE_TO_VEC.values():
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not bool(seen[rr, cc]):
                        seen[rr, cc] = True
                        q.append((rr, cc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((int(r0), int(c)))
        cells.add((int(r1), int(c)))
    for r in range(r0, r1 + 1):
        cells.add((int(r), int(c0)))
        cells.add((int(r), int(c1)))
    return cells


def _gap_side(gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> str:
    gr, gc = gap
    if gr == r0:
        return "top"
    if gc == c1:
        return "right"
    if gr == r1:
        return "bottom"
    if gc == c0:
        return "left"
    raise ValueError("gap must lie on the border")


def _axis_distance(r: int, c: int, r0: int, c0: int, r1: int, c1: int, gap_side: str) -> int:
    if gap_side == "top":
        return int(r - r0)
    if gap_side == "bottom":
        return int(r1 - r)
    if gap_side == "left":
        return int(c - c0)
    return int(c1 - c)


def _seed_wall_side(grid: np.ndarray, r: int, c: int, fence_color: int, box: tuple[int, int, int, int]) -> str | None:
    r0, c0, r1, c1 = box
    sides: list[str] = []
    if r == r0 + 1 and int(grid[r - 1, c]) == int(fence_color):
        sides.append("top")
    if c == c1 - 1 and int(grid[r, c + 1]) == int(fence_color):
        sides.append("right")
    if r == r1 - 1 and int(grid[r + 1, c]) == int(fence_color):
        sides.append("bottom")
    if c == c0 + 1 and int(grid[r, c - 1]) == int(fence_color):
        sides.append("left")
    if len(sides) != 1:
        return None
    return sides[0]


def _detect_fences(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    fences: list[dict] = []
    colors = [int(v) for v in np.unique(g) if int(v) != BG]
    for color in colors:
        for comp in _components(g == int(color)):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            border = _border_cells(r0, c0, r1, c1)
            comp_set = set(comp)
            if len(comp_set) != len(border) - 1 or not comp_set.issubset(border):
                continue
            missing = list(border - comp_set)
            if len(missing) != 1:
                continue
            gap = missing[0]
            if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
                continue
            gap_side = _gap_side(gap, r0, c0, r1, c1)
            seeds: list[dict] = []
            used_walls: set[str] = set()
            valid = True
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    val = int(g[rr, cc])
                    if val == BG:
                        continue
                    wall = _seed_wall_side(g, rr, cc, color, (r0, c0, r1, c1))
                    if wall is None or wall in used_walls:
                        valid = False
                        break
                    used_walls.add(wall)
                    seeds.append(
                        {
                            "r": int(rr),
                            "c": int(cc),
                            "color": int(val),
                            "wall": wall,
                            "distance": _axis_distance(rr, cc, r0, c0, r1, c1, gap_side),
                        }
                    )
                if not valid:
                    break
            if not valid or not (1 <= len(seeds) <= 3):
                continue
            fences.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "gap": (int(gap[0]), int(gap[1])),
                    "gap_side": gap_side,
                    "seeds": seeds,
                }
            )
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a fence-colored trunk from each rectangular fence gap and attach seed-colored branches.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = out.shape

    for fence in _detect_fences(g):
        r0, c0, r1, c1 = fence["bbox"]
        gap_r, gap_c = fence["gap"]
        trunk_dr, trunk_dc = SIDE_TO_VEC[fence["gap_side"]]
        seeds = sorted(
            fence["seeds"],
            key=lambda s: (int(s["distance"]), SIDES.index(s["wall"]), int(s["r"]), int(s["c"])),
        )

        for step in range(2 * len(seeds)):
            rr = int(gap_r + step * trunk_dr)
            cc = int(gap_c + step * trunk_dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(fence["color"])

        for idx, seed in enumerate(seeds):
            node_r = int(gap_r + (2 * idx + 1) * trunk_dr)
            node_c = int(gap_c + (2 * idx + 1) * trunk_dc)
            br, bc = SIDE_TO_VEC[seed["wall"]]
            for step in range(1, int(seed["distance"]) + 1):
                rr = int(node_r + step * br)
                cc = int(node_c + step * bc)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = int(seed["color"])

    return out


def _allowed_seed_sides(gap_side: str) -> list[str]:
    return [side for side in SIDES if side != OPPOSITE[gap_side]]


def _sample_seed_layout(
    rng: random.Random, gap_side: str, seed_sides: list[str], ih: int, iw: int
) -> tuple[dict[str, tuple[int, int]], tuple[int, int]]:
    positions: dict[str, tuple[int, int]] = {}

    if gap_side == "top":
        if "top" in seed_sides:
            positions["top"] = (1, int(rng.choice(list(range(2, iw)))))
        free_rows = list(range(2, ih))
        rng.shuffle(free_rows)
        for side in ("left", "right"):
            if side in seed_sides:
                positions[side] = (int(free_rows.pop()), 1 if side == "left" else int(iw))
        blocked = {positions["top"][1]} if "top" in positions else set()
        gap = (0, int(rng.choice([c for c in range(1, iw + 1) if c not in blocked])))
    elif gap_side == "bottom":
        if "bottom" in seed_sides:
            positions["bottom"] = (int(ih), int(rng.choice(list(range(2, iw)))))
        free_rows = list(range(2, ih))
        rng.shuffle(free_rows)
        for side in ("left", "right"):
            if side in seed_sides:
                positions[side] = (int(free_rows.pop()), 1 if side == "left" else int(iw))
        blocked = {positions["bottom"][1]} if "bottom" in positions else set()
        gap = (int(ih + 1), int(rng.choice([c for c in range(1, iw + 1) if c not in blocked])))
    elif gap_side == "left":
        if "left" in seed_sides:
            positions["left"] = (int(rng.choice(list(range(2, ih)))), 1)
        free_cols = list(range(2, iw))
        rng.shuffle(free_cols)
        for side in ("top", "bottom"):
            if side in seed_sides:
                positions[side] = (1 if side == "top" else int(ih), int(free_cols.pop()))
        blocked = {positions["left"][0]} if "left" in positions else set()
        gap = (int(rng.choice([r for r in range(1, ih + 1) if r not in blocked])), 0)
    else:
        if "right" in seed_sides:
            positions["right"] = (int(rng.choice(list(range(2, ih)))), int(iw))
        free_cols = list(range(2, iw))
        rng.shuffle(free_cols)
        for side in ("top", "bottom"):
            if side in seed_sides:
                positions[side] = (1 if side == "top" else int(ih), int(free_cols.pop()))
        blocked = {positions["right"][0]} if "right" in positions else set()
        gap = (int(rng.choice([r for r in range(1, ih + 1) if r not in blocked])), int(iw + 1))

    return positions, gap


def _paint_object_maps(
    fence_color: int,
    gap_side: str,
    gap: tuple[int, int],
    seed_specs: list[dict],
    ih: int,
    iw: int,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    border_h = ih + 2
    border_w = iw + 2
    input_map: dict[tuple[int, int], int] = {}
    output_map: dict[tuple[int, int], int] = {}

    for r in range(border_h):
        for c in range(border_w):
            on_border = r in (0, border_h - 1) or c in (0, border_w - 1)
            if on_border and (r, c) != gap:
                input_map[(int(r), int(c))] = int(fence_color)
                output_map[(int(r), int(c))] = int(fence_color)

    for spec in seed_specs:
        coord = (int(spec["r"]), int(spec["c"]))
        input_map[coord] = int(spec["color"])
        output_map[coord] = int(spec["color"])

    gap_r, gap_c = gap
    trunk_dr, trunk_dc = SIDE_TO_VEC[gap_side]
    ordered = sorted(seed_specs, key=lambda s: (int(s["distance"]), SIDES.index(s["wall"])))

    for step in range(2 * len(ordered)):
        coord = (int(gap_r + step * trunk_dr), int(gap_c + step * trunk_dc))
        output_map[coord] = int(fence_color)

    for idx, spec in enumerate(ordered):
        node_r = int(gap_r + (2 * idx + 1) * trunk_dr)
        node_c = int(gap_c + (2 * idx + 1) * trunk_dc)
        br, bc = SIDE_TO_VEC[spec["wall"]]
        for step in range(1, int(spec["distance"]) + 1):
            coord = (int(node_r + step * br), int(node_c + step * bc))
            output_map[coord] = int(spec["color"])

    return input_map, output_map


def _build_object_spec(rng: random.Random) -> dict:
    gap_side = rng.choice(SIDES)
    seed_count = rng.randint(1, 3)
    seed_sides = rng.sample(_allowed_seed_sides(gap_side), seed_count)
    ih = rng.randint(4, 8)
    iw = rng.randint(4, 8)

    fence_color = int(rng.choice(list(range(1, 10))))
    seed_colors = rng.sample([c for c in range(1, 10) if c != fence_color], seed_count)
    positions, gap = _sample_seed_layout(rng, gap_side, seed_sides, ih, iw)

    seed_specs: list[dict] = []
    for color, side in zip(seed_colors, seed_sides, strict=False):
        r, c = positions[side]
        seed_specs.append(
            {
                "color": int(color),
                "wall": side,
                "r": int(r),
                "c": int(c),
                "distance": _axis_distance(int(r), int(c), 0, 0, ih + 1, iw + 1, gap_side),
            }
        )

    input_map, output_map = _paint_object_maps(fence_color, gap_side, gap, seed_specs, ih, iw)
    footprint = set(output_map)
    min_r = min(r for r, _ in footprint)
    max_r = max(r for r, _ in footprint)
    min_c = min(c for _, c in footprint)
    max_c = max(c for _, c in footprint)

    return {
        "input_map": input_map,
        "output_map": output_map,
        "footprint": footprint,
        "bbox": (int(min_r), int(min_c), int(max_r), int(max_c)),
    }


def _padded(coords: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in coords:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    padded.add((int(rr), int(cc)))
    return padded


def _place_objects(rng: random.Random, h: int, w: int, specs: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    reserved: set[tuple[int, int]] = set()

    ordered_specs = sorted(
        specs,
        key=lambda spec: (
            -len(spec["footprint"]),
            -(spec["bbox"][2] - spec["bbox"][0] + 1) * (spec["bbox"][3] - spec["bbox"][1] + 1),
        ),
    )

    for spec in ordered_specs:
        min_r, min_c, max_r, max_c = spec["bbox"]
        row_lo = -min_r
        row_hi = h - 1 - max_r
        col_lo = -min_c
        col_hi = w - 1 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            return None

        placed = False
        for _ in range(PLACE_RETRIES):
            shift_r = rng.randint(row_lo, row_hi)
            shift_c = rng.randint(col_lo, col_hi)
            translated = {(int(r + shift_r), int(c + shift_c)) for r, c in spec["footprint"]}
            if any(cell in reserved for cell in translated):
                continue

            for (r, c), color in spec["input_map"].items():
                grid[int(r + shift_r), int(c + shift_c)] = int(color)
            for (r, c), color in spec["output_map"].items():
                expected[int(r + shift_r), int(c + shift_c)] = int(color)
            reserved |= _padded(translated, h, w)
            placed = True
            break

        if not placed:
            return None

    return grid, expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    obj_count = rng.randint(1, 3)
    specs = [_build_object_spec(rng) for _ in range(obj_count)]
    placed = _place_objects(rng, h, w, specs)
    if placed is None:
        return None
    grid, expected = placed

    try:
        observed = main(grid)
    except Exception:
        return None
    if np.array_equal(observed, grid):
        return None
    if not np.array_equal(observed, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
