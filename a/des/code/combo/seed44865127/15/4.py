import random

import numpy as np


BG = 0
GREY = 5
RED = 2
BLUE = 1
GREEN = 3
YELLOW = 4

FRAME_COLOR_POOL = (6, 7, 8, 9)
DIR_TO_OFFSET = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
POINTER_POS = {
    "up": (0, 1),
    "right": (1, 2),
    "down": (2, 1),
    "left": (1, 0),
}
ROTATE_CW = {
    "up": "right",
    "right": "down",
    "down": "left",
    "left": "up",
}
DIR_TO_COLOR = {
    "up": RED,
    "right": BLUE,
    "down": GREEN,
    "left": YELLOW,
}


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _fill_color_for_pointer(direction: str) -> int:
    return DIR_TO_COLOR[ROTATE_CW[direction]]


def _find_clocks(grid: np.ndarray, bg: int) -> list[dict]:
    g = grid
    h, w = g.shape
    clocks: list[dict] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            if int(sub[1, 1]) != GREY:
                continue

            corners = [int(sub[0, 0]), int(sub[0, 2]), int(sub[2, 0]), int(sub[2, 2])]
            frame_color = corners[0]
            if frame_color in {bg, GREY} or any(v != frame_color for v in corners[1:]):
                continue

            direction = None
            pointer_color = None
            ok = True
            for name, (rr, cc) in POINTER_POS.items():
                v = int(sub[rr, cc])
                if v == frame_color:
                    continue
                if v in {bg, GREY} or direction is not None:
                    ok = False
                    break
                direction = name
                pointer_color = v
            if not ok or direction is None or pointer_color is None:
                continue

            border_vals = [
                int(sub[rr, cc])
                for rr in range(3)
                for cc in range(3)
                if (rr, cc) != (1, 1)
            ]
            if border_vals.count(frame_color) != 7 or border_vals.count(pointer_color) != 1:
                continue

            clocks.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r0 + 2),
                    "c1": int(c0 + 2),
                    "center_r": int(r0 + 1),
                    "center_c": int(c0 + 1),
                    "frame_color": int(frame_color),
                    "direction": direction,
                }
            )
    return clocks


def _find_regions(grid: np.ndarray, bg: int) -> list[dict]:
    g = grid
    regions: list[dict] = []
    for color in sorted(set(map(int, np.unique(g))) - {bg, GREY}):
        for cells in _components(g == color):
            r0, c0, r1, c1 = _bbox(cells)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue

            expected = set()
            for c in range(c0, c1 + 1):
                expected.add((r0, c))
                expected.add((r1, c))
            for r in range(r0, r1 + 1):
                expected.add((r, c0))
                expected.add((r, c1))

            cell_set = set(cells)
            if cell_set != expected:
                continue

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0 or not np.all(interior == bg):
                continue

            regions.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                }
            )
    return regions


def _clock_region_distance(clock: dict, region: dict) -> int | None:
    if int(clock["frame_color"]) != int(region["color"]):
        return None

    center_r = int(clock["center_r"])
    center_c = int(clock["center_c"])
    r0, c0, r1, c1 = (int(region[k]) for k in ("r0", "c0", "r1", "c1"))

    if r0 <= center_r <= r1:
        if c1 < int(clock["c0"]):
            return int(clock["c0"] - c1)
        if c0 > int(clock["c1"]):
            return int(c0 - clock["c1"])

    if c0 <= center_c <= c1:
        if r1 < int(clock["r0"]):
            return int(clock["r0"] - r1)
        if r0 > int(clock["r1"]):
            return int(r0 - clock["r1"])

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    In grids containing 3x3 clock sprites, fill each aligned hollow rectangle with a color
    determined by the clock's pointer direction after one clockwise rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    clocks = _find_clocks(g, bg)
    regions = _find_regions(g, bg)
    out = g.copy()
    used_regions: set[int] = set()

    for clock in sorted(clocks, key=lambda item: (item["center_r"], item["center_c"])):
        candidates = []
        for idx, region in enumerate(regions):
            if idx in used_regions:
                continue
            dist = _clock_region_distance(clock, region)
            if dist is not None:
                candidates.append((int(dist), idx))
        if not candidates:
            continue

        _, idx = min(candidates)
        used_regions.add(idx)
        region = regions[idx]
        fill_color = _fill_color_for_pointer(clock["direction"])
        out[int(region["r0"]) + 1 : int(region["r1"]), int(region["c0"]) + 1 : int(region["c1"])] = fill_color

    return out


def _make_clock(frame_color: int, pointer_color: int, direction: str) -> np.ndarray:
    patch = np.full((3, 3), frame_color, dtype=int)
    patch[1, 1] = GREY
    pr, pc = POINTER_POS[direction]
    patch[pr, pc] = pointer_color
    return patch


def _make_region(frame_color: int, height: int, width: int) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    return patch


def _pair_region_origin(
    clock_r0: int,
    clock_c0: int,
    region_h: int,
    region_w: int,
    region_side: str,
    gap: int,
) -> tuple[int, int]:
    if region_side == "up":
        return clock_r0 - gap - region_h, clock_c0 + 1 - region_w // 2
    if region_side == "down":
        return clock_r0 + 3 + gap, clock_c0 + 1 - region_w // 2
    if region_side == "left":
        return clock_r0 + 1 - region_h // 2, clock_c0 - gap - region_w
    return clock_r0 + 1 - region_h // 2, clock_c0 + 3 + gap


def _stamp(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]


def _place_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    spec: dict,
    rng: random.Random,
    max_tries: int = 250,
) -> bool:
    h, w = grid.shape
    region_h = int(spec["region_h"])
    region_w = int(spec["region_w"])
    region_side = str(spec["region_side"])
    gap = int(spec["gap"])

    for _ in range(max_tries):
        clock_r0 = rng.randint(0, h - 3)
        clock_c0 = rng.randint(0, w - 3)
        region_r0, region_c0 = _pair_region_origin(clock_r0, clock_c0, region_h, region_w, region_side, gap)
        region_r1 = region_r0 + region_h - 1
        region_c1 = region_c0 + region_w - 1
        if not (0 <= region_r0 and 0 <= region_c0 and region_r1 < h and region_c1 < w):
            continue

        union_r0 = min(clock_r0, region_r0)
        union_c0 = min(clock_c0, region_c0)
        union_r1 = max(clock_r0 + 2, region_r1)
        union_c1 = max(clock_c0 + 2, region_c1)
        if union_r0 < 1 or union_c0 < 1 or union_r1 > h - 2 or union_c1 > w - 2:
            continue
        if np.any(blocked[union_r0 - 1 : union_r1 + 2, union_c0 - 1 : union_c1 + 2]):
            continue

        clock_patch = _make_clock(int(spec["frame_color"]), int(spec["pointer_color"]), str(spec["pointer_dir"]))
        region_patch = _make_region(int(spec["frame_color"]), region_h, region_w)
        _stamp(grid, clock_patch, clock_r0, clock_c0)
        _stamp(grid, region_patch, region_r0, region_c0)
        blocked[union_r0 - 1 : union_r1 + 2, union_c0 - 1 : union_c1 + 2] = True
        spec["clock_r0"] = int(clock_r0)
        spec["clock_c0"] = int(clock_c0)
        spec["region_r0"] = int(region_r0)
        spec["region_c0"] = int(region_c0)
        return True

    return False


def _sample_specs(rng: random.Random, num_pairs: int) -> list[dict]:
    frame_colors = list(rng.sample(FRAME_COLOR_POOL, num_pairs))
    rng.shuffle(frame_colors)
    size_choices = (5, 7, 9) if num_pairs <= 2 else (5, 7)
    gap_max = 4 if num_pairs <= 2 else 3

    specs: list[dict] = []
    for frame_color in frame_colors:
        pointer_choices = [c for c in FRAME_COLOR_POOL if c != frame_color]
        specs.append(
            {
                "frame_color": int(frame_color),
                "pointer_color": int(rng.choice(pointer_choices)),
                "pointer_dir": rng.choice(tuple(POINTER_POS)),
                "region_side": rng.choice(tuple(POINTER_POS)),
                "region_h": rng.choice(size_choices),
                "region_w": rng.choice(size_choices),
                "gap": rng.randint(2, gap_max),
            }
        )
    specs.sort(key=lambda item: int(item["region_h"]) * int(item["region_w"]), reverse=True)
    return specs


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        fill_color = _fill_color_for_pointer(str(spec["pointer_dir"]))
        r0 = int(spec["region_r0"])
        c0 = int(spec["region_c0"])
        h = int(spec["region_h"])
        w = int(spec["region_w"])
        out[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = fill_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_pairs = rng.randint(1, 4)
    min_size = 18 + max(0, num_pairs - 1) * 4
    max_size = 26 + max(0, num_pairs - 1) * 3
    height = rng.randint(min_size, max_size)
    width = rng.randint(min_size, max_size)
    inp = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    specs = _sample_specs(rng, num_pairs)

    for spec in specs:
        if not _place_pair(inp, blocked, spec, rng):
            return None

    expected = _expected_output(inp, specs)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid clock-fill puzzle")
