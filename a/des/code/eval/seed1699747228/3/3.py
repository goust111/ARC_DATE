import random
from collections import Counter

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SEED_DIRECTIONS = ["up", "down", "left", "right"]
BG = 0
GREY = 5
GEN_RETRIES = 400


PATTERNS = [
    {
        "mask": np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            dtype=bool,
        ),
        "remove": {
            "left": [(1, 0)],
            "right": [(1, 2)],
            "up": [(0, 1)],
            "down": [(2, 1)],
        },
    },
    {
        "mask": np.array(
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ],
            dtype=bool,
        ),
        "remove": {
            "left": [(1, 0)],
            "right": [(1, 2)],
            "up": [(0, 1)],
            "down": [(2, 1)],
        },
    },
    {
        "mask": np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        ),
        "remove": {
            "left": [(1, 0), (2, 0)],
            "right": [(1, 3), (2, 3)],
            "up": [(0, 1), (0, 2)],
            "down": [(3, 1), (3, 2)],
        },
    },
    {
        "mask": np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        ),
        "remove": {
            "left": [(1, 0), (2, 0)],
            "right": [(1, 3), (2, 3)],
            "up": [(0, 1), (0, 2)],
            "down": [(3, 1), (3, 2)],
        },
    },
]


def _get_bg(grid: np.ndarray) -> int:
    values = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(values))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _all_components(grid: np.ndarray, bg: int) -> list[dict]:
    components = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        for coords in _components_mask(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                    "size": len(coords),
                }
            )
    return components


def _mirror_missing(mask: np.ndarray, direction: str) -> list[tuple[int, int]] | None:
    h, w = mask.shape
    missing: list[tuple[int, int]] = []

    if direction in ("left", "right"):
        for c in range(w // 2):
            mc = w - 1 - c
            left = mask[:, c]
            right = mask[:, mc]
            if direction == "left":
                if np.any(left & ~right):
                    return None
                rows = np.where(~left & right)[0]
                missing.extend((int(r), int(c)) for r in rows)
            else:
                if np.any(right & ~left):
                    return None
                rows = np.where(left & ~right)[0]
                missing.extend((int(r), int(mc)) for r in rows)
        if not missing:
            return None
        repaired = mask.copy()
        for r, c in missing:
            repaired[r, c] = True
        if not np.array_equal(repaired, np.fliplr(repaired)):
            return None
        return missing

    for r in range(h // 2):
        mr = h - 1 - r
        top = mask[r, :]
        bottom = mask[mr, :]
        if direction == "up":
            if np.any(top & ~bottom):
                return None
            cols = np.where(~top & bottom)[0]
            missing.extend((int(r), int(c)) for c in cols)
        else:
            if np.any(bottom & ~top):
                return None
            cols = np.where(top & ~bottom)[0]
            missing.extend((int(mr), int(c)) for c in cols)
    if not missing:
        return None
    repaired = mask.copy()
    for r, c in missing:
        repaired[r, c] = True
    if not np.array_equal(repaired, np.flipud(repaired)):
        return None
    return missing


def _analyze_seed(comp: dict) -> dict | None:
    if comp["h"] != comp["w"] or comp["h"] not in (3, 4):
        return None
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((comp["h"], comp["w"]), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True

    candidates = []
    for direction in SEED_DIRECTIONS:
        missing = _mirror_missing(mask, direction)
        if missing:
            candidates.append((direction, missing))
    if len(candidates) != 1:
        return None

    direction, missing_local = candidates[0]
    missing_global = [(r0 + r, c0 + c) for r, c in missing_local]
    return {
        "color": comp["color"],
        "bbox": comp["bbox"],
        "direction": direction,
        "missing": missing_global,
        "size": comp["h"],
    }


def _is_dye(comp: dict, grid: np.ndarray) -> bool:
    if comp["h"] != 2 or comp["w"] != 2 or comp["size"] != 4:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == comp["color"]))


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _line_clear(grey_mask: np.ndarray, seed: dict, dye: dict, direction: str) -> bool:
    sr0, sc0, sr1, sc1 = seed["bbox"]
    dr0, dc0, dr1, dc1 = dye["bbox"]

    if direction in ("left", "right"):
        lo = max(sr0, dr0)
        hi = min(sr1, dr1)
        if lo > hi:
            return False
        c_start = dc1 + 1 if direction == "left" else sc1 + 1
        c_end = sc0 - 1 if direction == "left" else dc0 - 1
        for r in range(lo, hi + 1):
            if c_start > c_end or not grey_mask[r, c_start : c_end + 1].any():
                return True
        return False

    lo = max(sc0, dc0)
    hi = min(sc1, dc1)
    if lo > hi:
        return False
    r_start = dr1 + 1 if direction == "up" else sr1 + 1
    r_end = sr0 - 1 if direction == "up" else dr0 - 1
    for c in range(lo, hi + 1):
        if r_start > r_end or not grey_mask[r_start : r_end + 1, c].any():
            return True
    return False


def _find_dye(seed: dict, dyes: list[dict], grey_mask: np.ndarray) -> dict:
    sr0, sc0, sr1, sc1 = seed["bbox"]
    direction = seed["direction"]
    candidates = []
    for dye in dyes:
        if dye["color"] == seed["color"]:
            continue
        dr0, dc0, dr1, dc1 = dye["bbox"]
        if direction == "left":
            if dc1 >= sc0 or not _interval_overlap(sr0, sr1, dr0, dr1):
                continue
            dist = sc0 - dc1 - 1
        elif direction == "right":
            if dc0 <= sc1 or not _interval_overlap(sr0, sr1, dr0, dr1):
                continue
            dist = dc0 - sc1 - 1
        elif direction == "up":
            if dr1 >= sr0 or not _interval_overlap(sc0, sc1, dc0, dc1):
                continue
            dist = sr0 - dr1 - 1
        else:
            if dr0 <= sr1 or not _interval_overlap(sc0, sc1, dc0, dc1):
                continue
            dist = dr0 - sr1 - 1
        if dist < 0 or not _line_clear(grey_mask, seed, dye, direction):
            continue
        candidates.append((dist, dye["bbox"], dye))

    assert candidates, "expected a dye block on the missing side"
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _paint_ray(out: np.ndarray, grey_mask: np.ndarray, seed: dict, dye_color: int, bg: int) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = seed["bbox"]
    direction = seed["direction"]

    if direction == "left":
        for c in range(c0 - 1, -1, -1):
            if grey_mask[r0 : r1 + 1, c].any():
                break
            sl = out[r0 : r1 + 1, c]
            sl[sl == bg] = dye_color
    elif direction == "right":
        for c in range(c1 + 1, w):
            if grey_mask[r0 : r1 + 1, c].any():
                break
            sl = out[r0 : r1 + 1, c]
            sl[sl == bg] = dye_color
    elif direction == "up":
        for r in range(r0 - 1, -1, -1):
            if grey_mask[r, c0 : c1 + 1].any():
                break
            sl = out[r, c0 : c1 + 1]
            sl[sl == bg] = dye_color
    else:
        for r in range(r1 + 1, h):
            if grey_mask[r, c0 : c1 + 1].any():
                break
            sl = out[r, c0 : c1 + 1]
            sl[sl == bg] = dye_color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    components = _all_components(g, bg)
    dyes = [comp for comp in components if _is_dye(comp, g)]

    seeds = []
    seed_cells = set()
    dye_cells = set()
    for dye in dyes:
        dye_cells.update(dye["coords"])

    for comp in components:
        if comp in dyes:
            continue
        info = _analyze_seed(comp)
        if info is None:
            continue
        seeds.append(info)
        seed_cells.update(comp["coords"])

    grey_mask = np.zeros_like(g, dtype=bool)
    for r, c in np.argwhere(g != bg):
        pt = (int(r), int(c))
        if pt not in seed_cells and pt not in dye_cells:
            grey_mask[r, c] = True

    for seed in seeds:
        for r, c in seed["missing"]:
            out[r, c] = seed["color"]

    for seed in seeds:
        dye = _find_dye(seed, dyes, grey_mask)
        _paint_ray(out, grey_mask, seed, dye["color"], bg)

    return out


def _build_regions(h: int, w: int, rng: random.Random) -> tuple[list[dict], np.ndarray]:
    layout = rng.choice(["v", "h", "vh"])
    grid = np.full((h, w), BG, dtype=int)

    split_rows: list[int] = []
    split_cols: list[int] = []
    if layout in ("h", "vh"):
        split_rows = [rng.randint(8, h - 9)]
    if layout in ("v", "vh"):
        split_cols = [rng.randint(8, w - 9)]

    for r in split_rows:
        grid[r, :] = GREY
    for c in split_cols:
        grid[:, c] = GREY

    row_parts = [-1] + split_rows + [h]
    col_parts = [-1] + split_cols + [w]
    regions = []
    for i in range(len(row_parts) - 1):
        r0 = row_parts[i] + 1
        r1 = row_parts[i + 1] - 1
        for j in range(len(col_parts) - 1):
            c0 = col_parts[j] + 1
            c1 = col_parts[j + 1] - 1
            if r1 - r0 + 1 >= 8 and c1 - c0 + 1 >= 8:
                regions.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1})
    return regions, grid


def _draw_seed(arr: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    n = mask.shape[0]
    for r in range(n):
        for c in range(n):
            if mask[r, c]:
                arr[top + r, left + c] = color


def _place_object(
    inp: np.ndarray,
    out: np.ndarray,
    region: dict,
    seed_color: int,
    dye_color: int,
    rng: random.Random,
) -> bool:
    for _ in range(80):
        spec = rng.choice(PATTERNS)
        full_mask = spec["mask"]
        n = int(full_mask.shape[0])
        direction = rng.choice(SEED_DIRECTIONS)

        r0, r1, c0, c1 = region["r0"], region["r1"], region["c0"], region["c1"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1

        if direction in ("left", "right"):
            if width < n + 4 or height < n:
                continue
            row_lo = r0
            row_hi = r1 - n + 1
            if direction == "right":
                col_lo = c0
                col_hi = c1 - n - 3
            else:
                col_lo = c0 + 3
                col_hi = c1 - n + 1
            if row_lo > row_hi or col_lo > col_hi:
                continue
            top = rng.randint(row_lo, row_hi)
            left = rng.randint(col_lo, col_hi)
            available = c1 - (left + n - 1) if direction == "right" else left - c0
            if available < 4:
                continue
            gap = rng.randint(1, available - 3)
            dye_top = rng.randint(top, top + n - 2)
            if direction == "right":
                dye_left = left + n + gap
                ray = (top, top + n - 1, left + n, c1)
            else:
                dye_left = left - gap - 2
                ray = (top, top + n - 1, c0, left - 1)
        else:
            if height < n + 4 or width < n:
                continue
            col_lo = c0
            col_hi = c1 - n + 1
            if direction == "down":
                row_lo = r0
                row_hi = r1 - n - 3
            else:
                row_lo = r0 + 3
                row_hi = r1 - n + 1
            if row_lo > row_hi or col_lo > col_hi:
                continue
            top = rng.randint(row_lo, row_hi)
            left = rng.randint(col_lo, col_hi)
            available = r1 - (top + n - 1) if direction == "down" else top - r0
            if available < 4:
                continue
            gap = rng.randint(1, available - 3)
            dye_left = rng.randint(left, left + n - 2)
            if direction == "down":
                dye_top = top + n + gap
                ray = (top + n, r1, left, left + n - 1)
            else:
                dye_top = top - gap - 2
                ray = (r0, top - 1, left, left + n - 1)

        partial_mask = full_mask.copy()
        for rr, cc in spec["remove"][direction]:
            partial_mask[rr, cc] = False

        seed_box = inp[top : top + n, left : left + n]
        dye_box = inp[dye_top : dye_top + 2, dye_left : dye_left + 2]
        if np.any(seed_box != BG) or np.any(dye_box != BG):
            continue

        _draw_seed(inp, top, left, partial_mask, seed_color)
        _draw_seed(out, top, left, full_mask, seed_color)
        inp[dye_top : dye_top + 2, dye_left : dye_left + 2] = dye_color
        out[dye_top : dye_top + 2, dye_left : dye_left + 2] = dye_color

        rr0, rr1, cc0, cc1 = ray
        if rr0 <= rr1 and cc0 <= cc1:
            patch = out[rr0 : rr1 + 1, cc0 : cc1 + 1]
            patch[patch == BG] = dye_color
        return True

    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 28)
    w = rng.randint(18, 28)
    regions, base = _build_regions(h, w, rng)
    if len(regions) < 2:
        return None

    inp = base.copy()
    out = base.copy()

    n_objects = rng.randint(2, min(4, len(regions)))
    chosen_regions = rng.sample(regions, n_objects)
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    rng.shuffle(palette)

    for idx, region in enumerate(chosen_regions):
        seed_color = palette[2 * idx]
        dye_color = palette[2 * idx + 1]
        if not _place_object(inp, out, region, seed_color, dye_color, rng):
            return None

    if Counter(map(int, inp.ravel()))[BG] <= Counter(map(int, inp.ravel())).most_common(1)[0][1] - 1:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
