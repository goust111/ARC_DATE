import random
from collections import defaultdict, deque

import numpy as np


BG = 0

# Generator configuration.
GRID_SIDE_MIN = 14
GRID_SIDE_MAX = 24
MIN_OBJECT_SIDE = 3
MAX_OBJECT_SIDE = 7
PAIR_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
PLACEMENT_PAD = 1
PLACEMENT_TRIES = 250
GEN_ATTEMPTS = 400
SIZE_BUDGET_RATIO = 0.55


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]

            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch,
                    "size": patch.shape,
                }
            )

    return objects


def _classify_object(obj: dict, bg: int = BG) -> dict:
    color = int(obj["color"])
    patch = obj["patch"]

    if np.all(patch == color):
        result = dict(obj)
        result["kind"] = "cast"
        result["hole_mask"] = np.zeros_like(patch, dtype=bool)
        return result

    border_is_solid = (
        np.all(patch[0, :] == color)
        and np.all(patch[-1, :] == color)
        and np.all(patch[:, 0] == color)
        and np.all(patch[:, -1] == color)
    )
    only_color_and_bg = np.all((patch == color) | (patch == bg))
    has_hole = np.any(patch == bg)

    assert border_is_solid and only_color_and_bg and has_hole, "unexpected foreground object"

    result = dict(obj)
    result["kind"] = "mold"
    result["hole_mask"] = patch == bg
    return result


def _pair_objects(objects: list[dict]) -> list[tuple[dict, dict]]:
    molds: dict[tuple[int, int], list[dict]] = defaultdict(list)
    casts: dict[tuple[int, int], list[dict]] = defaultdict(list)

    for obj in objects:
        info = _classify_object(obj)
        if info["kind"] == "mold":
            molds[info["size"]].append(info)
        else:
            casts[info["size"]].append(info)

    assert molds, "expected at least one mold"
    assert set(molds) == set(casts), "molds and casts must share the same size set"

    pairs: list[tuple[dict, dict]] = []
    for size in sorted(molds):
        assert len(molds[size]) == 1, "each mold size must be unique"
        assert len(casts[size]) == 1, "each cast size must be unique"
        pairs.append((molds[size][0], casts[size][0]))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each perforated mold with the solid cast of the same bbox size, then
    stamp the mold's hole pattern into that cast using the mold color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pairs = _pair_objects(_find_components(g, bg=BG))
    out = g.copy()

    for mold, cast in pairs:
        r0, c0, r1, c1 = cast["bbox"]
        cast_patch = out[r0 : r1 + 1, c0 : c1 + 1]
        cast_patch[mold["hole_mask"]] = int(mold["color"])

    return out


def _single_rect_holes(h: int, w: int) -> np.ndarray:
    holes = np.zeros((h, w), dtype=bool)
    ih, iw = h - 2, w - 2
    hole_h = random.randint(1, ih)
    hole_w = random.randint(1, iw)
    r0 = random.randint(1, h - hole_h - 1)
    c0 = random.randint(1, w - hole_w - 1)
    holes[r0 : r0 + hole_h, c0 : c0 + hole_w] = True
    return holes


def _double_rect_holes(h: int, w: int) -> np.ndarray | None:
    if (h - 2) * (w - 2) < 4:
        return None

    holes = np.zeros((h, w), dtype=bool)
    for _ in range(30):
        temp = np.zeros((h, w), dtype=bool)
        for _ in range(2):
            hole_h = random.randint(1, max(1, (h - 2 + 1) // 2))
            hole_w = random.randint(1, max(1, (w - 2 + 1) // 2))
            r0 = random.randint(1, h - hole_h - 1)
            c0 = random.randint(1, w - hole_w - 1)
            temp[r0 : r0 + hole_h, c0 : c0 + hole_w] = True
        if temp.any() and np.sum(temp) >= 2:
            holes = temp
            break

    return holes if holes.any() else None


def _cross_holes(h: int, w: int) -> np.ndarray:
    holes = np.zeros((h, w), dtype=bool)
    center_r = random.randint(1, h - 2)
    center_c = random.randint(1, w - 2)

    row_start = random.randint(1, center_c)
    row_end = random.randint(center_c, w - 2)
    col_start = random.randint(1, center_r)
    col_end = random.randint(center_r, h - 2)

    holes[center_r, row_start : row_end + 1] = True
    holes[col_start : col_end + 1, center_c] = True
    return holes


def _point_holes(h: int, w: int) -> np.ndarray:
    holes = np.zeros((h, w), dtype=bool)
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    count = random.randint(1, max(1, len(interior) // 2))
    for r, c in random.sample(interior, count):
        holes[r, c] = True
    return holes


def _make_hole_mask(h: int, w: int) -> np.ndarray:
    builders = [_single_rect_holes, _cross_holes, _point_holes]
    if (h - 2) * (w - 2) >= 4:
        builders.append(_double_rect_holes)

    random.shuffle(builders)
    for builder in builders:
        holes = builder(h, w)
        if holes is not None and np.any(holes[1:-1, 1:-1]):
            return holes

    holes = np.zeros((h, w), dtype=bool)
    holes[random.randint(1, h - 2), random.randint(1, w - 2)] = True
    return holes


def _make_mold_patch(h: int, w: int, color: int) -> tuple[np.ndarray, np.ndarray]:
    patch = np.full((h, w), color, dtype=int)
    hole_mask = _make_hole_mask(h, w)
    patch[hole_mask] = BG
    return patch, hole_mask


def _make_cast_patch(h: int, w: int, color: int) -> np.ndarray:
    return np.full((h, w), color, dtype=int)


def _reserve_is_clear(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = patch.shape
    gh, gw = grid.shape
    if gh - h < 2 or gw - w < 2:
        return None

    for _ in range(PLACEMENT_TRIES):
        r0 = random.randint(1, gh - h - 1)
        c0 = random.randint(1, gw - w - 1)
        if not _reserve_is_clear(occupied, r0, c0, h, w, PLACEMENT_PAD):
            continue

        mask = patch != BG
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]

        rr0 = max(0, r0 - PLACEMENT_PAD)
        cc0 = max(0, c0 - PLACEMENT_PAD)
        rr1 = min(gh, r0 + h + PLACEMENT_PAD)
        cc1 = min(gw, c0 + w + PLACEMENT_PAD)
        occupied[rr0:rr1, cc0:cc1] = True
        return r0, c0

    return None


def _choose_pair_count(h: int, w: int) -> int:
    max_pairs = 2
    if h * w >= 280:
        max_pairs = 3
    if h * w >= 420:
        max_pairs = 4

    options = [count for count in PAIR_COUNT_CHOICES if count <= max_pairs]
    return random.choice(options)


def _choose_unique_sizes(h: int, w: int, num_pairs: int) -> list[tuple[int, int]] | None:
    candidates = [
        (hh, ww)
        for hh in range(MIN_OBJECT_SIDE, min(MAX_OBJECT_SIDE, h - 2) + 1)
        for ww in range(MIN_OBJECT_SIDE, min(MAX_OBJECT_SIDE, w - 2) + 1)
    ]
    random.shuffle(candidates)

    sizes: list[tuple[int, int]] = []
    budget = int(h * w * SIZE_BUDGET_RATIO)
    used = 0
    for size in candidates:
        hh, ww = size
        reserve = 2 * (hh + 2 * PLACEMENT_PAD) * (ww + 2 * PLACEMENT_PAD)
        if used + reserve > budget:
            continue
        sizes.append(size)
        used += reserve
        if len(sizes) == num_pairs:
            return sizes
    return None


def _expected_output_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0 = spec["cast_pos"]
        h, w = spec["size"]
        out[r0 : r0 + h, c0 : c0 + w][spec["hole_mask"]] = int(spec["mold_color"])
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    w = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    num_pairs = _choose_pair_count(h, w)
    sizes = _choose_unique_sizes(h, w, num_pairs)
    if sizes is None:
        return None

    colors = random.sample(range(1, 10), 2 * num_pairs)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    specs: list[dict] = []
    placements: list[tuple[str, tuple[int, int], np.ndarray, dict]] = []

    for index, size in enumerate(sizes):
        hh, ww = size
        mold_color = colors[2 * index]
        cast_color = colors[2 * index + 1]
        mold_patch, hole_mask = _make_mold_patch(hh, ww, mold_color)
        cast_patch = _make_cast_patch(hh, ww, cast_color)
        spec = {
            "size": size,
            "mold_color": mold_color,
            "cast_color": cast_color,
            "hole_mask": hole_mask,
        }
        specs.append(spec)
        placements.append(("mold", size, mold_patch, spec))
        placements.append(("cast", size, cast_patch, spec))

    random.shuffle(placements)

    for kind, _size, patch, spec in placements:
        pos = _place_patch(grid, occupied, patch)
        if pos is None:
            return None
        if kind == "mold":
            spec["mold_pos"] = pos
        else:
            spec["cast_pos"] = pos

    parsed_pairs = _pair_objects(_find_components(grid, bg=BG))
    if len(parsed_pairs) != num_pairs:
        return None

    expected = _expected_output_from_specs(grid, specs)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is None:
            continue
        grid, _expected = result
        return grid
    raise RuntimeError("failed to generate a valid mold/cast puzzle")
