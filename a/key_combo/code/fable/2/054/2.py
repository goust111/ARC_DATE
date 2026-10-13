from __future__ import annotations

import numpy as np

BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

BBox = tuple[int, int, int, int]

GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    out: list[np.ndarray] = []
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
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = BLACK) -> None:
    h, w = patch.shape
    r1, c1 = r0 + h, c0 + w
    assert 0 <= r0 <= r1 <= dst.shape[0]
    assert 0 <= c0 <= c1 <= dst.shape[1]
    view = dst[r0:r1, c0:c1]
    mask = patch != transparent
    assert not np.any(mask & (view != BLACK)), "unexpected overlap in rendered scene"
    view[mask] = patch[mask]


def _is_lr_symmetric(sprite: np.ndarray) -> bool:
    mask = sprite != BLACK
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _extract_objects(grid: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    colors = sorted(int(x) for x in np.unique(grid) if int(x) not in (BLACK, GREY))
    for color in colors:
        for coords in _components(grid == color, connectivity=8):
            r0, c0, r1, c1 = _bbox_coords(coords)
            sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
            sprite[coords[:, 0] - r0, coords[:, 1] - c0] = color
            objects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "sprite": sprite,
                    "symmetric": _is_lr_symmetric(sprite),
                }
            )
    objects.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0], obj["color"]))
    return objects


def _extract_planks(grid: np.ndarray) -> list[dict]:
    planks: list[dict] = []
    for coords in _components(grid == GREY, connectivity=4):
        rs = coords[:, 0]
        cs = coords[:, 1]
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        assert r0 == r1, "grey planks must be one-cell tall horizontal bars"
        assert int(coords.shape[0]) == c1 - c0 + 1, "grey plank must be a solid segment"
        planks.append({"row": r0, "c0": c0, "c1": c1})
    planks.sort(key=lambda plank: (plank["row"], plank["c0"]))
    return planks


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _planks_below(obj: dict, planks: list[dict]) -> list[dict]:
    _r0, c0, r1, c1 = obj["bbox"]
    return [
        plank
        for plank in planks
        if plank["row"] > r1 and _intervals_overlap(c0, c1, plank["c0"], plank["c1"])
    ]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _extract_objects(g)
    planks = _extract_planks(g)
    assert objects, "expected at least one falling object"

    for left, right in zip(objects, objects[1:]):
        assert left["bbox"][3] < right["bbox"][1], "object column intervals must be disjoint"

    out = np.full_like(g, BLACK)
    out[g == GREY] = GREY

    for obj in objects:
        sprite = obj["sprite"]
        h, w = sprite.shape
        _r0, c0, _r1, _c1 = obj["bbox"]
        below = _planks_below(obj, planks)

        if obj["symmetric"] and below:
            top = below[0]["row"] - h
            final_sprite = sprite
        else:
            top = g.shape[0] - h
            final_sprite = np.fliplr(sprite) if (not obj["symmetric"] and len(below) % 2 == 1) else sprite

        _paste(out, final_sprite, top, c0)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each floating object vertically, letting only left-right symmetric ones stop at the first grey plank.

    Input structure:
    - Black background with several disconnected single-color objects suspended in the upper area.
    - Each object occupies its own non-overlapping column interval.
    - Lower in the grid are one-cell-tall grey horizontal planks, possibly at several rows.

    Rule:
    - Every object falls straight down without changing color.
    - If an object's shape is left-right symmetric around its own vertical axis, it stops on top of the first grey plank directly below its column interval.
    - Otherwise it passes through every grey plank below it, mirroring left-right once per crossed plank, and finally bottom-aligns.
    - Grey planks stay fixed.

    Key insight:
    - Symmetry is the control signal: planks catch only self-mirroring objects, while non-symmetric ones treat each crossed plank as a left-right mirror that toggles the final orientation by parity.
    """
    return _solve(np.asarray(grid, dtype=int))


def _sprite_from_heights(heights: np.ndarray, color: int) -> np.ndarray:
    h = int(heights.max())
    w = int(heights.size)
    sprite = np.zeros((h, w), dtype=int)
    for c, col_h in enumerate(heights.tolist()):
        sprite[h - int(col_h) :, c] = color
    return sprite


def _sample_heights(rng: np.random.Generator, height: int, width: int, symmetric: bool) -> np.ndarray:
    for _ in range(200):
        if symmetric:
            half = (width + 1) // 2
            left = rng.integers(1, height + 1, size=half)
            if width % 2 == 0:
                heights = np.concatenate([left, left[::-1]])
            else:
                heights = np.concatenate([left, left[-2::-1]])
        else:
            heights = rng.integers(1, height + 1, size=width)

        heights = heights.astype(int)
        heights[int(rng.integers(0, width))] = height
        if len(set(int(x) for x in heights.tolist())) < 2:
            continue
        if symmetric != bool(np.array_equal(heights, heights[::-1])):
            continue
        return heights
    raise RuntimeError("failed to sample skyline heights")


def _build_scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    odd_count = int(rng.choice([1, 3]))
    roles = ["sym", "asym_even", "asym_odd"]
    rng.shuffle(roles)
    colors = rng.choice(np.array(PALETTE, dtype=int), size=3, replace=False)

    metas: list[dict] = []
    widths: list[int] = []
    for role, color in zip(roles, colors.tolist()):
        width = int(rng.integers(3, 6))
        height = int(rng.integers(2, 6))
        heights = _sample_heights(rng, height=height, width=width, symmetric=(role == "sym"))
        sprite = _sprite_from_heights(heights, int(color))
        widths.append(int(sprite.shape[1]))
        metas.append({"role": role, "color": int(color), "sprite": sprite})

    slot_widths = [int(rng.integers(1, 4)), int(rng.integers(2, 5)), int(rng.integers(2, 5)), int(rng.integers(1, 4))]
    base_w = slot_widths[0] + widths[0] + slot_widths[1] + widths[1] + slot_widths[2] + widths[2] + slot_widths[3]
    extra_w = int(rng.integers(0, 31 - base_w))
    for _ in range(extra_w):
        slot_widths[int(rng.integers(0, len(slot_widths)))] += 1

    lefts: list[int] = []
    cur = slot_widths[0]
    for idx, width in enumerate(widths):
        lefts.append(cur)
        cur += width
        if idx < 2:
            cur += slot_widths[idx + 1]
    canvas_w = cur + slot_widths[3]

    tops = [int(rng.integers(1, 4)) for _ in range(3)]
    max_bottom = max(top + metas[idx]["sprite"].shape[0] - 1 for idx, top in enumerate(tops))
    max_height = max(int(meta["sprite"].shape[0]) for meta in metas)

    plank_rows = [max_bottom + int(rng.integers(2, 5))]
    plank_rows.append(plank_rows[-1] + int(rng.integers(2, 5)))
    if odd_count == 3:
        plank_rows.append(plank_rows[-1] + int(rng.integers(2, 5)))

    canvas_h = plank_rows[-1] + max_height + int(rng.integers(2, 5)) + 1
    assert canvas_h <= 30 and canvas_w <= 30

    odd_single_row = [plank_rows[int(rng.integers(0, 2))]]
    for idx, meta in enumerate(metas):
        meta["top"] = tops[idx]
        meta["left"] = lefts[idx]
        if meta["role"] == "sym":
            rows = plank_rows[:2]
        elif meta["role"] == "asym_even":
            rows = plank_rows[:2]
        else:
            rows = plank_rows[:] if odd_count == 3 else odd_single_row
        meta["plank_rows"] = list(rows)

    grid = np.full((canvas_h, canvas_w), BLACK, dtype=int)
    for meta in metas:
        _paste(grid, meta["sprite"], meta["top"], meta["left"])
    for meta in metas:
        row_span = meta["plank_rows"]
        width = int(meta["sprite"].shape[1])
        left = int(meta["left"])
        for row in row_span:
            grid[int(row), left : left + width] = GREY

    expected = np.full_like(grid, BLACK)
    expected[grid == GREY] = GREY
    for meta in metas:
        sprite = meta["sprite"]
        if meta["role"] == "sym":
            final = sprite
            top = int(meta["plank_rows"][0]) - sprite.shape[0]
        else:
            final = np.fliplr(sprite) if (len(meta["plank_rows"]) % 2 == 1) else sprite
            top = canvas_h - sprite.shape[0]
        _paste(expected, final, int(top), int(meta["left"]))

    return grid, expected, metas


def _validate_scene(grid: np.ndarray, expected: np.ndarray, metas: list[dict]) -> bool:
    assert grid.shape == expected.shape
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert not np.array_equal(grid, expected), "input and output should differ"

    parsed_objects = _extract_objects(grid)
    parsed_planks = _extract_planks(grid)
    if len(parsed_objects) != 3:
        return False
    if len(parsed_planks) != sum(len(meta["plank_rows"]) for meta in metas):
        return False

    meta_by_color = {int(meta["color"]): meta for meta in metas}
    parsed_by_color = {int(obj["color"]): obj for obj in parsed_objects}
    if set(meta_by_color) != set(parsed_by_color):
        return False

    for meta in metas:
        color = int(meta["color"])
        obj = parsed_by_color[color]
        sprite = meta["sprite"]
        top = int(meta["top"])
        left = int(meta["left"])
        bbox = (top, left, top + sprite.shape[0] - 1, left + sprite.shape[1] - 1)
        if obj["bbox"] != bbox:
            return False
        if obj["symmetric"] != (meta["role"] == "sym"):
            return False
        if len(_planks_below(obj, parsed_planks)) != len(meta["plank_rows"]):
            return False

    role_map = {meta["role"]: meta for meta in metas}
    if role_map["sym"]["plank_rows"] != role_map["asym_even"]["plank_rows"]:
        return False
    if len(role_map["sym"]["plank_rows"]) != 2:
        return False
    if len(role_map["asym_even"]["plank_rows"]) != 2:
        return False
    if len(role_map["asym_odd"]["plank_rows"]) not in (1, 3):
        return False

    odd_sprite = role_map["asym_odd"]["sprite"]
    if np.array_equal(odd_sprite, np.fliplr(odd_sprite)):
        return False

    odd_left = int(role_map["asym_odd"]["left"])
    odd_h, odd_w = odd_sprite.shape
    odd_final = expected[expected.shape[0] - odd_h :, odd_left : odd_left + odd_w]
    if len(role_map["asym_odd"]["plank_rows"]) % 2 == 1 and not np.array_equal(odd_final, np.fliplr(odd_sprite)):
        return False

    even_sprite = role_map["asym_even"]["sprite"]
    even_left = int(role_map["asym_even"]["left"])
    even_h, even_w = even_sprite.shape
    even_final = expected[expected.shape[0] - even_h :, even_left : even_left + even_w]
    if not np.array_equal(even_final, even_sprite):
        return False

    sym_sprite = role_map["sym"]["sprite"]
    sym_left = int(role_map["sym"]["left"])
    sym_top = int(role_map["sym"]["plank_rows"][0]) - sym_sprite.shape[0]
    sym_final = expected[sym_top : sym_top + sym_sprite.shape[0], sym_left : sym_left + sym_sprite.shape[1]]
    if not np.array_equal(sym_final, sym_sprite):
        return False
    if sym_top >= expected.shape[0] - sym_sprite.shape[0]:
        return False

    return bool(np.array_equal(main(grid), expected))


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid, expected, metas = _build_scene(rng)
        if _validate_scene(grid, expected, metas):
            return grid
    raise RuntimeError(f"failed to generate a valid mirror-plank scene after {GEN_ATTEMPTS} attempts")
