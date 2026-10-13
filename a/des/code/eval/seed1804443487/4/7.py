import random
from collections import Counter, deque

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
SHAPES = ("line", "L", "T")
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_TRIES = 200


def _make_template(name: str, shape: str, coords: list[tuple[int, int]], gap_width: int) -> dict:
    coords = tuple(sorted((int(r), int(c)) for r, c in coords))
    return {
        "name": name,
        "shape": shape,
        "coords": coords,
        "gap_width": int(gap_width),
        "max_r": max(r for r, _c in coords),
        "max_c": max(c for _r, c in coords),
        "inside_count": sum(1 for r, _c in coords if r > 0),
    }


TEMPLATES = [
    _make_template("line2", "line", [(0, 0), (1, 0)], 1),
    _make_template("line3", "line", [(0, 0), (1, 0), (2, 0)], 1),
    _make_template("line4", "line", [(0, 0), (1, 0), (2, 0), (3, 0)], 1),
    _make_template("l2_short", "L", [(0, 0), (0, 1), (1, 0)], 2),
    _make_template("l2_tall", "L", [(0, 0), (0, 1), (1, 0), (2, 0)], 2),
    _make_template("l2_hook", "L", [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)], 2),
    _make_template("l3_short", "L", [(0, 0), (0, 1), (0, 2), (1, 0)], 3),
    _make_template("l3_tall", "L", [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)], 3),
    _make_template("l3_hook", "L", [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)], 3),
    _make_template("t3_short", "T", [(0, 0), (0, 1), (0, 2), (1, 1)], 3),
    _make_template("t3_tall", "T", [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)], 3),
    _make_template("t3_long", "T", [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (3, 1)], 3),
]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _normalize_coords(coords: list[tuple[int, int]] | np.ndarray) -> list[tuple[int, int]]:
    if isinstance(coords, np.ndarray):
        rows = coords[:, 0]
        cols = coords[:, 1]
        coords_list = [(int(r), int(c)) for r, c in coords]
    else:
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        coords_list = [(int(r), int(c)) for r, c in coords]
    min_r = int(min(rows))
    min_c = int(min(cols))
    return sorted((r - min_r, c - min_c) for r, c in coords_list)


def _palette_component(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[int]]:
    candidates = []
    for comp in _components(grid != bg):
        colors = [int(grid[r, c]) for r, c in comp]
        if len(set(colors)) < 2:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if c0 != c1:
            continue
        if len(comp) != (r1 - r0 + 1):
            continue
        h, w = grid.shape
        if not ((r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)):
            continue
        order = np.argsort(comp[:, 0], kind="stable")
        ordered = comp[order]
        palette = [int(grid[r, c]) for r, c in ordered]
        candidates.append((ordered, palette))

    assert len(candidates) == 1, "expected a unique multicolor corner palette"
    return candidates[0]


def _analyze_frame(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    coord_set = {(int(r), int(c)) for r, c in coords}
    for r, c in coord_set:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    sides = {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }
    missing = {
        side: [cell for cell in cells if cell not in coord_set]
        for side, cells in sides.items()
    }
    gap_sides = [side for side, miss in missing.items() if miss]
    if len(gap_sides) != 1:
        return None

    side = gap_sides[0]
    gap = missing[side]
    if not (1 <= len(gap) <= 3):
        return None

    if side in ("top", "bottom"):
        ordered = sorted(gap, key=lambda p: p[1])
        if any(ordered[i + 1][1] != ordered[i][1] + 1 for i in range(len(ordered) - 1)):
            return None
    else:
        ordered = sorted(gap, key=lambda p: p[0])
        if any(ordered[i + 1][0] != ordered[i][0] + 1 for i in range(len(ordered) - 1)):
            return None

    return {
        "bbox": (r0, c0, r1, c1),
        "coords": coord_set,
        "side": side,
        "gap": set(gap),
        "interior": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
    }


def _valid_translation(
    translated: set[tuple[int, int]], frame: dict
) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    gap = frame["gap"]
    border = (
        {(r0, c) for c in range(c0, c1 + 1)}
        | {(r1, c) for c in range(c0, c1 + 1)}
        | {(r, c0) for r in range(r0, r1 + 1)}
        | {(r, c1) for r in range(r0, r1 + 1)}
    )

    if not gap.issubset(translated):
        return False
    if translated & border != gap:
        return False

    interior = translated - gap
    if not interior:
        return False
    for r, c in translated:
        if not (r0 <= r <= r1 and c0 <= c <= c1):
            return False
    for r, c in interior:
        if not (r0 < r < r1 and c0 < c < c1):
            return False
    return True


def _place_key(frame: dict, key_coords: np.ndarray) -> set[tuple[int, int]]:
    key = [(int(r), int(c)) for r, c in key_coords]
    deltas = set()
    for kr, kc in key:
        for gr, gc in frame["gap"]:
            deltas.add((gr - kr, gc - kc))

    valid = []
    for dr, dc in deltas:
        translated = {(r + dr, c + dc) for r, c in key}
        if _valid_translation(translated, frame):
            valid.append(tuple(sorted(translated)))

    uniq = {placement for placement in valid}
    assert len(uniq) == 1, "expected a unique key insertion"
    return set(next(iter(uniq)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert each same-colored key into its frame gap, then fill the enclosed area via the palette index.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    palette_coords, palette = _palette_component(g, bg)
    palette_mask = np.zeros_like(g, dtype=bool)
    palette_mask[palette_coords[:, 0], palette_coords[:, 1]] = True

    colors = sorted(int(c) for c in np.unique(g[(g != bg) & (~palette_mask)]))
    out = g.copy()

    for color in colors:
        comps = _components((g == color) & (~palette_mask))
        assert len(comps) == 2, "expected one frame and one key for each door color"

        analyzed = [(_analyze_frame(comp), comp) for comp in comps]
        frames = [(frame, comp) for frame, comp in analyzed if frame is not None]
        assert len(frames) == 1, "expected a unique frame component"
        frame, frame_comp = frames[0]
        key_comp = next(comp for probe, comp in analyzed if probe is None)

        out[key_comp[:, 0], key_comp[:, 1]] = bg
        placed = _place_key(frame, key_comp)
        for r, c in placed:
            out[r, c] = int(color)

        ir0, ic0, ir1, ic1 = frame["interior"]
        interior = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
        area = int(np.sum(interior == bg))
        assert 1 <= area <= len(palette), "palette must cover the enclosed area"
        interior[interior == bg] = int(palette[area - 1])

    return out


def _reflect_template(coords: tuple[tuple[int, int], ...], gap_width: int) -> list[tuple[int, int]]:
    return sorted((r, gap_width - 1 - c) for r, c in coords)


def _door_dims_for_side(template: dict, side: str) -> tuple[int, int]:
    if side in ("top", "bottom"):
        min_h = int(template["max_r"])
        min_w = int(max(template["gap_width"], template["max_c"] + 1))
    else:
        min_h = int(max(template["gap_width"], template["max_c"] + 1))
        min_w = int(template["max_r"])
    return min_h, min_w


def _transform_inserted(
    coords: list[tuple[int, int]],
    side: str,
    inner_h: int,
    inner_w: int,
    gap_width: int,
    start: int,
) -> list[tuple[int, int]]:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    transformed = []
    for r, c in coords:
        if side == "top":
            transformed.append((r, start + c))
        elif side == "bottom":
            transformed.append((outer_h - 1 - r, start + c))
        elif side == "left":
            transformed.append((start + c, r))
        else:
            transformed.append((start + c, outer_w - 1 - r))
    return sorted(transformed)


def _door_spec(rng: random.Random) -> dict | None:
    shape = rng.choice(SHAPES)
    candidates = [tpl for tpl in TEMPLATES if tpl["shape"] == shape]
    rng.shuffle(candidates)
    for template in candidates:
        coords = list(template["coords"])
        if shape == "L" and rng.random() < 0.5:
            coords = _reflect_template(template["coords"], template["gap_width"])
        side = rng.choice(SIDES)
        min_h, min_w = _door_dims_for_side(template, side)
        dim_choices = [
            (inner_h, inner_w)
            for inner_h in range(max(2, min_h), 5)
            for inner_w in range(max(2, min_w), 5)
        ]
        rng.shuffle(dim_choices)
        for inner_h, inner_w in dim_choices:
            area = int(inner_h * inner_w - template["inside_count"])
            if not (1 <= area <= 8):
                continue

            if side in ("top", "bottom"):
                start = rng.randint(1, inner_w - template["gap_width"] + 1)
            else:
                start = rng.randint(1, inner_h - template["gap_width"] + 1)

            inserted = _transform_inserted(
                coords,
                side,
                inner_h,
                inner_w,
                template["gap_width"],
                start,
            )
            key_coords = _normalize_coords(inserted)
            frame_h = inner_h + 2
            frame_w = inner_w + 2
            gap = set()
            if side == "top":
                gap = {(0, c) for c in range(start, start + template["gap_width"])}
            elif side == "bottom":
                gap = {(frame_h - 1, c) for c in range(start, start + template["gap_width"])}
            elif side == "left":
                gap = {(r, 0) for r in range(start, start + template["gap_width"])}
            else:
                gap = {(r, frame_w - 1) for r in range(start, start + template["gap_width"])}

            frame = {
                (0, c) for c in range(frame_w)
            } | {
                (frame_h - 1, c) for c in range(frame_w)
            } | {
                (r, 0) for r in range(frame_h)
            } | {
                (r, frame_w - 1) for r in range(frame_h)
            }
            frame -= gap

            return {
                "shape": shape,
                "side": side,
                "area": area,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "frame_h": frame_h,
                "frame_w": frame_w,
                "frame_coords": sorted(frame),
                "inserted_coords": inserted,
                "key_coords": key_coords,
            }
    return None


def _palette_colors(rng: random.Random, count: int, pool: list[int]) -> list[int]:
    assert len(pool) >= 2
    colors: list[int] = []
    bag = pool[:]
    while len(colors) < count:
        rng.shuffle(bag)
        for color in bag:
            if colors and colors[-1] == color:
                continue
            colors.append(int(color))
            if len(colors) == count:
                break
    assert len(set(colors)) >= 2
    return colors


def _can_place(
    occupied: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    pad: int,
) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_rect(
    rng: random.Random,
    occupied: np.ndarray,
    height: int,
    width: int,
    pad: int,
) -> tuple[int, int] | None:
    h, w = occupied.shape
    if height > h - 2 or width > w - 2:
        return None
    for _ in range(PLACEMENT_TRIES):
        top = rng.randint(1, h - height - 1)
        left = rng.randint(1, w - width - 1)
        if _can_place(occupied, top, left, height, width, pad):
            return top, left
    return None


def _paint_coords(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    top: int,
    left: int,
    color: int,
) -> None:
    for r, c in coords:
        grid[top + r, left + c] = int(color)


def _build_expected(
    inp: np.ndarray,
    placed_doors: list[dict],
    palette_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    for door in placed_doors:
        color = int(door["color"])
        key_top, key_left = door["key_pos"]
        for r, c in door["key_coords"]:
            out[key_top + r, key_left + c] = BG

        frame_top, frame_left = door["frame_pos"]
        for r, c in door["inserted_coords"]:
            out[frame_top + r, frame_left + c] = color

        ir0 = frame_top + 1
        ic0 = frame_left + 1
        ir1 = ir0 + door["inner_h"]
        ic1 = ic0 + door["inner_w"]
        interior = out[ir0:ir1, ic0:ic1]
        area = int(np.sum(interior == BG))
        assert area == int(door["area"])
        interior[interior == BG] = int(palette_colors[area - 1])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    door_count = rng.choice([2, 3, 4])
    doors = []
    for _ in range(door_count):
        spec = _door_spec(rng)
        if spec is None:
            return None
        doors.append(spec)

    max_area = max(int(door["area"]) for door in doors)
    palette_len = max(3, max_area + rng.choice([0, 1]))

    non_bg_colors = list(range(1, 10))
    rng.shuffle(non_bg_colors)
    frame_colors = non_bg_colors[:door_count]
    palette_pool = non_bg_colors[door_count:]
    if len(palette_pool) < 2:
        return None
    palette_colors = _palette_colors(rng, palette_len, palette_pool)

    min_h = max(22, palette_len + 8)
    height = rng.randint(min_h, min_h + 6)
    width = rng.randint(22, 30)
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    corner = rng.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        palette_top, palette_left = 0, 0
    elif corner == "tr":
        palette_top, palette_left = 0, width - 1
    elif corner == "bl":
        palette_top, palette_left = height - palette_len, 0
    else:
        palette_top, palette_left = height - palette_len, width - 1
    for idx, color in enumerate(palette_colors):
        grid[palette_top + idx, palette_left] = int(color)
        occupied[palette_top + idx, palette_left] = True

    placed_doors = []
    door_indices = list(range(door_count))
    rng.shuffle(door_indices)
    for idx in door_indices:
        door = doors[idx]
        frame_pos = _place_rect(rng, occupied, door["frame_h"], door["frame_w"], pad=1)
        if frame_pos is None:
            return None
        frame_top, frame_left = frame_pos
        _paint_coords(grid, door["frame_coords"], frame_top, frame_left, frame_colors[idx])
        for r, c in door["frame_coords"]:
            occupied[frame_top + r, frame_left + c] = True

        key_rows = [r for r, _c in door["key_coords"]]
        key_cols = [c for _r, c in door["key_coords"]]
        key_h = max(key_rows) + 1
        key_w = max(key_cols) + 1
        key_pos = _place_rect(rng, occupied, key_h, key_w, pad=2)
        if key_pos is None:
            return None
        key_top, key_left = key_pos
        _paint_coords(grid, door["key_coords"], key_top, key_left, frame_colors[idx])
        for r, c in door["key_coords"]:
            occupied[key_top + r, key_left + c] = True

        placed = dict(door)
        placed["color"] = frame_colors[idx]
        placed["frame_pos"] = frame_pos
        placed["key_pos"] = key_pos
        placed_doors.append(placed)

    expected = _build_expected(grid, placed_doors, palette_colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
