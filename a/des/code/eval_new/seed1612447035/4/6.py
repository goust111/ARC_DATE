from collections import Counter, defaultdict, deque

import numpy as np


_RNG = np.random.default_rng()

_BG = 0
_GROUND = 5
_PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_GEN_RETRIES = 400
_PAIR_OFFSETS = (-3, -2, 2, 3)
_INPUT_SEPARATION = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _mode_color(values: np.ndarray) -> int:
    counts = np.bincount(np.asarray(values, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _shape_from_rows(name: str, rows: tuple[str, ...]) -> dict:
    coords = []
    width = len(rows[0])
    assert width % 2 == 1
    for r, row in enumerate(rows):
        assert len(row) == width
        for c, ch in enumerate(row):
            if ch == "1":
                coords.append((r, c))
    arr = np.asarray(coords, dtype=int)
    return {
        "name": name,
        "coords": arr,
        "h": len(rows),
        "w": width,
        "center": width // 2,
        "area": int(arr.shape[0]),
    }


_SHAPES = [
    _shape_from_rows("h3", ("111",)),
    _shape_from_rows("h5", ("11111",)),
    _shape_from_rows("v3", ("1", "1", "1")),
    _shape_from_rows("v4", ("1", "1", "1", "1")),
    _shape_from_rows("rect6", ("111", "111")),
    _shape_from_rows("t_down4", ("010", "111")),
    _shape_from_rows("t_up4", ("111", "010")),
    _shape_from_rows("plus5", ("010", "111", "010")),
    _shape_from_rows("arch5", ("101", "111")),
    _shape_from_rows("cup5", ("111", "101")),
    _shape_from_rows("zig5", ("011", "110", "100")),
    _shape_from_rows("block8", ("111", "111", "110")),
    _shape_from_rows("tower7", ("010", "111", "111")),
]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.asarray(coords, dtype=int).copy()
    out[:, 0] += int(dr)
    out[:, 1] += int(dc)
    return out


def _place_shape(shape: dict, top: int, center_col: int) -> np.ndarray:
    left = int(center_col) - int(shape["center"])
    return _translate(shape["coords"], int(top), int(left))


def _drop_piece(coords: np.ndarray, occupied: np.ndarray, ground_r: int) -> np.ndarray:
    cur = np.asarray(coords, dtype=int).copy()
    while True:
        nxt = cur.copy()
        nxt[:, 0] += 1
        if int(np.max(nxt[:, 0])) >= int(ground_r):
            return cur
        if bool(occupied[nxt[:, 0], nxt[:, 1]].any()):
            return cur
        cur = nxt


def _contact_width(a: np.ndarray, b: np.ndarray) -> int:
    aset = {(int(r), int(c)) for r, c in a}
    bset = {(int(r), int(c)) for r, c in b}
    cols = set()
    for r, c in aset:
        if (int(r) + 1, int(c)) in bset or (int(r) - 1, int(c)) in bset:
            cols.add(int(c))
    return int(len(cols))


def _find_piece_components(grid: np.ndarray, bg: int, ground: int, ground_r: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    pieces = []
    for r in range(int(ground_r)):
        for c in range(w):
            if bool(seen[r, c]):
                continue
            color = int(g[r, c])
            if color in {int(bg), int(ground)}:
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr) + int(dr)
                    nc = int(cc) + int(dc)
                    if not (0 <= nr < int(ground_r) and 0 <= nc < w):
                        continue
                    if bool(seen[nr, nc]) or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))
            arr = np.asarray(coords, dtype=int)
            r0, c0, r1, c1 = _bbox(arr)
            pieces.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (r0, c0, r1, c1),
                    "center_col": int((c0 + c1) // 2),
                }
            )
    return pieces


def _settle_and_fuse(grid: np.ndarray) -> tuple[list[dict], int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode_color(g)
    ground_r = int(h - 1)
    ground = _mode_color(g[ground_r])

    pieces = _find_piece_components(g, bg=bg, ground=ground, ground_r=ground_r)
    assert pieces

    markers: dict[int, int] = {}
    for c in range(w):
        color = int(g[ground_r, c])
        if color == int(ground):
            continue
        assert color not in markers
        markers[color] = int(c)

    assert len(markers) == len(pieces)
    order = sorted(
        pieces,
        key=lambda p: (-int(p["bbox"][2]), -int(p["bbox"][0]), int(p["bbox"][1]), int(p["color"])),
    )

    occupied = np.zeros((h, w), dtype=bool)
    settled = []
    for idx, piece in enumerate(order):
        color = int(piece["color"])
        assert color in markers
        dc = int(markers[color] - int(piece["center_col"]))
        shifted = _translate(piece["coords"], 0, dc)
        assert int(np.min(shifted[:, 1])) >= 0 and int(np.max(shifted[:, 1])) < w
        final_coords = _drop_piece(shifted, occupied, ground_r)
        occupied[final_coords[:, 0], final_coords[:, 1]] = True
        settled.append(
            {
                "id": idx,
                "color": color,
                "area": int(piece["area"]),
                "coords": final_coords,
            }
        )

    proposals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i in range(len(settled)):
        for j in range(i + 1, len(settled)):
            width = _contact_width(settled[i]["coords"], settled[j]["coords"])
            if width < 2:
                continue
            area_i = int(settled[i]["area"])
            area_j = int(settled[j]["area"])
            if area_i == area_j:
                continue
            if area_i < area_j:
                proposals[i].append((area_j, int(settled[j]["color"])))
            else:
                proposals[j].append((area_i, int(settled[i]["color"])))

    for idx, piece in enumerate(settled):
        if idx not in proposals:
            piece["out_color"] = int(piece["color"])
            continue
        best_area = max(area for area, _color in proposals[idx])
        colors = {color for area, color in proposals[idx] if area == best_area}
        piece["out_color"] = int(colors.pop()) if len(colors) == 1 else int(piece["color"])

    return settled, bg, ground


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    settled, bg, ground = _settle_and_fuse(g)
    out = np.full((h, w), int(bg), dtype=int)
    out[h - 1, :] = int(ground)
    for piece in settled:
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = int(piece["out_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop scattered colored fragments onto the ground, then fuse colors when wide vertical contacts form.

    Input structure:
    - bg = most common color. The bottom row is a grey ground line with a few single-pixel colored markers embedded in it.
    - Above the ground sit several monochrome 4-connected fragments, each using a unique non-bg, non-grey color.
    - Each fragment's color appears once on the ground line; that marker gives the target column for the fragment's bbox midpoint.

    Rule:
    - Match every fragment to the ground marker of the same color, translate it horizontally so the bbox midpoint column aligns to that marker, then let it fall vertically.
    - Fragments settle one by one from lowest original position to highest original position, stopping just above the ground or just above already settled fragments.
    - After all fragments settle, inspect every directly touching upper/lower pair. If their horizontal contact width is at least 2 cells, the smaller-area fragment recolors to the larger fragment's color; equal-area pairs keep both colors.
    - The output keeps the ground line but erases the colored markers back to grey.

    Key insight: Color-matched ground markers determine horizontal alignment, original height determines drop order, and only after stacking does a hidden contact-width threshold decide whether color fusion occurs.
    """
    return _solve(np.asarray(grid, dtype=int))


def _build_pair_catalog() -> dict[str, list[dict]]:
    catalog: dict[str, list[dict]] = defaultdict(list)
    probe_ground = 20
    probe_center = 15
    probe_occ = np.zeros((probe_ground + 1, 40), dtype=bool)

    for lower in _SHAPES:
        for upper in _SHAPES:
            for offset in _PAIR_OFFSETS:
                lower_coords = _drop_piece(_place_shape(lower, 0, probe_center), probe_occ, probe_ground)
                occupied = np.zeros_like(probe_occ)
                occupied[lower_coords[:, 0], lower_coords[:, 1]] = True
                upper_coords = _drop_piece(_place_shape(upper, 0, probe_center + int(offset)), occupied, probe_ground)
                width = _contact_width(lower_coords, upper_coords)
                if width == 0:
                    continue

                lower_area = int(lower["area"])
                upper_area = int(upper["area"])
                lower_rel = lower["coords"][:, 1] - int(lower["center"])
                upper_rel = upper["coords"][:, 1] - int(upper["center"]) + int(offset)
                min_rel = int(min(np.min(lower_rel), np.min(upper_rel), 0, int(offset)))
                max_rel = int(max(np.max(lower_rel), np.max(upper_rel), 0, int(offset)))
                spec = {
                    "lower_shape": lower,
                    "upper_shape": upper,
                    "offset": int(offset),
                    "contact_width": int(width),
                    "lower_area": lower_area,
                    "upper_area": upper_area,
                    "min_rel_col": min_rel,
                    "max_rel_col": max_rel,
                }
                if width >= 2:
                    if lower_area > upper_area:
                        catalog["trigger_upper"].append(spec)
                    elif lower_area < upper_area:
                        catalog["trigger_lower"].append(spec)
                    else:
                        catalog["equal_wide"].append(spec)
                else:
                    catalog["narrow"].append(spec)
    return catalog


_PAIR_CATALOG = _build_pair_catalog()


def _single_group() -> dict:
    shape = _RNG.choice(_SHAPES)
    rel = shape["coords"][:, 1] - int(shape["center"])
    return {
        "pieces": [{"shape": shape, "marker_rel": 0, "role": "single"}],
        "min_rel_col": int(min(np.min(rel), 0)),
        "max_rel_col": int(max(np.max(rel), 0)),
    }


def _pair_group(kind: str) -> dict:
    cand = dict(_RNG.choice(_PAIR_CATALOG[kind]))
    return {
        "pieces": [
            {"shape": cand["lower_shape"], "marker_rel": 0, "role": "lower"},
            {"shape": cand["upper_shape"], "marker_rel": int(cand["offset"]), "role": "upper"},
        ],
        "min_rel_col": int(cand["min_rel_col"]),
        "max_rel_col": int(cand["max_rel_col"]),
        "kind": kind,
    }


def _sample_groups() -> list[dict]:
    trigger_kind = str(_RNG.choice(("trigger_upper", "trigger_lower")))
    nontrigger_kind = str(_RNG.choice(("equal_wide", "narrow")))
    groups = [_pair_group(trigger_kind), _pair_group(nontrigger_kind)]
    extra_roll = float(_RNG.random())
    if extra_roll < 0.2:
        groups.append(_single_group())
    elif extra_roll < 0.35:
        groups.append(_pair_group(str(_RNG.choice(("narrow", "equal_wide")))))
    _RNG.shuffle(groups)
    return groups


def _pack_groups(groups: list[dict]) -> tuple[list[dict], int] | None:
    cursor = int(_RNG.integers(1, 4))
    laid_out = []
    for group in groups:
        shift = int(cursor - int(group["min_rel_col"]))
        placed = dict(group)
        placed["shift"] = shift
        laid_out.append(placed)
        cursor = int(shift + int(group["max_rel_col"]) + int(_RNG.integers(3, 6)))
    min_width = int(cursor + int(_RNG.integers(1, 4)))
    low = int(max(18, min_width))
    if low > 30:
        return None
    width = low if low == 30 else int(_RNG.integers(low, 31))
    return laid_out, width


def _final_specs_from_groups(groups: list[dict], width: int, height: int) -> list[dict] | None:
    ground_r = int(height - 1)
    occupied = np.zeros((height, width), dtype=bool)
    specs = []
    color_pool = list(_RNG.choice(_PIECE_COLORS, size=sum(len(g["pieces"]) for g in groups), replace=False))
    color_i = 0

    for group in groups:
        for piece in group["pieces"]:
            shape = piece["shape"]
            marker_col = int(group["shift"] + int(piece["marker_rel"]))
            cols = shape["coords"][:, 1] + int(marker_col) - int(shape["center"])
            if int(np.min(cols)) < 0 or int(np.max(cols)) >= width:
                return None
            dropped = _drop_piece(_place_shape(shape, 0, marker_col), occupied, ground_r)
            occupied[dropped[:, 0], dropped[:, 1]] = True
            specs.append(
                {
                    "shape": shape,
                    "color": int(color_pool[color_i]),
                    "marker_col": marker_col,
                    "final_coords": dropped,
                }
            )
            color_i += 1
    return specs


def _paint_piece(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _fits_input(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        for dr, dc in _INPUT_SEPARATION:
            nr = int(rr) + int(dr)
            nc = int(cc) + int(dc)
            if 0 <= nr < h and 0 <= nc < w and bool(occupied[nr, nc]):
                return False
    return True


def _scatter_input(specs: list[dict], width: int, height: int) -> np.ndarray | None:
    ground_r = int(height - 1)
    n = len(specs)
    if n == 0:
        return None

    order = list(range(n))
    bottoms = sorted(_RNG.choice(np.arange(5, ground_r - 1), size=n, replace=False), reverse=True)
    occupied = np.zeros((height, width), dtype=bool)
    placements = {}
    misaligned = False

    for idx, bottom in zip(order, bottoms):
        shape = specs[idx]["shape"]
        h = int(shape["h"])
        top = int(bottom) - int(h) + 1
        if top < 0:
            return None
        choices = list(range(0, width - int(shape["w"]) + 1))
        _RNG.shuffle(choices)
        marker_center = int(specs[idx]["marker_col"])
        placed = False
        for left in choices:
            center = int(left + int(shape["center"]))
            if center == marker_center and float(_RNG.random()) < 0.8:
                continue
            coords = _translate(shape["coords"], top, left)
            if not _fits_input(coords, occupied):
                continue
            placements[idx] = coords
            occupied[coords[:, 0], coords[:, 1]] = True
            misaligned = misaligned or center != marker_center
            placed = True
            break
        if not placed:
            return None

    if not misaligned:
        return None

    grid = np.full((height, width), _BG, dtype=int)
    grid[ground_r, :] = _GROUND
    for spec in specs:
        marker_col = int(spec["marker_col"])
        if int(grid[ground_r, marker_col]) != _GROUND:
            return None
        grid[ground_r, marker_col] = int(spec["color"])
    for idx, spec in enumerate(specs):
        _paint_piece(grid, placements[idx], int(spec["color"]))
    return grid


def _planned_output(specs: list[dict], width: int, height: int) -> np.ndarray:
    grid = np.full((height, width), _BG, dtype=int)
    grid[height - 1, :] = _GROUND
    tmp = []
    for spec in specs:
        tmp.append(
            {
                "color": int(spec["color"]),
                "area": int(spec["shape"]["area"]),
                "coords": np.asarray(spec["final_coords"], dtype=int),
            }
        )

    proposals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i in range(len(tmp)):
        for j in range(i + 1, len(tmp)):
            width_ij = _contact_width(tmp[i]["coords"], tmp[j]["coords"])
            if width_ij < 2:
                continue
            if int(tmp[i]["area"]) == int(tmp[j]["area"]):
                continue
            if int(tmp[i]["area"]) < int(tmp[j]["area"]):
                proposals[i].append((int(tmp[j]["area"]), int(tmp[j]["color"])))
            else:
                proposals[j].append((int(tmp[i]["area"]), int(tmp[i]["color"])))

    for i, piece in enumerate(tmp):
        if i not in proposals:
            out_color = int(piece["color"])
        else:
            best = max(area for area, _ in proposals[i])
            colors = {color for area, color in proposals[i] if area == best}
            out_color = int(colors.pop()) if len(colors) == 1 else int(piece["color"])
        _paint_piece(grid, piece["coords"], out_color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    groups = _sample_groups()
    packed = _pack_groups(groups)
    if packed is None:
        return None
    laid_out, width = packed

    height = int(_RNG.integers(18, 29))
    specs = _final_specs_from_groups(laid_out, width=width, height=height)
    if specs is None:
        return None

    planned = _planned_output(specs, width=width, height=height)
    grid = _scatter_input(specs, width=width, height=height)
    if grid is None:
        return None

    if int(np.bincount(grid.ravel(), minlength=10)[_BG]) <= int(np.bincount(grid.ravel(), minlength=10)[_GROUND]):
        return None
    if not np.array_equal(_solve(grid), planned):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid stacking-and-fusion sample")
