from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
AZURE = 8
PIECE_COLORS = (1, 2, 3, 4, 6, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _pattern(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


DIGIT_TEMPLATES: dict[int, np.ndarray] = {
    1: _pattern(
        (
            ".#.",
            "##.",
            ".#.",
            ".#.",
            "###",
        )
    ),
    2: _pattern(
        (
            "###",
            "..#",
            "###",
            "#..",
            "###",
        )
    ),
    3: _pattern(
        (
            "###",
            "..#",
            "###",
            "..#",
            "###",
        )
    ),
    4: _pattern(
        (
            "#.#",
            "#.#",
            "###",
            "..#",
            "..#",
        )
    ),
}

SHAPE_LIBRARY: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0),),
    ((0, -1), (0, 0), (0, 1)),
    ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)),
    tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)),
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if not bool(mask[nr, nc]) or bool(seen[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _digit_value_from_patch(patch: np.ndarray) -> int:
    for value, template in DIGIT_TEMPLATES.items():
        if patch.shape == template.shape and np.array_equal(patch, template):
            return int(value)
    raise AssertionError("unknown legend digit")


def _shape_offsets_from_component(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
    assert h % 2 == 1 and w % 2 == 1, "shape bbox must have an odd center"
    cr, cc = h // 2, w // 2
    offsets: list[tuple[int, int]] = []
    for r, c in coords:
        offsets.append((int(r - r0 - cr), int(c - c0 - cc)))
    offsets.sort()
    return tuple(offsets)


def _parse_legend(legend: np.ndarray) -> dict[int, tuple[tuple[int, int], ...]]:
    orange_comps = _components(legend == ORANGE)
    azure_comps = _components(legend == AZURE)
    assert orange_comps and len(orange_comps) == len(azure_comps), "legend rows must pair digits with shapes"

    azure_entries = []
    for comp in azure_comps:
        r0, c0, r1, c1 = _bbox(comp)
        azure_entries.append({"bbox": (r0, c0, r1, c1), "coords": comp})

    mapping: dict[int, tuple[tuple[int, int], ...]] = {}
    used_azure: set[int] = set()
    for comp in sorted(orange_comps, key=lambda item: _bbox(item)[0]):
        r0, c0, r1, c1 = _bbox(comp)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        patch[comp[:, 0] - r0, comp[:, 1] - c0] = True
        value = _digit_value_from_patch(patch)

        overlaps: list[tuple[int, int]] = []
        for idx, entry in enumerate(azure_entries):
            ar0, _ac0, ar1, _ac1 = entry["bbox"]
            overlap = min(r1, ar1) - max(r0, ar0) + 1
            if overlap > 0:
                overlaps.append((int(overlap), int(idx)))
        assert len(overlaps) == 1, "each digit row must match exactly one shape row"
        idx = overlaps[0][1]
        assert idx not in used_azure, "shape rows must be unique"
        used_azure.add(idx)
        mapping[value] = _shape_offsets_from_component(azure_entries[idx]["coords"])

    assert mapping and set(mapping) == set(range(1, max(mapping) + 1)), "legend values must be contiguous from 1"
    return mapping


def _find_separator(grid: np.ndarray) -> int:
    cols = np.where(np.all(grid == GREY, axis=0))[0]
    assert cols.size, "missing grey separator column"
    return int(cols[0])


def _visible_same_color(base: np.ndarray, r: int, c: int, color: int) -> int:
    h, w = base.shape
    total = 0
    for dr, dc in DIRS:
        rr, cc = int(r + dr), int(c + dc)
        while 0 <= rr < h and 0 <= cc < w:
            if int(base[rr, cc]) == GREY:
                break
            if int(base[rr, cc]) == int(color):
                total += 1
            rr += dr
            cc += dc
    return int(total)


def _paint_centered(out: np.ndarray, r: int, c: int, offsets: tuple[tuple[int, int], ...], color: int) -> None:
    h, w = out.shape
    for dr, dc in offsets:
        rr, cc = int(r + dr), int(c + dc)
        if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) != GREY:
            out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep = _find_separator(g)
    legend = g[:, :sep]
    base = g[:, sep + 1 :].copy()
    mapping = _parse_legend(legend)
    max_value = int(max(mapping))

    out = np.full_like(base, BLACK)
    out[base == GREY] = GREY

    piece_coords = np.argwhere((base != BLACK) & (base != GREY))
    for r_raw, c_raw in piece_coords:
        r, c = int(r_raw), int(c_raw)
        color = int(base[r, c])
        visible = _visible_same_color(base, r, c, color)
        if visible == 0:
            continue
        if visible > max_value:
            offsets = mapping[max_value]
        else:
            offsets = mapping[visible]
        _paint_centered(out, r, c, offsets, color)

    return out


def _shape_patch(offsets: tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = [dr for dr, _dc in offsets]
    cols = [dc for _dr, dc in offsets]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for dr, dc in offsets:
        patch[dr - r0, dc - c0] = True
    return patch


def _scenario_isolated() -> dict[str, object]:
    return {
        "pieces": [(0, 0)],
        "walls": [],
        "height": 1,
        "width": 1,
        "expected_counts": [0],
    }


def _scenario_open_line(count: int, orientation: str) -> dict[str, object]:
    step = 4
    n = int(count + 1)
    if orientation == "h":
        pieces = [(0, i * step) for i in range(n)]
        height, width = 1, 1 + step * (n - 1)
    else:
        pieces = [(i * step, 0) for i in range(n)]
        height, width = 1 + step * (n - 1), 1
    return {
        "pieces": pieces,
        "walls": [],
        "height": int(height),
        "width": int(width),
        "expected_counts": [int(count)] * n,
    }


def _scenario_blocked(seg_a: int, seg_b: int, orientation: str) -> dict[str, object]:
    step = 4
    wall_h = int(np.random.randint(1, 4))
    wall_w = int(np.random.randint(1, 4))

    if orientation == "h":
        row = wall_h // 2
        last_left = int((seg_a - 1) * step)
        right_start = int(seg_a * step)
        wall_center_c = int(last_left + 2)
        wall_left = int(wall_center_c - wall_w // 2)
        pieces = [(row, i * step) for i in range(seg_a)] + [
            (row, right_start + i * step) for i in range(seg_b)
        ]
        walls = [(rr, cc) for rr in range(wall_h) for cc in range(wall_left, wall_left + wall_w)]
        height = max(wall_h, row + 1)
        width = max(wall_left + wall_w, right_start + (seg_b - 1) * step + 1)
    else:
        col = wall_w // 2
        last_top = int((seg_a - 1) * step)
        lower_start = int(seg_a * step)
        wall_center_r = int(last_top + 2)
        wall_top = int(wall_center_r - wall_h // 2)
        pieces = [(i * step, col) for i in range(seg_a)] + [
            (lower_start + i * step, col) for i in range(seg_b)
        ]
        walls = [(rr, cc) for rr in range(wall_top, wall_top + wall_h) for cc in range(wall_w)]
        height = max(wall_top + wall_h, lower_start + (seg_b - 1) * step + 1)
        width = max(wall_w, col + 1)

    return {
        "pieces": pieces,
        "walls": walls,
        "height": int(height),
        "width": int(width),
        "expected_counts": [int(seg_a - 1)] * int(seg_a) + [int(seg_b - 1)] * int(seg_b),
    }


def _build_scenario(kind: str, a: int = 0, b: int = 0, h: int = 0, w: int = 0) -> dict[str, object] | None:
    orientations = list(np.random.permutation(["h", "v"]))
    if kind == "isolated":
        return _scenario_isolated()
    for orientation in orientations:
        scenario = _scenario_open_line(a, orientation) if kind == "open" else _scenario_blocked(a, b, orientation)
        if int(scenario["height"]) <= h and int(scenario["width"]) <= w:
            return scenario
    return None


def _expanded_slices(r0: int, c0: int, h: int, w: int, grid_h: int, grid_w: int, pad: int = 2) -> tuple[slice, slice]:
    return slice(max(0, r0 - pad), min(grid_h, r0 + h + pad)), slice(max(0, c0 - pad), min(grid_w, c0 + w + pad))


def _place_scenario(
    canvas: np.ndarray,
    reserved: np.ndarray,
    scenario: dict[str, object],
    color: int,
    *,
    edge_bias: bool,
) -> list[tuple[int, int]] | None:
    h, w = canvas.shape
    sh, sw = int(scenario["height"]), int(scenario["width"])
    assert sh <= h and sw <= w

    for _ in range(200):
        if edge_bias and np.random.rand() < 0.65:
            if np.random.rand() < 0.5:
                r0 = int(np.random.choice([0, h - sh]))
                c0 = int(np.random.randint(0, w - sw + 1))
            else:
                r0 = int(np.random.randint(0, h - sh + 1))
                c0 = int(np.random.choice([0, w - sw]))
        else:
            r0 = int(np.random.randint(0, h - sh + 1))
            c0 = int(np.random.randint(0, w - sw + 1))

        rs, cs = _expanded_slices(r0, c0, sh, sw, h, w)
        if bool(reserved[rs, cs].any()):
            continue

        ok = True
        for dr, dc in scenario["walls"]:
            if int(canvas[r0 + int(dr), c0 + int(dc)]) != BLACK:
                ok = False
                break
        if not ok:
            continue
        for dr, dc in scenario["pieces"]:
            if int(canvas[r0 + int(dr), c0 + int(dc)]) != BLACK:
                ok = False
                break
        if not ok:
            continue

        placed: list[tuple[int, int]] = []
        for dr, dc in scenario["walls"]:
            canvas[r0 + int(dr), c0 + int(dc)] = GREY
        for dr, dc in scenario["pieces"]:
            rr, cc = int(r0 + int(dr)), int(c0 + int(dc))
            canvas[rr, cc] = int(color)
            placed.append((rr, cc))
        reserved[rs, cs] = True
        return placed

    return None


def _place_noise_wall(canvas: np.ndarray, reserved: np.ndarray) -> bool:
    h, w = canvas.shape
    wall_h = int(np.random.randint(1, 4))
    wall_w = int(np.random.randint(1, 4))
    for _ in range(120):
        r0 = int(np.random.randint(0, h - wall_h + 1))
        c0 = int(np.random.randint(0, w - wall_w + 1))
        rs, cs = _expanded_slices(r0, c0, wall_h, wall_w, h, w, pad=1)
        if bool(reserved[rs, cs].any()):
            continue
        if bool((canvas[r0 : r0 + wall_h, c0 : c0 + wall_w] != BLACK).any()):
            continue
        canvas[r0 : r0 + wall_h, c0 : c0 + wall_w] = GREY
        reserved[rs, cs] = True
        return True
    return False


def _compute_piece_counts(base: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for r_raw, c_raw in np.argwhere((base != BLACK) & (base != GREY)):
        r, c = int(r_raw), int(c_raw)
        counts[(r, c)] = _visible_same_color(base, r, c, int(base[r, c]))
    return counts


def _paint_legend(legend: np.ndarray, mapping: dict[int, tuple[tuple[int, int], ...]]) -> None:
    row_h = 5
    row_gap = 1
    used_h = len(mapping) * row_h + (len(mapping) - 1) * row_gap
    top = int(np.random.randint(0, legend.shape[0] - used_h + 1))
    digit_x = int(np.random.randint(0, 2))
    shape_x = int(np.random.randint(6, 8))

    order = list(np.random.permutation(sorted(mapping)))
    for idx, value in enumerate(order):
        band_r = int(top + idx * (row_h + row_gap))
        digit = DIGIT_TEMPLATES[int(value)]
        legend[band_r : band_r + digit.shape[0], digit_x : digit_x + digit.shape[1]][digit] = ORANGE

        shape = _shape_patch(mapping[int(value)])
        sr = int(band_r + (row_h - shape.shape[0]) // 2)
        legend[sr : sr + shape.shape[0], shape_x : shape_x + shape.shape[1]][shape] = AZURE


def _gen_attempt() -> np.ndarray | None:
    max_value = int(np.random.randint(2, 5))
    shape_order = list(np.random.permutation(len(SHAPE_LIBRARY))[:max_value])
    mapping = {value: SHAPE_LIBRARY[shape_order[value - 1]] for value in range(1, max_value + 1)}

    legend_w = 10
    main_h = int(np.random.randint(max(24, 6 * max_value - 1), max(24, 6 * max_value - 1) + 8))
    main_w = int(np.random.randint(20, 27))
    main_area = np.full((main_h, main_w), BLACK, dtype=int)
    reserved = np.zeros_like(main_area, dtype=bool)

    high_count = int(np.random.randint(max_value + 1, 6))
    open_low = int(np.random.randint(1, max_value + 1))
    open_mid = int(np.random.randint(1, min(4, max_value + 1)))
    blocked_left = int(np.random.randint(2, 5))
    blocked_right = 1 if np.random.rand() < 0.7 else 2

    specs: list[tuple[str, int, int]] = [
        ("isolated", 0, 0),
        ("blocked", blocked_left, blocked_right),
        ("open", open_low, 0),
        ("open", high_count, 0),
    ]
    if np.random.rand() < 0.6:
        if np.random.rand() < 0.5:
            specs.append(("open", open_mid, 0))
        else:
            specs.append(("blocked", int(np.random.randint(2, 4)), 1))

    if len(specs) > len(PIECE_COLORS):
        return None

    colors = list(np.random.permutation(PIECE_COLORS)[: len(specs)])
    groups: list[dict[str, object]] = []
    for idx, (kind, a, b) in enumerate(specs):
        scenario = _build_scenario(kind, a=a, b=b, h=main_h, w=main_w)
        if scenario is None:
            return None
        placed = _place_scenario(
            main_area,
            reserved,
            scenario,
            colors[idx],
            edge_bias=bool(idx == 0 or (kind == "open" and np.random.rand() < 0.35)),
        )
        if placed is None:
            return None
        groups.append(
            {
                "color": int(colors[idx]),
                "coords": placed,
                "expected_counts": list(scenario["expected_counts"]),
                "kind": kind,
            }
        )

    extra_walls = int(np.random.randint(1, 4))
    for _ in range(extra_walls):
        _place_noise_wall(main_area, reserved)

    counts = _compute_piece_counts(main_area)
    all_counts = list(counts.values())
    if not all_counts or 0 not in all_counts or not any(value > 0 for value in all_counts):
        return None
    if not any(value > max_value for value in all_counts):
        return None

    for group in groups:
        actual = sorted(counts[tuple(coord)] for coord in group["coords"])
        expected = sorted(int(v) for v in group["expected_counts"])
        if actual != expected:
            return None

    full = np.full((main_h, legend_w + 1 + main_w), BLACK, dtype=int)
    full[:, legend_w] = GREY
    _paint_legend(full[:, :legend_w], mapping)
    full[:, legend_w + 1 :] = main_area

    try:
        out = main(full)
    except Exception:
        return None

    if out.shape != main_area.shape:
        return None
    if not bool((out == GREY).any()):
        return None
    if not bool((out != BLACK).any()):
        return None
    return full


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
