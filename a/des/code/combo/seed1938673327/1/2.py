from __future__ import annotations

import numpy as np


BG = 0

SYMMETRY_TYPES = ("horizontal", "vertical", "rotation", "none")
GROUP_SLOT = {
    "horizontal": (0, 0),
    "vertical": (0, 1),
    "rotation": (1, 0),
    "none": (1, 1),
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c, h, w):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _symmetry_flags(mask: np.ndarray) -> tuple[bool, bool, bool]:
    m = np.asarray(mask, dtype=bool)
    horizontal = bool(np.array_equal(m, np.flipud(m)))
    vertical = bool(np.array_equal(m, np.fliplr(m)))
    rotation = bool(np.array_equal(m, np.rot90(m, 2)))
    return horizontal, vertical, rotation


def _classify_mask(mask: np.ndarray) -> str:
    horizontal, vertical, rotation = _symmetry_flags(mask)
    if horizontal and not vertical:
        return "horizontal"
    if vertical and not horizontal:
        return "vertical"
    if rotation and not horizontal and not vertical:
        return "rotation"
    if not horizontal and not vertical and not rotation:
        return "none"
    raise AssertionError("mask does not have exactly one supported symmetry type")


def _majority_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    assert colors.size > 0, "expected at least one foreground color"
    best_idx = int(np.argmax(counts))
    assert int(np.sum(counts == counts[best_idx])) == 1, "dominant color must be unique"
    return int(colors[best_idx])


def _extract_runs(is_sep: np.ndarray) -> tuple[list[tuple[int, int]], int]:
    bands: list[tuple[int, int]] = []
    sep_widths: list[int] = []
    n = int(is_sep.size)
    i = 0
    while i < n:
        j = i
        while j < n and bool(is_sep[j]) == bool(is_sep[i]):
            j += 1
        if is_sep[i]:
            sep_widths.append(j - i)
        else:
            bands.append((i, j))
        i = j
    assert bands and sep_widths, "expected both panels and separators"
    assert not bool(is_sep[0]) and not bool(is_sep[-1]), "outer border separators are not supported"
    assert len(set(sep_widths)) == 1, "separator thickness must be uniform"
    return bands, sep_widths[0]


def _parse_panel_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int, int]:
    g = np.asarray(grid, dtype=int)
    row_candidates = {int(g[r, 0]) for r in range(g.shape[0]) if bool(np.all(g[r, :] == g[r, 0])) and int(g[r, 0]) != BG}
    col_candidates = {int(g[0, c]) for c in range(g.shape[1]) if bool(np.all(g[:, c] == g[0, c])) and int(g[0, c]) != BG}
    separator_candidates = row_candidates & col_candidates
    assert len(separator_candidates) == 1, "expected a unique separator color"
    separator_color = int(next(iter(separator_candidates)))

    row_is_sep = np.all(g == separator_color, axis=1)
    col_is_sep = np.all(g == separator_color, axis=0)
    row_bands, row_sep = _extract_runs(row_is_sep)
    col_bands, col_sep = _extract_runs(col_is_sep)
    assert row_sep == col_sep, "separator thickness must match in both directions"

    panel_sizes = {r1 - r0 for r0, r1 in row_bands} | {c1 - c0 for c0, c1 in col_bands}
    assert len(panel_sizes) == 1, "all panels must have the same square size"
    panel_size = int(next(iter(panel_sizes)))
    return separator_color, row_bands, col_bands, panel_size, row_sep


def _extract_object(panel: np.ndarray) -> dict[str, object]:
    mask = np.asarray(panel != BG, dtype=bool)
    assert bool(mask.any()), "each panel must contain one foreground object"
    assert _is_connected(mask), "foreground object must be 4-connected"
    tight_mask = _crop_mask(mask)
    values = np.asarray(panel[mask], dtype=int)
    return {
        "mask": tight_mask,
        "area": int(tight_mask.sum()),
        "color": _majority_color(values),
        "kind": _classify_mask(tight_mask),
    }


def _render_grouped_output(
    groups: dict[str, list[dict[str, object]]],
    panel_size: int,
    sep_width: int,
    separator_color: int,
) -> np.ndarray:
    side = 2 * panel_size + sep_width
    out = np.full((side, side), BG, dtype=int)
    out[panel_size : panel_size + sep_width, :] = separator_color
    out[:, panel_size : panel_size + sep_width] = separator_color

    for kind, (slot_r, slot_c) in GROUP_SLOT.items():
        items = groups[kind]
        if not items:
            continue
        union = np.zeros((panel_size, panel_size), dtype=bool)
        chosen = max(items, key=lambda item: int(item["area"]))
        for item in items:
            mask = np.asarray(item["mask"], dtype=bool)
            h, w = mask.shape
            union[:h, :w] |= mask
        panel = np.full((panel_size, panel_size), BG, dtype=int)
        panel[union] = int(chosen["color"])
        r0 = slot_r * (panel_size + sep_width)
        c0 = slot_c * (panel_size + sep_width)
        out[r0 : r0 + panel_size, c0 : c0 + panel_size] = panel

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group panel sprites by symmetry type and merge each group's aligned silhouettes.

    Input structure
    - The grid is partitioned by gray separator lines into equal square panels.
    - Each panel contains one 4-connected multicolor foreground object on a black background.
    - Each object's binary mask has exactly one of four symmetry types:
      horizontal, vertical, 180-degree rotation, or none.

    Rule
    - Classify each panel object by the symmetry of its cropped binary mask.
    - Send horizontal / vertical / rotational / asymmetric objects to the
      top-left / top-right / bottom-left / bottom-right output panel.
    - Within each group, align all cropped masks to the top-left corner and
      take their union.
    - Paint the merged mask using the dominant color of the largest-area object
      in that group, while keeping the separator lines gray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator_color, row_bands, col_bands, panel_size, sep_width = _parse_panel_layout(g)
    groups = {kind: [] for kind in SYMMETRY_TYPES}

    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            panel = g[r0:r1, c0:c1]
            obj = _extract_object(panel)
            groups[str(obj["kind"])].append(obj)

    return _render_grouped_output(groups, panel_size, sep_width, separator_color)


# --------------- generator ---------------

RNG = np.random.default_rng()

GEN_SEPARATOR_COLOR = 5
GEN_OBJECT_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_PANEL_SIZE_RANGE = (7, 11)  # inclusive-low, exclusive-high
GEN_PANEL_ROWS_RANGE = (2, 5)
GEN_PANEL_COLS_RANGE = (2, 5)
GEN_SEPARATOR_WIDTHS = (1, 2)
GEN_MIN_AREA = 5
GEN_MAX_AREA = 18
GEN_MAX_RETRIES = 400
GEN_MASK_RETRIES = 300


def _orbit_cells(kind: str, h: int, w: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    if kind == "horizontal":
        cells = {(r, c), (h - 1 - r, c)}
    elif kind == "vertical":
        cells = {(r, c), (r, w - 1 - c)}
    elif kind == "rotation":
        cells = {(r, c), (h - 1 - r, w - 1 - c)}
    else:
        cells = {(r, c)}
    return tuple(sorted(cells))


def _seed_options(kind: str, h: int, w: int) -> list[tuple[int, int]]:
    if kind == "horizontal":
        r = h // 2 if h % 2 == 1 else h // 2 - 1
        return [(r, c) for c in range(w)]
    if kind == "vertical":
        c = w // 2 if w % 2 == 1 else w // 2 - 1
        return [(r, c) for r in range(h)]
    if kind == "rotation":
        return [(h // 2, w // 2)]
    return [(h // 2, w // 2)]


def _choose_dims(kind: str, panel_size: int) -> tuple[int, int] | None:
    max_side = panel_size - 2
    if max_side < 5:
        return None
    if kind == "rotation":
        odd_choices = [size for size in range(5, max_side + 1) if size % 2 == 1]
        if not odd_choices:
            return None
        h = int(odd_choices[int(RNG.integers(len(odd_choices)))])
        w = int(odd_choices[int(RNG.integers(len(odd_choices)))])
        return h, w
    if kind == "horizontal":
        return int(RNG.integers(4, max_side + 1)), int(RNG.integers(3, max_side + 1))
    if kind == "vertical":
        return int(RNG.integers(3, max_side + 1)), int(RNG.integers(4, max_side + 1))
    return int(RNG.integers(3, max_side + 1)), int(RNG.integers(3, max_side + 1))


def _grow_mask(kind: str, h: int, w: int, target_area: int) -> np.ndarray:
    seeds = _seed_options(kind, h, w)
    seed = seeds[int(RNG.integers(len(seeds)))]
    cells = set(_orbit_cells(kind, h, w, seed[0], seed[1]))

    while len(cells) < target_area:
        choices: list[list[tuple[int, int]]] = []
        seen: set[tuple[tuple[int, int], ...]] = set()
        for r, c in tuple(cells):
            for rr, cc in _neighbors4(r, c, h, w):
                orbit = _orbit_cells(kind, h, w, rr, cc)
                if orbit in seen:
                    continue
                seen.add(orbit)
                new_cells = [cell for cell in orbit if cell not in cells]
                if not new_cells:
                    continue
                if len(cells) + len(new_cells) > target_area:
                    continue
                choices.append(new_cells)
        if not choices:
            break
        new_cells = choices[int(RNG.integers(len(choices)))]
        cells.update(new_cells)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _crop_mask(mask)


def _generate_mask(kind: str, panel_size: int) -> np.ndarray:
    for _ in range(GEN_MASK_RETRIES):
        dims = _choose_dims(kind, panel_size)
        if dims is None:
            continue
        h, w = dims
        max_area = min(GEN_MAX_AREA, h * w - 1)
        if max_area < GEN_MIN_AREA:
            continue
        target_area = int(RNG.integers(GEN_MIN_AREA, max_area + 1))
        mask = _grow_mask(kind, h, w, target_area)
        area = int(mask.sum())
        if area < GEN_MIN_AREA or area == int(mask.size):
            continue
        if not _is_connected(mask):
            continue
        try:
            actual_kind = _classify_mask(mask)
        except AssertionError:
            continue
        if actual_kind != kind:
            continue
        return mask
    raise RuntimeError(f"failed to build a {kind} mask")


def _colorize_mask(mask: np.ndarray, base_color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(base_color)

    coords = np.argwhere(mask)
    area = int(coords.shape[0])
    max_accent = min(3, area // 3)
    if max_accent <= 0:
        return patch

    accent_total = int(RNG.integers(1, max_accent + 1))
    accent_palette = [int(color) for color in GEN_OBJECT_COLORS if int(color) != int(base_color)]
    accent_color_count = 1 if accent_total == 1 else int(RNG.integers(1, min(2, accent_total) + 1))
    chosen_colors = list(map(int, RNG.choice(accent_palette, size=accent_color_count, replace=False)))
    chosen_idx = RNG.choice(area, size=accent_total, replace=False)

    if accent_color_count == 1:
        splits = [accent_total]
    else:
        split = int(RNG.integers(1, accent_total))
        splits = [split, accent_total - split]

    start = 0
    for color, count in zip(chosen_colors, splits):
        for idx in chosen_idx[start : start + count]:
            r, c = map(int, coords[int(idx)])
            patch[r, c] = color
        start += count

    return patch


def _generate_object_spec(kind: str, panel_size: int, used_areas: set[int]) -> dict[str, object]:
    for _ in range(GEN_MASK_RETRIES):
        mask = _generate_mask(kind, panel_size)
        area = int(mask.sum())
        if area in used_areas:
            continue
        base_color = int(RNG.choice(GEN_OBJECT_COLORS))
        patch = _colorize_mask(mask, base_color)
        return {
            "kind": kind,
            "mask": mask,
            "area": area,
            "color": base_color,
            "patch": patch,
        }
    raise RuntimeError(f"failed to build a unique-area {kind} object")


def _inner_offset(limit: int, span: int) -> int:
    slack = int(limit - span)
    options = list(range(slack + 1))
    if len(options) > 2:
        options = options[1:-1]
    if not options:
        options = list(range(slack + 1))
    return int(options[int(RNG.integers(len(options)))])


def _place_in_panel(panel_size: int, patch: np.ndarray) -> np.ndarray:
    h, w = patch.shape
    panel = np.full((panel_size, panel_size), BG, dtype=int)
    r0 = _inner_offset(panel_size, h)
    c0 = _inner_offset(panel_size, w)
    fg = patch != BG
    panel[r0 : r0 + h, c0 : c0 + w][fg] = patch[fg]
    return panel


def _compose_input(
    panel_rows: int,
    panel_cols: int,
    panel_size: int,
    sep_width: int,
    specs: list[dict[str, object]],
) -> np.ndarray:
    height = panel_rows * panel_size + (panel_rows - 1) * sep_width
    width = panel_cols * panel_size + (panel_cols - 1) * sep_width
    grid = np.full((height, width), BG, dtype=int)

    for r in range(1, panel_rows):
        rr = r * panel_size + (r - 1) * sep_width
        grid[rr : rr + sep_width, :] = GEN_SEPARATOR_COLOR
    for c in range(1, panel_cols):
        cc = c * panel_size + (c - 1) * sep_width
        grid[:, cc : cc + sep_width] = GEN_SEPARATOR_COLOR

    for idx, spec in enumerate(specs):
        panel = _place_in_panel(panel_size, np.asarray(spec["patch"], dtype=int))
        row = idx // panel_cols
        col = idx % panel_cols
        r0 = row * (panel_size + sep_width)
        c0 = col * (panel_size + sep_width)
        grid[r0 : r0 + panel_size, c0 : c0 + panel_size] = panel

    return grid


def _expected_output(specs: list[dict[str, object]], panel_size: int, sep_width: int) -> np.ndarray:
    groups = {kind: [] for kind in SYMMETRY_TYPES}
    for spec in specs:
        groups[str(spec["kind"])].append(
            {
                "mask": np.asarray(spec["mask"], dtype=bool),
                "area": int(spec["area"]),
                "color": int(spec["color"]),
            }
        )
    return _render_grouped_output(groups, panel_size, sep_width, GEN_SEPARATOR_COLOR)


def _sample_panel_layout() -> tuple[int, int, int, int]:
    panel_size = int(RNG.integers(*GEN_PANEL_SIZE_RANGE))
    panel_rows = int(RNG.integers(*GEN_PANEL_ROWS_RANGE))
    panel_cols = int(RNG.integers(*GEN_PANEL_COLS_RANGE))
    sep_width = int(GEN_SEPARATOR_WIDTHS[int(RNG.integers(len(GEN_SEPARATOR_WIDTHS)))])
    return panel_rows, panel_cols, panel_size, sep_width


def _gen_attempt() -> np.ndarray | None:
    panel_rows, panel_cols, panel_size, sep_width = _sample_panel_layout()
    panel_count = panel_rows * panel_cols
    if panel_count < len(SYMMETRY_TYPES):
        return None

    kinds = list(SYMMETRY_TYPES)
    extra_count = panel_count - len(kinds)
    if extra_count > 0:
        extra_kinds = RNG.choice(np.array(SYMMETRY_TYPES, dtype=object), size=extra_count, replace=True)
        kinds.extend(map(str, extra_kinds))
    RNG.shuffle(kinds)

    used_areas = {kind: set() for kind in SYMMETRY_TYPES}
    specs: list[dict[str, object]] = []
    for kind in kinds:
        spec = _generate_object_spec(kind, panel_size, used_areas[kind])
        used_areas[kind].add(int(spec["area"]))
        specs.append(spec)

    inp = _compose_input(panel_rows, panel_cols, panel_size, sep_width, specs)
    expected = _expected_output(specs, panel_size, sep_width)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
