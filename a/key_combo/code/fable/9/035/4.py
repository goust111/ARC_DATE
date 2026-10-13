from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


_BG = 0
_TEMPLATE_COLOR = 5
_PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

_ALL_TRANSFORMS = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "transpose",
    "anti",
)
_ROTATION_TRANSFORMS = ("rot90", "rot180", "rot270")
_MIRROR_TRANSFORMS = ("flip_lr", "flip_ud", "transpose", "anti")

_TEMPLATE_SIZES = (4, 5)
_LATTICE_DIMS = ((2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3))
_CELL_GAPS = (1, 2)
_LEGEND_COUNTS = (3, 4)
_LEGEND_GAPS = (1, 2)
_BORDERS = (1, 2)
_REGION_GAPS = (1, 2)
_EXTRA_SLACK = 2
_MAX_SIDE = 30
_GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    colors: list[int] | tuple[int, ...] | None = None,
) -> list[dict]:
    h, w = grid.shape
    allowed = None if colors is None else set(map(int, colors))
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            if allowed is not None and color not in allowed:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = component["coords"] - np.array([[r0, c0]], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    coords = tuple(tuple(map(int, rc)) for rc in np.argwhere(mask))
    return tuple(map(int, mask.shape)), coords


def _apply_transform(mask: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return mask.copy()
    if name == "rot90":
        return np.rot90(mask, 1)
    if name == "rot180":
        return np.rot90(mask, 2)
    if name == "rot270":
        return np.rot90(mask, 3)
    if name == "flip_lr":
        return np.fliplr(mask)
    if name == "flip_ud":
        return np.flipud(mask)
    if name == "transpose":
        return mask.T.copy()
    if name == "anti":
        return np.fliplr(np.rot90(mask, 1))
    raise ValueError(f"unknown transform: {name}")


def _transform_catalog(mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: _apply_transform(mask, name) for name in _ALL_TRANSFORMS}


def _match_transform(template_mask: np.ndarray, candidate_mask: np.ndarray) -> tuple[str, np.ndarray] | None:
    matches = [
        (name, transformed)
        for name, transformed in _transform_catalog(template_mask).items()
        if transformed.shape == candidate_mask.shape and np.array_equal(transformed, candidate_mask)
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _regular_lattice(
    components: list[dict],
) -> tuple[list[int], list[int], dict[tuple[int, int], dict], np.ndarray] | None:
    if len(components) < 4:
        return None

    heights = set()
    widths = set()
    signatures = set()
    by_anchor: dict[tuple[int, int], dict] = {}
    for component in components:
        r0, c0, r1, c1 = component["bbox"]
        heights.add(r1 - r0 + 1)
        widths.add(c1 - c0 + 1)
        signatures.add(_mask_signature(_component_mask(component)))
        by_anchor[(r0, c0)] = component

    if len(heights) != 1 or len(widths) != 1 or len(signatures) != 1:
        return None

    cell_h = heights.pop()
    cell_w = widths.pop()
    if cell_h < 2 or cell_w < 2:
        return None

    rows = sorted({r for r, _ in by_anchor})
    cols = sorted({c for _, c in by_anchor})
    if len(rows) < 2 or len(cols) < 2:
        return None
    if len(rows) * len(cols) != len(components):
        return None
    if any((r, c) not in by_anchor for r in rows for c in cols):
        return None

    row_steps = {rows[i + 1] - rows[i] for i in range(len(rows) - 1)}
    col_steps = {cols[i + 1] - cols[i] for i in range(len(cols) - 1)}
    if len(row_steps) != 1 or len(col_steps) != 1:
        return None
    if next(iter(row_steps)) <= cell_h or next(iter(col_steps)) <= cell_w:
        return None

    template_mask = _component_mask(components[0])
    return rows, cols, by_anchor, template_mask


def _find_control_and_legends(
    grid: np.ndarray,
    *,
    bg: int,
    template_color: int,
    lattice_rows: list[int],
    lattice_cols: list[int],
    template_mask: np.ndarray,
) -> tuple[int, int, dict[int, np.ndarray], dict[int, str]] | None:
    h, w = grid.shape
    rows_n = len(lattice_rows)
    cols_n = len(lattice_cols)
    candidates: list[tuple[int, int, dict[int, np.ndarray], dict[int, str]]] = []

    for r0 in range(h - rows_n + 1):
        for c0 in range(w - cols_n + 1):
            patch = grid[r0 : r0 + rows_n, c0 : c0 + cols_n]
            if np.any((patch == bg) | (patch == template_color)):
                continue
            colors = sorted(set(map(int, patch.ravel())))
            if len(colors) < 2:
                continue

            masked = grid.copy()
            masked[r0 : r0 + rows_n, c0 : c0 + cols_n] = bg
            components = _find_components(masked, bg=bg)
            legends = [component for component in components if component["color"] != template_color]
            if {component["color"] for component in legends} != set(colors):
                continue
            if len(legends) != len(colors):
                continue

            legend_masks: dict[int, np.ndarray] = {}
            legend_names: dict[int, str] = {}
            pose_signatures: set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = set()
            ok = True
            for component in legends:
                match = _match_transform(template_mask, _component_mask(component))
                if match is None:
                    ok = False
                    break
                name, transformed = match
                sig = _mask_signature(transformed)
                if sig in pose_signatures:
                    ok = False
                    break
                pose_signatures.add(sig)
                legend_masks[int(component["color"])] = transformed
                legend_names[int(component["color"])] = name

            if ok:
                candidates.append((r0, c0, legend_masks, legend_names))

    if len(candidates) != 1:
        return None
    return candidates[0]


def _parse_scene(grid: np.ndarray) -> dict:
    bg = _get_bg(grid)
    candidates = []
    for color in _colors_present(grid, exclude={bg}):
        components = _find_components(grid, bg=bg, colors=[color])
        lattice = _regular_lattice(components)
        if lattice is None:
            continue

        rows, cols, by_anchor, template_mask = lattice
        parsed = _find_control_and_legends(
            grid,
            bg=bg,
            template_color=color,
            lattice_rows=rows,
            lattice_cols=cols,
            template_mask=template_mask,
        )
        if parsed is None:
            continue

        control_r, control_c, legend_masks, legend_names = parsed
        candidates.append(
            {
                "bg": bg,
                "template_color": color,
                "rows": rows,
                "cols": cols,
                "by_anchor": by_anchor,
                "template_mask": template_mask,
                "control_r": control_r,
                "control_c": control_c,
                "legend_masks": legend_masks,
                "legend_names": legend_names,
            }
        )

    assert len(candidates) == 1, "expected exactly one lattice/control/legend interpretation"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a control matrix to recolor a gray lattice, then redraw each cell in the pose shown by the same-colored legend sprite.

    Input structure
    - A background canvas contains a regular lattice of identical gray sprite copies, all in the same pose and separated by background gaps.
    - A dense colored control matrix has the same row and column counts as that lattice.
    - Elsewhere, one colored legend sprite appears for every control color; each legend is the same base sprite under a distinct rotation or mirror pose.

    Rule
    - Match lattice slot (i, j) with control-matrix cell (i, j) to read a color.
    - Find the legend sprite of that color, clear the gray cell's original bbox, and redraw the sprite inside that bbox using the legend's pose and the matched color.
    - Leave the control matrix, legend sprites, background, and lattice spacing unchanged.

    Key insight
    - The control matrix does not directly tell you the geometry: it only selects a color, and that color points to a legend whose visible pose is the real transform to apply.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    scene = _parse_scene(g)
    out = g.copy()
    bg = int(scene["bg"])
    control_r = int(scene["control_r"])
    control_c = int(scene["control_c"])

    for i, row in enumerate(scene["rows"]):
        for j, col in enumerate(scene["cols"]):
            color = int(g[control_r + i, control_c + j])
            pose_mask = scene["legend_masks"][color]
            component = scene["by_anchor"][(row, col)]
            r0, c0, r1, c1 = component["bbox"]
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            assert patch.shape == pose_mask.shape, "legend pose must fit the template bbox"
            patch[:] = bg
            patch[pose_mask] = color

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _generate_template_mask(size: int) -> np.ndarray | None:
    target_area = random.randint(size + 2, size * size - 4)
    start = (random.randrange(size), random.randrange(size))
    cells = {start}

    while len(cells) < target_area:
        frontier = []
        for r, c in list(cells):
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    if not _mask_connected(mask):
        return None
    if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return None
    if np.all(mask):
        return None
    if size > 2 and not np.any(mask[1:-1, 1:-1]):
        return None

    transforms = {_mask_signature(_apply_transform(mask, name)) for name in _ALL_TRANSFORMS}
    if len(transforms) != len(_ALL_TRANSFORMS):
        return None
    return mask


def _choose_transforms(count: int) -> list[str]:
    selected = {
        random.choice(_ROTATION_TRANSFORMS),
        random.choice(_MIRROR_TRANSFORMS),
    }
    while len(selected) < count:
        selected.add(random.choice(_ALL_TRANSFORMS))
    return random.sample(sorted(selected), k=count)


def _generate_control(rows: int, cols: int, colors: list[int]) -> np.ndarray:
    total = rows * cols
    while True:
        flat = colors + random.choices(colors, k=total - len(colors))
        random.shuffle(flat)
        control = np.array(flat, dtype=int).reshape(rows, cols)
        if len(set(map(int, control[0, :]))) == 1 and len(set(map(int, control[:, 0]))) == 1:
            continue
        if len(set(map(int, control.ravel()))) != len(colors):
            continue
        return control


def _legend_line_size(size: int, count: int, gap: int, horizontal: bool) -> tuple[int, int]:
    if horizontal:
        return size, count * size + (count - 1) * gap
    return count * size + (count - 1) * gap, size


def _place_layout(
    lattice_h: int,
    lattice_w: int,
    control_h: int,
    control_w: int,
    legend_h: int,
    legend_w: int,
    *,
    layout: str,
    border: int,
    gap_a: int,
    gap_b: int,
) -> tuple[int, int, tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    if layout in {"bottom_band", "top_band"}:
        content_h = lattice_h + gap_a + max(control_h, legend_h)
        content_w = max(lattice_w, control_w + gap_b + legend_w)
    else:
        content_h = max(lattice_h, control_h + gap_b + legend_h)
        content_w = lattice_w + gap_a + max(control_w, legend_w)

    base_h = content_h + 2 * border
    base_w = content_w + 2 * border
    if base_h > _MAX_SIDE or base_w > _MAX_SIDE:
        return None

    height = random.randint(base_h, min(_MAX_SIDE, base_h + _EXTRA_SLACK))
    width = random.randint(base_w, min(_MAX_SIDE, base_w + _EXTRA_SLACK))
    content_r = border + random.randint(0, height - base_h)
    content_c = border + random.randint(0, width - base_w)

    if layout == "bottom_band":
        lattice_r = content_r
        lattice_c = content_c + random.randint(0, content_w - lattice_w)
        band_r = content_r + lattice_h + gap_a
        band_h = max(control_h, legend_h)
        band_c = content_c + random.randint(0, content_w - (control_w + gap_b + legend_w))
        control_r = band_r + random.randint(0, band_h - control_h)
        control_c = band_c
        legend_r = band_r + random.randint(0, band_h - legend_h)
        legend_c = band_c + control_w + gap_b
    elif layout == "top_band":
        band_h = max(control_h, legend_h)
        band_r = content_r
        band_c = content_c + random.randint(0, content_w - (control_w + gap_b + legend_w))
        control_r = band_r + random.randint(0, band_h - control_h)
        control_c = band_c
        legend_r = band_r + random.randint(0, band_h - legend_h)
        legend_c = band_c + control_w + gap_b
        lattice_r = content_r + band_h + gap_a
        lattice_c = content_c + random.randint(0, content_w - lattice_w)
    elif layout == "right_band":
        lattice_r = content_r + random.randint(0, content_h - lattice_h)
        lattice_c = content_c
        side_c = content_c + lattice_w + gap_a
        side_w = max(control_w, legend_w)
        side_r = content_r + random.randint(0, content_h - (control_h + gap_b + legend_h))
        control_r = side_r
        control_c = side_c + random.randint(0, side_w - control_w)
        legend_r = side_r + control_h + gap_b
        legend_c = side_c + random.randint(0, side_w - legend_w)
    else:
        lattice_r = content_r + random.randint(0, content_h - lattice_h)
        side_w = max(control_w, legend_w)
        side_r = content_r + random.randint(0, content_h - (control_h + gap_b + legend_h))
        side_c = content_c
        control_r = side_r
        control_c = side_c + random.randint(0, side_w - control_w)
        legend_r = side_r + control_h + gap_b
        legend_c = side_c + random.randint(0, side_w - legend_w)
        lattice_c = content_c + side_w + gap_a

    return height, width, (lattice_r, lattice_c), (control_r, control_c), (legend_r, legend_c)


def _render_mask(dst: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    patch = dst[top : top + h, left : left + w]
    patch[mask] = int(color)


def _render_expected(
    inp: np.ndarray,
    lattice_rows: int,
    lattice_cols: int,
    size: int,
    cell_gap: int,
    lattice_pos: tuple[int, int],
    control: np.ndarray,
    template_mask: np.ndarray,
    pose_masks: dict[int, np.ndarray],
) -> np.ndarray:
    out = inp.copy()
    stride = size + cell_gap
    lat_r, lat_c = lattice_pos
    for i in range(lattice_rows):
        for j in range(lattice_cols):
            r0 = lat_r + i * stride
            c0 = lat_c + j * stride
            patch = out[r0 : r0 + size, c0 : c0 + size]
            patch[:] = _BG
            patch[pose_masks[int(control[i, j])]] = int(control[i, j])
    return out


def _render_naive_recolor(
    inp: np.ndarray,
    lattice_rows: int,
    lattice_cols: int,
    size: int,
    cell_gap: int,
    lattice_pos: tuple[int, int],
    control: np.ndarray,
    template_mask: np.ndarray,
) -> np.ndarray:
    out = inp.copy()
    stride = size + cell_gap
    lat_r, lat_c = lattice_pos
    for i in range(lattice_rows):
        for j in range(lattice_cols):
            r0 = lat_r + i * stride
            c0 = lat_c + j * stride
            patch = out[r0 : r0 + size, c0 : c0 + size]
            patch[:] = _BG
            patch[template_mask] = int(control[i, j])
    return out


def _build_scene() -> tuple[np.ndarray, np.ndarray, dict] | None:
    size = random.choice(_TEMPLATE_SIZES)
    template_mask = _generate_template_mask(size)
    if template_mask is None:
        return None

    lattice_rows, lattice_cols = random.choice(_LATTICE_DIMS)
    cell_gap = random.choice(_CELL_GAPS)
    legend_count = random.choice(_LEGEND_COUNTS)
    legend_gap = random.choice(_LEGEND_GAPS)
    border = random.choice(_BORDERS)
    gap_a = random.choice(_REGION_GAPS)
    gap_b = random.choice(_REGION_GAPS)

    chosen_colors = list(map(int, np.random.choice(_PALETTE, size=legend_count, replace=False)))
    transform_names = _choose_transforms(legend_count)
    random.shuffle(chosen_colors)
    pose_by_color = {
        color: _apply_transform(template_mask, transform)
        for color, transform in zip(chosen_colors, transform_names)
    }
    transform_by_color = {color: transform for color, transform in zip(chosen_colors, transform_names)}

    control = _generate_control(lattice_rows, lattice_cols, chosen_colors)
    if not any(transform_by_color[int(color)] in _MIRROR_TRANSFORMS for color in chosen_colors):
        return None
    if not any(transform_by_color[int(color)] in _ROTATION_TRANSFORMS for color in chosen_colors):
        return None

    layout = random.choice(("bottom_band", "top_band", "right_band", "left_band"))
    legends_horizontal = layout in {"bottom_band", "top_band"}
    legend_h, legend_w = _legend_line_size(size, legend_count, legend_gap, legends_horizontal)
    lattice_h = lattice_rows * size + (lattice_rows - 1) * cell_gap
    lattice_w = lattice_cols * size + (lattice_cols - 1) * cell_gap
    placed = _place_layout(
        lattice_h,
        lattice_w,
        lattice_rows,
        lattice_cols,
        legend_h,
        legend_w,
        layout=layout,
        border=border,
        gap_a=gap_a,
        gap_b=gap_b,
    )
    if placed is None:
        return None

    height, width, lattice_pos, control_pos, legend_pos = placed
    inp = np.full((height, width), _BG, dtype=int)
    lat_r, lat_c = lattice_pos
    ctl_r, ctl_c = control_pos
    leg_r, leg_c = legend_pos
    stride = size + cell_gap

    for i in range(lattice_rows):
        for j in range(lattice_cols):
            _render_mask(inp, template_mask, _TEMPLATE_COLOR, lat_r + i * stride, lat_c + j * stride)

    inp[ctl_r : ctl_r + lattice_rows, ctl_c : ctl_c + lattice_cols] = control

    legend_order = chosen_colors[:]
    random.shuffle(legend_order)
    for idx, color in enumerate(legend_order):
        if legends_horizontal:
            rr = leg_r
            cc = leg_c + idx * (size + legend_gap)
        else:
            rr = leg_r + idx * (size + legend_gap)
            cc = leg_c
        _render_mask(inp, pose_by_color[color], color, rr, cc)

    expected = _render_expected(
        inp,
        lattice_rows,
        lattice_cols,
        size,
        cell_gap,
        lattice_pos,
        control,
        template_mask,
        pose_by_color,
    )
    naive = _render_naive_recolor(
        inp,
        lattice_rows,
        lattice_cols,
        size,
        cell_gap,
        lattice_pos,
        control,
        template_mask,
    )
    metadata = {
        "template_mask": template_mask,
        "control": control,
        "control_pos": control_pos,
        "lattice_rows": lattice_rows,
        "lattice_cols": lattice_cols,
        "lattice_pos": lattice_pos,
        "cell_gap": cell_gap,
        "pose_by_color": pose_by_color,
        "transform_by_color": transform_by_color,
    }
    return inp, expected, {"metadata": metadata, "naive": naive}


def _generation_valid(inp: np.ndarray, expected: np.ndarray, aux: dict) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > _MAX_SIDE or max(expected.shape) > _MAX_SIDE:
        return False
    if np.count_nonzero(inp != _BG) >= inp.size // 2:
        return False
    if np.array_equal(expected, aux["naive"]):
        return False

    parsed = _parse_scene(inp)
    metadata = aux["metadata"]
    if (parsed["control_r"], parsed["control_c"]) != tuple(map(int, metadata["control_pos"])):
        return False
    parsed_colors = set(map(int, parsed["legend_names"]))
    expected_colors = set(map(int, metadata["pose_by_color"]))
    if parsed_colors != expected_colors:
        return False
    for color, transform in metadata["transform_by_color"].items():
        if parsed["legend_names"].get(int(color)) != transform:
            return False

    return np.array_equal(main(inp), expected)


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        built = _build_scene()
        if built is None:
            continue
        inp, expected, aux = built
        try:
            if _generation_valid(inp, expected, aux):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
