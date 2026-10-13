import numpy as np


BG = 0
SEP = 2
PALETTE = np.array([1, 3, 4, 5, 6, 7, 8, 9], dtype=int)
_RNG = np.random.default_rng()

# --------------- generator (o5x) ---------------

PANEL_COUNT_CHOICES = (3, 4)
LAYOUT_CHOICES = ("row", "col")
TEMPLATE_SIDE_MIN = 5
TEMPLATE_SIDE_MAX = 8
INNER_MARGIN_MIN = 2
INNER_MARGIN_MAX = 4
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_bbox(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        mask = arr != BG
    box = _bbox(mask)
    assert box is not None, "expected at least one foreground pixel"
    r0, c0, r1, c1 = box
    return arr[r0:r1, c0:c1]


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    if not separators:
        return [(0, length)]

    bands: list[tuple[int, int]] = []
    start = 0
    for idx in sorted(separators):
        assert 0 <= idx < length, "separator index out of bounds"
        if idx > start:
            bands.append((start, idx))
        start = idx + 1
    if start < length:
        bands.append((start, length))
    assert bands, "separator lines must leave at least one panel"
    return bands


def _dihedral_transforms(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        out.append(rot)
        out.append(np.fliplr(rot))
    return out


def _shape_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    reps: list[tuple[int, int, tuple[int, ...]]] = []
    for transformed in _dihedral_transforms(mask):
        cropped = _crop_to_bbox(transformed.astype(np.uint8), transformed)
        reps.append(
            (
                int(cropped.shape[0]),
                int(cropped.shape[1]),
                tuple(int(v) for v in cropped.ravel()),
            )
        )
    return min(reps)


def _split_panels(grid: np.ndarray) -> tuple[list[dict[str, object]], int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_rows = [r for r in range(h) if np.all(g[r, :] == SEP)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == SEP)]
    assert sep_rows or sep_cols, "expected at least one separator line"

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)

    panels: list[dict[str, object]] = []
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            panel = g[r0:r1, c0:c1]
            assert panel.size > 0, "panels must be non-empty"
            panels.append({"row": ri, "col": ci, "grid": panel})
    return panels, len(row_bands), len(col_bands)


def _reference_panel(panels: list[dict[str, object]]) -> tuple[int, tuple[int, int, tuple[int, ...]]]:
    candidates: list[int] = []
    for idx, panel_info in enumerate(panels):
        panel = np.asarray(panel_info["grid"], dtype=int)
        fg = panel != BG
        assert fg.any(), "every panel must contain a template"
        colors = np.unique(panel[fg])
        if colors.size == 1:
            candidates.append(idx)
    assert len(candidates) == 1, "there must be exactly one unmasked reference panel"
    ref_idx = candidates[0]
    ref_mask = np.asarray(panels[ref_idx]["grid"], dtype=int) != BG
    return ref_idx, _shape_signature(ref_mask)


def _extract_mask_crop(panel: np.ndarray, ref_signature: tuple[int, int, tuple[int, ...]]) -> np.ndarray | None:
    fg = panel != BG
    assert _shape_signature(fg) == ref_signature, "all panels must share one template up to dihedral symmetry"
    colors = np.unique(panel[fg])
    if colors.size == 1:
        return None

    assert colors.size == 2, "masked panels must contain exactly two colors"
    count0 = int(np.sum(panel == int(colors[0])))
    count1 = int(np.sum(panel == int(colors[1])))
    assert count0 != count1, "mask and template colors must be distinguishable by area"

    mask_color = int(colors[0]) if count0 < count1 else int(colors[1])
    template_color = int(colors[1]) if mask_color == int(colors[0]) else int(colors[0])
    mask = panel == mask_color
    assert mask.any(), "mask layer must be non-empty"

    restored = panel.copy()
    restored[mask] = template_color
    assert np.unique(restored[fg]).size == 1, "restoring the mask must recover a monochrome template"

    layer = np.where(mask, mask_color, BG)
    return _crop_to_bbox(layer, mask)


def _assemble_output(items: list[dict[str, object]]) -> np.ndarray:
    assert items, "expected at least one non-reference panel"

    used_rows = sorted({int(item["row"]) for item in items})
    used_cols = sorted({int(item["col"]) for item in items})
    row_heights = {
        row: max(int(np.asarray(item["crop"]).shape[0]) for item in items if int(item["row"]) == row)
        for row in used_rows
    }
    col_widths = {
        col: max(int(np.asarray(item["crop"]).shape[1]) for item in items if int(item["col"]) == col)
        for col in used_cols
    }

    total_h = sum(row_heights[row] for row in used_rows) + max(0, len(used_rows) - 1)
    total_w = sum(col_widths[col] for col in used_cols) + max(0, len(used_cols) - 1)
    out = np.full((total_h, total_w), BG, dtype=int)

    row_offsets: dict[int, int] = {}
    cur = 0
    for i, row in enumerate(used_rows):
        row_offsets[row] = cur
        cur += row_heights[row]
        if i + 1 < len(used_rows):
            out[cur, :] = SEP
            cur += 1

    col_offsets: dict[int, int] = {}
    cur = 0
    for i, col in enumerate(used_cols):
        col_offsets[col] = cur
        cur += col_widths[col]
        if i + 1 < len(used_cols):
            out[:, cur] = SEP
            cur += 1

    for item in items:
        crop = np.asarray(item["crop"], dtype=int)
        r0 = row_offsets[int(item["row"])]
        c0 = col_offsets[int(item["col"])]
        h, w = crop.shape
        out[r0 : r0 + h, c0 : c0 + w] = crop
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split red-separated panels, recover the shared template, and output the cropped mask layers.
    """
    g = np.asarray(grid, dtype=int)
    panels, _rows, _cols = _split_panels(g)
    assert 3 <= len(panels) <= 4, "expected 3-4 panels"

    ref_idx, ref_signature = _reference_panel(panels)

    items: list[dict[str, object]] = []
    for idx, panel_info in enumerate(panels):
        crop = _extract_mask_crop(np.asarray(panel_info["grid"], dtype=int), ref_signature)
        if idx == ref_idx:
            assert crop is None, "reference panel must not contain a mask"
            continue
        assert crop is not None, "every non-reference panel must contain a mask"
        items.append({"row": int(panel_info["row"]), "col": int(panel_info["col"]), "crop": crop})

    return _assemble_output(items)


def _apply_transform(mask: np.ndarray, transform_id: int) -> np.ndarray:
    rotated = np.rot90(mask, transform_id % 4)
    if transform_id >= 4:
        rotated = np.fliplr(rotated)
    return rotated.copy()


def _make_template() -> np.ndarray | None:
    h = int(_RNG.integers(TEMPLATE_SIDE_MIN, TEMPLATE_SIDE_MAX + 1))
    w = int(_RNG.integers(TEMPLATE_SIDE_MIN, TEMPLATE_SIDE_MAX + 1))
    mask = np.ones((h, w), dtype=bool)

    corner_specs = [
        (slice(None), slice(None)),
        (slice(None), slice(None)),
        (slice(None), slice(None)),
        (slice(None), slice(None)),
    ]
    notch_count = int(_RNG.integers(1, 4))
    corners = _RNG.choice(4, size=notch_count, replace=False)
    max_notch_h = min(2, h - 2)
    max_notch_w = min(2, w - 2)
    if max_notch_h < 1 or max_notch_w < 1:
        return None

    for corner in corners:
        notch_h = int(_RNG.integers(1, max_notch_h + 1))
        notch_w = int(_RNG.integers(1, max_notch_w + 1))
        if corner == 0:
            corner_specs[corner] = (slice(0, notch_h), slice(0, notch_w))
        elif corner == 1:
            corner_specs[corner] = (slice(0, notch_h), slice(w - notch_w, w))
        elif corner == 2:
            corner_specs[corner] = (slice(h - notch_h, h), slice(0, notch_w))
        else:
            corner_specs[corner] = (slice(h - notch_h, h), slice(w - notch_w, w))

    for corner in corners:
        rs, cs = corner_specs[corner]
        mask[rs, cs] = False

    area = int(mask.sum())
    if area < max(14, (h * w) // 2):
        return None
    return mask


def _rect_positions(support: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for r0 in range(support.shape[0] - h + 1):
        for c0 in range(support.shape[1] - w + 1):
            if np.all(support[r0 : r0 + h, c0 : c0 + w]):
                positions.append((r0, c0))
    return positions


def _mask_pattern(h: int, w: int, max_area: int) -> np.ndarray | None:
    styles = np.array(["rect", "L", "stair", "T"], dtype=object)
    for style in _RNG.permutation(styles):
        mask = np.zeros((h, w), dtype=bool)

        if style == "rect":
            mask[:, :] = True
        elif style == "L":
            mask[:, 0] = True
            mask[-1, :] = True
            extra_h = int(_RNG.integers(1, h + 1))
            extra_w = int(_RNG.integers(1, w + 1))
            mask[:extra_h, :extra_w] = True
        elif style == "stair":
            widths = np.sort(_RNG.integers(1, w + 1, size=h))
            widths[-1] = w
            for r, width in enumerate(widths):
                mask[r, : int(width)] = True
            mask[:, 0] = True
        else:
            pivot = int(_RNG.integers(0, w))
            tail = int(_RNG.integers(0, w))
            mask[0, :] = True
            mask[:, pivot] = True
            mask[-1, tail:] = True

        if int(_RNG.integers(0, 2)):
            mask = np.flipud(mask)
        if int(_RNG.integers(0, 2)):
            mask = np.fliplr(mask)

        box = _bbox(mask)
        if box != (0, 0, h, w):
            continue
        area = int(mask.sum())
        if 2 <= area <= max_area:
            return mask
    return None


def _panel_crops_from_specs(
    layout: str,
    ref_idx: int,
    mask_specs: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for idx in range(len(mask_specs) + 1):
        if idx == ref_idx:
            continue
        spec = mask_specs[idx]
        crop = np.where(np.asarray(spec["mask_local"], dtype=bool), int(spec["mask_color"]), BG)
        if layout == "row":
            items.append({"row": 0, "col": idx, "crop": crop})
        else:
            items.append({"row": idx, "col": 0, "crop": crop})
    return items


def _gen_attempt() -> np.ndarray | None:
    template = _make_template()
    if template is None:
        return None

    panel_count = int(_RNG.choice(PANEL_COUNT_CHOICES))
    layout = str(_RNG.choice(LAYOUT_CHOICES))
    ref_idx = int(_RNG.integers(0, panel_count))
    support_area = int(template.sum())
    max_mask_area = (support_area - 1) // 2
    if max_mask_area < 4:
        return None

    transform_ids = [int(_RNG.integers(0, 8)) for _ in range(panel_count)]
    oriented_templates = [_apply_transform(template, tid) for tid in transform_ids]

    masked_indices = [idx for idx in range(panel_count) if idx != ref_idx]
    shared_limit = min(
        4,
        min(
            int(oriented_templates[idx].shape[0] if layout == "row" else oriented_templates[idx].shape[1])
            for idx in masked_indices
        ),
    )
    common_dims = [dim for dim in range(2, shared_limit + 1)]
    if not common_dims:
        return None

    mask_specs: dict[int, dict[str, object]] | None = None
    for common_dim in _RNG.permutation(common_dims):
        current_specs: dict[int, dict[str, object]] = {}
        ok = True
        for idx in masked_indices:
            support = oriented_templates[idx]
            variable_limit = int(support.shape[1] if layout == "row" else support.shape[0])
            variable_dims = [dim for dim in range(2, min(4, variable_limit) + 1)]
            _RNG.shuffle(variable_dims)

            chosen = None
            for variable_dim in variable_dims:
                mask_h, mask_w = (int(common_dim), int(variable_dim)) if layout == "row" else (int(variable_dim), int(common_dim))
                if mask_h * mask_w > max_mask_area:
                    continue
                positions = _rect_positions(support, mask_h, mask_w)
                if not positions:
                    continue
                mask_local = _mask_pattern(mask_h, mask_w, max_mask_area)
                if mask_local is None:
                    continue
                if int(mask_local.sum()) >= support_area - int(mask_local.sum()):
                    continue
                chosen = {
                    "mask_h": mask_h,
                    "mask_w": mask_w,
                    "mask_local": mask_local,
                    "origin": positions[int(_RNG.integers(0, len(positions)))],
                }
                break

            if chosen is None:
                ok = False
                break
            current_specs[idx] = chosen

        if ok:
            mask_specs = current_specs
            break

    if mask_specs is None:
        return None

    max_h = max(int(mask.shape[0]) for mask in oriented_templates)
    max_w = max(int(mask.shape[1]) for mask in oriented_templates)
    inner_h = max_h + int(_RNG.integers(INNER_MARGIN_MIN, INNER_MARGIN_MAX + 1))
    inner_w = max_w + int(_RNG.integers(INNER_MARGIN_MIN, INNER_MARGIN_MAX + 1))

    palette = list(map(int, _RNG.permutation(PALETTE)))
    needed = panel_count + len(masked_indices)
    if needed > len(palette):
        return None
    template_colors = palette[:panel_count]
    mask_colors = palette[panel_count:needed]

    panels: list[np.ndarray] = []
    mask_color_ptr = 0
    for idx in range(panel_count):
        support = oriented_templates[idx]
        sh, sw = support.shape
        panel = np.full((inner_h, inner_w), BG, dtype=int)
        top = int(_RNG.integers(0, inner_h - sh + 1))
        left = int(_RNG.integers(0, inner_w - sw + 1))

        panel_slice = panel[top : top + sh, left : left + sw]
        panel_slice[support] = int(template_colors[idx])

        if idx != ref_idx:
            spec = mask_specs[idx]
            mask_color = int(mask_colors[mask_color_ptr])
            mask_color_ptr += 1
            mr0, mc0 = map(int, spec["origin"])
            mask_local = np.asarray(spec["mask_local"], dtype=bool)
            mh, mw = mask_local.shape

            support_mask = np.zeros_like(support, dtype=bool)
            support_mask[mr0 : mr0 + mh, mc0 : mc0 + mw][mask_local] = True
            panel_slice[support_mask] = mask_color
            spec["mask_color"] = mask_color

        panels.append(panel)

    if layout == "row":
        width = panel_count * inner_w + (panel_count - 1)
        grid = np.full((inner_h, width), BG, dtype=int)
        c = 0
        for idx, panel in enumerate(panels):
            grid[:, c : c + inner_w] = panel
            c += inner_w
            if idx + 1 < panel_count:
                grid[:, c] = SEP
                c += 1
    else:
        height = panel_count * inner_h + (panel_count - 1)
        grid = np.full((height, inner_w), BG, dtype=int)
        r = 0
        for idx, panel in enumerate(panels):
            grid[r : r + inner_h, :] = panel
            r += inner_h
            if idx + 1 < panel_count:
                grid[r, :] = SEP
                r += 1

    expected = _assemble_output(_panel_crops_from_specs(layout, ref_idx, mask_specs))
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
