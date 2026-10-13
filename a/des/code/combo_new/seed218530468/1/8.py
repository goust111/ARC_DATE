from __future__ import annotations

import numpy as np


BG = 0
SEPARATOR_COLOR = 8
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
RNG = np.random.default_rng()
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
FLIP_NAMES = ("identity", "ud", "lr")
SCALE_OPTIONS = ((1, 1), (2, 1), (1, 2), (2, 2))
MAX_GENERATION_RETRIES = 400


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _collapse_repeats(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    changed = True
    while changed:
        changed = False
        if out.shape[0] > 1:
            keep_rows = [0]
            for idx in range(1, out.shape[0]):
                if not np.array_equal(out[idx], out[idx - 1]):
                    keep_rows.append(idx)
            if len(keep_rows) != out.shape[0]:
                out = out[keep_rows]
                changed = True
        if out.shape[1] > 1:
            keep_cols = [0]
            for idx in range(1, out.shape[1]):
                if not np.array_equal(out[:, idx], out[:, idx - 1]):
                    keep_cols.append(idx)
            if len(keep_cols) != out.shape[1]:
                out = out[:, keep_cols]
                changed = True
    return out


def _apply_flip(mask: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return mask.copy()
    if name == "ud":
        return np.flipud(mask)
    if name == "lr":
        return np.fliplr(mask)
    raise AssertionError(f"unknown flip: {name}")


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    base = _collapse_repeats(np.asarray(mask, dtype=bool))
    variants = (_apply_flip(base, name) for name in FLIP_NAMES)
    return min((tuple(map(int, variant.shape)), variant.tobytes()) for variant in variants)


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, r0, c0


def _stamp_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _reflect_rows(r0: int, h: int, total_rows: int) -> int:
    return int(total_rows - r0 - h)


def _find_separator_row(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    rows = [r for r in range(g.shape[0]) if bool(np.all(g[r] == SEPARATOR_COLOR))]
    if rows:
        return int(rows[len(rows) // 2])
    fallback = [r for r in range(g.shape[0]) if int(g[r, 0]) != BG and bool(np.all(g[r] == g[r, 0]))]
    assert fallback, "expected a full horizontal separator row"
    return int(fallback[len(fallback) // 2])


def _extract_objects(grid: np.ndarray, separator_row: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color in {BG, SEPARATOR_COLOR}:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            coord_arr = np.array(coords, dtype=int)
            side = "top" if int(coord_arr[:, 0].max()) < separator_row else "bottom"
            assert int(coord_arr[:, 0].min()) > separator_row or side == "top", "object cannot cross separator"
            mask, r0, c0 = _mask_from_coords(coord_arr)
            objects.append(
                {
                    "color": int(color),
                    "coords": coord_arr,
                    "area": int(coord_arr.shape[0]),
                    "side": side,
                    "mask": mask,
                    "r0": int(r0),
                    "c0": int(c0),
                    "shape_key": _shape_key(mask),
                }
            )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep isolates and equal-size cross-line pairs, but replace the smaller member of each unequal pair with the larger member's reflection.

    Input structure:
    - The background is black, and one full horizontal azure row acts as the separator.
    - Above and below the separator are 4-connected monochrome objects.
    - Some upper/lower objects share the same underlying silhouette after trimming and collapsing duplicated adjacent rows or columns, allowing a horizontal or vertical mirror; these paired objects use different colors.
    - Every silhouette key appears in at most one object per half, while any unmatched object is isolated.

    Rule:
    - Preserve the azure separator row.
    - Leave isolated objects unchanged.
    - For every unique top/bottom pair with the same silhouette key: if their areas match, keep both unchanged.
    - If their areas differ, keep the larger object in place, erase the smaller one, then reflect every pixel of the larger object across the separator row and repaint that reflected copy with the smaller object's color.

    Key insight:
    - The task is not color matching; it is cross-panel shape matching first, then an area comparison, and finally a separator-based reflection that reuses the smaller object's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator_row = _find_separator_row(g)
    objects = _extract_objects(g, separator_row)

    groups: dict[tuple[tuple[int, int], bytes], dict[str, list[dict]]] = {}
    for obj in objects:
        bucket = groups.setdefault(obj["shape_key"], {"top": [], "bottom": []})
        bucket[obj["side"]].append(obj)

    out = g.copy()
    for bucket in groups.values():
        if len(bucket["top"]) != 1 or len(bucket["bottom"]) != 1:
            continue
        top_obj = bucket["top"][0]
        bottom_obj = bucket["bottom"][0]
        if int(top_obj["area"]) == int(bottom_obj["area"]):
            continue

        large = top_obj if int(top_obj["area"]) > int(bottom_obj["area"]) else bottom_obj
        small = bottom_obj if large is top_obj else top_obj

        out[small["coords"][:, 0], small["coords"][:, 1]] = BG
        for rr_raw, cc_raw in large["coords"]:
            rr = int(rr_raw)
            cc = int(cc_raw)
            mirrored_row = 2 * int(separator_row) - rr
            assert 0 <= mirrored_row < g.shape[0], "reflected pixel must stay inside the grid"
            out[int(mirrored_row), int(cc)] = int(small["color"])

    return out


def _build_primitive_catalog() -> list[np.ndarray]:
    catalog: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for h in range(2, 4):
        for w in range(2, 4):
            for bits in range(1, 1 << (h * w)):
                mask = np.array([(bits >> idx) & 1 for idx in range(h * w)], dtype=bool).reshape(h, w)
                if int(mask.sum()) < 3 or not _is_connected(mask):
                    continue
                trimmed = _trim_mask(mask)
                if trimmed.shape != (h, w):
                    continue
                if any(np.array_equal(trimmed[idx], trimmed[idx + 1]) for idx in range(h - 1)):
                    continue
                if any(np.array_equal(trimmed[:, idx], trimmed[:, idx + 1]) for idx in range(w - 1)):
                    continue
                catalog.setdefault(_shape_key(trimmed), trimmed)
    return list(catalog.values())


PRIMITIVE_CATALOG = _build_primitive_catalog()


def _scale_mask(mask: np.ndarray, scale: tuple[int, int]) -> np.ndarray:
    rs, cs = scale
    return np.repeat(np.repeat(mask, int(rs), axis=0), int(cs), axis=1)


def _choose_unequal_scales() -> tuple[tuple[int, int], tuple[int, int]]:
    choices = [
        (small, large)
        for small in SCALE_OPTIONS
        for large in SCALE_OPTIONS
        if small[0] * small[1] < large[0] * large[1]
    ]
    return choices[int(RNG.integers(0, len(choices)))]


def _build_family(mode: str, primitive: np.ndarray, colors: list[int], color_idx: int) -> tuple[dict, int]:
    base_flip = FLIP_NAMES[int(RNG.integers(0, len(FLIP_NAMES)))]
    relation = FLIP_NAMES[int(RNG.integers(0, len(FLIP_NAMES)))]

    def oriented(mask: np.ndarray, extra_flip: str) -> np.ndarray:
        return _apply_flip(_apply_flip(mask, base_flip), extra_flip)

    family: dict[str, object] = {"mode": mode, "primitive_key": _shape_key(primitive)}
    if mode == "equal":
        scale = SCALE_OPTIONS[int(RNG.integers(0, len(SCALE_OPTIONS)))]
        shared = _scale_mask(primitive, scale)
        top_mask = oriented(shared, "identity")
        bottom_mask = oriented(shared, relation)
        family["top"] = {"mask": top_mask, "color": int(colors[color_idx])}
        family["bottom"] = {"mask": bottom_mask, "color": int(colors[color_idx + 1])}
        color_idx += 2
    elif mode == "top_large":
        small_scale, large_scale = _choose_unequal_scales()
        family["top"] = {
            "mask": oriented(_scale_mask(primitive, large_scale), "identity"),
            "color": int(colors[color_idx]),
        }
        family["bottom"] = {
            "mask": oriented(_scale_mask(primitive, small_scale), relation),
            "color": int(colors[color_idx + 1]),
        }
        color_idx += 2
    elif mode == "bottom_large":
        small_scale, large_scale = _choose_unequal_scales()
        family["top"] = {
            "mask": oriented(_scale_mask(primitive, small_scale), relation),
            "color": int(colors[color_idx]),
        }
        family["bottom"] = {
            "mask": oriented(_scale_mask(primitive, large_scale), "identity"),
            "color": int(colors[color_idx + 1]),
        }
        color_idx += 2
    elif mode == "isolated_top":
        scale = SCALE_OPTIONS[int(RNG.integers(0, len(SCALE_OPTIONS)))]
        family["top"] = {"mask": oriented(_scale_mask(primitive, scale), relation), "color": int(colors[color_idx])}
        family["bottom"] = None
        color_idx += 1
    elif mode == "isolated_bottom":
        scale = SCALE_OPTIONS[int(RNG.integers(0, len(SCALE_OPTIONS)))]
        family["top"] = None
        family["bottom"] = {
            "mask": oriented(_scale_mask(primitive, scale), relation),
            "color": int(colors[color_idx]),
        }
        color_idx += 1
    else:
        raise AssertionError(f"unknown family mode: {mode}")
    return family, color_idx


def _place_families(families: list[dict], height: int, width: int) -> bool:
    half = (height - 1) // 2
    min_gap = 1
    widths = []
    for family in families:
        masks = [obj["mask"] for obj in (family.get("top"), family.get("bottom")) if obj is not None]
        widths.append(max(int(mask.shape[1]) for mask in masks))
    required_width = int(sum(widths) + min_gap * (len(widths) - 1))
    if required_width > width:
        return False

    slack = int(width - required_width)
    extras = RNG.multinomial(slack, np.ones(len(widths) + 1) / (len(widths) + 1))
    cursor = int(extras[0])
    for idx, family in enumerate(families):
        family["c0"] = int(cursor)
        cursor += widths[idx]
        if idx < len(widths) - 1:
            cursor += min_gap + int(extras[idx + 1])

    for family in families:
        c0 = int(family["c0"])
        top_obj = family.get("top")
        bottom_obj = family.get("bottom")
        mode = str(family["mode"])

        if mode == "equal":
            h = int(top_obj["mask"].shape[0])
            top_r0 = int(RNG.integers(0, half - h + 1))
            bottom_r0 = _reflect_rows(top_r0, h, height)
            top_obj["r0"], top_obj["c0"] = int(top_r0), int(c0)
            bottom_obj["r0"], bottom_obj["c0"] = int(bottom_r0), int(c0)
        elif mode == "top_large":
            top_h = int(top_obj["mask"].shape[0])
            top_r0 = int(RNG.integers(0, half - top_h + 1))
            bottom_r0 = _reflect_rows(top_r0, top_h, height)
            top_obj["r0"], top_obj["c0"] = int(top_r0), int(c0)
            bottom_obj["r0"], bottom_obj["c0"] = int(bottom_r0), int(c0)
        elif mode == "bottom_large":
            bottom_h = int(bottom_obj["mask"].shape[0])
            bottom_r0 = int(half + 1 + RNG.integers(0, half - bottom_h + 1))
            top_r0 = _reflect_rows(bottom_r0, bottom_h, height)
            top_obj["r0"], top_obj["c0"] = int(top_r0), int(c0)
            bottom_obj["r0"], bottom_obj["c0"] = int(bottom_r0), int(c0)
        elif mode == "isolated_top":
            top_h = int(top_obj["mask"].shape[0])
            top_obj["r0"], top_obj["c0"] = int(RNG.integers(0, half - top_h + 1)), int(c0)
        elif mode == "isolated_bottom":
            bottom_h = int(bottom_obj["mask"].shape[0])
            bottom_obj["r0"], bottom_obj["c0"] = int(half + 1 + RNG.integers(0, half - bottom_h + 1)), int(c0)
        else:
            return False
    return True


def _render_input(height: int, width: int, families: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[(height - 1) // 2, :] = SEPARATOR_COLOR
    for family in families:
        for obj in (family.get("top"), family.get("bottom")):
            if obj is None:
                continue
            _stamp_mask(grid, obj["mask"], int(obj["r0"]), int(obj["c0"]), int(obj["color"]))
    return grid


def _expected_output(inp: np.ndarray, families: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    separator_row = (inp.shape[0] - 1) // 2
    for family in families:
        mode = str(family["mode"])
        if mode in {"equal", "isolated_top", "isolated_bottom"}:
            continue
        if mode == "top_large":
            large = family["top"]
            small = family["bottom"]
        else:
            large = family["bottom"]
            small = family["top"]

        mask = np.asarray(small["mask"], dtype=bool)
        r0 = int(small["r0"])
        c0 = int(small["c0"])
        out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = BG

        for rr_raw, cc_raw in np.argwhere(np.asarray(large["mask"], dtype=bool)):
            rr = int(rr_raw) + int(large["r0"])
            cc = int(cc_raw) + int(large["c0"])
            out[2 * separator_row - rr, cc] = int(small["color"])
    return out


def _make_candidate() -> np.ndarray | None:
    pair_modes_all = ["equal", "top_large", "bottom_large"]
    pair_count = 2 if bool(RNG.integers(0, 2)) else 3
    pair_modes = list(RNG.choice(np.array(pair_modes_all, dtype=object), size=pair_count, replace=False))

    isolate_limit = 2 if pair_count == 2 else 1
    isolate_count = 0
    if isolate_limit > 0 and float(RNG.random()) < 0.75:
        isolate_count = int(RNG.integers(1, isolate_limit + 1))

    modes = pair_modes[:]
    for _ in range(isolate_count):
        modes.append("isolated_top" if bool(RNG.integers(0, 2)) else "isolated_bottom")
    RNG.shuffle(modes)

    primitive_ids = RNG.choice(len(PRIMITIVE_CATALOG), size=len(modes), replace=False)
    object_count = 2 * pair_count + isolate_count
    colors = list(RNG.choice(np.array(PALETTE, dtype=int), size=object_count, replace=False))

    families: list[dict] = []
    color_idx = 0
    for mode, primitive_id in zip(modes, primitive_ids, strict=True):
        family, color_idx = _build_family(str(mode), PRIMITIVE_CATALOG[int(primitive_id)], colors, color_idx)
        families.append(family)

    max_height = 0
    for family in families:
        for obj in (family.get("top"), family.get("bottom")):
            if obj is not None:
                max_height = max(max_height, int(obj["mask"].shape[0]))
    half_min = max(7, max_height + 1)
    if half_min > 14:
        return None

    half = int(RNG.integers(half_min, 15))
    height = 2 * half + 1

    widths = []
    for family in families:
        masks = [obj["mask"] for obj in (family.get("top"), family.get("bottom")) if obj is not None]
        widths.append(max(int(mask.shape[1]) for mask in masks))
    required_width = int(sum(widths) + len(widths) - 1)
    if required_width > 30:
        return None
    width = int(RNG.integers(required_width, 31))

    if not _place_families(families, height, width):
        return None

    inp = _render_input(height, width, families)
    expected = _expected_output(inp, families)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _make_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 8")
