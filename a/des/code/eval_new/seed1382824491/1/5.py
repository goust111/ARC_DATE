from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200


def _scan_corner_legend(g: np.ndarray, corner: str) -> tuple[int, dict[int, int], np.ndarray] | None:
    h, w = g.shape
    if w < 2:
        return None

    if corner == "tl":
        rows = list(range(0, h, 2))
        c0, c1, step = 0, 1, 1
        margin_c = 2 if w > 2 else None
    elif corner == "tr":
        rows = list(range(0, h, 2))
        c0, c1, step = w - 2, w - 1, 1
        margin_c = w - 3 if w > 2 else None
    elif corner == "bl":
        rows = list(range(h - 1, -1, -2))
        c0, c1, step = 0, 1, -1
        margin_c = 2 if w > 2 else None
    else:
        rows = list(range(h - 1, -1, -2))
        c0, c1, step = w - 2, w - 1, -1
        margin_c = w - 3 if w > 2 else None

    pairs: list[tuple[int, int, int]] = []
    used_rows: list[int] = []
    for rr in rows:
        left = int(g[rr, c0])
        right = int(g[rr, c1])
        if left == BG or right == BG:
            break
        pairs.append((rr, left, right))
        used_rows.append(rr)

        sep = rr + step
        nxt = rr + 2 * step
        if 0 <= nxt < h:
            if not (0 <= sep < h and int(g[sep, c0]) == BG and int(g[sep, c1]) == BG):
                return None

    if len(pairs) < 2:
        return None

    if margin_c is not None:
        lo, hi = min(used_rows), max(used_rows)
        if np.any(g[lo : hi + 1, margin_c] != BG):
            return None

    mask = np.zeros_like(g, dtype=bool)
    mapping: dict[int, int] = {}
    for rr, src, dst in pairs:
        mask[rr, c0] = True
        mask[rr, c1] = True
        mapping[int(src)] = int(dst)
    return len(pairs), mapping, mask


def _extract_legend(g: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    best: tuple[int, dict[int, int], np.ndarray] | None = None
    for corner in ("tl", "tr", "bl", "br"):
        candidate = _scan_corner_legend(g, corner)
        if candidate is None:
            continue
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None, "expected a corner legend made of 2-pixel pairs"
    return best[1], best[2]


def _monochrome_components(g: np.ndarray, blocked: np.ndarray) -> list[dict[str, object]]:
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or blocked[r, c] or int(g[r, c]) == BG:
                continue

            color = int(g[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        if not blocked[nr, nc] and int(g[nr, nc]) == color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "r0": min(rows),
                    "r1": max(rows),
                    "c0": min(cols),
                    "c1": max(cols),
                }
            )
    return comps


def _extract_mirrors(g: np.ndarray, legend_mask: np.ndarray) -> tuple[list[dict[str, int]], np.ndarray]:
    mirror_mask = np.zeros_like(g, dtype=bool)
    mirrors: list[dict[str, int]] = []

    for comp in _monochrome_components(g, legend_mask):
        coords = comp["coords"]
        assert isinstance(coords, list)
        if len(coords) < 3:
            continue

        rows = sorted({rr for rr, _cc in coords})
        cols = sorted({cc for _rr, cc in coords})
        if len(rows) == 1:
            row = rows[0]
            if cols == list(range(cols[0], cols[-1] + 1)):
                for cc in cols:
                    mirror_mask[row, cc] = True
                mirrors.append({"orient": 0, "fixed": row, "start": cols[0], "end": cols[-1]})
        elif len(cols) == 1:
            col = cols[0]
            if rows == list(range(rows[0], rows[-1] + 1)):
                for rr in rows:
                    mirror_mask[rr, col] = True
                mirrors.append({"orient": 1, "fixed": col, "start": rows[0], "end": rows[-1]})

    mirrors.sort(key=lambda m: (m["fixed"] if m["orient"] == 1 else m["start"], m["start"] if m["orient"] == 1 else m["fixed"]))
    return mirrors, mirror_mask


def main(grid: np.ndarray) -> np.ndarray:
    """Reflect each side-pattern across its mirror line and recolor it via the corner legend.

    Input structure:
    - The background is black.
    - One corner contains a color legend made of horizontal 2-pixel pairs, with one blank row between consecutive pairs.
    - Elsewhere the grid contains one or more monochrome horizontal or vertical mirror segments of length at least 3, plus colored source pixels on exactly one side of each mirror.

    Rule:
    - Read the corner legend as a source-color -> reflected-color mapping.
    - For every mirror segment, determine which side is the source side by checking which side contains non-background, non-legend, non-mirror pixels aligned with that segment.
    - Reflect those source pixels across the mirror, recoloring each reflected pixel through the legend; colors without a legend entry stay unchanged.
    - Keep the original source pixels, mirror segments, and legend unchanged; if multiple reflections write to the same cell, later mirrors in left-to-right then top-to-bottom order overwrite earlier ones.

    Key insight:
    - The task combines geometry and lookup: first identify finite mirror segments and reflect only the pixels aligned to each segment, then remap reflected colors using the corner legend.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask = _extract_legend(g)
    mirrors, mirror_mask = _extract_mirrors(g, legend_mask)

    valid_source = (g != BG) & (~legend_mask) & (~mirror_mask)
    out = g.copy()
    h, w = g.shape

    for mirror in mirrors:
        if mirror["orient"] == 1:
            col = mirror["fixed"]
            r0, r1 = mirror["start"], mirror["end"]
            left_count = int(np.count_nonzero(valid_source[r0 : r1 + 1, :col]))
            right_count = int(np.count_nonzero(valid_source[r0 : r1 + 1, col + 1 :]))
            if left_count == 0 and right_count == 0:
                continue
            use_left = right_count == 0 or left_count >= right_count

            if use_left:
                src_pts = np.argwhere(valid_source[r0 : r1 + 1, :col])
                for rr_rel, cc in src_pts:
                    rr = int(rr_rel + r0)
                    src_c = int(cc)
                    dst_c = 2 * col - src_c
                    if 0 <= dst_c < w and not legend_mask[rr, dst_c] and not mirror_mask[rr, dst_c]:
                        color = int(g[rr, src_c])
                        out[rr, dst_c] = int(mapping.get(color, color))
            else:
                src_pts = np.argwhere(valid_source[r0 : r1 + 1, col + 1 :])
                for rr_rel, cc_rel in src_pts:
                    rr = int(rr_rel + r0)
                    src_c = int(col + 1 + cc_rel)
                    dst_c = 2 * col - src_c
                    if 0 <= dst_c < w and not legend_mask[rr, dst_c] and not mirror_mask[rr, dst_c]:
                        color = int(g[rr, src_c])
                        out[rr, dst_c] = int(mapping.get(color, color))
        else:
            row = mirror["fixed"]
            c0, c1 = mirror["start"], mirror["end"]
            up_count = int(np.count_nonzero(valid_source[:row, c0 : c1 + 1]))
            down_count = int(np.count_nonzero(valid_source[row + 1 :, c0 : c1 + 1]))
            if up_count == 0 and down_count == 0:
                continue
            use_up = down_count == 0 or up_count >= down_count

            if use_up:
                src_pts = np.argwhere(valid_source[:row, c0 : c1 + 1])
                for rr, cc_rel in src_pts:
                    src_r = int(rr)
                    cc = int(c0 + cc_rel)
                    dst_r = 2 * row - src_r
                    if 0 <= dst_r < h and not legend_mask[dst_r, cc] and not mirror_mask[dst_r, cc]:
                        color = int(g[src_r, cc])
                        out[dst_r, cc] = int(mapping.get(color, color))
            else:
                src_pts = np.argwhere(valid_source[row + 1 :, c0 : c1 + 1])
                for rr_rel, cc_rel in src_pts:
                    src_r = int(row + 1 + rr_rel)
                    cc = int(c0 + cc_rel)
                    dst_r = 2 * row - src_r
                    if 0 <= dst_r < h and not legend_mask[dst_r, cc] and not mirror_mask[dst_r, cc]:
                        color = int(g[src_r, cc])
                        out[dst_r, cc] = int(mapping.get(color, color))

    return out


def _mirror_sort_key(spec: dict[str, object]) -> tuple[int, int]:
    if spec["orient"] == "v":
        return int(spec["fixed"]), int(spec["start"])
    return int(spec["start"]), int(spec["fixed"])


def _expand_box(box: tuple[int, int, int, int], h: int, w: int, pad: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _legend_box(h: int, w: int, corner: str, n_pairs: int) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, 2 * n_pairs - 2, 1
    if corner == "tr":
        return 0, w - 2, 2 * n_pairs - 2, w - 1
    if corner == "bl":
        return h - (2 * n_pairs - 1), 0, h - 1, 1
    return h - (2 * n_pairs - 1), w - 2, h - 1, w - 1


def _place_legend(grid: np.ndarray, corner: str, pairs: list[tuple[int, int]]) -> np.ndarray:
    h, w = grid.shape
    mask = np.zeros_like(grid, dtype=bool)
    if corner in {"tl", "bl"}:
        c0, c1 = 0, 1
    else:
        c0, c1 = w - 2, w - 1

    for i, (src, dst) in enumerate(pairs):
        rr = 2 * i if corner in {"tl", "tr"} else h - 1 - 2 * i
        grid[rr, c0] = int(src)
        grid[rr, c1] = int(dst)
        mask[rr, c0] = True
        mask[rr, c1] = True
    return mask


def _source_rect(spec: dict[str, object]) -> tuple[int, int, int, int]:
    if spec["orient"] == "v":
        row0, row1 = int(spec["start"]), int(spec["end"])
        col = int(spec["fixed"])
        depth = int(spec["depth"])
        if spec["side"] == "left":
            return row0, col - depth, row1, col - 1
        return row0, col + 1, row1, col + depth

    col0, col1 = int(spec["start"]), int(spec["end"])
    row = int(spec["fixed"])
    depth = int(spec["depth"])
    if spec["side"] == "up":
        return row - depth, col0, row - 1, col1
    return row + 1, col0, row + depth, col1


def _affected_box(spec: dict[str, object]) -> tuple[int, int, int, int]:
    if spec["orient"] == "v":
        row0, row1 = int(spec["start"]), int(spec["end"])
        col, depth = int(spec["fixed"]), int(spec["depth"])
        return row0, col - depth, row1, col + depth
    col0, col1 = int(spec["start"]), int(spec["end"])
    row, depth = int(spec["fixed"]), int(spec["depth"])
    return row - depth, col0, row + depth, col1


def _sample_source_pixels(spec: dict[str, object], mapped_colors: list[int], unmapped_color: int) -> list[tuple[int, int, int]] | None:
    r0, c0, r1, c1 = _source_rect(spec)
    parity = random.randint(0, 1)
    candidates = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r + c) % 2 == parity]
    if len(candidates) < 4:
        return None

    count = random.randint(4, min(10, len(candidates)))
    chosen = random.sample(candidates, count)
    colors = [random.choice(mapped_colors), unmapped_color]
    source_palette = mapped_colors + [unmapped_color]
    while len(colors) < count:
        colors.append(random.choice(source_palette))
    random.shuffle(colors)
    return [(r, c, int(color)) for (r, c), color in zip(chosen, colors, strict=False)]


def _random_vertical_spec(
    h: int,
    w: int,
    mirror_color: int,
    mapped_colors: list[int],
    unmapped_color: int,
    blocked_boxes: list[tuple[int, int, int, int]],
    blocked_rows: list[tuple[int, int]],
) -> dict[str, object] | None:
    max_depth = min(4, (w - 1) // 2)
    if max_depth < 2:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        depth = random.randint(2, max_depth)
        length = random.randint(4, min(9, h))
        col = random.randint(depth, w - depth - 1)
        row0 = random.randint(0, h - length)
        row1 = row0 + length - 1
        box = _expand_box((row0, col - depth, row1, col + depth), h, w)
        if any(_boxes_overlap(box, other) for other in blocked_boxes):
            continue
        if any(_intervals_overlap((row0, row1), other) for other in blocked_rows):
            continue

        spec: dict[str, object] = {
            "orient": "v",
            "fixed": col,
            "start": row0,
            "end": row1,
            "depth": depth,
            "side": random.choice(["left", "right"]),
            "mirror_color": mirror_color,
        }
        pixels = _sample_source_pixels(spec, mapped_colors, unmapped_color)
        if pixels is None:
            continue
        spec["pixels"] = pixels
        return spec
    return None


def _random_horizontal_spec(
    h: int,
    w: int,
    mirror_color: int,
    mapped_colors: list[int],
    unmapped_color: int,
    blocked_boxes: list[tuple[int, int, int, int]],
    blocked_cols: list[tuple[int, int]],
) -> dict[str, object] | None:
    max_depth = min(4, (h - 1) // 2)
    if max_depth < 2:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        depth = random.randint(2, max_depth)
        length = random.randint(4, min(9, w))
        row = random.randint(depth, h - depth - 1)
        col0 = random.randint(0, w - length)
        col1 = col0 + length - 1
        box = _expand_box((row - depth, col0, row + depth, col1), h, w)
        if any(_boxes_overlap(box, other) for other in blocked_boxes):
            continue
        if any(_intervals_overlap((col0, col1), other) for other in blocked_cols):
            continue

        spec: dict[str, object] = {
            "orient": "h",
            "fixed": row,
            "start": col0,
            "end": col1,
            "depth": depth,
            "side": random.choice(["up", "down"]),
            "mirror_color": mirror_color,
        }
        pixels = _sample_source_pixels(spec, mapped_colors, unmapped_color)
        if pixels is None:
            continue
        spec["pixels"] = pixels
        return spec
    return None


def _overlap_specs(
    h: int,
    w: int,
    mirror_colors: list[int],
    mapped_colors: list[int],
    unmapped_color: int,
    blocked_boxes: list[tuple[int, int, int, int]],
) -> list[dict[str, object]] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        v_side = random.choice(["left", "right"])
        h_side = random.choice(["up", "down"])
        v_depth = random.randint(2, min(4, (w - 1) // 2))
        h_depth = random.randint(2, min(4, (h - 1) // 2))
        v_len = random.randint(6, min(10, h - 2))
        h_len = random.randint(6, min(10, w - 2))
        delta = random.randint(1, min(v_depth, h_depth))

        if v_side == "left":
            col_max = min(w - v_depth - 1, w - h_len - 1)
            if col_max < v_depth:
                continue
            v_col = random.randint(v_depth, col_max)
            h_col0_lo = v_col + 1
            h_col0_hi = min(v_col + v_depth, w - h_len)
            if h_col0_hi < h_col0_lo:
                continue
            h_col0 = random.randint(h_col0_lo, h_col0_hi)
            h_col1 = h_col0 + h_len - 1
        else:
            col_min = max(v_depth, h_len)
            col_max = w - v_depth - 1
            if col_max < col_min:
                continue
            v_col = random.randint(col_min, col_max)
            h_col1_lo = max(v_col - v_depth, h_len - 1)
            h_col1_hi = v_col - 1
            if h_col1_hi < h_col1_lo:
                continue
            h_col1 = random.randint(h_col1_lo, h_col1_hi)
            h_col0 = h_col1 - h_len + 1

        if h_side == "up":
            row0_min = h_depth + delta
            row0_max = h - v_len
            if row0_max < row0_min:
                continue
            v_row0 = random.randint(row0_min, row0_max)
            v_row1 = v_row0 + v_len - 1
            h_row = v_row0 - delta
        else:
            row0_min = 0
            row0_max = h - v_len - h_depth - delta
            if row0_max < row0_min:
                continue
            v_row0 = random.randint(row0_min, row0_max)
            v_row1 = v_row0 + v_len - 1
            h_row = v_row1 + delta

        v_spec: dict[str, object] = {
            "orient": "v",
            "fixed": v_col,
            "start": v_row0,
            "end": v_row1,
            "depth": v_depth,
            "side": v_side,
            "mirror_color": mirror_colors[0],
        }
        h_spec: dict[str, object] = {
            "orient": "h",
            "fixed": h_row,
            "start": h_col0,
            "end": h_col1,
            "depth": h_depth,
            "side": h_side,
            "mirror_color": mirror_colors[1],
        }

        v_box = _expand_box(_affected_box(v_spec), h, w)
        h_box = _expand_box(_affected_box(h_spec), h, w)
        if any(_boxes_overlap(v_box, other) or _boxes_overlap(h_box, other) for other in blocked_boxes):
            continue

        v_pixels = _sample_source_pixels(v_spec, mapped_colors, unmapped_color)
        h_pixels = _sample_source_pixels(h_spec, mapped_colors, unmapped_color)
        if v_pixels is None or h_pixels is None:
            continue

        v_spec["pixels"] = v_pixels
        h_spec["pixels"] = h_pixels
        return [v_spec, h_spec]
    return None


def _paint_spec(grid: np.ndarray, spec: dict[str, object]) -> None:
    color = int(spec["mirror_color"])
    if spec["orient"] == "v":
        col = int(spec["fixed"])
        row0, row1 = int(spec["start"]), int(spec["end"])
        grid[row0 : row1 + 1, col] = color
    else:
        row = int(spec["fixed"])
        col0, col1 = int(spec["start"]), int(spec["end"])
        grid[row, col0 : col1 + 1] = color

    pixels = spec["pixels"]
    assert isinstance(pixels, list)
    for r, c, pix_color in pixels:
        grid[int(r), int(c)] = int(pix_color)


def _spec_mask(shape: tuple[int, int], spec: dict[str, object]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if spec["orient"] == "v":
        col = int(spec["fixed"])
        row0, row1 = int(spec["start"]), int(spec["end"])
        mask[row0 : row1 + 1, col] = True
    else:
        row = int(spec["fixed"])
        col0, col1 = int(spec["start"]), int(spec["end"])
        mask[row, col0 : col1 + 1] = True

    pixels = spec["pixels"]
    assert isinstance(pixels, list)
    for r, c, _pix_color in pixels:
        mask[int(r), int(c)] = True
    return mask


def _expected_from_specs(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    mapping: dict[int, int],
    specs: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    mirror_mask = np.zeros_like(grid, dtype=bool)
    for spec in specs:
        if spec["orient"] == "v":
            col = int(spec["fixed"])
            row0, row1 = int(spec["start"]), int(spec["end"])
            mirror_mask[row0 : row1 + 1, col] = True
        else:
            row = int(spec["fixed"])
            col0, col1 = int(spec["start"]), int(spec["end"])
            mirror_mask[row, col0 : col1 + 1] = True

    for spec in sorted(specs, key=_mirror_sort_key):
        pixels = spec["pixels"]
        assert isinstance(pixels, list)
        for r, c, color in pixels:
            rr, cc = int(r), int(c)
            if spec["orient"] == "v":
                dst_r = rr
                dst_c = 2 * int(spec["fixed"]) - cc
            else:
                dst_r = 2 * int(spec["fixed"]) - rr
                dst_c = cc

            if 0 <= dst_r < grid.shape[0] and 0 <= dst_c < grid.shape[1]:
                if not legend_mask[dst_r, dst_c] and not mirror_mask[dst_r, dst_c]:
                    out[dst_r, dst_c] = int(mapping.get(int(color), int(color)))
    return out


def _choose_roles(n_mirrors: int) -> tuple[list[tuple[int, int]], list[int], int, list[int]]:
    mapped_count = random.randint(2, 3)
    colors = list(COLOR_POOL)
    random.shuffle(colors)
    mapped_sources = colors[:mapped_count]
    mapped_targets = colors[mapped_count : 2 * mapped_count]
    mirror_colors = colors[2 * mapped_count : 2 * mapped_count + n_mirrors]
    unmapped_color = colors[2 * mapped_count + n_mirrors]
    return list(zip(mapped_sources, mapped_targets, strict=False)), mapped_sources, int(unmapped_color), mirror_colors


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(["single", "double_disjoint", "double_overlap"])
    n_mirrors = 1 if mode == "single" else 2
    h_min = 18 if mode != "double_overlap" else 20
    w_min = 18 if mode != "double_overlap" else 20
    h = random.randint(h_min, 30)
    w = random.randint(w_min, 30)

    mapping_pairs, mapped_sources, unmapped_color, mirror_colors = _choose_roles(n_mirrors)
    mapping = {int(src): int(dst) for src, dst in mapping_pairs}
    corner = random.choice(["tl", "tr", "bl", "br"])

    grid = np.zeros((h, w), dtype=int)
    legend_mask = _place_legend(grid, corner, mapping_pairs)
    legend_guard = _expand_box(_legend_box(h, w, corner, len(mapping_pairs)), h, w, pad=1)

    specs: list[dict[str, object]] | None
    if mode == "single":
        if random.choice([False, True]):
            spec = _random_vertical_spec(h, w, mirror_colors[0], mapped_sources, unmapped_color, [legend_guard], [])
        else:
            spec = _random_horizontal_spec(h, w, mirror_colors[0], mapped_sources, unmapped_color, [legend_guard], [])
        specs = [spec] if spec is not None else None
    elif mode == "double_disjoint":
        if random.choice([False, True]):
            first = _random_vertical_spec(h, w, mirror_colors[0], mapped_sources, unmapped_color, [legend_guard], [])
            if first is None:
                return None
            first_rows = [(int(first["start"]), int(first["end"]))]
            second = _random_vertical_spec(
                h,
                w,
                mirror_colors[1],
                mapped_sources,
                unmapped_color,
                [legend_guard, _expand_box(_affected_box(first), h, w)],
                first_rows,
            )
            specs = [first, second] if second is not None else None
        else:
            first = _random_horizontal_spec(h, w, mirror_colors[0], mapped_sources, unmapped_color, [legend_guard], [])
            if first is None:
                return None
            first_cols = [(int(first["start"]), int(first["end"]))]
            second = _random_horizontal_spec(
                h,
                w,
                mirror_colors[1],
                mapped_sources,
                unmapped_color,
                [legend_guard, _expand_box(_affected_box(first), h, w)],
                first_cols,
            )
            specs = [first, second] if second is not None else None
    else:
        specs = _overlap_specs(h, w, mirror_colors, mapped_sources, unmapped_color, [legend_guard])

    if specs is None:
        return None

    painted = np.zeros_like(grid, dtype=bool)
    painted |= legend_mask
    for spec in specs:
        spec_mask = _spec_mask(grid.shape, spec)
        if np.any(spec_mask & painted):
            return None
        _paint_spec(grid, spec)
        painted |= spec_mask

    expected = _expected_from_specs(grid, legend_mask, mapping, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid mirror-reflection task")
