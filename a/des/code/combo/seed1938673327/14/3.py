from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


def _mode_color(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _touches_edge(coords: np.ndarray, height: int, width: int) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any((rows == 0) | (rows == height - 1) | (cols == 0) | (cols == width - 1))
    )


def _count_corners_from_mask(mask: np.ndarray) -> int:
    h, w = map(int, mask.shape)
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if r == 0 or not mask[r - 1, c]:
                edges.append(((r, c), (r, c + 1)))
            if c == w - 1 or not mask[r, c + 1]:
                edges.append(((r, c + 1), (r + 1, c + 1)))
            if r == h - 1 or not mask[r + 1, c]:
                edges.append(((r + 1, c + 1), (r + 1, c)))
            if c == 0 or not mask[r, c - 1]:
                edges.append(((r + 1, c), (r, c)))

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    assert adjacency, "expected a nonempty boundary"
    assert all(len(neighbors) == 2 for neighbors in adjacency.values()), "expected a single simple cycle"

    start = min(adjacency)
    prev: tuple[int, int] | None = None
    cur = start
    path = [start]
    while True:
        candidates = adjacency[cur]
        if prev is None:
            nxt = min(candidates)
        else:
            nxt = candidates[0] if candidates[1] == prev else candidates[1]
        path.append(nxt)
        prev, cur = cur, nxt
        if cur == start:
            break
        assert len(path) <= len(edges) + 2, "boundary walk should close promptly"

    dirs: list[tuple[int, int]] = []
    for a, b in zip(path, path[1:], strict=False):
        dr = int(b[0] - a[0])
        dc = int(b[1] - a[1])
        dirs.append((0, 1) if dc > 0 else (0, -1) if dc < 0 else (1, 0) if dr > 0 else (-1, 0))

    corners = 0
    for d1, d2 in zip(dirs, dirs[1:] + dirs[:1], strict=False):
        if d1 != d2:
            corners += 1
    return corners


def _count_region_corners(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return _count_corners_from_mask(local)


def _choose_center(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    target_r = (r0 + r1) / 2.0
    target_c = (c0 + c1) / 2.0
    best = min(
        (tuple(map(int, cell)) for cell in coords.tolist()),
        key=lambda rc: (
            (rc[0] - target_r) ** 2 + (rc[1] - target_c) ** 2,
            abs(rc[0] - target_r),
            abs(rc[1] - target_c),
            rc[0],
            rc[1],
        ),
    )
    return int(best[0]), int(best[1])


def _draw_cross(out: np.ndarray, coords: np.ndarray, color: int) -> None:
    region = {tuple(map(int, cell)) for cell in coords.tolist()}
    cr, cc = _choose_center(coords)
    out[cr, cc] = int(color)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r, c = cr, cc
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region:
                break
            out[nr, nc] = int(color)
            r, c = nr, nc


def _legend_mapping(grid: np.ndarray, bg: int, wall: int) -> dict[int, int]:
    special = (grid != int(bg)) & (grid != int(wall))
    comps = _components(special)
    assert len(comps) == 6, "expected exactly three legend rows"

    info = []
    for coords in comps:
        colors = np.unique(grid[coords[:, 0], coords[:, 1]])
        assert colors.size == 1, "legend parts should be monochrome"
        info.append(
            {
                "coords": coords,
                "color": int(colors[0]),
                "area": int(coords.shape[0]),
                "bbox": _bbox(coords),
            }
        )

    all_c0 = min(item["bbox"][1] for item in info)
    all_c1 = max(item["bbox"][3] for item in info)
    legend_on_left = all_c0 <= (grid.shape[1] - 1 - all_c1)

    info.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    rows = [info[i : i + 2] for i in range(0, len(info), 2)]
    assert len(rows) == 3 and all(len(row) == 2 for row in rows), "legend must have three paired rows"

    mapping: dict[int, int] = {}
    for row in rows:
        row = sorted(row, key=lambda item: item["bbox"][1])
        assert abs(row[0]["bbox"][0] - row[1]["bbox"][0]) <= 1, "legend pairs should share row bands"
        if legend_on_left:
            label, swatch = row[0], row[1]
        else:
            label, swatch = row[1], row[0]
        mapping[int(label["area"])] = int(swatch["color"])

    assert len(mapping) == 3, "legend counts must be unique"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    For each enclosed region, count the corners of its wall boundary, read the matching color
    from the corner legend, and draw a cross from the region's bbox-centered cell to the walls.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    h, w = map(int, g.shape)

    bg = _mode_color(g)
    enclosed_regions = [coords for coords in _components(g == int(bg)) if not _touches_edge(coords, h, w)]
    assert enclosed_regions, "expected enclosed background regions"

    boundary_counts: dict[int, int] = defaultdict(int)
    for coords in enclosed_regions:
        for r, c in coords.tolist():
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != bg:
                    boundary_counts[int(g[nr, nc])] += 1
    wall = max(boundary_counts, key=boundary_counts.get)

    mapping = _legend_mapping(g, bg, wall)
    out = g.copy()
    for coords in enclosed_regions:
        corners = _count_region_corners(coords)
        assert corners in mapping, f"missing legend entry for {corners} corners"
        _draw_cross(out, coords, mapping[corners])
    return out


# --------------- generator ---------------

BG = 0
WALL = 5
LABEL = 1
ROOM_COUNTS = (4, 6, 8)
GRID_SIDE_MIN = 24
GRID_SIDE_MAX = 30
ROOM_TOTAL_MIN = 3
ROOM_TOTAL_MAX = 5
ROOM_PLACE_TRIES = 120
GENERATION_ATTEMPTS = 300
LEGEND_ROW_H = 2
LEGEND_ROW_GAP = 1
LEGEND_COL_GAP = 1
SWATCH_H = 2
SWATCH_W = 2


def _rect_shape() -> np.ndarray:
    h = random.choice((3, 5, 7, 9))
    w = random.choice((3, 5, 7, 9))
    return np.ones((h, w), dtype=bool)


def _corner_cut_shape() -> np.ndarray:
    for _ in range(100):
        h = random.choice((5, 7, 9))
        w = random.choice((5, 7, 9))
        cut_h = random.randint(1, h // 2)
        cut_w = random.randint(1, w // 2)
        corner = random.choice(("tl", "tr", "bl", "br"))
        mask = np.ones((h, w), dtype=bool)
        if corner == "tl":
            mask[:cut_h, :cut_w] = False
        elif corner == "tr":
            mask[:cut_h, w - cut_w :] = False
        elif corner == "bl":
            mask[h - cut_h :, :cut_w] = False
        else:
            mask[h - cut_h :, w - cut_w :] = False
        if mask[h // 2, w // 2] and _count_corners_from_mask(mask) == 6:
            return mask
    raise RuntimeError("failed to build 6-corner room")


def _notch_shape() -> np.ndarray:
    for _ in range(100):
        h = random.choice((5, 7, 9, 11))
        w = random.choice((5, 7, 9, 11))
        side = random.choice(("top", "bottom", "left", "right"))
        mask = np.ones((h, w), dtype=bool)
        if side in {"top", "bottom"}:
            depth = random.randint(1, h // 2)
            notch_w = random.randint(1, w - 2)
            start = random.randint(1, w - 1 - notch_w)
            if side == "top":
                mask[:depth, start : start + notch_w] = False
            else:
                mask[h - depth :, start : start + notch_w] = False
        else:
            depth = random.randint(1, w // 2)
            notch_h = random.randint(1, h - 2)
            start = random.randint(1, h - 1 - notch_h)
            if side == "left":
                mask[start : start + notch_h, :depth] = False
            else:
                mask[start : start + notch_h, w - depth :] = False
        if mask[h // 2, w // 2] and _count_corners_from_mask(mask) == 8:
            return mask
    raise RuntimeError("failed to build 8-corner room")


def _interior_shape(corners: int) -> np.ndarray:
    if corners == 4:
        return _rect_shape()
    if corners == 6:
        return _corner_cut_shape()
    if corners == 8:
        return _notch_shape()
    raise AssertionError("unsupported corner count")


def _wall_patch(interior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    region = np.pad(interior.astype(bool), 1)
    wall = np.zeros_like(region, dtype=bool)
    for r, c in np.argwhere(region):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < region.shape[0] and 0 <= nc < region.shape[1] and not region[nr, nc]:
                    wall[nr, nc] = True
    wall &= ~region
    return region, wall


def _label_shape(count: int) -> np.ndarray:
    assert count % 2 == 0
    return np.ones((2, count // 2), dtype=bool)


def _build_legend(mapping: dict[int, int], corner: str) -> np.ndarray:
    labels = {count: _label_shape(count) for count in sorted(mapping)}
    max_label_w = max(label.shape[1] for label in labels.values())
    height = 3 * LEGEND_ROW_H + 2 * LEGEND_ROW_GAP
    width = max_label_w + LEGEND_COL_GAP + SWATCH_W
    legend = np.full((height, width), BG, dtype=int)
    left_side = corner.endswith("l")

    for idx, count in enumerate(sorted(mapping)):
        r0 = idx * (LEGEND_ROW_H + LEGEND_ROW_GAP)
        label = labels[count]
        if left_side:
            label_c0 = 0
            swatch_c0 = max_label_w + LEGEND_COL_GAP
        else:
            swatch_c0 = 0
            label_c0 = width - label.shape[1]
        legend[r0 : r0 + label.shape[0], label_c0 : label_c0 + label.shape[1]][label] = LABEL
        legend[r0 : r0 + SWATCH_H, swatch_c0 : swatch_c0 + SWATCH_W] = mapping[count]
    return legend


def _paste_legend(grid: np.ndarray, legend: np.ndarray, corner: str) -> tuple[int, int, int, int]:
    h, w = map(int, grid.shape)
    lh, lw = map(int, legend.shape)
    top = 0 if corner.startswith("t") else h - lh
    left = 0 if corner.endswith("l") else w - lw
    grid[top : top + lh, left : left + lw] = legend
    return top, left, top + lh - 1, left + lw - 1


def _draw_expected(grid: np.ndarray, regions: list[dict], mapping: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for region in regions:
        _draw_cross(out, region["coords"], mapping[region["corners"]])
    return out


def _sample_room_specs(total_rooms: int) -> list[dict]:
    counts = list(ROOM_COUNTS)
    counts.extend(random.choice(ROOM_COUNTS) for _ in range(total_rooms - len(ROOM_COUNTS)))
    random.shuffle(counts)

    specs = []
    for count in counts:
        interior = _interior_shape(count)
        region, wall = _wall_patch(interior)
        specs.append(
            {
                "corners": count,
                "region": region,
                "wall": wall,
                "footprint_h": int(region.shape[0]),
                "footprint_w": int(region.shape[1]),
                "area": int(region.size),
            }
        )
    specs.sort(key=lambda item: (item["footprint_h"] * item["footprint_w"]), reverse=True)
    return specs


def _reserve(occupied: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _expanded_box(top: int, left: int, height: int, width: int, grid_h: int, grid_w: int) -> tuple[int, int, int, int]:
    return (
        max(0, top - 1),
        max(0, left - 1),
        min(grid_h - 1, top + height),
        min(grid_w - 1, left + width),
    )


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    width = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    grid = np.full((height, width), BG, dtype=int)

    swatches = random.sample([c for c in range(10) if c not in (BG, WALL, LABEL)], 3)
    mapping = dict(zip(ROOM_COUNTS, swatches, strict=False))
    legend_corner = random.choice(("tl", "tr", "bl", "br"))
    legend = _build_legend(mapping, legend_corner)
    if legend.shape[0] + 6 > height or legend.shape[1] + 6 > width:
        return None
    legend_box = _paste_legend(grid, legend, legend_corner)

    occupied = np.zeros((height, width), dtype=bool)
    _reserve(
        occupied,
        (
            max(0, legend_box[0] - 1),
            max(0, legend_box[1] - 1),
            min(height - 1, legend_box[2] + 1),
            min(width - 1, legend_box[3] + 1),
        ),
    )

    total_rooms = random.randint(ROOM_TOTAL_MIN, ROOM_TOTAL_MAX)
    specs = _sample_room_specs(total_rooms)
    regions: list[dict] = []

    for spec in specs:
        fh = spec["footprint_h"]
        fw = spec["footprint_w"]
        max_top = height - fh - 2
        max_left = width - fw - 2
        if max_top < 1 or max_left < 1:
            return None

        placed = False
        for _ in range(ROOM_PLACE_TRIES):
            top = random.randint(1, max_top)
            left = random.randint(1, max_left)
            reserve_box = _expanded_box(top, left, fh, fw, height, width)
            if occupied[reserve_box[0] : reserve_box[2] + 1, reserve_box[1] : reserve_box[3] + 1].any():
                continue

            patch = grid[top : top + fh, left : left + fw]
            patch[spec["wall"]] = WALL
            coords = np.argwhere(spec["region"]) + np.array([top, left])
            assert _count_region_corners(coords) == spec["corners"]
            regions.append({"coords": coords.astype(int), "corners": spec["corners"]})
            _reserve(occupied, reserve_box)
            placed = True
            break

        if not placed:
            return None

    expected = _draw_expected(grid, regions, mapping)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid sample after {GENERATION_ATTEMPTS} attempts")
