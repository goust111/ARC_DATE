from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _corner_candidates(grid: np.ndarray) -> list[tuple[str, dict[int, int], np.ndarray]]:
    h, w = grid.shape
    candidates: list[tuple[str, dict[int, int], np.ndarray]] = []

    def build(name: str, rows: list[int], left_col: int) -> None:
        if len(rows) < 2:
            return
        mapping: dict[int, int] = {}
        for r in rows:
            src = int(grid[r, left_col])
            dst = int(grid[r, left_col + 1])
            if src == BG or dst == BG or src == dst or src in mapping:
                return
            mapping[src] = dst
        mask = np.zeros_like(grid, dtype=bool)
        for r in rows:
            mask[r, left_col : left_col + 2] = True
        candidates.append((name, mapping, mask))

    rows = []
    for r in range(h):
        if int(grid[r, 0]) != BG and int(grid[r, 1]) != BG:
            rows.append(r)
        else:
            break
    build("tl", rows, 0)

    rows = []
    for r in range(h):
        if int(grid[r, w - 2]) != BG and int(grid[r, w - 1]) != BG:
            rows.append(r)
        else:
            break
    build("tr", rows, w - 2)

    rows = []
    for r in range(h - 1, -1, -1):
        if int(grid[r, 0]) != BG and int(grid[r, 1]) != BG:
            rows.append(r)
        else:
            break
    build("bl", list(reversed(rows)), 0)

    rows = []
    for r in range(h - 1, -1, -1):
        if int(grid[r, w - 2]) != BG and int(grid[r, w - 1]) != BG:
            rows.append(r)
        else:
            break
    build("br", list(reversed(rows)), w - 2)

    return candidates


def _detect_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    candidates = _corner_candidates(grid)
    if not candidates:
        raise ValueError("legend not found")
    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and len(candidates[1][1]) == len(best[1]):
        raise ValueError("legend is ambiguous")
    return best[1], best[2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge each source color's objects by union bbox, keep source pixels on the bbox border,
    recolor source pixels in the bbox interior using the corner legend, preserve background,
    and erase the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mapping, legend_mask = _detect_legend(g)
    out = g.copy()
    out[legend_mask] = BG

    for src, dst in mapping.items():
        coords = np.argwhere((g == src) & (~legend_mask))
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if rr in (int(r0), int(r1)) or cc in (int(c0), int(c1)):
                out[rr, cc] = src
            else:
                out[rr, cc] = dst
    return out


def _can_place(local_occ: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    ph, pw = patch.shape
    h, w = local_occ.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    for dr in range(ph):
        for dc in range(pw):
            if int(patch[dr, dc]) == 0:
                continue
            rr = top + dr
            cc = left + dc
            if local_occ[rr, cc]:
                return False
            for ddr, ddc in DIR4:
                nr = rr + ddr
                nc = cc + ddc
                if 0 <= nr < h and 0 <= nc < w and local_occ[nr, nc]:
                    return False
    return True


def _stamp(arr: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    ph, pw = patch.shape
    for dr in range(ph):
        for dc in range(pw):
            if int(patch[dr, dc]) != 0:
                arr[top + dr, left + dc] = color


def _collect_component(grid: np.ndarray, start: tuple[int, int], color: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _component_count(grid: np.ndarray, color: int) -> int:
    seen: set[tuple[int, int]] = set()
    count = 0
    for r, c in np.argwhere(grid == color):
        cell = (int(r), int(c))
        if cell in seen:
            continue
        seen |= _collect_component(grid, cell, color)
        count += 1
    return count


def _place_color_objects(
    rng: np.random.Generator,
    grid: np.ndarray,
    zone: tuple[int, int, int, int],
    src: int,
) -> bool:
    zr0, zc0, zr1, zc1 = zone
    zh = zr1 - zr0 + 1
    zw = zc1 - zc0 + 1
    if zh < 6 or zw < 6:
        return False

    bbox_h = int(rng.integers(6, min(10, zh) + 1))
    bbox_w = int(rng.integers(6, min(10, zw) + 1))
    btop = int(rng.integers(zr0, zr1 - bbox_h + 2))
    bleft = int(rng.integers(zc0, zc1 - bbox_w + 2))

    local = np.zeros((bbox_h, bbox_w), dtype=int)
    object_count = int(rng.integers(2, 5))
    diagonal = int(rng.integers(0, 2))

    anchor_patch = np.array([[1]], dtype=int)
    rich_patch = np.array([[1, 1], [1, 1]], dtype=int)
    if diagonal == 0:
        if not _can_place(local > 0, anchor_patch, 0, 0):
            return False
        _stamp(local, anchor_patch, 0, 0, 1)
        if not _can_place(local > 0, rich_patch, bbox_h - 2, bbox_w - 2):
            return False
        _stamp(local, rich_patch, bbox_h - 2, bbox_w - 2, 1)
    else:
        if not _can_place(local > 0, anchor_patch, 0, bbox_w - 1):
            return False
        _stamp(local, anchor_patch, 0, bbox_w - 1, 1)
        if not _can_place(local > 0, rich_patch, bbox_h - 2, 0):
            return False
        _stamp(local, rich_patch, bbox_h - 2, 0, 1)

    extra_candidates = [
        (anchor_patch, 1, bbox_w // 2),
        (anchor_patch, bbox_h // 2, max(1, bbox_w // 3)),
        (anchor_patch, bbox_h // 2, bbox_w // 2),
        (anchor_patch, max(1, bbox_h // 3), max(1, (2 * bbox_w) // 3 - 1)),
        (anchor_patch, bbox_h - 3, max(1, bbox_w // 2)),
    ]
    order = rng.permutation(len(extra_candidates))
    placed = 2
    for idx in order:
        if placed >= object_count:
            break
        patch, top, left = extra_candidates[int(idx)]
        if _can_place(local > 0, patch, int(top), int(left)):
            _stamp(local, patch, int(top), int(left), 1)
            placed += 1
    if placed != object_count:
        return False

    target = grid[btop : btop + bbox_h, bleft : bleft + bbox_w]
    if np.any((local > 0) & (target != BG)):
        return False
    target[local > 0] = src

    coords = np.argwhere(grid == src)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    transformed = 0
    for r, c in coords:
        if int(r) not in (int(r0), int(r1)) and int(c) not in (int(c0), int(c1)):
            transformed += 1
    return transformed > 0


def _legend_cells(h: int, w: int, n: int, corner: str) -> list[tuple[int, int, int]]:
    if corner == "tl":
        return [(r, 0, 1) for r in range(n)]
    if corner == "tr":
        return [(r, w - 2, w - 1) for r in range(n)]
    if corner == "bl":
        return [(h - n + r, 0, 1) for r in range(n)]
    return [(h - n + r, w - 2, w - 1) for r in range(n)]


def _zones(h: int, w: int, legend_n: int, corner: str) -> list[tuple[int, int, int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    zones = [
        [1, 1, mid_r - 2, mid_c - 2],
        [1, mid_c + 1, mid_r - 2, w - 2],
        [mid_r + 1, 1, h - 2, mid_c - 2],
        [mid_r + 1, mid_c + 1, h - 2, w - 2],
    ]

    if corner == "tl":
        zones[0][0] = max(zones[0][0], legend_n + 1)
        zones[0][1] = max(zones[0][1], 3)
    elif corner == "tr":
        zones[1][0] = max(zones[1][0], legend_n + 1)
        zones[1][3] = min(zones[1][3], w - 4)
    elif corner == "bl":
        zones[2][2] = min(zones[2][2], h - legend_n - 2)
        zones[2][1] = max(zones[2][1], 3)
    else:
        zones[3][2] = min(zones[3][2], h - legend_n - 2)
        zones[3][3] = min(zones[3][3], w - 4)

    out = []
    for r0, c0, r1, c1 in zones:
        if r1 - r0 + 1 >= 6 and c1 - c0 + 1 >= 6:
            out.append((int(r0), int(c0), int(r1), int(c1)))
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    colors = np.arange(1, 10)

    for _ in range(400):
        h = int(rng.integers(20, 27))
        w = int(rng.integers(20, 27))
        n_colors = int(rng.integers(2, 5))
        corner = ["tl", "tr", "bl", "br"][int(rng.integers(0, 4))]

        palette = rng.choice(colors, size=2 * n_colors, replace=False)
        src_colors = [int(x) for x in palette[:n_colors]]
        dst_colors = [int(x) for x in palette[n_colors:]]
        mapping = dict(zip(src_colors, dst_colors, strict=False))

        grid = np.zeros((h, w), dtype=int)
        for idx, (r, c0, c1) in enumerate(_legend_cells(h, w, n_colors, corner)):
            grid[r, c0] = src_colors[idx]
            grid[r, c1] = dst_colors[idx]

        zones = _zones(h, w, n_colors, corner)
        if len(zones) < n_colors:
            continue
        zone_order = rng.permutation(len(zones))[:n_colors]

        ok = True
        for src, zone_idx in zip(src_colors, zone_order, strict=False):
            if not _place_color_objects(rng, grid, zones[int(zone_idx)], src):
                ok = False
                break
        if not ok:
            continue

        parsed_mapping, legend_mask = _detect_legend(grid)
        if parsed_mapping != mapping:
            continue
        scene = grid.copy()
        scene[legend_mask] = BG
        if any(not (2 <= _component_count(scene, src) <= 4) for src in src_colors):
            continue

        out = main(grid)
        if np.array_equal(out, grid):
            continue
        if np.any(out[legend_mask] != BG):
            continue
        if any(not np.any(out == dst) for dst in dst_colors):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
