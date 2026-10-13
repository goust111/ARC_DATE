from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GEN_RETRIES = 500
GROUP_RETRIES = 120
MAX_SCENE_WIDTH = 36
MAX_SCENE_HEIGHT = 30


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _scale_nearest(patch: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    src_h, src_w = patch.shape
    row_idx = (np.arange(out_h) * src_h) // out_h
    col_idx = (np.arange(out_w) * src_w) // out_w
    return patch[row_idx[:, None], col_idx[None, :]]


def _is_frame(grid: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(patch[0, :] == GREY):
        return False
    if not np.all(patch[-1, :] == GREY):
        return False
    if not np.all(patch[:, 0] == GREY):
        return False
    if not np.all(patch[:, -1] == GREY):
        return False
    if not np.all(patch[1:-1, 1:-1] == BG):
        return False
    return True


def _find_containers(grid: np.ndarray) -> list[dict[str, object]]:
    containers: list[dict[str, object]] = []
    for coords in _components(grid == GREY, connectivity=4):
        if not _is_frame(grid, coords):
            continue
        containers.append({"bbox": _bbox(coords)})
    containers.sort(key=lambda item: tuple(item["bbox"]))  # type: ignore[arg-type]
    return containers


def _find_markers(grid: np.ndarray, containers: list[dict[str, object]]) -> list[dict[str, object]]:
    markers: list[dict[str, object]] = []
    for container in containers:
        r0, c0, r1, _c1 = container["bbox"]  # type: ignore[index]
        assert c0 > 0, "marker column must exist"
        marker_cells: list[tuple[int, int, int]] = []
        for r in range(r0, r1 + 1):
            value = int(grid[r, c0 - 1])
            if value not in (BG, GREY):
                marker_cells.append((r, c0 - 1, value))
        assert len(marker_cells) == 1, "expected exactly one left-wall marker per container"
        rr, cc, color = marker_cells[0]
        markers.append({"pos": (rr, cc), "color": color})
    return markers


def _seed_components(grid: np.ndarray, markers: list[dict[str, object]]) -> list[dict[str, object]]:
    mask = (grid != BG) & (grid != GREY)
    for marker in markers:
        rr, cc = marker["pos"]  # type: ignore[index]
        mask[rr, cc] = False

    seeds: list[dict[str, object]] = []
    for coords in _components(mask, connectivity=8):
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        seeds.append({"bbox": (r0, c0, r1, c1), "patch": patch})
    return seeds


def _pair_seeds(
    containers: list[dict[str, object]], seeds: list[dict[str, object]]
) -> list[tuple[dict[str, object], dict[str, object]]]:
    used: set[int] = set()
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []

    for container in containers:
        r0, c0, _r1, c1 = container["bbox"]  # type: ignore[index]
        best_idx = -1
        best_score: tuple[int, int, int, int, int, int] | None = None
        for idx, seed in enumerate(seeds):
            if idx in used:
                continue
            sr0, sc0, sr1, sc1 = seed["bbox"]  # type: ignore[index]
            if sr1 >= r0:
                continue
            overlap = _interval_overlap(sc0, sc1, c0, c1)
            if overlap <= 0:
                continue
            gap = r0 - sr1
            center_delta = abs((sc0 + sc1) - (c0 + c1))
            score = (0 if gap == 1 else 1, gap, -overlap, center_delta, sr0, sc0)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        assert best_idx >= 0, "every container must have a seed above it"
        used.add(best_idx)
        pairs.append((container, seeds[best_idx]))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale each seed pattern into the container directly below it, then hollow out
    the scaled cells whose color matches the left-side marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    containers = _find_containers(g)
    assert containers, "expected at least one grey container"
    markers = _find_markers(g, containers)
    seeds = _seed_components(g, markers)
    pairs = _pair_seeds(containers, seeds)
    marker_by_bbox = {
        tuple(container["bbox"]): marker for container, marker in zip(containers, markers, strict=True)
    }

    out = g.copy()
    for container, seed in pairs:
        r0, c0, r1, c1 = container["bbox"]  # type: ignore[index]
        sr0, sc0, sr1, sc1 = seed["bbox"]  # type: ignore[index]
        marker_color = int(marker_by_bbox[tuple(container["bbox"])]["color"])  # type: ignore[index]
        patch = np.asarray(seed["patch"], dtype=int)
        assert np.any(patch == marker_color), "marker color must appear in its seed"

        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = BG

        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        scaled = _scale_nearest(patch, inner_h, inner_w)
        filled = scaled.copy()
        filled[filled == marker_color] = BG
        out[r0 + 1 : r1, c0 + 1 : c1] = filled

    return out


def _sample_seed_dim(rng: np.random.Generator, target: int, mode: str, cap: int) -> int:
    if mode == "same":
        return target
    if mode == "up":
        lo = 2
        hi = max(2, target - 1)
        return int(rng.integers(lo, hi + 1))
    lo = target + 1
    hi = max(lo, min(cap, target + 3))
    return int(rng.integers(lo, hi + 1))


def _make_seed_pattern(
    rng: np.random.Generator, seed_h: int, seed_w: int, colors: np.ndarray
) -> np.ndarray:
    mode = str(rng.choice(["cells", "rows", "cols", "checker", "blocks"]))
    patch = np.empty((seed_h, seed_w), dtype=int)

    if mode == "cells":
        patch[:] = rng.choice(colors, size=(seed_h, seed_w))
    elif mode == "rows":
        row_colors = rng.choice(colors, size=seed_h)
        patch[:] = np.asarray(row_colors[:, None]).repeat(seed_w, axis=1)
    elif mode == "cols":
        col_colors = rng.choice(colors, size=seed_w)
        patch[:] = np.asarray(col_colors[None, :]).repeat(seed_h, axis=0)
    elif mode == "checker":
        shifts = rng.integers(0, len(colors), size=seed_h)
        for r in range(seed_h):
            for c in range(seed_w):
                patch[r, c] = int(colors[(r + c + int(shifts[r])) % len(colors)])
    else:
        split_r = 1 if seed_h == 2 else int(rng.integers(1, seed_h))
        split_c = 1 if seed_w == 2 else int(rng.integers(1, seed_w))
        a, b, c, d = rng.choice(colors, size=4, replace=True)
        patch[:split_r, :split_c] = int(a)
        patch[:split_r, split_c:] = int(b)
        patch[split_r:, :split_c] = int(c)
        patch[split_r:, split_c:] = int(d)

    slots = [(r, c) for r in range(seed_h) for c in range(seed_w)]
    rng.shuffle(slots)
    for color, (r, c) in zip(colors, slots, strict=False):
        patch[r, c] = int(color)

    if len(np.unique(patch)) < 2:
        patch[0, 0] = int(colors[0])
        patch[-1, -1] = int(colors[1])
    return patch


def _build_group(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    inner_h = int(rng.integers(3, 8))
    inner_w = int(rng.integers(4, 9))
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    width_mode = str(rng.choice(["up", "down", "same"]))
    height_mode = str(rng.choice(["up", "down", "same"]))
    seed_h = _sample_seed_dim(rng, inner_h, height_mode, cap=10)
    seed_w = _sample_seed_dim(rng, inner_w, width_mode, cap=11)

    color_count = int(rng.integers(2, 5))
    colors = rng.choice(np.array(PALETTE, dtype=int), size=color_count, replace=False)

    for _ in range(GROUP_RETRIES):
        seed = _make_seed_pattern(rng, seed_h, seed_w, colors)
        marker_color = int(rng.choice(np.unique(seed)))
        scaled = _scale_nearest(seed, inner_h, inner_w)
        if not np.any(scaled == marker_color):
            continue
        if not np.any(scaled != marker_color):
            continue

        shift_min = -(seed_w - 1)
        shift_max = outer_w - 1
        shift = int(rng.integers(shift_min, shift_max + 1))

        container_r0 = seed_h
        container_c0 = 1
        seed_r0 = 0
        seed_c0 = container_c0 + shift
        marker_r = container_r0 + int(rng.integers(1, outer_h - 1))
        marker_c = container_c0 - 1

        min_c = min(seed_c0, marker_c, container_c0)
        max_c = max(seed_c0 + seed_w - 1, marker_c, container_c0 + outer_w - 1)
        total_w = max_c - min_c + 1
        total_h = seed_h + outer_h
        col_shift = -min_c

        seed_c0 += col_shift
        container_c0 += col_shift
        marker_c += col_shift

        inp = np.full((total_h, total_w), BG, dtype=int)
        out = np.full((total_h, total_w), BG, dtype=int)

        r0 = container_r0
        c0 = container_c0
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        inp[r0, c0 : c1 + 1] = GREY
        inp[r1, c0 : c1 + 1] = GREY
        inp[r0 : r1 + 1, c0] = GREY
        inp[r0 : r1 + 1, c1] = GREY
        inp[seed_r0 : seed_r0 + seed_h, seed_c0 : seed_c0 + seed_w] = seed
        inp[marker_r, marker_c] = marker_color

        out[:] = inp
        out[seed_r0 : seed_r0 + seed_h, seed_c0 : seed_c0 + seed_w] = BG
        filled = scaled.copy()
        filled[filled == marker_color] = BG
        out[r0 + 1 : r1, c0 + 1 : c1] = filled
        return inp, out

    return None


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
    values: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        values.append(int(cut - prev))
        prev = int(cut)
    return values


def _layout_scene(
    groups: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray] | None:
    order = list(range(len(groups)))
    rng.shuffle(order)
    rows = int(rng.integers(1, min(3, len(groups)) + 1))
    counts = _random_composition(len(groups), rows, rng)
    top_margin = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))
    row_gaps = [int(rng.integers(2, 6)) for _ in range(max(0, rows - 1))]

    row_specs: list[dict[str, object]] = []
    cursor = 0
    max_width = 0
    total_height = top_margin

    for row_idx, count in enumerate(counts):
        ids = order[cursor : cursor + count]
        cursor += count
        left_margin = int(rng.integers(1, 4))
        right_margin = int(rng.integers(1, 4))
        gaps = [int(rng.integers(2, 5)) for _ in range(max(0, count - 1))]
        row_groups = [groups[idx] for idx in ids]
        row_h = max(inp.shape[0] for inp, _ in row_groups)
        row_w = left_margin + right_margin + sum(inp.shape[1] for inp, _ in row_groups) + sum(gaps)
        max_width = max(max_width, row_w)
        row_specs.append(
            {
                "ids": ids,
                "left_margin": left_margin,
                "gaps": gaps,
                "row_h": row_h,
            }
        )
        total_height += row_h
        if row_idx != len(counts) - 1:
            total_height += row_gaps[row_idx]

    total_height += bottom_margin
    if max_width > MAX_SCENE_WIDTH or total_height > MAX_SCENE_HEIGHT:
        return None

    inp_scene = np.full((total_height, max_width), BG, dtype=int)
    out_scene = np.full((total_height, max_width), BG, dtype=int)
    y = top_margin

    for row_idx, row_spec in enumerate(row_specs):
        row_h = int(row_spec["row_h"])  # type: ignore[index]
        x = int(row_spec["left_margin"])  # type: ignore[index]
        gaps = row_spec["gaps"]  # type: ignore[assignment]
        ids = row_spec["ids"]  # type: ignore[assignment]
        for pos, group_idx in enumerate(ids):
            inp_patch, out_patch = groups[int(group_idx)]
            gh, gw = inp_patch.shape
            y_off = int(rng.integers(0, row_h - gh + 1))
            inp_scene[y + y_off : y + y_off + gh, x : x + gw] = inp_patch
            out_scene[y + y_off : y + y_off + gh, x : x + gw] = out_patch
            x += gw
            if pos < len(gaps):
                x += int(gaps[pos])
        y += row_h
        if row_idx != len(row_specs) - 1:
            y += row_gaps[row_idx]

    return inp_scene, out_scene


def _generate_pair() -> tuple[np.ndarray, np.ndarray] | None:
    rng = np.random.default_rng()
    group_count = int(rng.integers(1, 5))
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(group_count):
        group = _build_group(rng)
        if group is None:
            return None
        groups.append(group)

    scene = _layout_scene(groups, rng)
    if scene is None:
        return None
    inp, out = scene
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        pair = _generate_pair()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid input")
