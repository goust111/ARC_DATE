from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _trim_mask_with_offset(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1], int(r0), int(c0)


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    trimmed = _trim_mask(mask)
    return trimmed.shape, tuple(int(v) for v in trimmed.astype(np.uint8).ravel())


def _match_positions(reference_mask: np.ndarray, probe_mask: np.ndarray) -> list[tuple[int, int]]:
    rh, rw = reference_mask.shape
    ph, pw = probe_mask.shape
    matches: list[tuple[int, int]] = []
    for r in range(rh - ph + 1):
        for c in range(rw - pw + 1):
            if np.array_equal(reference_mask[r : r + ph, c : c + pw], probe_mask):
                matches.append((r, c))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each small monochrome probe by copying colors from the unique
    shape-matching window inside the large multicolor reference pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _components(g != bg)
    assert len(comps) >= 2, "expected one reference object and at least one probe"

    ref_idx = max(range(len(comps)), key=lambda i: len(comps[i]))
    rr0, rc0, rr1, rc1 = _bbox(comps[ref_idx])
    reference = g[rr0 : rr1 + 1, rc0 : rc1 + 1]
    reference_mask = reference != bg

    out = g.copy()
    for i, coords in enumerate(comps):
        if i == ref_idx:
            continue
        pr0, pc0, pr1, pc1 = _bbox(coords)
        probe = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
        probe_mask = probe != bg
        matches = _match_positions(reference_mask, probe_mask)
        assert len(matches) == 1, "probe shape must match a unique reference window"
        mr, mc = matches[0]
        color_window = reference[mr : mr + probe_mask.shape[0], mc : mc + probe_mask.shape[1]]
        out_region = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
        out_region[probe_mask] = color_window[probe_mask]

    return out


def _random_shape(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        area_max = min(8, h * w - 1)
        if area_max < 3:
            continue
        target_area = rng.randint(3, area_max)

        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))

        if len(cells) != target_area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)

        if mask.sum() < 3 or mask.sum() > 8:
            continue
        if min(mask.shape) < 2:
            continue
        if mask.all():
            continue
        if len(_components(mask)) != 1:
            continue
        return mask

    raise RuntimeError("failed to sample a valid probe shape")


def _build_reference(
    masks: list[np.ndarray], rng: random.Random
) -> tuple[np.ndarray, list[dict[str, object]]]:
    max_h = max(mask.shape[0] for mask in masks)
    canvas_h = max_h + rng.randint(4, 6)

    placements: list[dict[str, object]] = []
    col = 1
    prev_w = 0
    for i, mask in enumerate(masks):
        h, w = mask.shape
        if i > 0:
            col += prev_w + rng.randint(2, 4)
        row = rng.randint(1, canvas_h - h - 1)
        placements.append({"mask": mask, "row": row, "col": col})
        prev_w = w

    canvas_w = int(placements[-1]["col"]) + placements[-1]["mask"].shape[1] + 1
    reference_mask = np.zeros((canvas_h, canvas_w), dtype=bool)

    for item in placements:
        mask = item["mask"]
        row = int(item["row"])
        col = int(item["col"])
        h, w = mask.shape
        reference_mask[row : row + h, col : col + w][mask] = True

    for left, right in zip(placements, placements[1:]):
        left_mask = left["mask"]
        right_mask = right["mask"]
        left_row = int(left["row"])
        right_row = int(right["row"])
        left_col = int(left["col"])
        right_col = int(right["col"])

        left_edge_rows = [r for r in range(left_mask.shape[0]) if left_mask[r, -1]]
        right_edge_rows = [r for r in range(right_mask.shape[0]) if right_mask[r, 0]]
        anchor_left_r = left_row + rng.choice(left_edge_rows)
        anchor_right_r = right_row + rng.choice(right_edge_rows)
        bridge_c = left_col + left_mask.shape[1]

        reference_mask[anchor_left_r, bridge_c] = True
        lo, hi = sorted((anchor_left_r, anchor_right_r))
        reference_mask[lo : hi + 1, bridge_c] = True
        reference_mask[anchor_right_r, bridge_c:right_col] = True

    assert len(_components(reference_mask)) == 1

    reference_mask, r_off, c_off = _trim_mask_with_offset(reference_mask)
    for item in placements:
        item["row"] = int(item["row"]) - r_off
        item["col"] = int(item["col"]) - c_off

    return reference_mask, placements


def _color_reference(
    reference_mask: np.ndarray, placements: list[dict[str, object]], bg: int, rng: random.Random
) -> tuple[np.ndarray, list[int]]:
    non_bg = [color for color in range(10) if color != bg]
    palette_size = rng.randint(3, min(5, len(non_bg)))
    palette = rng.sample(non_bg, palette_size)

    for _ in range(100):
        ref = np.full(reference_mask.shape, bg, dtype=int)
        coords = np.argwhere(reference_mask)
        for r, c in coords:
            ref[r, c] = rng.choice(palette)

        if len(set(int(v) for v in ref[reference_mask])) < 3:
            continue

        ok = True
        for item in placements:
            mask = item["mask"]
            row = int(item["row"])
            col = int(item["col"])
            patch = ref[row : row + mask.shape[0], col : col + mask.shape[1]]
            colors = {int(v) for v in patch[mask]}
            if len(colors) < 2:
                ok = False
                break
        if ok:
            return ref, palette

    raise RuntimeError("failed to color the reference with enough variety")


def _place_probes(
    grid: np.ndarray,
    masks: list[np.ndarray],
    probe_colors: list[int],
    right_left: int,
    right_top: int,
    cell_h: int,
    cell_w: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    slots = [(sr, sc) for sr in range(2) for sc in range(2)]
    rng.shuffle(slots)
    positions: list[tuple[int, int]] = []

    for mask, color, (slot_r, slot_c) in zip(masks, probe_colors, slots):
        h, w = mask.shape
        slot_top = right_top + slot_r * cell_h
        slot_left = right_left + slot_c * cell_w
        max_r = slot_top + cell_h - h - 1
        max_c = slot_left + cell_w - w - 1
        row = rng.randint(slot_top, max_r)
        col = rng.randint(slot_left, max_c)
        grid[row : row + h, col : col + w][mask] = color
        positions.append((row, col))

    return positions


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    num_probes = rng.randint(2, 4)

    masks: list[np.ndarray] = []
    seen_keys = set()
    for _ in range(num_probes):
        for _ in range(200):
            mask = _random_shape(rng)
            key = _shape_key(mask)
            if key not in seen_keys:
                seen_keys.add(key)
                masks.append(mask)
                break
        else:
            return None

    reference_mask, placements = _build_reference(masks, rng)
    for item in placements:
        if len(_match_positions(reference_mask, item["mask"])) != 1:
            return None

    reference, palette = _color_reference(reference_mask, placements, bg, rng)

    max_h = max(mask.shape[0] for mask in masks)
    max_w = max(mask.shape[1] for mask in masks)
    cell_h = max_h + 3
    cell_w = max_w + 3
    right_h = 2 * cell_h
    right_w = 2 * cell_w

    h = max(reference.shape[0] + 2, right_h + 2) + rng.randint(0, 3)
    w = reference.shape[1] + right_w + 5 + rng.randint(0, 3)
    if h < 12:
        h = 12
    if w < 18:
        w = 18

    grid = np.full((h, w), bg, dtype=int)
    ref_top = rng.randint(1, h - reference.shape[0] - 1)
    ref_left = 1
    grid[
        ref_top : ref_top + reference.shape[0],
        ref_left : ref_left + reference.shape[1],
    ] = reference

    right_left = w - right_w - 1
    right_top = rng.randint(1, h - right_h - 1)
    if right_left <= ref_left + reference.shape[1]:
        return None

    probe_pool = [color for color in range(10) if color != bg and color not in palette]
    if not probe_pool:
        probe_pool = [color for color in range(10) if color != bg]

    if rng.random() < 0.5:
        probe_colors = [rng.choice(probe_pool)] * num_probes
    else:
        probe_colors = [rng.choice(probe_pool) for _ in range(num_probes)]

    positions = _place_probes(
        grid, masks, probe_colors, right_left, right_top, cell_h, cell_w, rng
    )

    try:
        out = main(grid)
    except AssertionError:
        return None

    for (row, col), mask, item in zip(positions, masks, placements):
        patch = out[row : row + mask.shape[0], col : col + mask.shape[1]]
        if len({int(v) for v in patch[mask]}) < 2:
            return None
        if len(_match_positions(reference_mask, item["mask"])) != 1:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
