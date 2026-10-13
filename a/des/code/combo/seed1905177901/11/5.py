from __future__ import annotations

import random

import numpy as np


BG = 8
RECEIVER = 5
INTERFERENCE = 9

TOWER_COLORS = tuple(color for color in range(10) if color not in {BG, RECEIVER, INTERFERENCE})
GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
TOWER_COUNT_RANGE = (2, 4)
RECEIVER_COUNT_RANGE = (3, 5)
TOWER_CENTER_MARGIN = 3
TOWER_CENTER_GAP = 5
RECEIVER_OUTER_H_RANGE = (4, 8)
RECEIVER_OUTER_W_RANGE = (4, 8)
OBJECT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_tower(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == 9 and (r1 - r0 + 1) == 3 and (c1 - c0 + 1) == 3


def _is_receiver(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    window = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        window[r - r0, c - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(window, expected))


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _most_common_color(grid)
    towers: list[dict] = []
    receivers: list[dict] = []

    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            if _is_tower(coords):
                towers.append(
                    {
                        "color": color,
                        "bbox": (r0, c0, r1, c1),
                        "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                    }
                )
            elif _is_receiver(coords):
                receivers.append({"bbox": (r0, c0, r1, c1), "color": color})
            else:
                raise AssertionError("unexpected foreground object shape")

    assert towers, "expected at least one tower"
    assert receivers, "expected at least one receiver"
    return bg, towers, receivers


def _tower_block_mask(shape: tuple[int, int], towers: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for tower in towers:
        r0, c0, r1, c1 = tower["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _ray_mask(shape: tuple[int, int], tower: dict, blockers: np.ndarray) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = tower["bbox"]
    cr, cc = tower["center"]
    mask = np.zeros(shape, dtype=bool)
    starts = [
        (r0 - 1, cc, -1, 0),
        (r1 + 1, cc, 1, 0),
        (cr, c0 - 1, 0, -1),
        (cr, c1 + 1, 0, 1),
    ]
    for r, c, dr, dc in starts:
        while 0 <= r < h and 0 <= c < w:
            if blockers[r, c]:
                break
            mask[r, c] = True
            r += dr
            c += dc
    return mask


def _tower_ray_masks(shape: tuple[int, int], towers: list[dict]) -> list[np.ndarray]:
    blockers = _tower_block_mask(shape, towers)
    return [_ray_mask(shape, tower, blockers) for tower in towers]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each receiver interior according to the tower rays that cross it.

    Towers are 3x3 solid blocks. Receivers are one-pixel-thick hollow rectangles.
    Rays are cast from the middle of each tower side. Receiver frames act as
    transparent detectors; only towers block rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, towers, receivers = _parse_scene(g)
    ray_masks = _tower_ray_masks(g.shape, towers)

    out = np.array(g, copy=True)
    for receiver in receivers:
        r0, c0, r1, c1 = receiver["bbox"]
        interior = (slice(r0 + 1, r1), slice(c0 + 1, c1))
        hit_colors = [
            tower["color"]
            for tower, ray_mask in zip(towers, ray_masks, strict=False)
            if np.any(ray_mask[interior])
        ]
        if len(hit_colors) == 1:
            out[interior] = hit_colors[0]
        elif len(hit_colors) > 1:
            out[interior] = INTERFERENCE
        else:
            out[interior] = bg

    return out


def _bbox_conflict(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
    gap: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = bbox_a
    br0, bc0, br1, bc1 = bbox_b
    return not (
        ar1 < br0 - gap
        or br1 < ar0 - gap
        or ac1 < bc0 - gap
        or bc1 < ac0 - gap
    )


def _sample_spaced_values(count: int, low: int, high: int, min_gap: int) -> list[int] | None:
    if high < low:
        return None
    values = list(range(low, high + 1))
    for _ in range(80):
        random.shuffle(values)
        chosen: list[int] = []
        for value in values:
            if all(abs(value - other) >= min_gap for other in chosen):
                chosen.append(value)
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _sample_towers(height: int, width: int) -> list[dict] | None:
    for _ in range(120):
        count = random.randint(*TOWER_COUNT_RANGE)
        rows = _sample_spaced_values(
            count,
            TOWER_CENTER_MARGIN,
            height - TOWER_CENTER_MARGIN - 1,
            TOWER_CENTER_GAP,
        )
        cols = _sample_spaced_values(
            count,
            TOWER_CENTER_MARGIN,
            width - TOWER_CENTER_MARGIN - 1,
            TOWER_CENTER_GAP,
        )
        if rows is None or cols is None:
            continue
        random.shuffle(cols)
        colors = random.sample(TOWER_COLORS, count)
        towers = []
        for row, col, color in zip(rows, cols, colors, strict=False):
            towers.append(
                {
                    "color": int(color),
                    "bbox": (row - 1, col - 1, row + 1, col + 1),
                    "center": (row, col),
                }
            )
        return towers
    return None


def _candidate_hits(ray_masks: list[np.ndarray], bbox: tuple[int, int, int, int]) -> tuple[int, ...]:
    r0, c0, r1, c1 = bbox
    interior = (slice(r0 + 1, r1), slice(c0 + 1, c1))
    hits = [
        index
        for index, ray_mask in enumerate(ray_masks)
        if np.any(ray_mask[interior])
    ]
    return tuple(hits)


def _receiver_candidates(height: int, width: int, towers: list[dict]) -> list[dict]:
    ray_masks = _tower_ray_masks((height, width), towers)
    candidates: list[dict] = []
    for outer_h in range(RECEIVER_OUTER_H_RANGE[0], RECEIVER_OUTER_H_RANGE[1] + 1):
        for outer_w in range(RECEIVER_OUTER_W_RANGE[0], RECEIVER_OUTER_W_RANGE[1] + 1):
            for r0 in range(1, height - outer_h):
                r1 = r0 + outer_h - 1
                for c0 in range(1, width - outer_w):
                    c1 = c0 + outer_w - 1
                    bbox = (r0, c0, r1, c1)
                    if any(_bbox_conflict(bbox, tower["bbox"], gap=OBJECT_GAP) for tower in towers):
                        continue
                    hits = _candidate_hits(ray_masks, bbox)
                    mode = "multi" if len(hits) > 1 else "single" if len(hits) == 1 else "none"
                    candidates.append({"bbox": bbox, "hits": hits, "mode": mode})
    return candidates


def _choose_receivers(candidates: list[dict]) -> list[dict] | None:
    count = random.randint(*RECEIVER_COUNT_RANGE)
    desired = ["none", "single", "multi"]
    desired.extend(random.choice(desired) for _ in range(count - len(desired)))

    by_mode = {"none": [], "single": [], "multi": []}
    for candidate in candidates:
        by_mode[candidate["mode"]].append(candidate)
    if any(not by_mode[mode] for mode in {"none", "single", "multi"}):
        return None

    for mode in by_mode:
        random.shuffle(by_mode[mode])

    for _ in range(120):
        mode_order = desired[:]
        random.shuffle(mode_order)
        mode_order.sort(key=lambda mode: len(by_mode[mode]))
        chosen: list[dict] = []
        ok = True

        for mode in mode_order:
            pool = by_mode[mode]
            if len(pool) > 180:
                options = random.sample(pool, 180)
            else:
                options = pool[:]
            random.shuffle(options)
            fits = [
                candidate
                for candidate in options
                if all(not _bbox_conflict(candidate["bbox"], other["bbox"], gap=OBJECT_GAP) for other in chosen)
            ]
            if not fits:
                ok = False
                break
            chosen.append(random.choice(fits[: min(len(fits), 40)]))

        if ok:
            return chosen

    return None


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _render_input(height: int, width: int, towers: list[dict], receivers: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for tower in towers:
        r0, c0, r1, c1 = tower["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = tower["color"]
    for receiver in receivers:
        _draw_frame(grid, receiver["bbox"], RECEIVER)
    return grid


def _receiver_mode_summary(grid: np.ndarray) -> set[str]:
    _, towers, receivers = _parse_scene(grid)
    ray_masks = _tower_ray_masks(grid.shape, towers)
    seen: set[str] = set()
    for receiver in receivers:
        hits = _candidate_hits(ray_masks, receiver["bbox"])
        mode = "multi" if len(hits) > 1 else "single" if len(hits) == 1 else "none"
        seen.add(mode)
    return seen


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(*GRID_H_RANGE)
        width = random.randint(*GRID_W_RANGE)
        towers = _sample_towers(height, width)
        if towers is None:
            continue
        candidates = _receiver_candidates(height, width, towers)
        if not candidates:
            continue
        receivers = _choose_receivers(candidates)
        if receivers is None:
            continue
        grid = _render_input(height, width, towers, receivers)
        if _receiver_mode_summary(grid) != {"none", "single", "multi"}:
            continue
        try:
            main(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
