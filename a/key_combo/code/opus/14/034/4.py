from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

CYCLE = (BLUE, RED, GREEN, PINK)

GRID_HEIGHT_RANGE = (18, 24)
GRID_WIDTH_RANGE = (14, 18)
TOTAL_SPRITES_RANGE = (8, 12)
MAX_STACK = 4
GEN_ATTEMPTS = 400


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors8(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_cycle(grid: np.ndarray) -> tuple[int, ...]:
    top = [int(v) for v in grid[0] if int(v) != BACKGROUND]
    assert len(top) >= 4, "expected a 1x4 legend on the top row"
    cycle = tuple(top[:4])
    assert len(cycle) == 4
    return cycle


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    for comp in components_mask(grid == GREY):
        assert comp.shape[0] == 4, "each sprite must have 4 grey cells"
        r0, c0, r1, c1 = bbox_coords(comp)
        assert (r1 - r0, c1 - c0) == (1, 1), "each grey component must be a 2x2 square"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY), "sprite bbox must be solid"
        sprites.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1})
    assert TOTAL_SPRITES_RANGE[0] <= len(sprites) <= TOTAL_SPRITES_RANGE[1]
    return sprites


def _settle_sprites(sprites: list[dict], shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    occupied[h - 1, :] = True
    settled: list[dict] = []

    for sprite in sorted(sprites, key=lambda sp: (sp["r1"], sp["c0"]), reverse=True):
        top = int(sprite["r0"])
        c0 = int(sprite["c0"])
        while not occupied[top + 2, c0] and not occupied[top + 2, c0 + 1]:
            top += 1
        occupied[top : top + 2, c0 : c0 + 2] = True
        settled.append({"r0": top, "c0": c0, "r1": top + 1, "c1": c0 + 1, "src_r0": sprite["r0"]})

    return settled


def _paint_square(grid: np.ndarray, r0: int, c0: int, color: int) -> None:
    grid[r0 : r0 + 2, c0 : c0 + 2] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the grey 2x2 sprites onto the yellow floor, then recolor each stack by bottom-up cycle order.

    Input structure:
    - A black background contains a full-width yellow ground on the last row and a 1x4 legend on the top-left.
    - The legend's four visible colors define a cycle read left-to-right.
    - The remaining foreground objects are separated grey 2x2 square sprites located above the ground.

    Rule:
    - Simulate gravity for all grey sprites, processing the ones with lower initial bottoms first so each sprite falls until the next row would hit the yellow ground or a previously settled sprite.
    - Group settled sprites by their 2-column start position.
    - Within each group, sort sprites from lowest to highest and recolor the k-th sprite with the k-th legend color.
    - Keep the legend and the yellow ground unchanged, and preserve the original grid size.

    Key insight:
    - The final color of a sprite is not read from its starting height directly; it is determined by its bottom-up rank inside the stack created by the gravity simulation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"

    cycle = _extract_cycle(g)
    sprites = _extract_sprites(g)
    settled = _settle_sprites(sprites, g.shape)

    out = np.array(g, copy=True)
    out[g == GREY] = BACKGROUND

    groups: dict[int, list[dict]] = defaultdict(list)
    for sprite in settled:
        groups[int(sprite["c0"])].append(sprite)

    for c0, group in groups.items():
        ordered = sorted(group, key=lambda sp: sp["r1"], reverse=True)
        for k, sprite in enumerate(ordered):
            assert k < len(cycle), "stack exceeds legend cycle length"
            _paint_square(out, int(sprite["r0"]), int(c0), int(cycle[k]))

    return out


def _sample_columns(width: int, n_groups: int) -> list[int] | None:
    candidates = list(range(4, width - 1))
    for _ in range(200):
        cols = sorted(random.sample(candidates, n_groups))
        if all(b - a >= 3 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _sample_group_counts(max_groups: int) -> list[int] | None:
    for _ in range(300):
        total = random.randint(*TOTAL_SPRITES_RANGE)
        min_groups = max(3, (total + MAX_STACK - 1) // MAX_STACK)
        if min_groups > max_groups:
            continue
        n_groups = random.randint(min_groups, min(max_groups, 5))
        counts = [1] * n_groups
        remaining = total - n_groups
        while remaining > 0:
            idx = random.randrange(n_groups)
            if counts[idx] < MAX_STACK:
                counts[idx] += 1
                remaining -= 1
        if 1 not in counts:
            continue
        if max(counts) < 3:
            continue
        if len(set(counts)) < 2:
            continue
        if total >= 10 and 4 not in counts and random.random() < 0.55:
            continue
        random.shuffle(counts)
        return counts
    return None


def _canonical_rows(height: int, c0: int, count: int, group_index: int) -> list[int]:
    slots = list(range(2, height - 3, 3))
    assert len(slots) >= count
    shift = (2 * c0 + count + group_index) % len(slots)
    rotated = slots[shift:] + slots[:shift]
    rows = sorted(rotated[:count])
    return [int(r) for r in rows]


def _build_input(height: int, width: int, columns: list[int], counts: list[int]) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[0, :4] = np.array(CYCLE, dtype=int)
    grid[height - 1, :] = YELLOW
    for group_index, (c0, count) in enumerate(zip(columns, counts)):
        for r0 in _canonical_rows(height, c0, count, group_index):
            _paint_square(grid, int(r0), int(c0), GREY)
    return grid


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    h, w = inp.shape
    if not np.all(inp[h - 1, :] == YELLOW):
        return False
    if tuple(map(int, inp[0, :4])) != CYCLE:
        return False
    if np.any(inp[0, 4:] != BACKGROUND):
        return False
    if not np.all(out[h - 1, :] == YELLOW):
        return False
    if tuple(map(int, out[0, :4])) != CYCLE:
        return False
    if set(map(int, np.unique(inp))) - {BACKGROUND, YELLOW, GREY, *CYCLE}:
        return False

    sprites = _extract_sprites(inp)
    groups: dict[int, list[dict]] = defaultdict(list)
    for sp in sprites:
        if sp["r0"] < 2 or sp["c0"] < 4 or sp["r1"] >= h - 1:
            return False
        groups[int(sp["c0"])].append(sp)

    counts = [len(v) for v in groups.values()]
    if not counts or max(counts) > MAX_STACK or max(counts) < 3 or 1 not in counts:
        return False
    if len(set(counts)) < 2:
        return False

    # Anti-shortcut checks:
    # 1) same input top-row can map to different output colors
    # 2) same output color can arise from different input top-rows
    settled = _settle_sprites(sprites, inp.shape)
    by_col: dict[int, list[dict]] = defaultdict(list)
    for sp in settled:
        by_col[int(sp["c0"])].append(sp)
    input_row_to_colors: dict[int, set[int]] = defaultdict(set)
    color_to_input_rows: dict[int, set[int]] = defaultdict(set)
    for c0, group in by_col.items():
        ordered = sorted(group, key=lambda sp: sp["r1"], reverse=True)
        for k, sp in enumerate(ordered):
            final_color = int(CYCLE[k])
            input_row_to_colors[int(sp["src_r0"])].add(final_color)
            color_to_input_rows[final_color].add(int(sp["src_r0"]))
            expected_top = h - 3 - 2 * k
            if sp["r0"] != expected_top:
                return False
            if not np.all(out[expected_top : expected_top + 2, c0 : c0 + 2] == final_color):
                return False

    if not any(len(colors) >= 2 for colors in input_row_to_colors.values()):
        return False
    if not any(len(rows) >= 2 for rows in color_to_input_rows.values()):
        return False

    # Different column layouts should not collapse to the same settled output.
    color_mask = (out != BACKGROUND) & (out != YELLOW)
    if int(color_mask.sum()) <= 16:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    max_groups = max(0, (width - 6) // 3 + 1)
    if max_groups < 3:
        return None

    counts = _sample_group_counts(max_groups)
    if counts is None:
        return None
    columns = _sample_columns(width, len(counts))
    if columns is None:
        return None

    inp = _build_input(height, width, columns, counts)
    out = main(inp)

    if not _semantic_checks(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-stack color-cycle input")
