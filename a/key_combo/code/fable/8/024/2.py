from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, in_bounds, neighbors4


PANEL_COUNT = 5
SEPARATOR_COUNT = PANEL_COUNT - 1
BG = Color.BLACK
VOTE_THRESHOLD = 4
MAX_FRAME_NOISE = 2
GEN_RETRY_LIMIT = 400
GEN_PANEL_HEIGHTS = (4, 5)
GEN_PANEL_WIDTH_RANGE = (12, 24)
GEN_SPRITE_COLORS = tuple(c for c in Color.NOT_BLACK)


def _coords_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    seen = set()
    queue = deque([next(iter(coords))])
    seen.add(queue[0])
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            p = (int(nr), int(nc))
            if p in coords and p not in seen:
                seen.add(p)
                queue.append(p)
    return seen == coords


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    arr = np.array(sorted(cells), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return {tuple(map(int, row)) for row in arr}


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= PANEL_COUNT + SEPARATOR_COUNT, "grid too short"

    sep_rows = [
        r
        for r in range(height)
        if int(g[r, 0]) != BG and bool(np.all(g[r] == g[r, 0]))
    ]
    assert len(sep_rows) == SEPARATOR_COUNT, "expected four full-width separator rows"

    sep_color = int(g[sep_rows[0], 0])
    assert all(int(g[r, 0]) == sep_color for r in sep_rows), "separator color must be consistent"
    gaps = [b - a for a, b in zip(sep_rows, sep_rows[1:])]
    assert len(set(gaps)) == 1, "separator rows must be equally spaced"
    panel_h = gaps[0] - 1
    assert panel_h >= 1, "panel height must be positive"

    panels: list[np.ndarray] = []
    start = 0
    for row in sep_rows:
        panel = g[start:row, :]
        assert panel.shape == (panel_h, width), "all panels must have equal size"
        panels.append(panel.copy())
        start = row + 1
    last = g[start:, :]
    assert last.shape == (panel_h, width), "all panels must have equal size"
    panels.append(last.copy())
    assert len(panels) == PANEL_COUNT
    assert all(int(panel.max()) != sep_color for panel in panels), "separator color must stay outside panels"
    return panels, sep_color


def _render_panel(
    shape: tuple[int, int],
    sprite0: dict[tuple[int, int], int],
    shift: tuple[int, int],
    step: int,
) -> np.ndarray:
    panel = np.zeros(shape, dtype=int)
    dr, dc = shift
    for (r0, c0), color in sprite0.items():
        rr = int(r0 + step * dr)
        cc = int(c0 + step * dc)
        assert in_bounds(shape, rr, cc), "translated sprite left the panel"
        panel[rr, cc] = int(color)
    return panel


def _candidate_details(
    panels: list[np.ndarray],
    shift: tuple[int, int],
) -> dict | None:
    panel_h, panel_w = map(int, panels[0].shape)
    dr, dc = map(int, shift)
    aligned: dict[tuple[int, int], Counter] = defaultdict(Counter)
    support_hits: dict[tuple[int, int], int] = {}

    for step, panel in enumerate(panels):
        coords = np.argwhere(panel != BG)
        for r, c in coords:
            key = (int(r) - step * dr, int(c) - step * dc)
            aligned[key][int(panel[r, c])] += 1

    sprite0: dict[tuple[int, int], int] = {}
    for key, counts in aligned.items():
        color, count = counts.most_common(1)[0]
        if count >= VOTE_THRESHOLD:
            sprite0[key] = int(color)
            support_hits[key] = int(count)

    if not sprite0:
        return None

    sprite_coords = set(sprite0)
    if not _coords_connected(sprite_coords):
        return None
    if len(set(sprite0.values())) < 2:
        return None

    arr = np.array(sorted(sprite_coords), dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    if not (0 <= r0 <= r1 < panel_h and 0 <= c0 <= c1 < panel_w):
        return None

    for step in range(PANEL_COUNT + 1):
        for r, c in sprite_coords:
            if not in_bounds((panel_h, panel_w), int(r + step * dr), int(c + step * dc)):
                return None

    total_noise = 0
    noisy_frames = 0
    for step, observed in enumerate(panels):
        expected = _render_panel((panel_h, panel_w), sprite0, shift, step)
        if np.any((expected == BG) & (observed != BG)):
            return None
        mismatches = int(np.count_nonzero(observed != expected))
        if mismatches > MAX_FRAME_NOISE:
            return None
        total_noise += mismatches
        noisy_frames += int(mismatches > 0)

    output = _render_panel((panel_h, panel_w), sprite0, shift, PANEL_COUNT)
    score = (sum(support_hits.values()), len(sprite0), -total_noise, -abs(dr), -abs(dc))
    return {
        "shift": (dr, dc),
        "sprite0": sprite0,
        "output": output,
        "score": score,
        "noise": total_noise,
        "noisy_frames": noisy_frames,
    }


def _solve(grid: np.ndarray) -> dict:
    panels, _sep_color = _split_panels(grid)
    panel_h, panel_w = map(int, panels[0].shape)
    assert all(panel.shape == (panel_h, panel_w) for panel in panels)
    assert all(getattr(panel, "ndim", 0) == 2 for panel in panels)

    candidates = []
    for dr in range(-(panel_h - 1), panel_h):
        for dc in range(-(panel_w - 1), panel_w):
            if dr == 0 and dc == 0:
                continue
            details = _candidate_details(panels, (dr, dc))
            if details is not None:
                candidates.append(details)

    assert candidates, "no valid constant-translation explanation found"
    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    if len(candidates) > 1:
        assert best["score"] > candidates[1]["score"], "motion is not uniquely determined"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the clean translating sprite from five noisy panels and extrapolate the sixth frame.

    Input structure:
    - The grid is split by four full-width separator rows into five equally sized black-background panels.
    - Each panel contains one multicolor sprite, and the sprite moves by one constant translation vector from panel to panel.
    - Every panel may corrupt up to two sprite pixels by erasing them or recoloring them incorrectly.

    Rule:
    - Identify the unique constant translation that aligns the five noisy panels into one sprite coordinate system.
    - For each aligned sprite cell, take the majority color across the five frames to recover the clean sprite.
    - Place that clean sprite at the fifth-frame position plus one more copy of the same translation, and output just that sixth panel.

    Key insight:
    - The answer frame is absent from the input, so the task requires motion compensation first and denoising by cross-frame voting before extrapolating one more step.
    """
    g = np.asarray(grid, dtype=int)
    return _solve(g)["output"]


def _random_connected_shape(height: int, width: int, area: int) -> set[tuple[int, int]] | None:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (int(nr), int(nc)) not in cells:
                    frontier.append((int(nr), int(nc)))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return _normalize_cells(cells)


def _make_sprite(panel_h: int, panel_w: int) -> dict[tuple[int, int], int] | None:
    bbox_h = random.randint(2, min(panel_h, 4))
    bbox_w = random.randint(4, min(panel_w, 8))
    max_area = min(bbox_h * bbox_w, 12)
    if max_area < 8:
        return None
    area = random.randint(8, max_area)
    shape = _random_connected_shape(bbox_h, bbox_w, area)
    if shape is None or not _coords_connected(shape):
        return None

    arr = np.array(sorted(shape), dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 4:
        return None

    color_count = random.randint(2, min(4, len(shape)))
    palette = random.sample(list(GEN_SPRITE_COLORS), color_count)
    ordered = sorted(shape)
    random.shuffle(ordered)

    sprite: dict[tuple[int, int], int] = {}
    for idx, cell in enumerate(ordered):
        if idx < color_count:
            sprite[cell] = palette[idx]
        else:
            sprite[cell] = random.choice(palette)
    if len(set(sprite.values())) < 2:
        return None
    return sprite


def _sample_shift(panel_w: int, sprite_w: int) -> int | None:
    max_step = min(3, (panel_w - sprite_w) // PANEL_COUNT)
    if max_step < 1:
        return None
    step = random.randint(1, max_step)
    return random.choice((-step, step))


def _sample_noise_counts(sprite_area: int) -> list[int]:
    for _ in range(200):
        counts = [random.choice((0, 1, 2)) for _ in range(PANEL_COUNT)]
        counts[-1] = random.choice((1, 2))
        if 0 not in counts:
            counts[random.randrange(PANEL_COUNT - 1)] = 0
        if 2 not in counts:
            counts[random.randrange(PANEL_COUNT)] = 2
        total = sum(counts)
        if 3 <= total <= min(sprite_area, 8):
            return counts
    raise RuntimeError("failed to sample a valid noise schedule")


def _apply_noise(
    clean_panel: np.ndarray,
    sprite_abs: dict[tuple[int, int], int],
    rel_corruptions: list[tuple[int, int]],
    wrong_palette: list[int],
) -> tuple[np.ndarray, list[str]]:
    panel = clean_panel.copy()
    kinds = []
    for rel_cell in rel_corruptions:
        rr, cc = rel_cell
        r, c = rr, cc
        true_color = int(sprite_abs[(r, c)])
        if random.random() < 0.5:
            panel[r, c] = BG
            kinds.append("erase")
        else:
            wrong_choices = [color for color in wrong_palette if color != true_color]
            panel[r, c] = random.choice(wrong_choices)
            kinds.append("recolor")
    return panel, kinds


def _assemble_input(panels: list[np.ndarray], sep_color: int) -> np.ndarray:
    panel_h, panel_w = map(int, panels[0].shape)
    grid = np.zeros((PANEL_COUNT * panel_h + SEPARATOR_COUNT, panel_w), dtype=int)
    for idx, panel in enumerate(panels):
        top = idx * (panel_h + 1)
        grid[top : top + panel_h, :] = panel
        if idx < SEPARATOR_COUNT:
            grid[top + panel_h, :] = int(sep_color)
    return grid


def _shift_noisy_last_panel(panel: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    dr, dc = shift
    out = np.zeros_like(panel)
    coords = np.argwhere(panel != BG)
    for r, c in coords:
        rr = int(r + dr)
        cc = int(c + dc)
        if in_bounds(panel.shape, rr, cc):
            out[rr, cc] = int(panel[r, c])
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(GEN_RETRY_LIMIT):
        panel_h = random.choice(GEN_PANEL_HEIGHTS)
        panel_w = random.randint(*GEN_PANEL_WIDTH_RANGE)
        sprite_rel = _make_sprite(panel_h, panel_w)
        if sprite_rel is None:
            continue

        rel_arr = np.array(sorted(sprite_rel), dtype=int)
        _r0, _c0, _r1, rel_c1 = bbox_coords(rel_arr)
        sprite_w = int(rel_c1 - _c0 + 1)
        dc = _sample_shift(panel_w, sprite_w)
        if dc is None:
            continue
        shift = (0, dc)

        start_r = random.randint(0, panel_h - int(rel_arr[:, 0].max()) - 1)
        if dc > 0:
            start_c = random.randint(0, panel_w - sprite_w - PANEL_COUNT * dc)
        else:
            start_c = random.randint(PANEL_COUNT * abs(dc), panel_w - sprite_w)

        sprite0 = {
            (int(r + start_r), int(c + start_c)): int(color)
            for (r, c), color in sprite_rel.items()
        }
        sprite_area = len(sprite0)
        noise_counts = _sample_noise_counts(sprite_area)
        rel_cells = list(sprite_rel)
        random.shuffle(rel_cells)
        corrupt_rel = rel_cells[: sum(noise_counts)]

        sep_color = random.choice([c for c in GEN_SPRITE_COLORS if c not in set(sprite_rel.values())])
        wrong_palette = [c for c in GEN_SPRITE_COLORS if c not in {sep_color}]
        panels = []
        corruption_kinds: list[str] = []
        cursor = 0

        for step, count in enumerate(noise_counts):
            clean = _render_panel((panel_h, panel_w), sprite0, shift, step)
            sprite_abs = {
                (int(r + step * shift[0]), int(c + step * shift[1])): color
                for (r, c), color in sprite0.items()
            }
            frame_rel = corrupt_rel[cursor : cursor + count]
            cursor += count
            frame_abs = [
                (int(r + start_r + step * shift[0]), int(c + start_c + step * shift[1]))
                for r, c in frame_rel
            ]
            noisy, kinds = _apply_noise(clean, sprite_abs, frame_abs, wrong_palette)
            panels.append(noisy)
            corruption_kinds.extend(kinds)

        expected = _render_panel((panel_h, panel_w), sprite0, shift, PANEL_COUNT)
        inp = _assemble_input(panels, sep_color)

        if np.array_equal(_shift_noisy_last_panel(panels[-1], shift), expected):
            continue

        try:
            solved = _solve(inp)
        except AssertionError:
            continue
        if solved["shift"] != shift:
            continue
        if not np.array_equal(solved["output"], expected):
            continue

        meta = {
            "panel_shape": (panel_h, panel_w),
            "shift": shift,
            "noise_counts": tuple(noise_counts),
            "noise_kinds": tuple(corruption_kinds),
            "sprite_area": sprite_area,
            "separator_color": sep_color,
        }
        return inp, expected, meta
    raise RuntimeError("failed to generate a valid input")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
