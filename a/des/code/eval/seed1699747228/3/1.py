import random
from typing import Iterable

import numpy as np


GREY = 5
MAROON = 9
GEN_RETRIES = 300
MAX_LAYERS = 4


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _is_full_rectangle(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    cells = set(points)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) not in cells:
                return False
    return True


def _is_perimeter_rectangle(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    cells = set(points)
    if len(cells) != 2 * (h + w) - 4:
        return False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in cells):
                return False
    return True


def _collect_arm(
    grid: np.ndarray, start_r: int, start_c: int, dr: int, dc: int, color: int
) -> list[tuple[int, int]]:
    h, w = grid.shape
    pts: list[tuple[int, int]] = []
    r = start_r + dr
    c = start_c + dc
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == color:
        pts.append((r, c))
        r += dr
        c += dc
    return pts


def _read_legend_run(grid: np.ndarray, row: int, start_c: int, bg: int) -> list[int]:
    w = grid.shape[1]
    seq: list[int] = []
    c = start_c
    while c < w:
        value = int(grid[row, c])
        if value == bg:
            break
        seq.append(value)
        c += 1
    return seq


def _border_values(rect: np.ndarray) -> np.ndarray:
    if rect.shape[0] == 1:
        return rect[0, :].copy()
    if rect.shape[1] == 1:
        return rect[:, 0].copy()
    return np.concatenate(
        [rect[0, :], rect[-1, :], rect[1:-1, 0], rect[1:-1, -1]]
    )


def _count_layers(inner: np.ndarray, bg: int) -> int:
    assert inner.size > 0
    r0, c0 = 0, 0
    r1, c1 = inner.shape[0] - 1, inner.shape[1] - 1
    layers = 0
    while r0 <= r1 and c0 <= c1:
        rect = inner[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_values(rect)
        color = int(border[0])
        assert color not in (bg, GREY)
        assert np.all(border == color)
        layers += 1
        while True:
            rect = inner[r0 : r1 + 1, c0 : c1 + 1]
            if np.all(rect == color):
                return layers
            r0 += 1
            c0 += 1
            r1 -= 1
            c1 -= 1
            assert r0 <= r1 and c0 <= c1
            next_rect = inner[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(_border_values(next_rect) == color):
                break
    return layers


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    grey_components = _components4(g == GREY)
    legend_boxes: list[tuple[int, int, int, int]] = []
    frames: list[tuple[int, int, int, int, list[list[tuple[int, int]]]]] = []

    for comp in grey_components:
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h == 2 and w == 2 and len(comp) == 4 and _is_full_rectangle(comp):
            legend_boxes.append((r0, c0, r1, c1))
            continue
        if not _is_perimeter_rectangle(comp):
            continue
        mid_r = (r0 + r1) // 2
        mid_c = (c0 + c1) // 2
        arms = [
            _collect_arm(g, r0, mid_c, -1, 0, MAROON),
            _collect_arm(g, mid_r, c1, 0, 1, MAROON),
            _collect_arm(g, r1, mid_c, 1, 0, MAROON),
            _collect_arm(g, mid_r, c0, 0, -1, MAROON),
        ]
        if all(arms):
            frames.append((r0, c0, r1, c1, arms))

    assert legend_boxes, "legend rows not found"
    assert frames, "frames not found"

    legend_boxes.sort(key=lambda box: (box[0], box[1]))
    schemes: dict[int, list[int]] = {}
    for idx, (r0, _c0, r1, c1) in enumerate(legend_boxes, start=1):
        top_run = _read_legend_run(g, r0, c1 + 1, bg)
        bottom_run = _read_legend_run(g, r1, c1 + 1, bg)
        seq = top_run if len(top_run) >= len(bottom_run) else bottom_run
        assert seq, "legend row must provide a non-empty scheme"
        schemes[idx] = seq

    out = g.copy()
    for r0, c0, r1, c1, arms in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        layers = _count_layers(inner, bg)
        assert layers in schemes, "missing legend entry for frame layer count"
        scheme = schemes[layers]
        idx = 0
        for arm in arms:
            for rr, cc in arm:
                out[rr, cc] = scheme[idx % len(scheme)]
                idx += 1
    return out


def _sample_distinct(rng: random.Random, choices: list[int], count: int) -> list[int]:
    return rng.sample(choices, count)


def _build_frame_spec(rng: random.Random, bg: int) -> dict[str, object]:
    palette = [c for c in range(10) if c not in (bg, GREY, MAROON)]
    layers = rng.randint(1, MAX_LAYERS)
    band_thicknesses = [1] * max(0, layers - 1)
    if band_thicknesses and rng.random() < 0.45:
        band_thicknesses[rng.randrange(len(band_thicknesses))] = 2
    core_choices = [1, 3]
    if layers <= 2 and rng.random() < 0.35:
        core_choices.append(5)
    core_h = rng.choice(core_choices)
    core_w = rng.choice(core_choices)
    ring_margin = 2 * sum(band_thicknesses)
    inner_h = core_h + ring_margin
    inner_w = core_w + ring_margin
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    arm_lengths = {
        "top": rng.randint(1, 3),
        "right": rng.randint(1, 3),
        "bottom": rng.randint(1, 3),
        "left": rng.randint(1, 3),
    }
    layer_colors = _sample_distinct(rng, palette, layers)
    return {
        "layers": layers,
        "band_thicknesses": band_thicknesses,
        "core_h": core_h,
        "core_w": core_w,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "arm_lengths": arm_lengths,
        "layer_colors": layer_colors,
        "full_h": outer_h + arm_lengths["top"] + arm_lengths["bottom"],
        "full_w": outer_w + arm_lengths["left"] + arm_lengths["right"],
    }


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h - 1, r1)
    cc1 = min(w - 1, c1)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _region_is_free(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = blocked.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    return not blocked[r0 : r1 + 1, c0 : c1 + 1].any()


def _draw_frame(grid: np.ndarray, top: int, left: int, spec: dict[str, object]) -> None:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    r0, c0 = top, left
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    arm_lengths = spec["arm_lengths"]
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    for step in range(1, int(arm_lengths["top"]) + 1):
        grid[r0 - step, mid_c] = MAROON
    for step in range(1, int(arm_lengths["right"]) + 1):
        grid[mid_r, c1 + step] = MAROON
    for step in range(1, int(arm_lengths["bottom"]) + 1):
        grid[r1 + step, mid_c] = MAROON
    for step in range(1, int(arm_lengths["left"]) + 1):
        grid[mid_r, c0 - step] = MAROON

    ir0, ic0 = r0 + 1, c0 + 1
    ir1, ic1 = r1 - 1, c1 - 1
    layer_colors = list(spec["layer_colors"])
    band_thicknesses = list(spec["band_thicknesses"])
    for idx, color in enumerate(layer_colors):
        if idx == len(layer_colors) - 1:
            grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = color
            break
        thickness = int(band_thicknesses[idx])
        grid[ir0 : ir0 + thickness, ic0 : ic1 + 1] = color
        grid[ir1 - thickness + 1 : ir1 + 1, ic0 : ic1 + 1] = color
        grid[ir0 + thickness : ir1 - thickness + 1, ic0 : ic0 + thickness] = color
        grid[ir0 + thickness : ir1 - thickness + 1, ic1 - thickness + 1 : ic1 + 1] = color
        ir0 += thickness
        ic0 += thickness
        ir1 -= thickness
        ic1 -= thickness


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice([0, 1, 2, 3, 4, 6, 7, 8])
    palette = [c for c in range(10) if c not in (bg, GREY, MAROON)]

    schemes: dict[int, list[int]] = {}
    for layer_count in range(1, MAX_LAYERS + 1):
        seq_len = rng.randint(1, 4)
        schemes[layer_count] = _sample_distinct(rng, palette, seq_len)

    frame_count = rng.randint(1, 3)
    frame_specs = [_build_frame_spec(rng, bg) for _ in range(frame_count)]
    max_full_h = max(int(spec["full_h"]) for spec in frame_specs)
    max_full_w = max(int(spec["full_w"]) for spec in frame_specs)

    legend_gap = rng.choice([1, 2])
    row_step = 2 + legend_gap
    legend_top = rng.randint(0, 2)
    legend_rows = [legend_top + i * row_step for i in range(MAX_LAYERS)]
    legend_bottom = legend_rows[-1] + 1
    legend_w = 2 + max(len(seq) for seq in schemes.values())
    reserved_w = legend_w + 2

    min_h = max(24, legend_bottom + 4, max_full_h + 2)
    min_w = max(28, reserved_w + max_full_w + 2)
    if min_h > 32 or min_w > 36:
        return None
    h = rng.randint(min_h, 32)
    w = rng.randint(min_w, 36)

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    blocked[:, :reserved_w] = True

    for layer_count, row in enumerate(legend_rows, start=1):
        grid[row : row + 2, 0:2] = GREY
        scheme_row = row if rng.random() < 0.5 else row + 1
        seq = schemes[layer_count]
        grid[scheme_row, 2 : 2 + len(seq)] = seq

    placements: list[tuple[int, int] | None] = [None] * frame_count
    order = sorted(
        range(frame_count),
        key=lambda idx: int(frame_specs[idx]["full_h"]) * int(frame_specs[idx]["full_w"]),
        reverse=True,
    )
    for idx in order:
        spec = frame_specs[idx]
        arm_lengths = spec["arm_lengths"]
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        top_arm = int(arm_lengths["top"])
        right_arm = int(arm_lengths["right"])
        bottom_arm = int(arm_lengths["bottom"])
        left_arm = int(arm_lengths["left"])
        placed = False
        for _ in range(200):
            top = rng.randint(top_arm, h - outer_h - bottom_arm)
            left = rng.randint(max(reserved_w + left_arm, left_arm), w - outer_w - right_arm)
            bottom = top + outer_h - 1
            right = left + outer_w - 1
            full_r0 = top - top_arm
            full_c0 = left - left_arm
            full_r1 = bottom + bottom_arm
            full_c1 = right + right_arm
            if not _region_is_free(blocked, full_r0 - 1, full_c0 - 1, full_r1 + 1, full_c1 + 1):
                continue
            placements[idx] = (top, left)
            _mark_blocked(blocked, full_r0 - 1, full_c0 - 1, full_r1 + 1, full_c1 + 1)
            placed = True
            break
        if not placed:
            return None

    for spec, placement in zip(frame_specs, placements):
        assert placement is not None
        top, left = placement
        _draw_frame(grid, top, left, spec)

    if _most_common_color(grid) != bg:
        return None
    if not np.any(grid == MAROON):
        return None
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except AssertionError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
