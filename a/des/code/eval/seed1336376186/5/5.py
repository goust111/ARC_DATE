from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class FenceSpec:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int
    side: str
    relation: str
    gap: tuple[int, int]
    key: tuple[int, int]


BG = 0
COLORS = tuple(range(1, 10))
GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
OBJECT_COUNT_RANGE = (2, 4)
FRAME_H_RANGE = (5, 10)
FRAME_W_RANGE = (5, 10)
PLACEMENT_TRIES = 300
GENERATION_TRIES = 400

VERTICAL_PHASE_CASES = [
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
]
HORIZONTAL_PHASE_CASES = [
    ("left", "up"),
    ("left", "down"),
    ("right", "up"),
    ("right", "down"),
]
ALL_CASES = VERTICAL_PHASE_CASES + HORIZONTAL_PHASE_CASES


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _analyze_fence(grid: np.ndarray, color: int, bg: int) -> FenceSpec:
    comps = _components(grid == color)
    fence_comps = [comp for comp in comps if len(comp) > 1]
    key_comps = [comp for comp in comps if len(comp) == 1]
    assert len(fence_comps) == 1, "expected one fence component per color"
    assert len(key_comps) == 1, "expected one key pixel per color"

    fence = fence_comps[0]
    key = key_comps[0][0]
    r0, c0, r1, c1 = _bbox(fence)
    assert r1 - r0 + 1 >= 5 and c1 - c0 + 1 >= 5, "fence must have a visible interior"

    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in fence:
        local[r - r0, c - c0] = True
    expected = _border_mask(local.shape[0], local.shape[1])
    missing = np.argwhere(expected & ~local)
    extra = np.argwhere(local & ~expected)
    assert extra.size == 0, "fence component must lie on its bbox border"
    assert len(missing) == 1, "fence border must have exactly one missing pixel"

    gap_r = int(missing[0, 0] + r0)
    gap_c = int(missing[0, 1] + c0)
    if gap_r == r0:
        side = "top"
    elif gap_r == r1:
        side = "bottom"
    elif gap_c == c0:
        side = "left"
    else:
        side = "right"

    key_r, key_c = key
    if side in {"top", "bottom"}:
        assert key_r == gap_r, "top/bottom-gap keys must share the gap row"
        assert abs(key_c - gap_c) <= 3, "key must stay near the gap"
        relation = "left" if key_c < gap_c else "right"
    else:
        assert key_c == gap_c, "left/right-gap keys must share the gap column"
        assert abs(key_r - gap_r) <= 3, "key must stay near the gap"
        relation = "up" if key_r < gap_r else "down"

    assert grid[key_r, key_c] == color
    assert bg != color
    return FenceSpec(
        color=int(color),
        r0=int(r0),
        c0=int(c0),
        r1=int(r1),
        c1=int(c1),
        side=side,
        relation=relation,
        gap=(gap_r, gap_c),
        key=(int(key_r), int(key_c)),
    )


def _paint_stripes(out: np.ndarray, spec: FenceSpec, bg: int) -> None:
    # The phase cue is left/right for top/bottom gaps and up/down for left/right gaps,
    # so the coherent stripe orientation is columns for top/bottom gaps and rows for left/right gaps.
    if spec.side in {"top", "bottom"}:
        if spec.relation == "left":
            columns = range(spec.c1 - 1, spec.c0, -1)
        else:
            columns = range(spec.c0 + 1, spec.c1)
        for idx, c in enumerate(columns):
            out[spec.r0 + 1 : spec.r1, c] = spec.color if idx % 2 == 0 else bg
    else:
        if spec.relation == "up":
            rows = range(spec.r1 - 1, spec.r0, -1)
        else:
            rows = range(spec.r0 + 1, spec.r1)
        for idx, r in enumerate(rows):
            out[r, spec.c0 + 1 : spec.c1] = spec.color if idx % 2 == 0 else bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each open rectangular fence with alternating stripes keyed by its gap side and its same-color key pixel.

    Top/bottom gaps use column stripes because the key encodes a left/right phase;
    left/right gaps use row stripes because the key encodes an up/down phase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        spec = _analyze_fence(g, color, bg)
        out[spec.key] = bg
        _paint_stripes(out, spec, bg)

    return out


def _gap_and_key(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    side: str,
    relation: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if side == "top":
        gap = (r0, c0 + 1 if relation == "left" else c1 - 1)
        key = (r0, c0 - 2 if relation == "left" else c1 + 2)
    elif side == "bottom":
        gap = (r1, c0 + 1 if relation == "left" else c1 - 1)
        key = (r1, c0 - 2 if relation == "left" else c1 + 2)
    elif side == "left":
        gap = (r0 + 1 if relation == "up" else r1 - 1, c0)
        key = (r0 - 2 if relation == "up" else r1 + 2, c0)
    else:
        gap = (r0 + 1 if relation == "up" else r1 - 1, c1)
        key = (r0 - 2 if relation == "up" else r1 + 2, c1)
    return gap, key


def _reserve_box(spec: FenceSpec) -> tuple[int, int, int, int]:
    rows = [spec.r0, spec.r1, spec.key[0]]
    cols = [spec.c0, spec.c1, spec.key[1]]
    return min(rows) - 1, min(cols) - 1, max(rows) + 1, max(cols) + 1


def _draw_fence_outline(grid: np.ndarray, spec: FenceSpec, bg: int) -> None:
    grid[spec.r0 : spec.r1 + 1, spec.c0] = spec.color
    grid[spec.r0 : spec.r1 + 1, spec.c1] = spec.color
    grid[spec.r0, spec.c0 : spec.c1 + 1] = spec.color
    grid[spec.r1, spec.c0 : spec.c1 + 1] = spec.color
    grid[spec.gap] = bg


def _sample_cases(count: int) -> list[tuple[str, str]]:
    cases = [random.choice(VERTICAL_PHASE_CASES), random.choice(HORIZONTAL_PHASE_CASES)]
    while len(cases) < count:
        cases.append(random.choice(ALL_CASES))
    random.shuffle(cases)
    return cases[:count]


def _place_specs(height: int, width: int, cases: list[tuple[str, str]], colors: list[int]) -> list[FenceSpec] | None:
    occupied = np.zeros((height, width), dtype=bool)
    specs: list[FenceSpec] = []
    for color, (side, relation) in zip(colors, cases, strict=False):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            fh = random.randint(*FRAME_H_RANGE)
            fw = random.randint(*FRAME_W_RANGE)
            if fh >= height - 2 or fw >= width - 2:
                continue
            r0 = random.randint(0, height - fh)
            c0 = random.randint(0, width - fw)
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            gap, key = _gap_and_key(r0, c0, r1, c1, side, relation)
            spec = FenceSpec(color, r0, c0, r1, c1, side, relation, gap, key)
            rr0, cc0, rr1, cc1 = _reserve_box(spec)
            if rr0 < 0 or cc0 < 0 or rr1 >= height or cc1 >= width:
                continue
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            specs.append(spec)
            placed = True
            break
        if not placed:
            return None
    return specs


def _render_input(height: int, width: int, specs: list[FenceSpec], bg: int) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for spec in specs:
        _draw_fence_outline(grid, spec, bg)
        grid[spec.key] = spec.color
    return grid


def _render_output(inp: np.ndarray, specs: list[FenceSpec], bg: int) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        out[spec.key] = bg
        _paint_stripes(out, spec, bg)
    return out


def _valid_scene(inp: np.ndarray, out: np.ndarray, specs: list[FenceSpec], bg: int) -> bool:
    if not np.array_equal(main(inp), out):
        return False
    for spec in specs:
        if inp[spec.key] != spec.color or out[spec.key] != bg:
            return False
        if inp[spec.gap] != bg or out[spec.gap] != bg:
            return False
        analyzed = _analyze_fence(inp, spec.color, bg)
        if analyzed.side != spec.side or analyzed.relation != spec.relation:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(COLORS, count)
    cases = _sample_cases(count)
    specs = _place_specs(height, width, cases, colors)
    if specs is None:
        return None

    inp = _render_input(height, width, specs, BG)
    out = _render_output(inp, specs, BG)
    if _valid_scene(inp, out, specs, BG):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_TRIES} attempts")
