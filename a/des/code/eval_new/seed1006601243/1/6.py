from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 600


@dataclass
class Container:
    r0: int
    c0: int
    r1: int
    c1: int
    border_color: int
    gap_cols: tuple[int, ...]
    stacks: list[list[int]]

    @property
    def inner_h(self) -> int:
        return self.r1 - self.r0 - 1

    @property
    def inner_w(self) -> int:
        return self.c1 - self.c0 - 1


@dataclass
class ContainerSpec:
    r0: int
    c0: int
    outer_h: int
    outer_w: int
    border_color: int
    liquid_color: int
    gap_cols: tuple[int, ...]
    heights: list[int]

    @property
    def r1(self) -> int:
        return self.r0 + self.outer_h - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.outer_w - 1

    @property
    def inner_h(self) -> int:
        return self.outer_h - 2

    @property
    def inner_w(self) -> int:
        return self.outer_w - 2


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_container(grid: np.ndarray, color: int, comp: list[tuple[int, int]]) -> Container | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    gap_cols = [c for c in range(c0 + 1, c1) if int(grid[r1, c]) == BG]
    if gap_cols:
        expected_gap = list(range(gap_cols[0], gap_cols[-1] + 1))
        if gap_cols != expected_gap:
            return None
    for c in range(c0 + 1, c1):
        val = int(grid[r1, c])
        if val not in (BG, color):
            return None

    expected: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
    for r in range(r0 + 1, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0 + 1, c1):
        if c not in gap_cols:
            expected.add((r1, c))
    if set(comp) != expected:
        return None

    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
        return None

    stacks: list[list[int]] = []
    for c in range(c0 + 1, c1):
        stack: list[int] = []
        for r in range(r1 - 1, r0, -1):
            val = int(grid[r, c])
            if val == BG:
                continue
            if val == color:
                return None
            stack.append(val)
        stacks.append(stack)

    return Container(r0, c0, r1, c1, color, tuple(gap_cols), stacks)


def _find_containers(grid: np.ndarray) -> list[Container]:
    containers: list[Container] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for comp in _components(grid == color):
            container = _extract_container(grid, color, comp)
            if container is not None:
                containers.append(container)
    containers.sort(key=lambda box: (box.r0, box.c0, box.r1, box.c1))
    return containers


def _first_receiver(containers: list[Container], source_index: int, col: int) -> int | None:
    source = containers[source_index]
    for idx in range(source_index + 1, len(containers)):
        target = containers[idx]
        if target.r0 <= source.r1:
            continue
        if target.c0 < col < target.c1:
            return idx
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Leak liquid downward through bottom gaps of rectangular containers and restack it in lower containers.

    Input structure:
    - The background is black.
    - Several colored rectangular containers appear at different heights; each container has a 1-pixel border and may have one contiguous gap on its bottom edge.
    - Inside a container, non-black liquid pixels sit against the bottom and the border color does not appear in the interior.

    Rule:
    - For every container with a bottom gap, only the liquid columns directly above that gap leak out.
    - A leaked column falls straight down its absolute grid column and enters the first lower container whose left and right borders straddle that column.
    - Within each receiving container, liquid is tracked per interior column: incoming pixels are added to that same column, then every column is compacted from the bottom upward.
    - If no lower container catches a leaked column, that liquid disappears; containers without a bottom gap keep their liquid in place.

    Key insight:
    - The bottom gap serves two roles at once: it selects which source columns empty out, and the same absolute columns determine which lower container, if any, can catch and restack the falling liquid.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    containers = _find_containers(g)
    if not containers:
        return out

    for source_index, container in enumerate(containers):
        for col in container.gap_cols:
            local_col = col - (container.c0 + 1)
            leaked = container.stacks[local_col]
            if not leaked:
                continue
            container.stacks[local_col] = []
            receiver_index = _first_receiver(containers, source_index, col)
            if receiver_index is None:
                continue
            receiver = containers[receiver_index]
            receiver_local_col = col - (receiver.c0 + 1)
            receiver.stacks[receiver_local_col].extend(leaked)

    for container in containers:
        out[container.r0 + 1 : container.r1, container.c0 + 1 : container.c1] = BG
        for local_col, stack in enumerate(container.stacks):
            col = container.c0 + 1 + local_col
            for depth, color in enumerate(stack[: container.inner_h]):
                row = container.r1 - 1 - depth
                out[row, col] = color

    return out


def _overlap_with_next(specs: list[ContainerSpec], index: int) -> list[int]:
    source = specs[index]
    target = specs[index + 1]
    lo = max(source.c0 + 1, target.c0 + 1)
    hi = min(source.c1 - 1, target.c1 - 1)
    if lo > hi:
        return []
    return list(range(lo, hi + 1))


def _pick_gap_cols(rng: random.Random, cols: list[int], inner_w: int) -> tuple[int, ...]:
    max_width = min(3, len(cols))
    if inner_w > 1:
        max_width = min(max_width, inner_w - 1)
    width = rng.randint(1, max_width)
    start = rng.randint(0, len(cols) - width)
    return tuple(cols[start : start + width])


def _sample_heights(
    rng: random.Random,
    inner_w: int,
    inner_h: int,
    gap_cols: tuple[int, ...],
    c0: int,
    *,
    must_have_liquid: bool,
    must_fill_gap: bool,
    keep_non_gap: bool,
) -> list[int]:
    max_fill = max(1, inner_h - 2)
    heights = [rng.randint(0, max_fill) for _ in range(inner_w)]
    if not must_have_liquid and rng.random() < 0.25:
        heights = [0] * inner_w
    if must_have_liquid and max(heights) == 0:
        heights[rng.randrange(inner_w)] = rng.randint(1, max_fill)

    gap_local = [col - (c0 + 1) for col in gap_cols]
    if must_fill_gap and gap_local and all(heights[idx] == 0 for idx in gap_local):
        idx = rng.choice(gap_local)
        heights[idx] = rng.randint(1, max_fill)

    non_gap = [idx for idx in range(inner_w) if idx not in gap_local]
    if keep_non_gap and non_gap and all(heights[idx] == 0 for idx in non_gap):
        idx = rng.choice(non_gap)
        heights[idx] = rng.randint(1, max_fill)
    return heights


def _initial_stacks(specs: list[ContainerSpec]) -> list[list[list[int]]]:
    return [[[spec.liquid_color] * height for height in spec.heights] for spec in specs]


def _simulate_specs(specs: list[ContainerSpec]) -> tuple[list[list[list[int]]], dict[str, int | bool]]:
    stacks = _initial_stacks(specs)
    caught = 0
    lost = 0
    moved = 0
    for source_index, spec in enumerate(specs):
        for col in spec.gap_cols:
            local_col = col - (spec.c0 + 1)
            leaked = stacks[source_index][local_col]
            if not leaked:
                continue
            moved += len(leaked)
            stacks[source_index][local_col] = []
            receiver_index = None
            for idx in range(source_index + 1, len(specs)):
                target = specs[idx]
                if target.r0 <= spec.r1:
                    continue
                if target.c0 < col < target.c1:
                    receiver_index = idx
                    break
            if receiver_index is None:
                lost += len(leaked)
                continue
            target = specs[receiver_index]
            target_local_col = col - (target.c0 + 1)
            stacks[receiver_index][target_local_col].extend(leaked)
            caught += len(leaked)

    overflow = False
    for spec, spec_stacks in zip(specs, stacks):
        for stack in spec_stacks:
            if len(stack) > spec.inner_h:
                overflow = True
                break
        if overflow:
            break
    return stacks, {"caught": caught, "lost": lost, "moved": moved, "overflow": overflow}


def _render_specs(specs: list[ContainerSpec], height: int, width: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        grid[spec.r0, spec.c0 : spec.c1 + 1] = spec.border_color
        grid[spec.r0 : spec.r1 + 1, spec.c0] = spec.border_color
        grid[spec.r0 : spec.r1 + 1, spec.c1] = spec.border_color
        for col in range(spec.c0 + 1, spec.c1):
            if col not in spec.gap_cols:
                grid[spec.r1, col] = spec.border_color
        for local_col, liquid_height in enumerate(spec.heights):
            col = spec.c0 + 1 + local_col
            for depth in range(liquid_height):
                row = spec.r1 - 1 - depth
                grid[row, col] = spec.liquid_color
    return grid


def _build_specs(rng: random.Random) -> tuple[list[ContainerSpec], int, int] | None:
    width = rng.randint(16, 30)
    count = rng.randint(3, 5)

    outer_ws = [rng.randint(5, 9) for _ in range(count)]
    outer_hs = [rng.randint(5, 8) for _ in range(count)]
    if max(outer_ws) + 2 > width:
        return None

    top_margin = rng.randint(1, 3)
    row = top_margin
    positions: list[tuple[int, int, int, int]] = []
    prev_c0 = None
    for outer_h, outer_w in zip(outer_hs, outer_ws):
        max_c0 = width - outer_w - 1
        if max_c0 < 1:
            return None
        if prev_c0 is None:
            c0 = rng.randint(1, max_c0)
        else:
            lo = max(1, prev_c0 - 4)
            hi = min(max_c0, prev_c0 + 4)
            if lo > hi:
                return None
            c0 = rng.randint(lo, hi)
        positions.append((row, c0, outer_h, outer_w))
        row += outer_h + rng.randint(1, 3)
        prev_c0 = c0

    height = row + rng.randint(0, 2)
    if height > 30:
        return None

    border_colors = rng.sample(PALETTE, count)
    liquid_pool = [color for color in PALETTE if color not in border_colors]

    base_specs = [
        ContainerSpec(
            r0=r0,
            c0=c0,
            outer_h=outer_h,
            outer_w=outer_w,
            border_color=border_colors[idx],
            liquid_color=rng.choice(liquid_pool),
            gap_cols=(),
            heights=[],
        )
        for idx, (r0, c0, outer_h, outer_w) in enumerate(positions)
    ]

    caught_candidates = [idx for idx in range(count - 1) if _overlap_with_next(base_specs, idx)]
    if not caught_candidates:
        return None
    caught_idx = rng.choice(caught_candidates)
    uncaught_idx = count - 1
    sealed_candidates = [idx for idx in range(count) if idx not in {caught_idx, uncaught_idx}]
    if not sealed_candidates:
        return None
    sealed_idx = rng.choice(sealed_candidates)

    specs: list[ContainerSpec] = []
    for idx, spec in enumerate(base_specs):
        gap_cols: tuple[int, ...]
        if idx == caught_idx:
            gap_cols = _pick_gap_cols(rng, _overlap_with_next(base_specs, idx), spec.inner_w)
        elif idx == uncaught_idx:
            cols = list(range(spec.c0 + 1, spec.c1))
            gap_cols = _pick_gap_cols(rng, cols, spec.inner_w)
        elif idx == sealed_idx:
            gap_cols = ()
        else:
            can_leak = idx < count - 1 and bool(_overlap_with_next(base_specs, idx))
            if can_leak and rng.random() < 0.4:
                gap_cols = _pick_gap_cols(rng, _overlap_with_next(base_specs, idx), spec.inner_w)
            else:
                gap_cols = ()

        must_have_liquid = idx in {caught_idx, uncaught_idx, sealed_idx} or rng.random() < 0.6
        heights = _sample_heights(
            rng,
            spec.inner_w,
            spec.inner_h,
            gap_cols,
            spec.c0,
            must_have_liquid=must_have_liquid,
            must_fill_gap=bool(gap_cols),
            keep_non_gap=bool(gap_cols) and spec.inner_w > len(gap_cols),
        )
        specs.append(
            ContainerSpec(
                r0=spec.r0,
                c0=spec.c0,
                outer_h=spec.outer_h,
                outer_w=spec.outer_w,
                border_color=spec.border_color,
                liquid_color=spec.liquid_color,
                gap_cols=gap_cols,
                heights=heights,
            )
        )

    stacks, stats = _simulate_specs(specs)
    if stats["moved"] <= 0 or stats["caught"] <= 0 or stats["lost"] <= 0 or bool(stats["overflow"]):
        return None
    if max(len(stack) for spec_stacks in stacks for stack in spec_stacks) <= 0:
        return None

    return specs, height, width


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    built = _build_specs(rng)
    if built is None:
        return None
    specs, height, width = built
    grid = _render_specs(specs, height, width)

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape or max(out.shape) > 30 or max(grid.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
