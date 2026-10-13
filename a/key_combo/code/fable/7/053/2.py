from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components


BLACK = int(Color.BLACK)
PALETTE = [int(c) for c in Color.NOT_BLACK]
GEN_RETRIES = 400


@dataclass(frozen=True)
class Container:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int
    inner_w: int
    capacity: int


@dataclass(frozen=True)
class Bar:
    color: int
    row: int
    c0: int
    length: int


@dataclass(frozen=True)
class Placement:
    bar: Bar
    exact_idx: int
    target_idx: int
    slot: int
    overflow: bool


def _component_mask(grid: np.ndarray, comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    return region == int(comp["color"])


def _as_bar(grid: np.ndarray, comp: dict) -> Bar | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if r0 != r1:
        return None
    length = int(c1 - c0 + 1)
    if length < 2 or int(comp["area"]) != length:
        return None
    if not np.all(grid[r0, c0 : c1 + 1] == int(comp["color"])):
        return None
    return Bar(color=int(comp["color"]), row=int(r0), c0=int(c0), length=length)


def _as_container(grid: np.ndarray, comp: dict) -> Container | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 2 or width < 4:
        return None

    expected = np.zeros((height, width), dtype=bool)
    expected[:, 0] = True
    expected[:, -1] = True
    expected[-1, :] = True
    if not np.array_equal(_component_mask(grid, comp), expected):
        return None

    return Container(
        color=int(comp["color"]),
        r0=int(r0),
        c0=int(c0),
        r1=int(r1),
        c1=int(c1),
        inner_w=int(width - 2),
        capacity=int(height - 1),
    )


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, list[Container], list[Bar]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    comps = find_components(
        g,
        bg=BLACK,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert comps, "expected bars and containers"

    bars: list[Bar] = []
    containers: list[Container] = []
    for comp in comps:
        container = _as_container(g, comp)
        if container is not None:
            containers.append(container)
            continue
        bar = _as_bar(g, comp)
        if bar is not None:
            bars.append(bar)
            continue
        raise AssertionError("unexpected foreground component")

    assert len(containers) >= 2, "need multiple containers"
    assert bars, "need at least one bar"
    assert len({c.inner_w for c in containers}) == len(containers), "container inner widths must be unique"
    assert len({bar.row for bar in bars}) == len(bars), "bar rows must be unique"
    assert max(bar.row for bar in bars) < min(c.r0 for c in containers), "bars must float above containers"

    containers = sorted(containers, key=lambda c: c.inner_w)
    width_to_idx = {c.inner_w: i for i, c in enumerate(containers)}
    for bar in bars:
        assert bar.length in width_to_idx, "every bar length needs an exact-width container"

    bars = sorted(bars, key=lambda bar: bar.row, reverse=True)
    return g, containers, bars


def _assign_bars(containers: list[Container], bars: list[Bar]) -> list[Placement]:
    width_to_idx = {c.inner_w: i for i, c in enumerate(containers)}
    used = [0] * len(containers)
    placements: list[Placement] = []

    for bar in bars:
        exact_idx = width_to_idx[bar.length]
        target_idx = exact_idx
        overflow = False
        if used[exact_idx] >= containers[exact_idx].capacity:
            target_idx = -1
            for i in range(exact_idx + 1, len(containers)):
                if used[i] < containers[i].capacity:
                    target_idx = i
                    break
            assert target_idx != -1, "overflow must have a wider container with space"
            overflow = True

        slot = used[target_idx]
        assert slot < containers[target_idx].capacity
        used[target_idx] += 1
        placements.append(
            Placement(
                bar=bar,
                exact_idx=int(exact_idx),
                target_idx=int(target_idx),
                slot=int(slot),
                overflow=overflow,
            )
        )

    return placements


def _render_output(grid: np.ndarray, containers: list[Container], bars: list[Bar], placements: list[Placement]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for bar in bars:
        out[bar.row, bar.c0 : bar.c0 + bar.length] = BLACK

    for placement in placements:
        container = containers[placement.target_idx]
        row = int(container.r1 - 1 - placement.slot)
        start = int(container.c0 + 1)
        out[row, start : start + placement.bar.length] = int(placement.bar.color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack each horizontal bar into the container whose inner width matches its length, with overflow cascading into the narrowest wider container that still has room.

    Input structure:
    - Black is the background.
    - The lower part contains several upward-open monochrome hollow rectangles whose inner widths are all distinct and whose inner heights define capacities.
    - The upper part contains several one-cell-thick colored horizontal bars; multiple bars may share the same length, and their horizontal positions are unrelated to their destination containers.

    Rule:
    - Process the bars from lower initial row to higher initial row.
    - A bar first tries to enter the unique container whose inner width exactly equals the bar length, landing on that container's lowest empty interior row.
    - If that exact-width container is already full, overflow the bar into the narrowest wider container that still has an empty row, left-aligned against the inner left wall and again placed on the lowest empty row.
    - Remove the bars from their original floating positions; keep the containers and all other cells unchanged.

    Key insight:
    - Destination is determined by exact length matching plus mutable container occupancy, not by vertical falling under the bar's current x-position.
    """
    g, containers, bars = _parse_scene(grid)
    placements = _assign_bars(containers, bars)
    return _render_output(g, containers, bars, placements)


def _symbolic_assign(widths: list[int], capacities: list[int], lengths: list[int]) -> tuple[list[tuple[int, int, int, bool]], bool]:
    width_to_idx = {width: i for i, width in enumerate(widths)}
    used = [0] * len(widths)
    foreign = [0] * len(widths)
    placements: list[tuple[int, int, int, bool]] = []
    chain_seen = False

    for length in lengths:
        exact_idx = width_to_idx[length]
        target_idx = exact_idx
        overflow = False
        if used[exact_idx] >= capacities[exact_idx]:
            if foreign[exact_idx] > 0:
                chain_seen = True
            target_idx = -1
            for i in range(exact_idx + 1, len(widths)):
                if used[i] < capacities[i]:
                    target_idx = i
                    break
            if target_idx == -1:
                raise ValueError("invalid symbolic sequence")
            overflow = True

        slot = used[target_idx]
        used[target_idx] += 1
        if target_idx != exact_idx:
            foreign[target_idx] += 1
        placements.append((int(exact_idx), int(target_idx), int(slot), overflow))

    return placements, chain_seen


def _build_sequence(widths: list[int], capacities: list[int]) -> tuple[list[int], list[tuple[int, int, int, bool]], bool] | None:
    n = len(widths)
    want_chain = n >= 3 and random.random() < 0.7
    min_chain = 2 if want_chain else 1
    chain_len = random.randint(min_chain, n - 1)

    used = [0] * n
    lengths: list[int] = []

    for i in range(chain_len):
        fill_needed = max(0, capacities[i] - used[i])
        lengths.extend([widths[i]] * fill_needed)
        used[i] += fill_needed

        target_idx = -1
        for j in range(i + 1, n):
            if used[j] < capacities[j]:
                target_idx = j
                break
        if target_idx == -1:
            return None
        lengths.append(widths[i])
        used[target_idx] += 1

    for i in range(chain_len, n):
        remaining = capacities[i] - used[i]
        if remaining <= 0:
            continue
        lo = 1 if i == n - 1 or random.random() < 0.55 else 0
        add = random.randint(lo, remaining)
        lengths.extend([widths[i]] * add)
        used[i] += add

    if not (4 <= len(lengths) <= len(PALETTE)):
        return None
    if len(set(lengths)) < 2:
        return None

    try:
        placements, chain_seen = _symbolic_assign(widths, capacities, lengths)
    except ValueError:
        return None

    overflow_count = sum(1 for _, _, _, overflow in placements if overflow)
    if overflow_count == 0:
        return None
    if want_chain and not chain_seen:
        return None
    if not any(not overflow for _, _, _, overflow in placements):
        return None
    return lengths, placements, chain_seen


def _sample_layout(widths: list[int]) -> tuple[dict[int, int], int] | None:
    n = len(widths)
    outer_ws = {i: widths[i] + 2 for i in range(n)}
    order = list(range(n))
    random.shuffle(order)
    if n >= 3 and order == sorted(order):
        return None

    gaps = [random.randint(2, 4 if n == 3 else 3) for _ in range(n - 1)]
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    total_w = left_margin + right_margin + sum(outer_ws[i] for i in order) + sum(gaps)
    if total_w > 30:
        return None

    extra = random.randint(0, 30 - total_w)
    total_w += extra

    starts: dict[int, int] = {}
    c = left_margin
    for pos, idx in enumerate(order):
        starts[idx] = c
        c += outer_ws[idx]
        if pos < n - 1:
            c += gaps[pos]
    return starts, total_w


def _choose_bar_colors(lengths: list[int], container_colors: list[int]) -> list[int]:
    colors = random.sample(PALETTE, len(lengths))
    groups: dict[int, list[int]] = {}
    for i, length in enumerate(lengths):
        groups.setdefault(length, []).append(i)

    multi_groups = [idxs for idxs in groups.values() if len(idxs) >= 2]
    assert multi_groups
    chosen = random.choice(multi_groups)
    return [int(color) for color in colors]


def _bar_overlaps_container(bar: Bar, container: Container) -> bool:
    return not (bar.c0 + bar.length - 1 < container.c0 + 1 or bar.c0 > container.c1 - 1)


def _make_expected_output(grid: np.ndarray, containers: list[Container], bars: list[Bar], symbolic: list[tuple[int, int, int, bool]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for bar in bars:
        out[bar.row, bar.c0 : bar.c0 + bar.length] = BLACK
    for bar, (_, target_idx, slot, _) in zip(bars, symbolic):
        container = containers[target_idx]
        row = int(container.r1 - 1 - slot)
        start = int(container.c0 + 1)
        out[row, start : start + bar.length] = int(bar.color)
    return out


def _sample_bars(
    widths: list[int],
    symbolic: list[tuple[int, int, int, bool]],
    containers: list[Container],
    total_h: int,
    total_w: int,
    colors: list[int],
) -> list[Bar] | None:
    n_bars = len(symbolic)
    row_choices = list(range(1, total_h // 2))
    if len(row_choices) < n_bars:
        return None
    rows = sorted(random.sample(row_choices, n_bars), reverse=True)

    bars: list[Bar] = []
    misleading_done = False
    for i, (row, color, (exact_idx, target_idx, _slot, overflow)) in enumerate(zip(rows, colors, symbolic)):
        length = widths[exact_idx]

        start = None
        if not misleading_done:
            wrong = [
                k
                for k, container in enumerate(containers)
                if k != target_idx and container.inner_w >= length
            ]
            if wrong:
                chosen = random.choice(wrong)
                container = containers[chosen]
                offset = random.randint(0, container.inner_w - length)
                start = int(container.c0 + 1 + offset)
                misleading_done = True
        elif random.random() < 0.65:
            candidate_idxs = [k for k, container in enumerate(containers) if container.inner_w >= length]
            chosen = random.choice(candidate_idxs)
            container = containers[chosen]
            offset = random.randint(0, container.inner_w - length)
            start = int(container.c0 + 1 + offset)

        if start is None:
            start = random.randint(0, total_w - length)

        bars.append(Bar(color=int(color), row=int(row), c0=int(start), length=int(length)))

    if not misleading_done:
        return None
    if any(max(bar.row for bar in bars) >= container.r0 for container in containers):
        return None
    return bars


def _semantic_checks(
    grid: np.ndarray,
    out: np.ndarray,
    widths: list[int],
    capacities: list[int],
    symbolic: list[tuple[int, int, int, bool]],
    containers: list[Container],
    bars: list[Bar],
    chain_seen: bool,
) -> bool:
    if len(set(widths)) != len(widths):
        return False
    if len(set(capacities)) < 2:
        return False
    if len(set(bar.row for bar in bars)) != len(bars):
        return False
    if max(bar.row for bar in bars) >= grid.shape[0] // 2:
        return False
    if min(container.r0 for container in containers) < grid.shape[0] // 2:
        return False
    if not any(overflow for _, _, _, overflow in symbolic):
        return False
    if not any(not overflow for _, _, _, overflow in symbolic):
        return False
    if not chain_seen and sum(1 for _, _, _, overflow in symbolic if overflow) >= 2:
        return False

    group_colors: dict[int, set[int]] = {}
    for bar in bars:
        group_colors.setdefault(bar.length, set()).add(bar.color)
    if not any(len(colors) >= 2 for colors in group_colors.values()):
        return False

    misleading = False
    for bar, (_exact_idx, target_idx, _slot, _overflow) in zip(bars, symbolic):
        if any(_bar_overlaps_container(bar, container) and i != target_idx for i, container in enumerate(containers)):
            misleading = True
            break
    if not misleading:
        return False

    parsed_out = main(grid)
    if not np.array_equal(parsed_out, out):
        return False
    return True


def _draw_container(grid: np.ndarray, container: Container) -> None:
    grid[container.r0 : container.r1 + 1, container.c0] = container.color
    grid[container.r0 : container.r1 + 1, container.c1] = container.color
    grid[container.r1, container.c0 : container.c1 + 1] = container.color


def _build_attempt() -> np.ndarray | None:
    n_containers = random.randint(3, 4)
    widths = sorted(random.sample(range(2, 7), n_containers))
    capacities = [random.randint(1, 4) for _ in range(n_containers)]
    if len(set(capacities)) < 2:
        return None

    sequence_info = _build_sequence(widths, capacities)
    if sequence_info is None:
        return None
    lengths, symbolic, chain_seen = sequence_info

    layout = _sample_layout(widths)
    if layout is None:
        return None
    starts, total_w = layout

    max_outer_h = max(cap + 1 for cap in capacities)
    sep = random.randint(2, 4)
    bottom_margin = random.randint(1, 2)
    total_h = max(2 * len(lengths) + 4, max_outer_h + sep + bottom_margin + 10)
    if total_h > 30:
        return None

    bottom = total_h - 1 - bottom_margin
    container_colors = random.sample(PALETTE, n_containers)
    containers: list[Container] = []
    for idx, width in enumerate(widths):
        outer_h = capacities[idx] + 1
        r0 = bottom - outer_h + 1
        c0 = starts[idx]
        containers.append(
            Container(
                color=int(container_colors[idx]),
                r0=int(r0),
                c0=int(c0),
                r1=int(bottom),
                c1=int(c0 + width + 1),
                inner_w=int(width),
                capacity=int(capacities[idx]),
            )
        )

    colors = _choose_bar_colors(lengths, container_colors)
    bars = _sample_bars(widths, symbolic, containers, total_h, total_w, colors)
    if bars is None:
        return None

    grid = np.zeros((total_h, total_w), dtype=int)
    for container in containers:
        _draw_container(grid, container)
    for bar in bars:
        grid[bar.row, bar.c0 : bar.c0 + bar.length] = bar.color

    bars = sorted(bars, key=lambda bar: bar.row, reverse=True)
    expected = _make_expected_output(grid, containers, bars, symbolic)
    if not _semantic_checks(grid, expected, widths, capacities, symbolic, containers, bars, chain_seen):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overflow-container ARC input")
