from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import random
from typing import Iterable

import numpy as np

BG = 0
PATH = 8
PATH2 = 1
BORDER2 = 5
MARK = 9

SEED_A = 2
SEED_B = 3
SEED_C = 4
SEED_D = 6
SEED_E = 7
SEED_PALETTE = (SEED_A, SEED_B, SEED_C, SEED_D, SEED_E)

BASE_SHAPE = (27, 27)
SHORT_BRANCH_THRESHOLD = 14


def _rect(r0: int, r1: int, c0: int, c1: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _poly(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for idx, ((r0, c0), (r1, c1)) in enumerate(zip(points, points[1:])):
        seg: list[tuple[int, int]] = []
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            seg = [(r0, c) for c in range(c0, c1 + step, step)]
        elif c0 == c1:
            step = 1 if r1 >= r0 else -1
            seg = [(r, c0) for r in range(r0, r1 + step, step)]
        else:
            raise ValueError(f"non-orthogonal segment: {(r0, c0)} -> {(r1, c1)}")
        if idx:
            seg = seg[1:]
        out.extend(seg)
    return out


def _adjacent_border(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    border: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                border.add((rr, cc))
    return border


def _dilate_one_sided(cells: Iterable[tuple[int, int]], dr: int, dc: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set(cells)
    for r, c in cells:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.add((rr, cc))
    return out


def _fill_cells(grid: np.ndarray, cells: Iterable[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _ordered_cells(cells: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(cells, key=lambda rc: (rc[0], rc[1]))


def _paint_seed_counts(
    grid: np.ndarray,
    cells: set[tuple[int, int]],
    counts: list[tuple[int, int]],
    rng: random.Random,
) -> None:
    ordered = _ordered_cells(cells)
    rng.shuffle(ordered)
    flat: list[int] = []
    for color, count in counts:
        flat.extend([color] * count)
    assert len(flat) == len(ordered), (len(flat), len(ordered))
    for (r, c), color in zip(ordered, flat):
        grid[r, c] = color


def _seed_stats(grid: np.ndarray, cells: set[tuple[int, int]]) -> Counter[int]:
    vals = [int(grid[r, c]) for r, c in cells if int(grid[r, c]) != BG]
    return Counter(vals)


def _major_color(grid: np.ndarray, cells: set[tuple[int, int]]) -> int:
    return int(_seed_stats(grid, cells).most_common(1)[0][0])


def _minor_color(grid: np.ndarray, cells: set[tuple[int, int]]) -> int:
    counts = _seed_stats(grid, cells)
    items = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return int(items[0][0])


def _ordered_branch(cells: set[tuple[int, int]], reverse: bool = False) -> list[tuple[int, int]]:
    ordered = sorted(cells, key=lambda rc: (abs(rc[0] - 13) + abs(rc[1] - 13), rc[0], rc[1]))
    return list(reversed(ordered)) if reverse else ordered


def _ratio_pattern(c1: int, c2: int, n1: int, n2: int, length: int) -> list[int]:
    seq: list[int] = []
    left1, left2 = n1, n2
    prev = None
    while len(seq) < length:
        need_first = left1 >= left2
        pick = c1 if need_first and left1 else c2
        if pick == c2 and left2 == 0:
            pick = c1
        if pick == c1 and left1 == 0:
            pick = c2
        if prev is not None and pick == prev:
            other = c2 if pick == c1 else c1
            other_left = left2 if other == c2 else left1
            if other_left > 0:
                pick = other
        seq.append(pick)
        if pick == c1:
            left1 -= 1
        else:
            left2 -= 1
        prev = pick
        if left1 < 0 or left2 < 0:
            break
    return seq[:length]


@dataclass
class Layout:
    shape: tuple[int, int] = BASE_SHAPE
    seed_cells: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    border_cells: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    branch_cells: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    hub_cells: set[tuple[int, int]] = field(default_factory=set)
    center_cells: list[tuple[int, int]] = field(default_factory=list)
    marker_cells: dict[str, tuple[int, int]] = field(default_factory=dict)
    legend_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = field(default_factory=list)
    disconnected_seed_cells: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    noise_path_cells: set[tuple[int, int]] = field(default_factory=set)
    noise_pixels: list[tuple[int, int]] = field(default_factory=list)
    slot_to_hub: dict[str, int] = field(default_factory=dict)
    branch_lengths: dict[str, int] = field(default_factory=dict)
    branch_bends: dict[str, int] = field(default_factory=dict)
    branch_widths: dict[str, int] = field(default_factory=dict)
    path_colors: dict[str, int] = field(default_factory=dict)
    border_color: int = PATH
    shared_cells: set[tuple[int, int]] = field(default_factory=set)
    slot_colors_override: dict[str, int] = field(default_factory=dict)


def _base_layout() -> Layout:
    layout = Layout()
    layout.seed_cells = {
        "N": _rect(0, 2, 3, 7),
        "E": _rect(3, 7, 24, 26),
        "S": _rect(24, 26, 19, 23),
        "W": _rect(19, 23, 0, 2),
    }
    layout.border_cells = {k: _adjacent_border(v, layout.shape) for k, v in layout.seed_cells.items()}
    layout.branch_cells = {
        "N": set(_poly([(11, 13), (4, 13), (4, 5), (3, 5)])),
        "E": set(_poly([(13, 15), (13, 22), (5, 22), (5, 23)])),
        "S": set(_poly([(15, 13), (22, 13), (22, 21), (23, 21)])),
        "W": set(_poly([(13, 11), (13, 4), (21, 4), (21, 3)])),
    }
    layout.hub_cells = _rect(12, 14, 12, 14)
    layout.center_cells = [(13, 13)]
    layout.branch_lengths = {k: len(v) for k, v in layout.branch_cells.items()}
    layout.branch_bends = {k: 1 for k in layout.branch_cells}
    layout.branch_widths = {k: 1 for k in layout.branch_cells}
    layout.slot_to_hub = {k: 0 for k in layout.branch_cells}
    layout.path_colors = {k: PATH for k in layout.branch_cells}
    return layout


def _layout_variable_hub() -> Layout:
    layout = _base_layout()
    return layout


def _layout_cross_hub() -> Layout:
    layout = _base_layout()
    layout.hub_cells = {(13, 13), (12, 13), (14, 13), (13, 12), (13, 14)}
    return layout


def _layout_interior_seeds() -> Layout:
    layout = Layout()
    layout.seed_cells = {
        "N": _rect(2, 4, 4, 8),
        "E": _rect(4, 8, 20, 22),
        "S": _rect(20, 22, 18, 22),
        "W": _rect(18, 22, 4, 6),
    }
    layout.border_cells = {k: _adjacent_border(v, layout.shape) for k, v in layout.seed_cells.items()}
    layout.branch_cells = {
        "N": set(_poly([(11, 13), (6, 13), (6, 6), (5, 6)])),
        "E": set(_poly([(13, 15), (13, 18), (6, 18), (6, 19)])),
        "S": set(_poly([(15, 13), (18, 13), (18, 20), (19, 20)])),
        "W": set(_poly([(13, 11), (13, 8), (20, 8), (20, 7)])),
    }
    layout.hub_cells = _rect(12, 14, 12, 14)
    layout.center_cells = [(13, 13)]
    layout.branch_lengths = {k: len(v) for k, v in layout.branch_cells.items()}
    layout.branch_bends = {k: 1 for k in layout.branch_cells}
    layout.branch_widths = {k: 1 for k in layout.branch_cells}
    layout.slot_to_hub = {k: 0 for k in layout.branch_cells}
    layout.path_colors = {k: PATH for k in layout.branch_cells}
    return layout


def _layout_no_hub_point() -> Layout:
    layout = _base_layout()
    layout.hub_cells = {(13, 13)}
    layout.branch_cells = {
        "N": set(_poly([(12, 13), (4, 13), (4, 5), (3, 5)])),
        "E": set(_poly([(13, 14), (13, 22), (5, 22), (5, 23)])),
        "S": set(_poly([(14, 13), (22, 13), (22, 21), (23, 21)])),
        "W": set(_poly([(13, 12), (13, 4), (21, 4), (21, 3)])),
    }
    layout.branch_lengths = {k: len(v) for k, v in layout.branch_cells.items()}
    return layout


def _layout_dual_hub() -> Layout:
    layout = Layout()
    layout.hub_cells = _rect(12, 14, 7, 9) | _rect(12, 14, 17, 19)
    layout.center_cells = [(13, 8), (13, 18)]
    layout.seed_cells = {
        "NW": _rect(0, 2, 1, 5),
        "SW": _rect(24, 26, 1, 5),
        "NE": _rect(0, 2, 21, 25),
        "SE": _rect(24, 26, 21, 25),
    }
    layout.border_cells = {k: _adjacent_border(v, layout.shape) for k, v in layout.seed_cells.items()}
    layout.branch_cells = {
        "NW": set(_poly([(11, 8), (4, 8), (4, 3), (3, 3)])),
        "SW": set(_poly([(15, 8), (22, 8), (22, 3), (23, 3)])),
        "NE": set(_poly([(11, 18), (4, 18), (4, 23), (3, 23)])),
        "SE": set(_poly([(15, 18), (22, 18), (22, 23), (23, 23)])),
    }
    layout.slot_to_hub = {"NW": 0, "SW": 0, "NE": 1, "SE": 1}
    layout.branch_lengths = {k: len(v) for k, v in layout.branch_cells.items()}
    layout.branch_bends = {k: 1 for k in layout.branch_cells}
    layout.branch_widths = {k: 1 for k in layout.branch_cells}
    layout.path_colors = {k: PATH for k in layout.branch_cells}
    return layout


def _layout_length_order() -> Layout:
    layout = _base_layout()
    layout.branch_cells = {
        "N": set(_poly([(11, 13), (6, 13), (6, 7), (5, 7)])),
        "E": set(_poly([(13, 15), (13, 21), (6, 21), (6, 23)])),
        "S": set(_poly([(15, 13), (22, 13), (22, 22), (23, 22)])),
        "W": set(_poly([(13, 11), (13, 3), (22, 3), (22, 1)])),
    }
    layout.branch_lengths = {k: len(v) for k, v in layout.branch_cells.items()}
    return layout


def _layout_bend_parity() -> Layout:
    layout = _base_layout()
    layout.branch_cells = {
        "N": set(_poly([(11, 13), (4, 13), (4, 5), (3, 5)])),
        "E": set(_poly([(13, 15), (13, 20), (8, 20), (8, 22), (5, 22), (5, 23)])),
        "S": set(_poly([(15, 13), (19, 13), (19, 17), (21, 17), (21, 21), (23, 21)])),
        "W": set(_poly([(13, 11), (13, 7), (16, 7), (16, 3), (21, 3)])),
    }
    layout.branch_lengths = {k: len(v) for k, v in layout.branch_cells.items()}
    layout.branch_bends = {"N": 1, "E": 3, "S": 3, "W": 3}
    return layout


def _layout_width_coded() -> Layout:
    layout = _base_layout()
    layout.branch_cells["N"] = _dilate_one_sided(layout.branch_cells["N"], 0, 1, layout.shape)
    layout.branch_cells["S"] = _dilate_one_sided(layout.branch_cells["S"], 0, -1, layout.shape)
    layout.branch_widths = {"N": 2, "E": 1, "S": 2, "W": 1}
    layout.branch_lengths = {k: len(v) for k, v in layout.branch_cells.items()}
    return layout


def _layout_shared_path() -> Layout:
    layout = Layout()
    layout.hub_cells = _rect(12, 14, 12, 14)
    layout.center_cells = [(13, 13)]
    layout.seed_cells = {
        "NL": _rect(0, 2, 2, 6),
        "NR": _rect(0, 2, 20, 24),
        "S": _rect(24, 26, 11, 15),
        "W": _rect(18, 22, 0, 2),
    }
    layout.border_cells = {k: _adjacent_border(v, layout.shape) for k, v in layout.seed_cells.items()}
    layout.shared_cells = set(_poly([(11, 13), (6, 13), (6, 12)]))
    layout.branch_cells = {
        "NL": layout.shared_cells | set(_poly([(6, 12), (6, 4), (3, 4)])),
        "NR": layout.shared_cells | set(_poly([(6, 12), (6, 22), (3, 22)])),
        "S": set(_poly([(15, 13), (23, 13)])),
        "W": set(_poly([(13, 11), (13, 4), (20, 4), (20, 3)])),
    }
    layout.slot_to_hub = {k: 0 for k in layout.branch_cells}
    layout.branch_lengths = {k: len(v) for k, v in layout.branch_cells.items()}
    layout.branch_bends = {"NL": 2, "NR": 2, "S": 0, "W": 1}
    layout.branch_widths = {k: 1 for k in layout.branch_cells}
    layout.path_colors = {k: PATH for k in layout.branch_cells}
    return layout


def _layout_short_delete() -> Layout:
    layout = _base_layout()
    layout.branch_cells = {
        "N": set(_poly([(11, 13), (7, 13), (7, 6), (6, 6)])),
        "E": set(_poly([(13, 15), (13, 20), (6, 20), (6, 23)])),
        "S": set(_poly([(15, 13), (23, 13), (23, 21)])),
        "W": set(_poly([(13, 11), (13, 5), (22, 5), (22, 3)])),
    }
    layout.branch_lengths = {k: len(v) for k, v in layout.branch_cells.items()}
    return layout


def _layout_path_color_split() -> Layout:
    layout = _base_layout()
    layout.path_colors = {"N": PATH, "E": PATH2, "S": PATH, "W": PATH2}
    return layout


def _layout_border_split() -> Layout:
    layout = _base_layout()
    layout.border_color = BORDER2
    return layout


def _layout_no_border() -> Layout:
    layout = _base_layout()
    layout.border_cells = {k: set() for k in layout.seed_cells}
    layout.branch_cells = {
        "N": set(_poly([(11, 13), (4, 13), (4, 5), (2, 5)])),
        "E": set(_poly([(13, 15), (13, 22), (5, 22), (5, 24)])),
        "S": set(_poly([(15, 13), (22, 13), (22, 21), (24, 21)])),
        "W": set(_poly([(13, 11), (13, 4), (21, 4), (21, 2)])),
    }
    layout.branch_lengths = {k: len(v) for k, v in layout.branch_cells.items()}
    return layout


def _layout_irregular() -> Layout:
    layout = _base_layout()
    layout.seed_cells = {
        "N": {(0, 3), (0, 4), (0, 5), (1, 3), (2, 3), (2, 4), (2, 5)},
        "E": {(3, 24), (4, 24), (5, 24), (5, 25), (5, 26), (6, 26), (7, 26)},
        "S": {(24, 21), (24, 22), (24, 23), (25, 23), (26, 21), (26, 22), (26, 23)},
        "W": {(19, 0), (20, 0), (21, 0), (21, 1), (21, 2), (22, 2), (23, 2)},
    }
    layout.border_cells = {k: _adjacent_border(v, layout.shape) for k, v in layout.seed_cells.items()}
    return layout


def _layout_for_variant(variant_id: str) -> Layout:
    if variant_id == "D4_0":
        return _layout_dual_hub()
    if variant_id == "D4_1":
        return _layout_no_hub_point()
    if variant_id == "D7_1":
        return _layout_cross_hub()
    if variant_id == "D8_1":
        return _layout_width_coded()
    if variant_id == "D16_0":
        return _layout_interior_seeds()
    if variant_id == "D18_2":
        return _layout_shared_path()
    if variant_id == "D10_1":
        return _layout_short_delete()
    if variant_id == "D20_1":
        return _layout_path_color_split()
    if variant_id == "D20_2":
        return _layout_border_split()
    if variant_id == "D5_3":
        return _layout_no_border()
    if variant_id == "D5_2":
        return _layout_irregular()
    if variant_id == "D6_1":
        return _layout_length_order()
    if variant_id == "D14_2":
        return _layout_bend_parity()
    return _base_layout()


def _variant_major_plan(variant_id: str, slots: list[str]) -> dict[str, int]:
    if variant_id == "D2_0":
        colors = [SEED_A, SEED_A, SEED_A, SEED_B]
    elif variant_id in {"D6_0", "D19_2"}:
        colors = [SEED_A, SEED_B, SEED_C, SEED_D]
    elif variant_id == "D18_2":
        colors = [SEED_A, SEED_C, SEED_A, SEED_B]
    else:
        colors = [SEED_A, SEED_A, SEED_B, SEED_C]
    out = {}
    for slot, color in zip(slots, colors):
        out[slot] = color
    return out


def _seed_count_plan(variant_id: str, slot: str, area: int, major: int) -> list[tuple[int, int]]:
    if variant_id in {"D5_1", "D12_1"}:
        rest = [c for c in SEED_PALETTE if c != major][:2]
        counts = [area - 5, 3, 2]
        return [(major, counts[0]), (rest[0], counts[1]), (rest[1], counts[2])]
    if variant_id == "D1_1":
        other = SEED_D if major != SEED_D else SEED_E
        return [(major, area - 5), (other, 5)]
    if variant_id == "D2_0":
        other = SEED_C if major != SEED_C else SEED_D
        return [(major, area - 4), (other, 4)]
    other = {
        SEED_A: SEED_D,
        SEED_B: SEED_E,
        SEED_C: SEED_D,
        SEED_D: SEED_B,
        SEED_E: SEED_C,
    }.get(major, SEED_D)
    return [(major, area - 4), (other, 4)]


def _noise_seed_counts(area: int) -> list[tuple[int, int]]:
    return [(SEED_E, area - 3), (SEED_B, 3)]


def _paint_input_for_layout(grid: np.ndarray, layout: Layout, variant_id: str) -> None:
    hubs = layout.hub_cells
    for slot, cells in layout.branch_cells.items():
        _fill_cells(grid, cells, layout.path_colors.get(slot, PATH))
    for cells in layout.border_cells.values():
        _fill_cells(grid, cells, layout.border_color)
    _fill_cells(grid, hubs, PATH)
    if variant_id == "D20_2":
        for cells in layout.border_cells.values():
            _fill_cells(grid, cells, BORDER2)


def _add_variant_markers(grid: np.ndarray, layout: Layout, variant_id: str, seed_majors: dict[str, int]) -> None:
    if variant_id == "D6_0":
        layout.marker_cells = {
            slot: pt
            for slot, pt in zip(layout.branch_cells, [(10, 12), (12, 21), (21, 14), (14, 4)])
        }
        for slot, pt in layout.marker_cells.items():
            grid[pt] = seed_majors[slot]
    elif variant_id == "D19_2":
        layout.marker_cells = {
            slot: pt
            for slot, pt in zip(layout.branch_cells, [(10, 14), (12, 20), (21, 12), (14, 5)])
        }
        overrides = {
            slot: color
            for slot, color in zip(layout.branch_cells, [SEED_C, SEED_A, SEED_D, SEED_B])
        }
        layout.slot_colors_override = overrides
        for slot, pt in layout.marker_cells.items():
            grid[pt] = overrides[slot]
    elif variant_id == "D19_0":
        pairs = [
            ((0, 11), (0, 12), SEED_A, SEED_C),
            ((1, 11), (1, 12), SEED_B, SEED_A),
            ((0, 14), (0, 15), SEED_C, SEED_D),
            ((1, 14), (1, 15), SEED_D, SEED_B),
        ]
        layout.legend_pairs = [((a[0], a[1]), (b[0], b[1])) for a, b, _, _ in pairs]
        for (r0, c0), (r1, c1), src, dst in pairs:
            grid[r0, c0] = src
            grid[r1, c1] = dst


def _init_variant_metadata(layout: Layout, variant_id: str) -> None:
    if variant_id == "D6_0":
        layout.marker_cells = {
            slot: pt
            for slot, pt in zip(layout.branch_cells, [(10, 12), (12, 21), (21, 14), (14, 4)])
        }
    elif variant_id == "D19_2":
        layout.marker_cells = {
            slot: pt
            for slot, pt in zip(layout.branch_cells, [(10, 14), (12, 20), (21, 12), (14, 5)])
        }
    elif variant_id == "D19_0":
        layout.legend_pairs = [
            ((0, 11), (0, 12)),
            ((1, 11), (1, 12)),
            ((0, 14), (0, 15)),
            ((1, 14), (1, 15)),
        ]


def _add_variant_noise(grid: np.ndarray, layout: Layout, variant_id: str, rng: random.Random) -> None:
    if variant_id == "D17_0":
        layout.noise_path_cells = set(_poly([(7, 17), (7, 20), (9, 20)]))
        _fill_cells(grid, layout.noise_path_cells, PATH)
    elif variant_id == "D17_1":
        layout.disconnected_seed_cells = {
            "X1": _rect(0, 2, 18, 22),
            "X2": _rect(10, 12, 0, 2),
        }
        for cells in layout.disconnected_seed_cells.values():
            _paint_seed_counts(grid, cells, _noise_seed_counts(len(cells)), rng)
    elif variant_id == "D17_2":
        layout.noise_pixels = [(9, 9), (9, 10), (17, 17), (18, 17)]
        colors = [SEED_A, SEED_D, SEED_B, SEED_E]
        for (r, c), color in zip(layout.noise_pixels, colors):
            grid[r, c] = color


def _paint_seed_regions(grid: np.ndarray, layout: Layout, variant_id: str, rng: random.Random) -> dict[str, int]:
    slots = list(layout.seed_cells)
    majors = _variant_major_plan(variant_id, slots)
    for slot, cells in layout.seed_cells.items():
        counts = _seed_count_plan(variant_id, slot, len(cells), majors[slot])
        _paint_seed_counts(grid, cells, counts, rng)
    return majors


def _paint_disconnected_regions(grid: np.ndarray, layout: Layout, rng: random.Random) -> None:
    for cells in layout.disconnected_seed_cells.values():
        _paint_seed_counts(grid, cells, _noise_seed_counts(len(cells)), rng)


def _variable_hub_cells(k: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    r0 = 13 - k // 2
    c0 = 13 - k // 2
    hub = _rect(r0, r0 + k - 1, c0, c0 + k - 1)
    return hub, (13, 13)


def generate_input_for_variant(variant_id: str) -> np.ndarray:
    rng = random.Random()
    layout = _layout_for_variant(variant_id)
    _init_variant_metadata(layout, variant_id)
    grid = np.full(layout.shape, BG, dtype=int)

    if variant_id == "D7_0":
        k = rng.choice([2, 3, 4, 5])
        hub, center = _variable_hub_cells(k)
        layout.hub_cells = hub
        layout.center_cells = [center]
        anchors = {
            "N": (min(r for r, _ in hub) - 1, 13),
            "E": (13, max(c for _, c in hub) + 1),
            "S": (max(r for r, _ in hub) + 1, 13),
            "W": (13, min(c for _, c in hub) - 1),
        }
        layout.branch_cells = {
            "N": set(_poly([anchors["N"], (4, 13), (4, 5), (3, 5)])),
            "E": set(_poly([anchors["E"], (13, 22), (5, 22), (5, 23)])),
            "S": set(_poly([anchors["S"], (22, 13), (22, 21), (23, 21)])),
            "W": set(_poly([anchors["W"], (13, 4), (21, 4), (21, 3)])),
        }
        layout.branch_lengths = {k: len(v) for k, v in layout.branch_cells.items()}

    seed_majors = _paint_seed_regions(grid, layout, variant_id, rng)
    _paint_input_for_layout(grid, layout, variant_id)
    _add_variant_markers(grid, layout, variant_id, seed_majors)
    _add_variant_noise(grid, layout, variant_id, rng)
    _paint_disconnected_regions(grid, layout, rng)

    if variant_id == "D20_1":
        for slot, color in layout.path_colors.items():
            _fill_cells(grid, layout.branch_cells[slot], color)
    if variant_id == "D20_2":
        for cells in layout.border_cells.values():
            _fill_cells(grid, cells, BORDER2)
    if variant_id == "D5_3":
        for cells in layout.border_cells.values():
            assert not cells

    return grid


def _legend_mapping(grid: np.ndarray, layout: Layout) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for src_pt, dst_pt in layout.legend_pairs:
        mapping[int(grid[src_pt])] = int(grid[dst_pt])
    return mapping


def _branch_vote_color(colors: list[int], least: bool = False) -> int:
    cnt = Counter(colors)
    if least:
        return int(sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0][0])
    return int(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])


def _slot_output_color(
    grid: np.ndarray,
    layout: Layout,
    variant_id: str,
    slot: str,
    major_by_slot: dict[str, int],
    minor_by_slot: dict[str, int],
    seed_area_rank: dict[str, int],
) -> int:
    if variant_id == "D1_0":
        return minor_by_slot[slot]
    if variant_id == "D6_1":
        branch_order = sorted(layout.branch_lengths, key=lambda s: (layout.branch_lengths[s], s))
        seed_order = sorted(layout.seed_cells, key=lambda s: (len(layout.seed_cells[s]), s))
        idx = branch_order.index(slot)
        return major_by_slot[seed_order[idx]]
    if variant_id == "D8_1":
        return major_by_slot[slot] if layout.branch_widths[slot] == 2 else minor_by_slot[slot]
    if variant_id == "D14_2":
        return major_by_slot[slot] if layout.branch_bends[slot] % 2 == 0 else minor_by_slot[slot]
    if variant_id == "D19_0":
        return _legend_mapping(grid, layout).get(major_by_slot[slot], major_by_slot[slot])
    if variant_id == "D19_2":
        marker_pt = layout.marker_cells[slot]
        return int(grid[marker_pt])
    if variant_id == "D6_0":
        marker_color = int(grid[layout.marker_cells[slot]])
        candidates = sorted(layout.seed_cells, key=lambda s: (major_by_slot[s] != marker_color, s))
        return major_by_slot[candidates[0]]
    return major_by_slot[slot]


def _apply_ratio_branch(out: np.ndarray, grid: np.ndarray, cells: set[tuple[int, int]], seed_cells: set[tuple[int, int]]) -> None:
    counts = _seed_stats(grid, seed_cells)
    items = counts.most_common(2)
    major, minor = int(items[0][0]), int(items[1][0])
    seq = _ratio_pattern(major, minor, items[0][1], items[1][1], len(cells))
    ordered = _ordered_branch(cells)
    for (r, c), color in zip(ordered, seq):
        out[r, c] = color


def _hub_vote(
    out: np.ndarray,
    layout: Layout,
    hub_idx: int,
    by_seed_count: bool = False,
    least: bool = False,
) -> int:
    slots = [s for s, idx in layout.slot_to_hub.items() if idx == hub_idx]
    colors: list[int] = []
    if by_seed_count:
        for slot in slots:
            branch_color = int(out[next(iter(layout.branch_cells[slot]))])
            colors.append(branch_color)
    else:
        for slot in slots:
            branch_adj = sorted(layout.branch_cells[slot], key=lambda rc: (abs(rc[0] - layout.center_cells[hub_idx][0]) + abs(rc[1] - layout.center_cells[hub_idx][1])))
            colors.append(int(out[branch_adj[0]]))
    return _branch_vote_color(colors, least=least)


def main_for_variant(grid: np.ndarray, variant_id: str) -> np.ndarray:
    g = np.array(grid, dtype=int)
    layout = _layout_for_variant(variant_id)
    _init_variant_metadata(layout, variant_id)
    out = g.copy()

    if variant_id == "D7_0":
        path_cells = {(r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if int(g[r, c]) == PATH}
        rows = [r for r, c in path_cells if 9 <= r <= 17 and 9 <= c <= 17]
        cols = [c for r, c in path_cells if 9 <= r <= 17 and 9 <= c <= 17]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        layout.hub_cells = _rect(r0, r1, c0, c1)
        layout.center_cells = [(13, 13)]

    major_by_slot = {slot: _major_color(g, cells) for slot, cells in layout.seed_cells.items()}
    minor_by_slot = {slot: _minor_color(g, cells) for slot, cells in layout.seed_cells.items()}
    seed_area_rank = {slot: len(cells) for slot, cells in layout.seed_cells.items()}

    if variant_id == "D3_0":
        for slot, cells in layout.seed_cells.items():
            _fill_cells(out, cells, PATH)
    if variant_id == "D1_3":
        for slot, cells in layout.seed_cells.items():
            major = major_by_slot[slot]
            minor = minor_by_slot[slot]
            for r, c in cells:
                val = int(out[r, c])
                if val == major:
                    out[r, c] = minor
                elif val == minor:
                    out[r, c] = major

    if variant_id == "D4_0":
        slots = list(layout.seed_cells)
    else:
        slots = list(layout.branch_cells)

    for slot in slots:
        branch_color = _slot_output_color(g, layout, variant_id, slot, major_by_slot, minor_by_slot, seed_area_rank)
        if variant_id == "D1_1":
            _apply_ratio_branch(out, g, layout.branch_cells[slot], layout.seed_cells[slot])
        elif variant_id == "D3_0":
            pass
        else:
            _fill_cells(out, layout.branch_cells[slot], branch_color)

        if variant_id == "D13_0":
            _fill_cells(out, layout.border_cells[slot], PATH if variant_id != "D20_2" else BORDER2)
        elif variant_id == "D13_1":
            _fill_cells(out, layout.border_cells[slot], minor_by_slot[slot])
        elif variant_id == "D3_1":
            _fill_cells(out, layout.border_cells[slot], minor_by_slot[slot])
        elif variant_id == "D3_0":
            _fill_cells(out, layout.border_cells[slot], PATH if layout.border_color == PATH else layout.border_color)
        else:
            if variant_id != "D20_2":
                _fill_cells(out, layout.border_cells[slot], branch_color)
            else:
                _fill_cells(out, layout.border_cells[slot], branch_color)

    if variant_id == "D10_1":
        for slot, cells in layout.branch_cells.items():
            if len(cells) < SHORT_BRANCH_THRESHOLD:
                _fill_cells(out, cells, BG)

    if variant_id == "D18_2":
        top_slots = ["NL", "NR"]
        larger = max(top_slots, key=lambda s: len(layout.seed_cells[s]))
        shared_color = major_by_slot[larger]
        _fill_cells(out, layout.shared_cells, shared_color)
        _fill_cells(out, layout.branch_cells["NL"] - layout.shared_cells, major_by_slot["NL"])
        _fill_cells(out, layout.branch_cells["NR"] - layout.shared_cells, major_by_slot["NR"])
        _fill_cells(out, layout.branch_cells["S"], major_by_slot["S"])
        _fill_cells(out, layout.branch_cells["W"], major_by_slot["W"])
        for slot, cells in layout.border_cells.items():
            _fill_cells(out, cells, major_by_slot[slot])

    if variant_id == "D20_1":
        pass

    if variant_id == "D2_1":
        pass

    if variant_id == "D4_1":
        center = (13, 13)
        colors = [int(out[pt]) for pt in [(12, 13), (13, 14), (14, 13), (13, 12)] if int(out[pt]) not in {BG, PATH}]
        out[center] = _branch_vote_color(colors)
        return out

    if variant_id == "D7_1":
        center = (13, 13)
        colors = [int(out[pt]) for pt in [(11, 13), (13, 15), (15, 13), (13, 11)] if int(out[pt]) != BG]
        out[center] = _branch_vote_color(colors)
        return out

    if variant_id == "D4_0":
        for idx, center in enumerate(layout.center_cells):
            hub_color = _hub_vote(out, layout, idx)
            out[center] = hub_color
        return out

    if variant_id == "D2_3":
        out[layout.center_cells[0]] = _hub_vote(out, layout, 0, by_seed_count=True)
    elif variant_id == "D2_0":
        out[layout.center_cells[0]] = _hub_vote(out, layout, 0, least=True)
    else:
        out[layout.center_cells[0]] = _hub_vote(out, layout, 0)

    if variant_id == "D17_1":
        for cells in layout.disconnected_seed_cells.values():
            for r, c in cells:
                out[r, c] = g[r, c]

    return out
