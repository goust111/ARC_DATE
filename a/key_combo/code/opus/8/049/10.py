import random
from dataclasses import dataclass

import numpy as np


BG = 0
BAR_COLORS = (1, 2, 7)
GATE_CENTER = 3
UNMATCHED_COLORS = (4, 5, 6, 8)
GATE_END = 9

GRID_MIN = 12
GRID_MAX = 16
BAR_COUNT_CHOICES = (2, 3)
MARKER_COUNT_MIN = 3
MARKER_COUNT_MAX = 6
GENERATION_ATTEMPTS = 400


@dataclass(frozen=True)
class Bar:
    color: int
    primary: int


@dataclass(frozen=True)
class Gate:
    gate_id: int
    primary: int
    lane: int
    consumed: bool


@dataclass(frozen=True)
class MarkerPlan:
    color: int
    primary: int
    lane: int
    bar_color: int | None


def _coord(orientation: str, primary: int, lane: int) -> tuple[int, int]:
    return (primary, lane) if orientation == "horizontal" else (lane, primary)


def _primary_lane(orientation: str, r: int, c: int) -> tuple[int, int]:
    return (r, c) if orientation == "horizontal" else (c, r)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _full_row_color(grid: np.ndarray, r: int) -> int | None:
    color = int(grid[r, 0])
    if color != BG and np.all(grid[r, :] == color):
        return color
    return None


def _full_col_color(grid: np.ndarray, c: int) -> int | None:
    color = int(grid[0, c])
    if color != BG and np.all(grid[:, c] == color):
        return color
    return None


def _detect_bars(grid: np.ndarray) -> tuple[str, dict[int, Bar], set[tuple[int, int]]]:
    h, w = grid.shape
    row_bars = [(int(grid[r, 0]), r) for r in range(h) if _full_row_color(grid, r) is not None]
    col_bars = [(int(grid[0, c]), c) for c in range(w) if _full_col_color(grid, c) is not None]

    if row_bars and col_bars:
        raise AssertionError("bars must share a single orientation")
    if not row_bars and not col_bars:
        raise AssertionError("expected 2-3 full-span bars")

    if row_bars:
        orientation = "horizontal"
        bars = {color: Bar(color=color, primary=primary) for color, primary in row_bars}
        cells = {(primary, c) for _color, primary in row_bars for c in range(w)}
    else:
        orientation = "vertical"
        bars = {color: Bar(color=color, primary=primary) for color, primary in col_bars}
        cells = {(r, primary) for _color, primary in col_bars for r in range(h)}

    assert 2 <= len(bars) <= 3
    assert len(bars) == len(set(bars))
    return orientation, bars, cells


def _detect_doors(grid: np.ndarray, orientation: str) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    h, w = grid.shape
    doors: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != GATE_CENTER:
                continue
            if orientation == "horizontal":
                if 0 < c < w - 1 and int(grid[r, c - 1]) == GATE_END and int(grid[r, c + 1]) == GATE_END:
                    doors[(r, c)] = ((r, c - 1), (r, c), (r, c + 1))
            else:
                if 0 < r < h - 1 and int(grid[r - 1, c]) == GATE_END and int(grid[r + 1, c]) == GATE_END:
                    doors[(r, c)] = ((r - 1, c), (r, c), (r + 1, c))
    return doors


def _extract_markers(
    grid: np.ndarray,
    bar_cells: set[tuple[int, int]],
    door_cells: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    markers = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or (r, c) in bar_cells or (r, c) in door_cells:
                continue
            markers.append((r, c, color))
    return markers


def _marker_sort_key(
    orientation: str,
    marker: tuple[int, int, int],
    bars: dict[int, Bar],
) -> tuple[int, int, int]:
    r, c, color = marker
    if color not in bars:
        return (10**9, r, c)
    primary, _lane = _primary_lane(orientation, r, c)
    bar_primary = bars[color].primary
    distance = abs(primary - bar_primary) - 1
    return (distance, r, c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored marker toward its same-color full bar, but stop one cell before any maroon-green-maroon gate on that path and erase the gate.

    Input structure:
    - The grid has a black background with 2-3 same-orientation full-span bars whose colors are chosen from blue, red, and orange.
    - Additional single-pixel markers are scattered off the bars; some share a color with one bar and some use colors with no matching bar.
    - Some straight marker-to-bar paths contain one perpendicular maroon-green-maroon triplet whose green center lies on the sliding line.

    Rule:
    - Delete every marker whose color has no matching full-span bar.
    - For each remaining marker, slide straight toward the same-color bar: vertically if the bars are horizontal, otherwise horizontally.
    - If the path contains no gate, stop on the cell adjacent to the target bar; if the first active gate center lies on the path, stop on the cell immediately before that center and erase the whole gate triplet.
    - Process matched markers from smaller bar-distance to larger bar-distance, breaking ties by smaller marker row; settled markers never block later slides.

    Key insight:
    - Color selects the destination bar, but the perpendicular maroon-green-maroon motif acts as a one-use interrupt that changes the stopping point and then disappears for later markers.
    """
    g = np.asarray(grid, dtype=int)
    orientation, bars, bar_cells = _detect_bars(g)
    doors = _detect_doors(g, orientation)
    door_cells = {cell for triplet in doors.values() for cell in triplet}
    markers = _extract_markers(g, bar_cells, door_cells)

    out = g.copy()
    active_doors = dict(doors)

    for r, c, color in sorted(markers, key=lambda item: _marker_sort_key(orientation, item, bars)):
        out[r, c] = BG
        bar = bars.get(color)
        if bar is None:
            continue

        primary, lane = _primary_lane(orientation, r, c)
        step = 1 if bar.primary > primary else -1
        stop_primary = bar.primary - step
        hit_center: tuple[int, int] | None = None

        probe = primary + step
        while probe != bar.primary:
            coord = _coord(orientation, probe, lane)
            if coord in active_doors:
                stop_primary = probe - step
                hit_center = coord
                break
            probe += step

        stop_r, stop_c = _coord(orientation, stop_primary, lane)
        out[stop_r, stop_c] = color

        if hit_center is not None:
            for cell in active_doors.pop(hit_center):
                out[cell] = BG

    return out


def _sample_bar_positions(length: int, count: int) -> list[int] | None:
    candidates = list(range(1, length - 1))
    for _ in range(120):
        positions = sorted(random.sample(candidates, count))
        if min(b - a for a, b in zip(positions, positions[1:])) < 4:
            continue
        return positions
    return None


def _side_specs(bar_positions: list[int], primary_len: int) -> list[dict[str, int]]:
    specs = []
    for idx, bar_primary in enumerate(bar_positions):
        prev_bar = bar_positions[idx - 1] if idx > 0 else None
        next_bar = bar_positions[idx + 1] if idx + 1 < len(bar_positions) else None

        low = 0 if prev_bar is None else prev_bar + 1
        high = bar_primary - 2
        if low <= high:
            specs.append(
                {
                    "bar_primary": bar_primary,
                    "step": 1,
                    "start_low": low,
                    "start_high": high,
                    "start_count": high - low + 1,
                }
            )

        low = bar_primary + 2
        high = primary_len - 1 if next_bar is None else next_bar - 1
        if low <= high:
            specs.append(
                {
                    "bar_primary": bar_primary,
                    "step": -1,
                    "start_low": low,
                    "start_high": high,
                    "start_count": high - low + 1,
                }
            )
    return specs


def _scenario_cells(
    orientation: str,
    markers: list[tuple[int, int]],
    gate: tuple[int, int] | None,
    lane: int,
) -> set[tuple[int, int]]:
    cells = { _coord(orientation, primary, lane) for primary, lane in markers }
    if gate is not None:
        gate_primary, gate_lane = gate
        if orientation == "horizontal":
            cells.update({(gate_primary, gate_lane - 1), (gate_primary, gate_lane), (gate_primary, gate_lane + 1)})
        else:
            cells.update({(gate_lane - 1, gate_primary), (gate_lane, gate_primary), (gate_lane + 1, gate_primary)})
    return cells


def _clear_cells(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _choose_plain(spec: dict[str, int]) -> int:
    candidates = list(range(spec["start_low"], spec["start_high"] + 1))
    return random.choice(candidates)


def _choose_gated(spec: dict[str, int]) -> tuple[int, int] | None:
    bar_primary = spec["bar_primary"]
    step = spec["step"]
    starts = [
        start
        for start in range(spec["start_low"], spec["start_high"] + 1)
        if abs(bar_primary - start) >= 3
    ]
    if not starts:
        return None
    start = random.choice(starts)
    if step == 1:
        gate_candidates = list(range(start + 2, bar_primary))
    else:
        gate_candidates = list(range(bar_primary + 1, start - 1))
    if not gate_candidates:
        return None
    gate_primary = random.choice(gate_candidates)
    return start, gate_primary


def _choose_shared(spec: dict[str, int]) -> tuple[int, int, int] | None:
    bar_primary = spec["bar_primary"]
    step = spec["step"]
    gate_candidates = []
    if step == 1:
        for gate_primary in range(spec["start_low"] + 4, bar_primary):
            near_primary = gate_primary - 2
            far_primary = gate_primary - 4 - random.randint(0, 1)
            if spec["start_low"] <= far_primary < near_primary <= spec["start_high"]:
                gate_candidates.append((far_primary, near_primary, gate_primary))
    else:
        for gate_primary in range(bar_primary + 1, spec["start_high"] - 3):
            near_primary = gate_primary + 2
            far_primary = gate_primary + 4 + random.randint(0, 1)
            if spec["start_low"] <= near_primary < far_primary <= spec["start_high"]:
                gate_candidates.append((far_primary, near_primary, gate_primary))
    if not gate_candidates:
        return None
    return random.choice(gate_candidates)


def _place_gate(
    grid: np.ndarray,
    orientation: str,
    gate_primary: int,
    lane: int,
    gate_id: int,
    occupied: set[tuple[int, int]],
    gate_records: list[Gate],
) -> bool:
    if orientation == "horizontal":
        if not (1 <= lane < grid.shape[1] - 1):
            return False
        cells = {(gate_primary, lane - 1), (gate_primary, lane), (gate_primary, lane + 1)}
    else:
        if not (1 <= lane < grid.shape[0] - 1):
            return False
        cells = {(lane - 1, gate_primary), (lane, gate_primary), (lane + 1, gate_primary)}
    if not _clear_cells(cells, occupied):
        return False
    for r, c in cells:
        if grid[r, c] != BG:
            return False
    occupied.update(cells)
    for cell in cells:
        grid[cell] = GATE_CENTER if cell == _coord(orientation, gate_primary, lane) else GATE_END
    gate_records.append(Gate(gate_id=gate_id, primary=gate_primary, lane=lane, consumed=False))
    return True


def _place_marker(
    grid: np.ndarray,
    orientation: str,
    primary: int,
    lane: int,
    color: int,
    occupied: set[tuple[int, int]],
    marker_records: list[MarkerPlan],
) -> bool:
    r, c = _coord(orientation, primary, lane)
    cells = {(r, c)}
    if not _clear_cells(cells, occupied):
        return False
    if grid[r, c] != BG:
        return False
    grid[r, c] = color
    occupied.add((r, c))
    marker_records.append(MarkerPlan(color=color, primary=primary, lane=lane, bar_color=color if color in BAR_COLORS else None))
    return True


def _unused_gate_ok(
    orientation: str,
    lane: int,
    gate_primary: int,
    marker_specs: list[dict[str, int]],
) -> bool:
    for spec in marker_specs:
        if spec["lane"] != lane or spec["bar_color"] is None:
            continue
        start = spec["primary"]
        bar_primary = spec["bar_primary"]
        low = min(start, bar_primary)
        high = max(start, bar_primary)
        if low < gate_primary < high:
            return False
    return True


def _build_expected(
    shape: tuple[int, int],
    orientation: str,
    bars: list[Bar],
    markers: list[dict[str, int]],
    gates: list[dict[str, int]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    if orientation == "horizontal":
        for bar in bars:
            out[bar.primary, :] = bar.color
    else:
        for bar in bars:
            out[:, bar.primary] = bar.color

    active_gates = {gate["gate_id"]: dict(gate) for gate in gates}
    bar_by_color = {bar.color: bar for bar in bars}
    order = sorted(
        [m for m in markers if m["bar_color"] is not None],
        key=lambda m: (abs(m["primary"] - m["bar_primary"]) - 1, _coord(orientation, m["primary"], m["lane"])[0]),
    )

    for marker in order:
        bar_primary = marker["bar_primary"]
        step = 1 if bar_primary > marker["primary"] else -1
        stop_primary = bar_primary - step
        probe = marker["primary"] + step
        while probe != bar_primary:
            hit_gate = None
            for gate in active_gates.values():
                if gate["primary"] == probe and gate["lane"] == marker["lane"]:
                    hit_gate = gate
                    break
            if hit_gate is not None:
                stop_primary = probe - step
                hit_gate["consumed"] = True
                del active_gates[hit_gate["gate_id"]]
                break
            probe += step
        r, c = _coord(orientation, stop_primary, marker["lane"])
        out[r, c] = marker["color"]

    for gate in active_gates.values():
        gate_primary = gate["primary"]
        lane = gate["lane"]
        if orientation == "horizontal":
            out[gate_primary, lane - 1] = GATE_END
            out[gate_primary, lane] = GATE_CENTER
            out[gate_primary, lane + 1] = GATE_END
        else:
            out[lane - 1, gate_primary] = GATE_END
            out[lane, gate_primary] = GATE_CENTER
            out[lane + 1, gate_primary] = GATE_END

    return out


def _validate_instance(
    grid: np.ndarray,
    orientation: str,
    bars: list[Bar],
    markers: list[dict[str, int]],
    gates: list[dict[str, int]],
) -> bool:
    try:
        parsed_orientation, parsed_bars, bar_cells = _detect_bars(grid)
    except AssertionError:
        return False
    if parsed_orientation != orientation:
        return False
    if sorted((bar.color, bar.primary) for bar in bars) != sorted((bar.color, bar.primary) for bar in parsed_bars.values()):
        return False

    doors = _detect_doors(grid, orientation)
    door_cells = {cell for triplet in doors.values() for cell in triplet}
    marker_pixels = _extract_markers(grid, bar_cells, door_cells)
    if not (MARKER_COUNT_MIN <= len(marker_pixels) <= MARKER_COUNT_MAX):
        return False

    seen_order_keys = set()
    for marker in markers:
        if marker["bar_color"] is None:
            continue
        row, _col = _coord(orientation, marker["primary"], marker["lane"])
        key = (abs(marker["primary"] - marker["bar_primary"]) - 1, row)
        if key in seen_order_keys:
            return False
        seen_order_keys.add(key)

        path_gate_count = sum(
            gate["lane"] == marker["lane"]
            and min(marker["primary"], marker["bar_primary"]) < gate["primary"] < max(marker["primary"], marker["bar_primary"])
            for gate in gates
        )
        if path_gate_count > 1:
            return False

    expected = _build_expected(grid.shape, orientation, bars, markers, gates)
    actual = main(grid)
    return np.array_equal(expected, actual)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    orientation = random.choice(("horizontal", "vertical"))
    primary_len, lane_len = (h, w) if orientation == "horizontal" else (w, h)

    bar_count_options = [count for count in BAR_COUNT_CHOICES if count <= len(BAR_COLORS)]
    random.shuffle(bar_count_options)

    for bar_count in bar_count_options:
        bar_positions = _sample_bar_positions(primary_len, bar_count)
        if bar_positions is None:
            continue
        side_specs = _side_specs(bar_positions, primary_len)
        plain_specs = [spec for spec in side_specs if spec["start_count"] >= 1]
        gated_specs = [spec for spec in side_specs if spec["start_count"] >= 2]
        shared_specs = [spec for spec in side_specs if spec["start_count"] >= 4]
        if not plain_specs or not gated_specs:
            continue

        colors = random.sample(list(BAR_COLORS), bar_count)
        bars = [Bar(color=color, primary=primary) for color, primary in zip(colors, bar_positions)]
        color_by_primary = {bar.primary: bar.color for bar in bars}
        grid = np.zeros((h, w), dtype=int)
        if orientation == "horizontal":
            for bar in bars:
                grid[bar.primary, :] = bar.color
        else:
            for bar in bars:
                grid[:, bar.primary] = bar.color

        occupied: set[tuple[int, int]] = set()
        markers: list[dict[str, int]] = []
        gates: list[dict[str, int]] = []
        gate_id = 0

        scenario_choices = ["plain", "gated"]
        if shared_specs:
            scenario_choices.append("shared")
        gate_scenario = random.choice(scenario_choices)

        plans_to_place: list[dict[str, object]] = [{"kind": "plain"}]
        plans_to_place.append({"kind": gate_scenario})
        while len(plans_to_place) < random.randint(3, 5):
            plans_to_place.append(
                {
                    "kind": random.choices(
                        ["plain", "gated", "unmatched"],
                        weights=[0.35, 0.35, 0.30],
                        k=1,
                    )[0]
                }
            )
        if not any(plan["kind"] == "unmatched" for plan in plans_to_place):
            plans_to_place.append({"kind": "unmatched"})

        total_marker_count = 0
        placed_ok = True
        used_bar_primaries: set[int] = set()
        used_matched_lanes: set[int] = set()

        for plan in plans_to_place:
            kind = plan["kind"]
            if kind == "unmatched":
                continue

            candidate_specs = plain_specs if kind == "plain" else gated_specs if kind == "gated" else shared_specs
            random.shuffle(candidate_specs)
            placed_this = False
            for spec in candidate_specs:
                bar_primary = spec["bar_primary"]
                bar_color = color_by_primary[bar_primary]
                candidate_lanes = list(range(1, lane_len - 1))
                random.shuffle(candidate_lanes)
                if bar_primary in used_bar_primaries and len(used_bar_primaries) < len(bars):
                    continue
                for lane in candidate_lanes:
                    if lane in used_matched_lanes:
                        continue
                    if kind == "plain":
                        start_primary = _choose_plain(spec)
                        if not _place_marker(grid, orientation, start_primary, lane, bar_color, occupied, []):
                            continue
                        occupied.remove(_coord(orientation, start_primary, lane))
                        grid[_coord(orientation, start_primary, lane)] = BG
                        if not _place_marker(grid, orientation, start_primary, lane, bar_color, occupied, []):
                            continue
                        markers.append(
                            {
                                "color": bar_color,
                                "primary": start_primary,
                                "lane": lane,
                                "bar_color": bar_color,
                                "bar_primary": bar_primary,
                            }
                        )
                        placed_this = True
                        total_marker_count += 1
                    elif kind == "gated":
                        choice = _choose_gated(spec)
                        if choice is None:
                            continue
                        start_primary, gate_primary = choice
                        if not _place_gate(grid, orientation, gate_primary, lane, gate_id, occupied, []):
                            continue
                        gate_cells = _scenario_cells(orientation, [], (gate_primary, lane), lane)
                        for cell in gate_cells:
                            occupied.remove(cell)
                            grid[cell] = BG
                        if not _place_gate(grid, orientation, gate_primary, lane, gate_id, occupied, []):
                            continue
                        if not _place_marker(grid, orientation, start_primary, lane, bar_color, occupied, []):
                            for cell in gate_cells:
                                occupied.remove(cell)
                                grid[cell] = BG
                            continue
                        gates.append({"gate_id": gate_id, "primary": gate_primary, "lane": lane})
                        gate_id += 1
                        markers.append(
                            {
                                "color": bar_color,
                                "primary": start_primary,
                                "lane": lane,
                                "bar_color": bar_color,
                                "bar_primary": bar_primary,
                            }
                        )
                        placed_this = True
                        total_marker_count += 1
                    else:
                        choice = _choose_shared(spec)
                        if choice is None:
                            continue
                        far_primary, near_primary, gate_primary = choice
                        if not _place_gate(grid, orientation, gate_primary, lane, gate_id, occupied, []):
                            continue
                        gate_cells = _scenario_cells(orientation, [], (gate_primary, lane), lane)
                        for cell in gate_cells:
                            occupied.remove(cell)
                            grid[cell] = BG
                        if not _place_gate(grid, orientation, gate_primary, lane, gate_id, occupied, []):
                            continue
                        if not _place_marker(grid, orientation, near_primary, lane, bar_color, occupied, []):
                            for cell in gate_cells:
                                occupied.remove(cell)
                                grid[cell] = BG
                            continue
                        if not _place_marker(grid, orientation, far_primary, lane, bar_color, occupied, []):
                            cell = _coord(orientation, near_primary, lane)
                            occupied.remove(cell)
                            grid[cell] = BG
                            for cell in gate_cells:
                                occupied.remove(cell)
                                grid[cell] = BG
                            continue
                        gates.append({"gate_id": gate_id, "primary": gate_primary, "lane": lane})
                        gate_id += 1
                        markers.extend(
                            [
                                {
                                    "color": bar_color,
                                    "primary": near_primary,
                                    "lane": lane,
                                    "bar_color": bar_color,
                                    "bar_primary": bar_primary,
                                },
                                {
                                    "color": bar_color,
                                    "primary": far_primary,
                                    "lane": lane,
                                    "bar_color": bar_color,
                                    "bar_primary": bar_primary,
                                },
                            ]
                        )
                        placed_this = True
                        total_marker_count += 2

                    if placed_this:
                        used_bar_primaries.add(bar_primary)
                        used_matched_lanes.add(lane)
                        break
                if placed_this:
                    break

            if not placed_this:
                placed_ok = False
                break

        if not placed_ok:
            continue

        target_total = max(total_marker_count + 1, random.randint(MARKER_COUNT_MIN, MARKER_COUNT_MAX))
        if target_total > MARKER_COUNT_MAX:
            continue

        while len(markers) < target_total:
            candidate_positions = [
                (primary, lane)
                for primary in range(primary_len)
                for lane in range(lane_len)
                if _coord(orientation, primary, lane) not in occupied
            ]
            random.shuffle(candidate_positions)
            placed_unmatched = False
            for primary, lane in candidate_positions:
                r, c = _coord(orientation, primary, lane)
                if grid[r, c] != BG:
                    continue
                if any((nr, nc) in occupied for nr, nc in _neighbors4(r, c)):
                    continue
                color = random.choice(UNMATCHED_COLORS)
                grid[r, c] = color
                occupied.add((r, c))
                markers.append(
                    {
                        "color": color,
                        "primary": primary,
                        "lane": lane,
                        "bar_color": None,
                    }
                )
                placed_unmatched = True
                break
            if not placed_unmatched:
                placed_ok = False
                break

        if not placed_ok:
            continue

        if random.random() < 0.65:
            available_lanes = [lane for lane in range(1, lane_len - 1) if lane not in used_matched_lanes]
            random.shuffle(available_lanes)
            for lane in available_lanes[:]:
                gate_primaries = list(range(1, primary_len - 1))
                random.shuffle(gate_primaries)
                for gate_primary in gate_primaries:
                    if not _unused_gate_ok(orientation, lane, gate_primary, markers):
                        continue
                    if _place_gate(grid, orientation, gate_primary, lane, gate_id, occupied, []):
                        gates.append({"gate_id": gate_id, "primary": gate_primary, "lane": lane})
                        gate_id += 1
                        break
                else:
                    continue
                break

        if len(markers) < MARKER_COUNT_MIN or len(markers) > MARKER_COUNT_MAX:
            continue
        if not any(marker["bar_color"] is None for marker in markers):
            continue
        if not any(marker["bar_color"] is not None for marker in markers):
            continue
        if not gates:
            continue

        if _validate_instance(grid, orientation, bars, markers, gates):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-gate ARC input")
