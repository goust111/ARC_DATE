import random
from collections import Counter

import numpy as np


BG = 0
CARGO = 4
BELT_PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).reshape(-1), minlength=10)
    return int(np.argmax(counts))


def _axis_value(belt: dict, pos: tuple[int, int]) -> int:
    return int(pos[1] if belt["orientation"] == "H" else pos[0])


def _position_key_on_belt(belt: dict, pos: tuple[int, int]) -> int:
    return int(_axis_value(belt, pos) * belt["direction"])


def _traversable_positions(belt: dict) -> list[tuple[int, int]]:
    low = int(belt["low"])
    high = int(belt["high"])
    line = int(belt["line"])
    direction = int(belt["direction"])
    head_axis = high if direction > 0 else low
    tail_axis = low if direction > 0 else high

    if belt["orientation"] == "H":
        return [(line, c) for c in range(tail_axis, head_axis, direction)]
    return [(r, line) for r in range(tail_axis, head_axis, direction)]


def _parse_single_belt(grid: np.ndarray, color: int) -> dict:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
    assert coords, f"missing belt color {color}"

    row_counts = Counter(r for r, _ in coords)
    col_counts = Counter(c for _, c in coords)
    row, row_count = max(row_counts.items(), key=lambda kv: (kv[1], -kv[0]))
    col, col_count = max(col_counts.items(), key=lambda kv: (kv[1], -kv[0]))

    horizontal = row_count >= len(coords) - 1
    vertical = col_count >= len(coords) - 1
    assert horizontal or vertical, f"cannot infer orientation for color {color}"

    if horizontal and (not vertical or row_count >= col_count):
        axis_cells = [(r, c) for r, c in coords if r == row]
        off_axis = [(r, c) for r, c in coords if r != row]
        assert len(off_axis) == 1, f"expected one thick-head pixel for horizontal belt {color}"
        head_extra = off_axis[0]
        low = min(c for _, c in axis_cells)
        high = max(c for _, c in axis_cells)
        head_axis = int(head_extra[1])
        assert head_axis in {low, high}, f"head column mismatch for horizontal belt {color}"
        direction = 1 if head_axis == high else -1
        head_terminal = (int(row), int(head_axis))
        return {
            "color": int(color),
            "orientation": "H",
            "line": int(row),
            "low": int(low),
            "high": int(high),
            "direction": int(direction),
            "head_terminal": head_terminal,
            "head_extra": (int(head_extra[0]), int(head_extra[1])),
        }

    axis_cells = [(r, c) for r, c in coords if c == col]
    off_axis = [(r, c) for r, c in coords if c != col]
    assert len(off_axis) == 1, f"expected one thick-head pixel for vertical belt {color}"
    head_extra = off_axis[0]
    low = min(r for r, _ in axis_cells)
    high = max(r for r, _ in axis_cells)
    head_axis = int(head_extra[0])
    assert head_axis in {low, high}, f"head row mismatch for vertical belt {color}"
    direction = 1 if head_axis == high else -1
    head_terminal = (int(head_axis), int(col))
    return {
        "color": int(color),
        "orientation": "V",
        "line": int(col),
        "low": int(low),
        "high": int(high),
        "direction": int(direction),
        "head_terminal": head_terminal,
        "head_extra": (int(head_extra[0]), int(head_extra[1])),
    }


def _build_network(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) not in {bg, CARGO}]
    belts = {_color: _parse_single_belt(g, _color) for _color in colors}

    horizontals = [belt for belt in belts.values() if belt["orientation"] == "H"]
    verticals = [belt for belt in belts.values() if belt["orientation"] == "V"]
    assert horizontals and verticals, "expected at least one horizontal and one vertical belt"

    crossings: dict[tuple[int, int], dict] = {}
    for hb in horizontals:
        for vb in verticals:
            row = int(hb["line"])
            col = int(vb["line"])
            if hb["low"] <= col <= hb["high"] and vb["low"] <= row <= vb["high"]:
                pos = (row, col)
                value = int(g[pos])
                assert value in {hb["color"], vb["color"]}, "crossing priority must be one of the two belt colors"
                crossings[pos] = {
                    "horizontal": int(hb["color"]),
                    "vertical": int(vb["color"]),
                    "priority": int(value),
                }

    return {
        "grid": g,
        "bg": int(bg),
        "belts": belts,
        "crossings": crossings,
    }


def _covering_belts(network: dict, pos: tuple[int, int]) -> list[int]:
    r, c = map(int, pos)
    hits: list[int] = []
    for color, belt in network["belts"].items():
        if pos == belt["head_terminal"]:
            continue
        if belt["orientation"] == "H":
            if r == belt["line"] and belt["low"] <= c <= belt["high"]:
                hits.append(int(color))
        else:
            if c == belt["line"] and belt["low"] <= r <= belt["high"]:
                hits.append(int(color))
    return hits


def _next_state(state: tuple[int, tuple[int, int]], network: dict) -> tuple[tuple[int, tuple[int, int]] | None, str | None]:
    belt_color, pos = int(state[0]), tuple(map(int, state[1]))
    belt = network["belts"][belt_color]
    event = None
    next_belt = belt
    crossing = network["crossings"].get(pos)
    if crossing is not None:
        if crossing["priority"] == belt_color:
            event = "pass"
        else:
            event = "transfer"
            if crossing["horizontal"] == belt_color:
                next_belt = network["belts"][crossing["vertical"]]
            else:
                next_belt = network["belts"][crossing["horizontal"]]

    if next_belt["orientation"] == "H":
        step = (0, int(next_belt["direction"]))
    else:
        step = (int(next_belt["direction"]), 0)

    nxt = (int(pos[0] + step[0]), int(pos[1] + step[1]))
    if nxt == next_belt["head_terminal"]:
        return None, event
    return (int(next_belt["color"]), nxt), event


def _trace_path(start_color: int, start_pos: tuple[int, int], network: dict) -> dict:
    state = (int(start_color), (int(start_pos[0]), int(start_pos[1])))
    seen = set()
    positions: list[tuple[int, int]] = []
    events: list[str | None] = []

    while True:
        assert state not in seen, "belt routing loop detected"
        seen.add(state)
        positions.append(state[1])
        nxt, event = _next_state(state, network)
        events.append(event)
        if nxt is None:
            break
        state = nxt

    return {
        "positions": positions,
        "events": events,
        "distance": len(positions) - 1,
    }


def _simulate(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    network = _build_network(g)

    cargo_positions = [tuple(map(int, rc)) for rc in np.argwhere(g == CARGO)]
    cargo_infos: list[dict] = []
    out = g.copy()

    for pos in cargo_positions:
        belts_here = _covering_belts(network, pos)
        assert len(belts_here) == 1, f"cargo at {pos} must belong to exactly one belt"
        belt_color = int(belts_here[0])
        trace = _trace_path(belt_color, pos, network)
        cargo_infos.append(
            {
                "start": pos,
                "belt_color": belt_color,
                "positions": trace["positions"],
                "events": trace["events"],
                "distance": int(trace["distance"]),
            }
        )
        out[pos] = belt_color

    order = sorted(
        range(len(cargo_infos)),
        key=lambda i: (
            cargo_infos[i]["distance"],
            cargo_infos[i]["start"][0],
            cargo_infos[i]["start"][1],
            cargo_infos[i]["belt_color"],
        ),
    )

    occupied: set[tuple[int, int]] = set()
    meta_by_index: dict[int, dict] = {}
    for idx in order:
        info = cargo_infos[idx]
        path = info["positions"]
        stop_idx = len(path) - 1
        blocked = False
        for i in range(1, len(path)):
            if path[i] in occupied:
                stop_idx = i - 1
                blocked = True
                break

        final = tuple(path[stop_idx])
        occupied.add(final)
        traversed_events = info["events"][: stop_idx + 1]
        meta_by_index[idx] = {
            "start": info["start"],
            "final": final,
            "belt_color": info["belt_color"],
            "distance": info["distance"],
            "blocked": blocked,
            "transfer": any(event == "transfer" for event in traversed_events),
            "priority_pass": any(event == "pass" for event in traversed_events),
            "head_stop": (not blocked) and final == tuple(path[-1]),
            "moved": final != info["start"],
        }

    for idx in range(len(cargo_infos)):
        out[meta_by_index[idx]["final"]] = CARGO

    if return_meta:
        return out, [meta_by_index[i] for i in range(len(cargo_infos))]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the yellow cargo blocks along directed single-pixel conveyor belts.

    Belts are straight horizontal or vertical lines. Each belt has one thickened
    same-color end, so movement goes from the thin end toward the thick end.
    At a crossing, the visible crossing color marks the priority belt: cargo on
    that belt continues straight through, while cargo from the other belt
    switches onto the priority belt's perpendicular partner and keeps moving.
    Cargo blocks one another only after stopping, so the final state is obtained
    by resolving cargos in downstream-first order.
    """
    return _simulate(grid, return_meta=False)


def _sample_spaced(lo: int, hi: int, count: int, min_gap: int, rng: random.Random) -> list[int] | None:
    choices = list(range(lo, hi + 1))
    if len(choices) < count:
        return None
    for _ in range(200):
        picked = sorted(rng.sample(choices, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _choose_side(index: int, forbidden: set[int], lo: int, hi: int, rng: random.Random) -> int | None:
    options = []
    for delta in (-1, 1):
        candidate = index + delta
        if lo <= candidate <= hi and candidate not in forbidden:
            options.append(delta)
    if not options:
        return None
    return int(rng.choice(options))


def _make_base_grid(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 28)
    w = rng.randint(20, 28)
    top = rng.randint(2, 4)
    bottom = h - 1 - rng.randint(2, 4)
    left = rng.randint(2, 4)
    right = w - 1 - rng.randint(2, 4)
    if bottom - top < 12 or right - left < 12:
        return None

    n_h = rng.randint(2, 3)
    n_v = rng.randint(2, 3)
    rows = _sample_spaced(top + 4, bottom - 4, n_h, 3, rng)
    cols = _sample_spaced(left + 4, right - 4, n_v, 3, rng)
    if rows is None or cols is None:
        return None

    h_dir = rng.choice((-1, 1))
    v_dir = rng.choice((-1, 1))
    colors = rng.sample(BELT_PALETTE, n_h + n_v)
    h_colors = colors[:n_h]
    v_colors = colors[n_h:]

    grid = np.full((h, w), BG, dtype=int)
    belts: list[dict] = []

    row_set = set(rows)
    col_set = set(cols)

    for row, color in zip(rows, h_colors):
        grid[row, left : right + 1] = color
        head_col = right if h_dir > 0 else left
        side = _choose_side(row, row_set, 0, h - 1, rng)
        if side is None or grid[row + side, head_col] != BG:
            return None
        grid[row + side, head_col] = color
        belts.append(
            {
                "color": int(color),
                "orientation": "H",
                "line": int(row),
                "low": int(left),
                "high": int(right),
                "direction": int(h_dir),
            }
        )

    for col, color in zip(cols, v_colors):
        grid[top : bottom + 1, col] = color
        head_row = bottom if v_dir > 0 else top
        side = _choose_side(col, col_set, 0, w - 1, rng)
        if side is None or grid[head_row, col + side] != BG:
            return None
        grid[head_row, col + side] = color
        belts.append(
            {
                "color": int(color),
                "orientation": "V",
                "line": int(col),
                "low": int(top),
                "high": int(bottom),
                "direction": int(v_dir),
            }
        )

    crossings: list[tuple[tuple[int, int], int, int]] = []
    for row, h_color in zip(rows, h_colors):
        for col, v_color in zip(cols, v_colors):
            crossings.append(((int(row), int(col)), int(h_color), int(v_color)))

    priorities = ["H", "V"] + [rng.choice(("H", "V")) for _ in range(max(0, len(crossings) - 2))]
    rng.shuffle(priorities)
    for (pos, h_color, v_color), pref in zip(crossings, priorities):
        grid[pos] = h_color if pref == "H" else v_color

    try:
        network = _build_network(grid)
    except AssertionError:
        return None

    # Require both crossing-priority branches to exist in the rendered network.
    priority_orientations = {
        "H" if info["priority"] == info["horizontal"] else "V" for info in network["crossings"].values()
    }
    if priority_orientations != {"H", "V"}:
        return None
    return grid


def _candidate_infos(base_grid: np.ndarray) -> tuple[dict, dict]:
    network = _build_network(base_grid)
    crossing_positions = set(network["crossings"])
    infos: dict[tuple[int, int], dict] = {}

    for belt in network["belts"].values():
        for pos in _traversable_positions(belt):
            if pos in crossing_positions:
                continue
            trace = _trace_path(belt["color"], pos, network)
            infos[pos] = {
                "start": pos,
                "belt_color": int(belt["color"]),
                "distance": int(trace["distance"]),
                "has_transfer": any(event == "transfer" for event in trace["events"]),
                "has_pass": any(event == "pass" for event in trace["events"]),
                "path_positions": tuple(trace["positions"]),
                "path_set": set(trace["positions"]),
            }
    return network, infos


def _tail_queue_segments(network: dict) -> list[list[tuple[int, int]]]:
    crossing_positions = set(network["crossings"])
    segments: list[list[tuple[int, int]]] = []
    for belt in network["belts"].values():
        cells = _traversable_positions(belt)
        last_cross = -1
        for i, pos in enumerate(cells):
            if pos in crossing_positions:
                last_cross = i
        tail = cells[last_cross + 1 :]
        if len(tail) >= 3:
            segments.append(tail)
    return segments


def _has_ambiguous_equal_distance(chosen: list[dict]) -> bool:
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if chosen[i]["distance"] != chosen[j]["distance"]:
                continue
            if chosen[i]["path_set"] & chosen[j]["path_set"]:
                return True
    return False


def _coverage_ok(meta: list[dict]) -> bool:
    return (
        any(item["transfer"] for item in meta)
        and any(item["priority_pass"] for item in meta)
        and any(item["blocked"] for item in meta)
        and any(item["head_stop"] for item in meta)
        and sum(item["moved"] for item in meta) >= 3
    )


def _place_cargoes(base_grid: np.ndarray, rng: random.Random) -> np.ndarray | None:
    network, infos = _candidate_infos(base_grid)
    movable = [info for info in infos.values() if info["distance"] >= 1]
    transfer_pool = [info for info in movable if info["has_transfer"]]
    pass_pool = [info for info in movable if info["has_pass"]]
    queue_segments = _tail_queue_segments(network)

    if not movable or not transfer_pool or not pass_pool or not queue_segments:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        segment = list(rng.choice(queue_segments))
        pair_pool = segment[:-1]
        if len(pair_pool) < 2:
            continue
        pair_positions = rng.sample(pair_pool, 2)

        chosen_positions = list(pair_positions)
        chosen_set = set(chosen_positions)

        transfer_choices = [info for info in transfer_pool if info["start"] not in chosen_set]
        if not transfer_choices:
            continue
        transfer_pick = rng.choice(transfer_choices)
        chosen_positions.append(transfer_pick["start"])
        chosen_set.add(transfer_pick["start"])

        pass_choices = [info for info in pass_pool if info["start"] not in chosen_set]
        if not pass_choices:
            continue
        pass_pick = rng.choice(pass_choices)
        chosen_positions.append(pass_pick["start"])
        chosen_set.add(pass_pick["start"])

        extra_count = rng.randint(0, 2)
        extra_pool = [info for info in movable if info["start"] not in chosen_set]
        rng.shuffle(extra_pool)
        for info in extra_pool[:extra_count]:
            chosen_positions.append(info["start"])
            chosen_set.add(info["start"])

        chosen_infos = [infos[pos] for pos in chosen_positions]
        if _has_ambiguous_equal_distance(chosen_infos):
            continue

        grid = base_grid.copy()
        for pos in chosen_positions:
            grid[pos] = CARGO

        try:
            out, meta = _simulate(grid, return_meta=True)
        except AssertionError:
            continue

        if len({tuple(item["final"]) for item in meta}) != len(meta):
            continue
        if np.array_equal(out, grid):
            continue
        if _coverage_ok(meta):
            return grid

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    base = _make_base_grid(rng)
    if base is None:
        return None
    grid = _place_cargoes(base, rng)
    if grid is None:
        return None
    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid conveyor puzzle input")
