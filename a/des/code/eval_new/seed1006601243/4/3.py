from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

PRIMARY_COLORS = (RED, BLUE, YELLOW)
VISIBLE_WIRE_COLORS = {RED, BLUE, YELLOW, GREEN, MAGENTA, ORANGE}
MIX_MAP = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}

GENERATION_ATTEMPTS = 400
MODES = ("solo", "pair", "fan", "mesh")


@dataclass(frozen=True)
class Source:
    color: int
    block: tuple[int, int]
    start: tuple[int, int]
    direction: tuple[int, int]


@dataclass(frozen=True)
class Terminal:
    block: tuple[int, int]


@dataclass(frozen=True)
class WireSpec:
    orientation: str
    color: int
    start: tuple[int, int]
    end: tuple[int, int]
    source_block: tuple[int, int]
    terminal_block: tuple[int, int]

    @property
    def direction(self) -> tuple[int, int]:
        if self.orientation == "h":
            return (0, 1) if self.end[1] > self.start[1] else (0, -1)
        return (1, 0) if self.end[0] > self.start[0] else (-1, 0)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _block_cells(r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]


def _side_cells(r0: int, c0: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "up": [(r0 - 1, c0), (r0 - 1, c0 + 1)],
        "down": [(r0 + 2, c0), (r0 + 2, c0 + 1)],
        "left": [(r0, c0 - 1), (r0 + 1, c0 - 1)],
        "right": [(r0, c0 + 2), (r0 + 1, c0 + 2)],
    }


def _border_cells(shape: tuple[int, int], r0: int, c0: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(r0 - 1, r0 + 3):
        for c in range(c0 - 1, c0 + 3):
            if r0 <= r <= r0 + 1 and c0 <= c <= c0 + 1:
                continue
            if _in_bounds(shape, r, c):
                cells.append((r, c))
    return cells


def _path_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    assert c0 == c1
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _mix_signal(signal: int, other_base: int) -> int:
    if signal not in PRIMARY_COLORS:
        return signal
    if signal == other_base:
        return signal
    key = frozenset((int(signal), int(other_base)))
    assert key in MIX_MAP, "crossing wires must use different primary colors"
    return MIX_MAP[key]


def _find_sources(grid: np.ndarray) -> list[Source]:
    h, w = grid.shape
    sources: list[Source] = []
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            color = int(grid[r0, c0])
            if color not in PRIMARY_COLORS:
                continue
            if not np.all(grid[r0 : r0 + 2, c0 : c0 + 2] == color):
                continue

            border = _border_cells(grid.shape, r0, c0)
            same_outside = [(r, c) for r, c in border if int(grid[r, c]) == color]
            if len(same_outside) != 1:
                continue

            attachment = same_outside[0]
            direction: tuple[int, int] | None = None
            for side, cells in _side_cells(r0, c0).items():
                if attachment not in cells:
                    continue
                if side == "up":
                    direction = (-1, 0)
                elif side == "down":
                    direction = (1, 0)
                elif side == "left":
                    direction = (0, -1)
                else:
                    direction = (0, 1)
                break
            if direction is None:
                continue

            sources.append(Source(color=color, block=(r0, c0), start=attachment, direction=direction))

    assert sources, "expected at least one source node"
    return sources


def _find_terminals(grid: np.ndarray) -> list[Terminal]:
    h, w = grid.shape
    terminals: list[Terminal] = []
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            if not np.all(grid[r0 : r0 + 2, c0 : c0 + 2] == GREY):
                continue
            border = _border_cells(grid.shape, r0, c0)
            if any(int(grid[r, c]) == GREY for r, c in border):
                continue
            side_hits = []
            for cells in _side_cells(r0, c0).values():
                for r, c in cells:
                    if _in_bounds(grid.shape, r, c) and int(grid[r, c]) in VISIBLE_WIRE_COLORS:
                        side_hits.append((r, c))
            if len(side_hits) != 1:
                continue
            terminals.append(Terminal(block=(r0, c0)))

    assert terminals, "expected at least one terminal node"
    return terminals


def _trace_wire(
    grid: np.ndarray,
    source: Source,
    terminal_cells: dict[tuple[int, int], int],
) -> tuple[list[tuple[int, int]], int]:
    h, w = grid.shape
    r, c = source.start
    dr, dc = source.direction
    path: list[tuple[int, int]] = []

    for _ in range(h * w):
        value = int(grid[r, c])
        assert value != BLACK and value != GREY, "wire path cannot enter background or terminal"
        path.append((r, c))

        nr, nc = r + dr, c + dc
        assert _in_bounds(grid.shape, nr, nc), "wire must terminate at a terminal before the edge"
        if (nr, nc) in terminal_cells:
            return path, terminal_cells[(nr, nc)]

        assert int(grid[nr, nc]) != BLACK, "wire cannot break before reaching a terminal"
        r, c = nr, nc

    raise AssertionError("wire trace exceeded grid area")


def _crossings_by_wire(wires: list[dict[str, object]]) -> list[dict[tuple[int, int], int]]:
    crossings = [dict() for _ in wires]
    for i, wire_a in enumerate(wires):
        for j in range(i + 1, len(wires)):
            wire_b = wires[j]
            if wire_a["orientation"] == wire_b["orientation"]:
                continue
            cells_a = wire_a["path_set"]
            cells_b = wire_b["path_set"]
            common = cells_a & cells_b
            if not common:
                continue
            assert len(common) == 1, "two straight orthogonal wires may share at most one cell"
            cell = next(iter(common))
            crossings[i][cell] = int(wire_b["color"])
            crossings[j][cell] = int(wire_a["color"])
    return crossings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each source color along its straight wire and recolor the adjacent terminal after ordered crossings.

    Input structure:
    - The background is black.
    - Colored wires are one-cell-wide straight horizontal or vertical segments whose base colors are red, blue, or yellow.
    - Each wire touches a same-colored 2x2 source block at one end and ends next to a grey 2x2 terminal block at the other end.
    - Some horizontal and vertical wires cross at a shared pixel, while the rest of the grid stays black.

    Rule:
    - Follow every wire from its source block to its terminal-adjacent free end.
    - The wire starts with its own base color as the carried signal.
    - Whenever the path reaches an orthogonal crossing, combine the current signal with the other wire's original color:
      red+blue -> magenta, red+yellow -> orange, blue+yellow -> green.
    - Once a wire has already become a mixed color, later crossings do not change it again.
    - Keep all source blocks and wire pixels unchanged, and recolor only the terminal block next to each wire's free end with that wire's final signal.

    Key insight:
    - The task is solved by reconstructing each straight wire from source to terminal, ordering the crossings along that path, and mixing only with the crossed wire's base color while treating mixed signals as saturated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sources = _find_sources(g)
    terminals = _find_terminals(g)
    terminal_cells: dict[tuple[int, int], int] = {}
    for idx, terminal in enumerate(terminals):
        for cell in _block_cells(*terminal.block):
            terminal_cells[cell] = idx

    wires: list[dict[str, object]] = []
    used_terminals: set[int] = set()
    for source in sources:
        path, terminal_id = _trace_wire(g, source, terminal_cells)
        assert terminal_id not in used_terminals, "each terminal must belong to exactly one wire"
        used_terminals.add(terminal_id)
        wires.append(
            {
                "color": source.color,
                "orientation": "h" if source.direction[0] == 0 else "v",
                "path": path,
                "path_set": set(path),
                "terminal_id": terminal_id,
            }
        )

    assert len(wires) == len(terminals), "every terminal should be matched to one wire"
    crossings = _crossings_by_wire(wires)

    out = g.copy()
    for wire, crossing_map in zip(wires, crossings):
        signal = int(wire["color"])
        for cell in wire["path"]:
            other_color = crossing_map.get(cell)
            if other_color is not None:
                signal = _mix_signal(signal, other_color)

        terminal = terminals[int(wire["terminal_id"])]
        r0, c0 = terminal.block
        out[r0 : r0 + 2, c0 : c0 + 2] = signal

    return out


def _sample_positions(low: int, high: int, count: int, min_gap: int) -> list[int] | None:
    if count == 0:
        return []
    candidates = list(range(low, high + 1))
    random.shuffle(candidates)
    chosen: list[int] = []
    for pos in candidates:
        if all(abs(pos - other) >= min_gap for other in chosen):
            chosen.append(pos)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _build_horizontal_wire(row: int, width: int, color: int) -> WireSpec:
    if random.random() < 0.5:
        src_off = random.randint(0, 1)
        term_off = random.randint(0, 1)
        return WireSpec(
            orientation="h",
            color=color,
            start=(row, 2),
            end=(row, width - 3),
            source_block=(row - src_off, 0),
            terminal_block=(row - term_off, width - 2),
        )

    src_off = random.randint(0, 1)
    term_off = random.randint(0, 1)
    return WireSpec(
        orientation="h",
        color=color,
        start=(row, width - 3),
        end=(row, 2),
        source_block=(row - src_off, width - 2),
        terminal_block=(row - term_off, 0),
    )


def _build_vertical_wire(col: int, height: int, color: int) -> WireSpec:
    if random.random() < 0.5:
        src_off = random.randint(0, 1)
        term_off = random.randint(0, 1)
        return WireSpec(
            orientation="v",
            color=color,
            start=(2, col),
            end=(height - 3, col),
            source_block=(0, col - src_off),
            terminal_block=(height - 2, col - term_off),
        )

    src_off = random.randint(0, 1)
    term_off = random.randint(0, 1)
    return WireSpec(
        orientation="v",
        color=color,
        start=(height - 3, col),
        end=(2, col),
        source_block=(height - 2, col - src_off),
        terminal_block=(0, col - term_off),
    )


def _mode_counts(mode: str) -> tuple[int, int]:
    if mode == "solo":
        if random.random() < 0.5:
            return random.randint(1, 3), 0
        return 0, random.randint(1, 3)
    if mode == "pair":
        return 1, 1
    if mode == "fan":
        return (1, 2) if random.random() < 0.5 else (2, 1)
    assert mode == "mesh"
    return 2, 2


def _assign_colors(mode: str, h_count: int, v_count: int) -> tuple[list[int], list[int]]:
    primaries = list(PRIMARY_COLORS)
    if h_count == 0:
        return [], [random.choice(primaries) for _ in range(v_count)]
    if v_count == 0:
        return [random.choice(primaries) for _ in range(h_count)], []

    if mode == "pair":
        a, b = random.sample(primaries, 2)
        return [a], [b]

    if mode == "fan":
        main_color = random.choice(primaries)
        others = [color for color in primaries if color != main_color]
        random.shuffle(others)
        if h_count == 1:
            return [main_color], others[:v_count]
        return others[:h_count], [main_color]

    shared_orientation = random.choice(("h", "v"))
    shared_color = random.choice(primaries)
    others = [color for color in primaries if color != shared_color]
    random.shuffle(others)
    if shared_orientation == "h":
        return [shared_color for _ in range(h_count)], others[:v_count]
    return others[:h_count], [shared_color for _ in range(v_count)]


def _wire_crossings(specs: list[WireSpec]) -> list[dict[tuple[int, int], int]]:
    data = [
        {
            "orientation": spec.orientation,
            "color": spec.color,
            "path_set": set(_path_cells(spec.start, spec.end)),
        }
        for spec in specs
    ]
    return _crossings_by_wire(data)


def _render_input(height: int, width: int, specs: list[WireSpec]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for spec in specs:
        sr, sc = spec.source_block
        tr, tc = spec.terminal_block
        grid[sr : sr + 2, sc : sc + 2] = spec.color
        grid[tr : tr + 2, tc : tc + 2] = GREY
        for r, c in _path_cells(spec.start, spec.end):
            grid[r, c] = spec.color

    crossings = _wire_crossings(specs)
    for spec, crossing_map in zip(specs, crossings):
        for cell, other_color in crossing_map.items():
            grid[cell] = MIX_MAP[frozenset((spec.color, other_color))]
    return grid


def _expected_output(grid: np.ndarray, specs: list[WireSpec]) -> np.ndarray:
    out = grid.copy()
    crossings = _wire_crossings(specs)
    for spec, crossing_map in zip(specs, crossings):
        signal = spec.color
        for cell in _path_cells(spec.start, spec.end):
            other_color = crossing_map.get(cell)
            if other_color is not None:
                signal = _mix_signal(signal, other_color)
        tr, tc = spec.terminal_block
        out[tr : tr + 2, tc : tc + 2] = signal
    return out


def _build_specs(height: int, width: int, mode: str) -> list[WireSpec] | None:
    h_count, v_count = _mode_counts(mode)
    rows = _sample_positions(3, height - 4, h_count, min_gap=3)
    cols = _sample_positions(3, width - 4, v_count, min_gap=3)
    if rows is None or cols is None:
        return None

    h_colors, v_colors = _assign_colors(mode, h_count, v_count)
    specs = [_build_horizontal_wire(row, width, color) for row, color in zip(rows, h_colors)]
    specs.extend(_build_vertical_wire(col, height, color) for col, color in zip(cols, v_colors))
    return specs


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(MODES)
    height = random.randint(13, 24)
    width = random.randint(13, 24)
    specs = _build_specs(height, width, mode)
    if specs is None:
        return None

    grid = _render_input(height, width, specs)
    expected = _expected_output(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
