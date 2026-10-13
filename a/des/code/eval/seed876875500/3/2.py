from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PIPE_COLORS = tuple(c for c in range(10) if c not in (BG, GREY))
MAX_GENERATION_ATTEMPTS = 300

SPACING_CHOICES = (2, 3)
TAIL_CHOICES = (1, 2, 3)
GAP_CHOICES = (1, 2, 3)
OUTER_MARGIN_CHOICES = (0, 1, 2, 3)
ORIENTATIONS = ("H", "V")


def _find_hint_box(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.size > 0, "expected a grey hint box"

    r0, c0 = grey_coords.min(axis=0)
    r1, c1 = grey_coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert (r0, c0) == (0, 0), "hint box must sit in the top-left corner"
    assert c1 - c0 == 2 and r1 - r0 >= 2, "hint box must be a 3-wide vertical frame"

    expected = np.zeros_like(grid, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    assert np.array_equal(expected, grid == GREY), "grey pixels must form exactly one frame"

    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    hint_colors = [int(grid[r, c0 + 1]) for r in range(r0 + 1, r1)]
    assert hint_colors and all(color not in (BG, GREY) for color in hint_colors)
    return hint_colors, mask


def _extract_pipes(grid: np.ndarray, hint_mask: np.ndarray) -> dict[int, dict[str, int | str]]:
    colors = [
        int(color)
        for color in np.unique(grid[~hint_mask])
        if int(color) not in (BG, GREY)
    ]
    pipes: dict[int, dict[str, int | str]] = {}
    for color in colors:
        coords = np.argwhere((grid == color) & (~hint_mask))
        assert coords.size > 0, "every pipe color must appear outside the hint box"

        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) == 1 and len(cols) >= 2:
            row = int(rows[0])
            c0 = int(coords[:, 1].min())
            c1 = int(coords[:, 1].max())
            pipes[color] = {
                "color": color,
                "orientation": "H",
                "row": row,
                "c0": c0,
                "c1": c1,
            }
        elif len(cols) == 1 and len(rows) >= 2:
            col = int(cols[0])
            r0 = int(coords[:, 0].min())
            r1 = int(coords[:, 0].max())
            pipes[color] = {
                "color": color,
                "orientation": "V",
                "col": col,
                "r0": r0,
                "r1": r1,
            }
        else:
            raise AssertionError("each color must correspond to one straight 1-pixel pipe")
    assert pipes, "expected at least one pipe"
    return pipes


def _hint_edges(hint_colors: list[int]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i, higher in enumerate(hint_colors):
        for lower in hint_colors[i + 1 :]:
            edges.add((int(higher), int(lower)))
    return edges


def _crossing_edges(grid: np.ndarray, pipes: dict[int, dict[str, int | str]]) -> set[tuple[int, int]]:
    horizontals = [pipe for pipe in pipes.values() if pipe["orientation"] == "H"]
    verticals = [pipe for pipe in pipes.values() if pipe["orientation"] == "V"]
    edges: set[tuple[int, int]] = set()

    for h in horizontals:
        for v in verticals:
            rr = int(h["row"])
            cc = int(v["col"])
            if not (int(h["c0"]) <= cc <= int(h["c1"]) and int(v["r0"]) <= rr <= int(v["r1"])):
                continue

            visible = int(grid[rr, cc])
            h_color = int(h["color"])
            v_color = int(v["color"])
            if visible == h_color:
                edges.add((h_color, v_color))
            elif visible == v_color:
                edges.add((v_color, h_color))
            else:
                raise AssertionError("crossings must show one of the two pipe colors")
    return edges


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    adjacency = {int(color): set() for color in colors}
    indegree = {int(color): 0 for color in colors}
    for higher, lower in edges:
        if higher == lower or lower in adjacency[higher]:
            continue
        adjacency[higher].add(lower)
        indegree[lower] += 1

    ready = deque([color for color in colors if indegree[color] == 0])
    order: list[int] = []
    while ready:
        assert len(ready) == 1, "priority order must be uniquely determined"
        current = ready.popleft()
        order.append(current)
        for nxt in colors:
            if nxt not in adjacency[current]:
                continue
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)

    assert len(order) == len(colors), "priority graph must be acyclic and complete enough"
    return order


def _has_unique_order(colors: list[int], edges: set[tuple[int, int]]) -> bool:
    adjacency = {int(color): set() for color in colors}
    indegree = {int(color): 0 for color in colors}
    for higher, lower in edges:
        if higher == lower or lower in adjacency[higher]:
            continue
        adjacency[higher].add(lower)
        indegree[lower] += 1

    ready = deque([color for color in colors if indegree[color] == 0])
    seen = 0
    while ready:
        if len(ready) != 1:
            return False
        current = ready.popleft()
        seen += 1
        for nxt in colors:
            if nxt not in adjacency[current]:
                continue
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    return seen == len(colors)


def _draw_pipe(grid: np.ndarray, pipe: dict[str, int | str]) -> None:
    color = int(pipe["color"])
    if pipe["orientation"] == "H":
        grid[int(pipe["row"]), int(pipe["c0"]) : int(pipe["c1"]) + 1] = color
    else:
        grid[int(pipe["r0"]) : int(pipe["r1"]) + 1, int(pipe["col"])] = color


def _render_scene(
    shape: tuple[int, int],
    hint_colors: list[int],
    pipes: dict[int, dict[str, int | str]],
    draw_order: list[int],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    hint_h = len(hint_colors) + 2
    grid[:, :] = BG
    grid[0, :3] = GREY
    grid[hint_h - 1, :3] = GREY
    grid[:hint_h, 0] = GREY
    grid[:hint_h, 2] = GREY
    for i, color in enumerate(hint_colors, start=1):
        grid[i, 1] = int(color)
    for color in draw_order:
        _draw_pipe(grid, pipes[int(color)])
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the visible pipe priority at every crossing.

    The grey top-left frame lists a partial high-to-low priority subsequence.
    Each remaining relation comes from crossings: the visible pipe at a crossing
    had higher original priority than the interrupted one. Reconstruct every
    pipe as a full straight segment, infer the unique high-to-low order, then
    redraw from high to low so that lower-priority pipes overwrite higher ones.
    """
    g = np.asarray(grid, dtype=int)
    hint_colors, hint_mask = _find_hint_box(g)
    pipes = _extract_pipes(g, hint_mask)
    colors = list(pipes)
    assert set(hint_colors).issubset(set(colors)), "hint colors must all correspond to pipes"

    edges = _hint_edges(hint_colors) | _crossing_edges(g, pipes)
    high_to_low = _unique_topological_order(colors, edges)

    out = np.full_like(g, BG)
    out[hint_mask] = g[hint_mask]
    for color in high_to_low:
        _draw_pipe(out, pipes[color])
    return out


def _hint_block_partitions(total: int) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []

    def rec(remaining: int, so_far: list[int]) -> None:
        if remaining == 0:
            if any(part == 2 for part in so_far):
                result.append(tuple(so_far))
            return
        for part in (1, 2):
            if part <= remaining:
                so_far.append(part)
                rec(remaining - part, so_far)
                so_far.pop()

    rec(total, [])
    assert result
    return result


def _spaced_positions(count: int, start: int) -> list[int]:
    positions = [start]
    for _ in range(count - 1):
        positions.append(positions[-1] + random.choice(SPACING_CHOICES))
    return positions


def _build_orientation_sequence(hint_orientation: str, block_sizes: tuple[int, ...]) -> list[str]:
    other = "V" if hint_orientation == "H" else "H"
    start_options = [(0, 1), (1, 0), (1, 1)] if len(block_sizes) == 1 else [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    start_other, end_other = random.choice(start_options)

    sequence: list[str] = []
    if start_other:
        sequence.append(other)
    for i, block in enumerate(block_sizes):
        sequence.extend([hint_orientation] * block)
        if i + 1 < len(block_sizes):
            sequence.append(other)
    if end_other:
        sequence.append(other)
    return sequence


def _gen_attempt() -> np.ndarray | None:
    hint_orientation = random.choice(ORIENTATIONS)
    hint_count = random.randint(2, 4)
    block_sizes = random.choice(_hint_block_partitions(hint_count))
    orientation_sequence = _build_orientation_sequence(hint_orientation, block_sizes)

    total_pipes = len(orientation_sequence)
    if total_pipes > len(PIPE_COLORS):
        return None

    priority_colors = random.sample(PIPE_COLORS, total_pipes)
    h_colors = [color for color, orient in zip(priority_colors, orientation_sequence) if orient == "H"]
    v_colors = [color for color, orient in zip(priority_colors, orientation_sequence) if orient == "V"]
    if not h_colors or not v_colors:
        return None

    hint_colors = [color for color, orient in zip(priority_colors, orientation_sequence) if orient == hint_orientation]
    other_count = total_pipes - len(hint_colors)
    if len(hint_colors) < 2 or other_count < 1:
        return None

    h_left = {color: random.choice(TAIL_CHOICES) for color in h_colors}
    h_right = {color: random.choice(TAIL_CHOICES) for color in h_colors}
    v_top = {color: random.choice(TAIL_CHOICES) for color in v_colors}
    v_bottom = {color: random.choice(TAIL_CHOICES) for color in v_colors}

    hint_h = len(hint_colors) + 2
    first_row = hint_h + random.choice(GAP_CHOICES) + max(v_top.values())
    first_col = 3 + random.choice(GAP_CHOICES) + max(h_left.values())
    row_positions = _spaced_positions(len(h_colors), first_row)
    col_positions = _spaced_positions(len(v_colors), first_col)

    random.shuffle(h_colors)
    random.shuffle(v_colors)
    row_by_color = {color: row for color, row in zip(h_colors, row_positions)}
    col_by_color = {color: col for color, col in zip(v_colors, col_positions)}

    pipes: dict[int, dict[str, int | str]] = {}
    for color in row_by_color:
        row = row_by_color[color]
        pipes[int(color)] = {
            "color": int(color),
            "orientation": "H",
            "row": int(row),
            "c0": int(col_positions[0] - h_left[color]),
            "c1": int(col_positions[-1] + h_right[color]),
        }
    for color in col_by_color:
        col = col_by_color[color]
        pipes[int(color)] = {
            "color": int(color),
            "orientation": "V",
            "col": int(col),
            "r0": int(row_positions[0] - v_top[color]),
            "r1": int(row_positions[-1] + v_bottom[color]),
        }

    max_r = hint_h - 1
    max_c = 2
    for pipe in pipes.values():
        if pipe["orientation"] == "H":
            max_r = max(max_r, int(pipe["row"]))
            max_c = max(max_c, int(pipe["c1"]))
        else:
            max_r = max(max_r, int(pipe["r1"]))
            max_c = max(max_c, int(pipe["col"]))

    shape = (
        max_r + 1 + random.choice(OUTER_MARGIN_CHOICES),
        max_c + 1 + random.choice(OUTER_MARGIN_CHOICES),
    )
    inp = _render_scene(shape, hint_colors, pipes, list(reversed(priority_colors)))

    inferred_hint, hint_mask = _find_hint_box(inp)
    extracted_pipes = _extract_pipes(inp, hint_mask)
    colors = list(extracted_pipes)

    combined_edges = _hint_edges(inferred_hint) | _crossing_edges(inp, extracted_pipes)
    crossing_only_edges = _crossing_edges(inp, extracted_pipes)
    hint_only_edges = _hint_edges(inferred_hint)

    if not _has_unique_order(colors, combined_edges):
        return None
    if _has_unique_order(colors, crossing_only_edges):
        return None
    if _has_unique_order(colors, hint_only_edges):
        return None

    expected = _render_scene(shape, hint_colors, pipes, priority_colors)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
