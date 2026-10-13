import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 0
WALL = 5
VISIBLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> Tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _playfield_bbox(grid: np.ndarray) -> Tuple[int, int, int, int]:
    wall_pts = np.argwhere(grid == WALL)
    assert wall_pts.size > 0, "expected grey wall columns"
    return _bbox(wall_pts)


def _legend_mapping(grid: np.ndarray, play_bbox: Tuple[int, int, int, int]) -> Dict[int, int]:
    r0, c0, r1, c1 = play_bbox
    rr, cc = np.indices(grid.shape)
    inside_play = (r0 <= rr) & (rr <= r1) & (c0 <= cc) & (cc <= c1)
    legend_pts = np.argwhere((grid != BG) & (grid != WALL) & (~inside_play))
    assert legend_pts.size > 0, "expected a 2xN legend block outside the pipe area"
    lr0, lc0, lr1, lc1 = _bbox(legend_pts)
    assert lr1 - lr0 + 1 == 2, "legend must be a 2xN block"
    mapping: Dict[int, int] = {}
    for c in range(lc0, lc1 + 1):
        gate = int(grid[lr0, c])
        passed = int(grid[lr0 + 1, c])
        assert gate not in (BG, WALL) and passed not in (BG, WALL), "legend cells must be colored"
        assert gate not in mapping, "legend gate colors must be unique"
        mapping[gate] = passed
    return mapping


def _pipe_spans(grid: np.ndarray, play_bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    r0, c0, r1, c1 = play_bbox
    wall_cols = [c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == WALL)]
    spans: List[Tuple[int, int]] = []
    for left, right in zip(wall_cols, wall_cols[1:]):
        if right - left > 1:
            spans.append((left, right))
    assert spans, "expected at least one pipe between wall columns"
    return spans


def _pipe_contents(
    grid: np.ndarray, play_bbox: Tuple[int, int, int, int], span: Tuple[int, int]
) -> Tuple[Tuple[int, int, int], List[Tuple[int, int]]]:
    r0, _c0, r1, _c1 = play_bbox
    left, right = span
    token = None
    gates: List[Tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        segment = grid[r, left + 1 : right]
        non_bg = np.argwhere(segment != BG).flatten()
        if non_bg.size == 0:
            continue
        colors = {int(x) for x in segment if int(x) != BG}
        if non_bg.size == 1:
            c = int(left + 1 + non_bg[0])
            color = int(segment[non_bg[0]])
            assert token is None, "each pipe must contain exactly one token"
            token = (r, c, color)
        else:
            assert non_bg.size == right - left - 1, "gates must span the full pipe width"
            assert len(colors) == 1, "each gate row must be monochrome"
            gates.append((r, int(segment[0])))
    assert token is not None, "missing token in pipe"
    gates.sort()
    return token, gates


def _simulate(grid: np.ndarray, recolor: bool = True) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    play_bbox = _playfield_bbox(out)
    mapping = _legend_mapping(out, play_bbox)
    spans = _pipe_spans(out, play_bbox)
    _r0, _c0, play_bottom, _c1 = play_bbox

    finals: List[Tuple[int, int, int]] = []
    for span in spans:
        token, gates = _pipe_contents(out, play_bbox, span)
        tr, tc, current = token
        out[tr, tc] = BG
        final_row = play_bottom
        current_color = int(current)
        for gate_row, gate_color in gates:
            if mapping[int(gate_color)] != current_color:
                final_row = gate_row - 1
                break
            if recolor:
                current_color = int(gate_color)
        finals.append((final_row, tc, current_color))

    for r, c, color in finals:
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each token through vertical pipes, checking gate passports from the corner legend.

    Input structure:
    - Grey wall columns carve the play area into several disconnected vertical black pipes.
    - Each pipe contains one colored single-pixel token near the top.
    - Colored one-row gates span the full width of a pipe at different depths.
    - A 2xN legend block sits in a corner outside the pipe area; its top row lists gate colors and its bottom row lists the matching pass colors.

    Rule:
    - Remove the tokens from their starting cells and let each one fall straight down inside its own pipe.
    - When a token reaches a gate, compare the token's current color with the legend's pass color for that gate.
    - If the colors match, the token passes through and is recolored to the gate color before continuing downward.
    - If the colors do not match, the token stops one cell above that gate; if every checked gate passes, the token stops at the pipe bottom.
    - Walls, gates, and the legend stay fixed; only the tokens are redrawn at their final positions in their final colors.

    Key insight:
    - The legend is a chained passport rule, not a static recolor table: every successful gate rewrites the token's color, and that new color is what the next gate checks.
    """
    return _simulate(grid, recolor=True)


def _legend_position(corner: str, height: int, width: int, legend_w: int) -> Tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - legend_w
    if corner == "bl":
        return height - 2, 0
    if corner == "br":
        return height - 2, width - legend_w
    raise ValueError("invalid corner")


def _place_legend(grid: np.ndarray, mapping: Dict[int, int], corner: str) -> None:
    cols = list(mapping)
    r0, c0 = _legend_position(corner, grid.shape[0], grid.shape[1], len(cols))
    for i, gate in enumerate(cols):
        grid[r0, c0 + i] = int(gate)
        grid[r0 + 1, c0 + i] = int(mapping[gate])


def _token_center(left: int, right: int) -> int:
    cols = list(range(left + 1, right))
    return int(cols[len(cols) // 2])


def _simulate_specs(
    shape: Tuple[int, int],
    play_bbox: Tuple[int, int, int, int],
    widths: Sequence[int],
    pipe_specs: Sequence[dict],
    mapping: Dict[int, int],
    legend_corner: str,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = shape
    pr0, pc0, pr1, _pc1 = play_bbox
    grid = np.full((height, width), BG, dtype=int)

    wall_c = pc0
    spans: List[Tuple[int, int]] = []
    for pipe_w in widths:
        grid[pr0 : pr1 + 1, wall_c] = WALL
        next_wall = wall_c + pipe_w + 1
        grid[pr0 : pr1 + 1, next_wall] = WALL
        spans.append((wall_c, next_wall))
        wall_c = next_wall

    expected = grid.copy()
    final_tokens: List[Tuple[int, int, int]] = []
    for span, spec in zip(spans, pipe_specs):
        left, right = span
        token_c = _token_center(left, right)
        token_r = pr0 + int(spec["token_offset"])
        init_color = int(spec["init"])
        grid[token_r, token_c] = init_color
        current = init_color
        final_r = pr1
        for gate_r, gate_color in spec["gates"]:
            gate_rr = pr0 + int(gate_r)
            grid[gate_rr, left + 1 : right] = int(gate_color)
            if final_r != pr1:
                continue
            if mapping[int(gate_color)] != current:
                final_r = gate_rr - 1
            else:
                current = int(gate_color)
        final_tokens.append((final_r, token_c, current))

    _place_legend(grid, mapping, legend_corner)
    expected[:, :] = grid
    for span, spec in zip(spans, pipe_specs):
        token_r = pr0 + int(spec["token_offset"])
        token_c = _token_center(*span)
        expected[token_r, token_c] = BG
    for r, c, color in final_tokens:
        expected[r, c] = int(color)
    return grid, expected


def _naive_no_recolor(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, recolor=False)


def _pipe_outcome(spec: dict, mapping: Dict[int, int]) -> Tuple[int, int]:
    current = int(spec["init"])
    passed = 0
    for gate_r, gate_color in spec["gates"]:
        if mapping[int(gate_color)] != current:
            return passed, int(gate_r)
        current = int(gate_color)
        passed += 1
    return passed, -1


def _mandatory_specs(rng: random.Random, gate_colors: Sequence[int], token_colors: Sequence[int]) -> Tuple[Dict[int, int], List[dict]]:
    t0, t1 = map(int, token_colors[:2])
    if len(gate_colors) == 5:
        g_first, g_alt, g_second, g_third, g_block = map(int, gate_colors)
        mapping = {
            g_first: t0,
            g_alt: t0,
            g_second: g_first,
            g_third: g_second,
            g_block: g_alt,
        }
        specs = [
            {"init": t0, "gates": [], "token_offset": 0},
            {"init": t0, "gates": [], "token_offset": 0},
            {"init": t1, "gates": [], "token_offset": 0},
        ]
        specs[0]["gates"] = [(0, g_first), (0, g_second), (0, g_third)]
        specs[1]["gates"] = [(0, g_alt), (0, g_second)]
        specs[2]["gates"] = [(0, g_block)]
        return mapping, specs

    g_first, g_alt, g_second, g_block = map(int, gate_colors)
    mapping = {
        g_first: t0,
        g_alt: t0,
        g_second: g_first,
        g_block: g_alt,
    }
    specs = [
        {"init": t0, "gates": [(0, g_first), (0, g_second)], "token_offset": 0},
        {"init": t0, "gates": [(0, g_alt), (0, g_second)], "token_offset": 0},
        {"init": t1, "gates": [(0, g_block)], "token_offset": 0},
    ]
    return mapping, specs


def _random_pipe_spec(
    rng: random.Random, mapping: Dict[int, int], token_colors: Sequence[int], max_depth: int
) -> dict:
    color_pool = list(token_colors) + list(mapping)
    init = int(rng.choice(token_colors))
    gates: List[Tuple[int, int]] = []
    current = init
    depth = int(rng.randint(0, max_depth))
    available = list(mapping)
    rng.shuffle(available)
    for _ in range(depth):
        choices = [gate for gate in available if mapping[int(gate)] == current]
        if not choices:
            break
        gate = int(rng.choice(choices))
        gates.append((0, gate))
        current = gate
        if rng.random() < 0.35:
            break
    if rng.random() < 0.75:
        blockers = [gate for gate in mapping if mapping[int(gate)] != current]
        if blockers:
            gates.append((0, int(rng.choice(blockers))))
    elif not gates:
        pass_gates = [gate for gate in mapping if mapping[int(gate)] == current]
        if pass_gates:
            gates.append((0, int(rng.choice(pass_gates))))
    return {"init": init, "gates": gates, "token_offset": 0}


def _assign_gate_rows(rng: random.Random, pipe_specs: List[dict], play_h: int) -> None:
    candidate_rows = list(range(2, play_h - 1, 2))
    for spec in pipe_specs:
        need = len(spec["gates"])
        assert need <= len(candidate_rows), "not enough vertical slots for gates"
        chosen = sorted(rng.sample(candidate_rows, need))
        spec["token_offset"] = int(rng.choice([0, 1]))
        spec["gates"] = [(chosen[i], int(spec["gates"][i][1])) for i in range(need)]
        if chosen:
            assert chosen[0] - spec["token_offset"] >= 2, "first gate must not touch the token"


def _counterexample_present(pipe_specs: Sequence[dict], mapping: Dict[int, int]) -> bool:
    for i, a in enumerate(pipe_specs):
        for b in pipe_specs[i + 1 :]:
            if int(a["init"]) != int(b["init"]):
                continue
            if len(a["gates"]) < 2 or len(b["gates"]) < 2:
                continue
            if int(a["gates"][1][1]) != int(b["gates"][1][1]):
                continue
            if int(a["gates"][0][1]) == int(b["gates"][0][1]):
                continue
            a_pass, a_stop = _pipe_outcome(a, mapping)
            b_pass, b_stop = _pipe_outcome(b, mapping)
            if a_pass >= 2 and b_stop == int(b["gates"][1][0]):
                return True
            if b_pass >= 2 and a_stop == int(a["gates"][1][0]):
                return True
    return False


def _valid_generated_pair(inp: np.ndarray, expected: np.ndarray, pipe_specs: Sequence[dict], mapping: Dict[int, int]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(_naive_no_recolor(inp), expected):
        return False
    if not _counterexample_present(pipe_specs, mapping):
        return False
    outcomes = [_pipe_outcome(spec, mapping)[0] for spec in pipe_specs]
    if max(outcomes) < 2:
        return False
    if not any(_pipe_outcome(spec, mapping)[1] != -1 for spec in pipe_specs):
        return False
    if not any(_pipe_outcome(spec, mapping)[1] == -1 for spec in pipe_specs):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    token_count = int(rng.choice([2, 3]))
    gate_count = int(rng.choice([4, 5]))
    shuffled = VISIBLE_COLORS[:]
    rng.shuffle(shuffled)
    token_colors = list(map(int, shuffled[:token_count]))
    gate_colors = list(map(int, shuffled[token_count : token_count + gate_count]))

    if len(token_colors) < 2:
        return None

    mapping, pipe_specs = _mandatory_specs(rng, gate_colors, token_colors)
    extra_pipes = int(rng.choice([0, 1, 2]))
    max_depth = 3 if gate_count == 5 else 2
    for _ in range(extra_pipes):
        pipe_specs.append(_random_pipe_spec(rng, mapping, token_colors, max_depth))

    pipe_count = len(pipe_specs)
    widths = [int(rng.randint(2, 4)) for _ in range(pipe_count)]

    max_gates = max(len(spec["gates"]) for spec in pipe_specs)
    play_h = max(10, 2 * max_gates + 5 + int(rng.choice([0, 1, 2])))
    if play_h > 18:
        play_h = 18
    try:
        _assign_gate_rows(rng, pipe_specs, play_h)
    except AssertionError:
        return None

    play_w = sum(widths) + pipe_count + 1
    legend_w = len(mapping)

    grid_h = max(play_h + 4, 14)
    grid_w = max(play_w + 4, legend_w + 6)
    if grid_h > 30 or grid_w > 30:
        return None

    legend_corner = str(rng.choice(["tl", "tr", "bl", "br"]))
    if legend_corner in ("tl", "tr"):
        pr0 = int(rng.randint(3, grid_h - play_h))
    else:
        pr0 = int(rng.randint(0, grid_h - play_h - 3))
    pc0 = int(rng.randint(1, grid_w - play_w - 1))
    play_bbox = (pr0, pc0, pr0 + play_h - 1, pc0 + play_w - 1)

    inp, expected = _simulate_specs((grid_h, grid_w), play_bbox, widths, pipe_specs, mapping, legend_corner)
    if not _valid_generated_pair(inp, expected, pipe_specs, mapping):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gate-passport puzzle input")
