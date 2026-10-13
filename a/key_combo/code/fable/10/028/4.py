import random
from collections import Counter

import numpy as np


EMPTY = -1
BG_DEFAULT = 0
WALL_DEFAULT = 5
TOKEN_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _most_common(values) -> int:
    return Counter(int(v) for v in values).most_common(1)[0][0]


def _infer_bg_wall(grid: np.ndarray) -> tuple[int, int]:
    bg = _most_common(grid.flatten())
    top = [int(x) for x in grid[0, :]]
    bottom = [int(x) for x in grid[-1, :]]
    left = [int(x) for x in grid[:, 0]]
    right = [int(x) for x in grid[:, -1]]
    edge = top + bottom + left + right
    edge_counts = Counter(v for v in edge if v != bg)
    if not edge_counts:
        raise AssertionError("expected a non-background wall color on the border")
    wall = edge_counts.most_common(1)[0][0]
    return bg, wall


def _gate_id(gate: dict) -> tuple[int, int, int, int]:
    return int(gate["row"]), int(gate["c0"]), int(gate["c1"]), int(gate["color"])


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg, wall = _infer_bg_wall(g)

    wall_mask = g == wall
    gate_mask = np.zeros_like(g, dtype=bool)
    gates: list[dict] = []

    for r in range(1, h - 1):
        c = 0
        while c < w:
            color = int(g[r, c])
            if color in (bg, wall):
                c += 1
                continue
            c0 = c
            while c + 1 < w and int(g[r, c + 1]) == color:
                c += 1
            c1 = c
            if c1 - c0 + 1 >= 2 and np.all(g[r + 1, c0 : c1 + 1] == bg):
                gates.append(
                    {
                        "row": r,
                        "c0": c0,
                        "c1": c1,
                        "color": color,
                        "active": True,
                    }
                )
                gate_mask[r, c0 : c1 + 1] = True
            c += 1

    token_grid = np.full((h, w), EMPTY, dtype=int)
    source_tokens: list[tuple[int, int]] = []
    for r, c in np.argwhere((g != bg) & (g != wall) & (~gate_mask)):
        color = int(g[r, c])
        if int(r) == 0:
            source_tokens.append((int(c), color))
        else:
            token_grid[int(r), int(c)] = color
    source_tokens.sort()

    return {
        "grid": g,
        "bg": bg,
        "wall": wall,
        "walls": wall_mask,
        "gates": gates,
        "tokens": token_grid,
        "sources": source_tokens,
    }


def _build_blocker_mask(walls: np.ndarray, gates: list[dict]) -> np.ndarray:
    blocker = np.array(walls, copy=True)
    for gate in gates:
        if gate["active"]:
            r = int(gate["row"])
            blocker[r, int(gate["c0"]) : int(gate["c1"]) + 1] = True
    return blocker


def _settle_all_tokens(tokens: np.ndarray, walls: np.ndarray, gates: list[dict]) -> None:
    h, w = tokens.shape
    blocker = _build_blocker_mask(walls, gates)
    for c in range(w):
        token_rows = [int(r) for r in range(h) if int(tokens[r, c]) != EMPTY]
        if not token_rows:
            continue
        colors_by_row = {r: int(tokens[r, c]) for r in token_rows}
        for r in token_rows:
            tokens[r, c] = EMPTY

        r = 0
        while r < h:
            while r < h and blocker[r, c]:
                r += 1
            if r >= h:
                break
            start = r
            while r + 1 < h and not blocker[r + 1, c]:
                r += 1
            end = r
            colors = [colors_by_row[row] for row in token_rows if start <= row <= end]
            for dst_r, color in zip(range(end, end - len(colors), -1), reversed(colors)):
                tokens[dst_r, c] = color
            r += 1


def _drop_source_token(
    color: int,
    col: int,
    tokens: np.ndarray,
    walls: np.ndarray,
    gates: list[dict],
) -> int:
    blocker = _build_blocker_mask(walls, gates)
    assert not blocker[0, col], "source token column must be open at the top"
    assert int(tokens[0, col]) == EMPTY, "source token start cell must be empty"
    r = 0
    while r + 1 < tokens.shape[0] and not blocker[r + 1, col] and int(tokens[r + 1, col]) == EMPTY:
        r += 1
    tokens[r, col] = int(color)
    return r


def _resolve_cascades(tokens: np.ndarray, walls: np.ndarray, gates: list[dict]) -> list[list[tuple[int, int, int, int]]]:
    rounds: list[list[tuple[int, int, int, int]]] = []
    while True:
        fuseable: list[dict] = []
        for gate in gates:
            if not gate["active"]:
                continue
            r = int(gate["row"])
            color = int(gate["color"])
            if r == 0:
                continue
            ok = True
            for c in range(int(gate["c0"]), int(gate["c1"]) + 1):
                if int(tokens[r - 1, c]) != color:
                    ok = False
                    break
            if ok:
                fuseable.append(gate)
        if not fuseable:
            break

        rounds.append([_gate_id(gate) for gate in fuseable])
        for gate in fuseable:
            gate["active"] = False
            r = int(gate["row"]) - 1
            for c in range(int(gate["c0"]), int(gate["c1"]) + 1):
                tokens[r, c] = EMPTY
        _settle_all_tokens(tokens, walls, gates)
    return rounds


def _render_state(bg: int, wall: int, walls: np.ndarray, gates: list[dict], tokens: np.ndarray) -> np.ndarray:
    out = np.full(tokens.shape, int(bg), dtype=int)
    out[walls] = int(wall)
    for gate in gates:
        if gate["active"]:
            out[int(gate["row"]), int(gate["c0"]) : int(gate["c1"]) + 1] = int(gate["color"])
    for r, c in np.argwhere(tokens != EMPTY):
        out[int(r), int(c)] = int(tokens[int(r), int(c)])
    return out


def _simulate(grid: np.ndarray, *, trace: bool = False):
    parsed = _parse_input(grid)
    gates = [dict(gate) for gate in parsed["gates"]]
    tokens = np.array(parsed["tokens"], copy=True)
    walls = parsed["walls"]

    _settle_all_tokens(tokens, walls, gates)
    initial_rounds = _resolve_cascades(tokens, walls, gates)
    drops = []
    for col, color in parsed["sources"]:
        landing_row = _drop_source_token(color, col, tokens, walls, gates)
        rounds = _resolve_cascades(tokens, walls, gates)
        drops.append(
            {
                "col": int(col),
                "color": int(color),
                "landing_row": int(landing_row),
                "rounds": rounds,
            }
        )

    out = _render_state(parsed["bg"], parsed["wall"], walls, gates, tokens)
    if trace:
        return out, {
            "bg": int(parsed["bg"]),
            "wall": int(parsed["wall"]),
            "initial_rounds": initial_rounds,
            "drops": drops,
            "remaining_gates": [_gate_id(gate) for gate in gates if gate["active"]],
            "removed_gates": [_gate_id(gate) for gate in gates if not gate["active"]],
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop colored tokens through wall-bounded shafts until same-colored gate covers fuse away.

    Input structure:
    - A wall color forms one or more vertical shafts with open tops and a closed floor.
    - Inside the shafts are horizontal single-color gate bars of varying lengths.
    - The top row contains single-pixel colored tokens, one per occupied column; gates and tokens share the same foreground palette.

    Rule:
    - Process the top-row tokens from left to right, dropping each straight down its own column until it lands on a wall, a gate, or a previously settled token.
    - Whenever every cell of a gate has a directly adjacent token of the same color above it, that gate and those covering tokens vanish together; then all remaining tokens above re-drop under gravity, and further fusions continue until the whole grid is stable.
    - A wrong-colored token resting on a gate cell blocks that gate from ever fusing, so the output keeps walls, any unfused gates, and all tokens still present after all cascading collapses finish.

    Key insight:
    - Gates do not open from a single matching token: an entire same-colored cover must be present at once, and each fusion destroys its own matching cover before gravity can reorganize the rest.
    """
    return _simulate(np.asarray(grid, dtype=int), trace=False)


def _draw_walls(grid: np.ndarray, wall_cols: list[int], wall: int) -> None:
    grid[:, wall_cols] = int(wall)
    grid[-1, :] = int(wall)


def _place_gate(grid: np.ndarray, row: int, cols: list[int], color: int) -> tuple[int, int, int, int]:
    assert len(cols) >= 2
    assert cols == list(range(cols[0], cols[-1] + 1))
    assert np.all(grid[row, cols[0] : cols[-1] + 1] == BG_DEFAULT)
    assert np.all(grid[row + 1, cols[0] : cols[-1] + 1] == BG_DEFAULT)
    grid[row, cols[0] : cols[-1] + 1] = int(color)
    return int(row), int(cols[0]), int(cols[-1]), int(color)


def _place_token(grid: np.ndarray, row: int, col: int, color: int) -> None:
    assert grid[row, col] == BG_DEFAULT
    grid[row, col] = int(color)


def _random_colors(k: int) -> list[int]:
    return random.sample(TOKEN_POOL, k=k)


def _make_layout() -> tuple[int, int, list[list[int]], list[int]]:
    n_shafts = random.choice([2, 3])
    widths: list[int] = [random.randint(2, 4) for _ in range(n_shafts)]
    widths[random.randrange(n_shafts)] = random.randint(3, 4)
    if n_shafts == 3 and sum(widths) > 10:
        widths[random.randrange(n_shafts)] = 2

    shaft_cols: list[list[int]] = []
    wall_cols = [0]
    cur = 1
    for width in widths:
        shaft = list(range(cur, cur + width))
        shaft_cols.append(shaft)
        cur += width
        wall_cols.append(cur)
        cur += 1

    width = wall_cols[-1] + 1
    height = random.randint(12, 18)
    return height, width, shaft_cols, wall_cols


def _pattern_cascade(
    grid: np.ndarray,
    shaft: list[int],
    colors: list[int],
    top_tokens: dict[int, int],
) -> dict:
    assert len(shaft) >= 3
    start = random.randint(0, len(shaft) - 3)
    upper = shaft[start : start + 3]
    lower = upper[:2]
    lower_color, upper_color = random.sample(colors, 2)
    upper_row = random.randint(3, grid.shape[0] - 8)
    lower_row = random.randint(upper_row + 4, grid.shape[0] - 4)

    upper_id = _place_gate(grid, upper_row, upper, upper_color)
    lower_id = _place_gate(grid, lower_row, lower, lower_color)
    for col in lower:
        _place_token(grid, upper_row - 1, col, upper_color)
        top_tokens[col] = lower_color
    top_tokens[upper[2]] = upper_color

    return {
        "kind": "cascade",
        "removed": {upper_id, lower_id},
        "survive": set(),
        "expect_chain": True,
    }


def _pattern_blocked(
    grid: np.ndarray,
    shaft: list[int],
    colors: list[int],
    top_tokens: dict[int, int],
) -> dict:
    span_len = random.randint(2, len(shaft))
    start = random.randint(0, len(shaft) - span_len)
    span = shaft[start : start + span_len]
    gate_color, wrong_color = random.sample(colors, 2)
    gate_row = random.randint(4, grid.shape[0] - 5)
    gate_id = _place_gate(grid, gate_row, span, gate_color)
    wrong_col = random.choice(span)
    for col in span:
        _place_token(grid, gate_row - 1, col, wrong_color if col == wrong_col else gate_color)
    for col in random.sample(span, k=random.randint(0, min(2, len(span)))):
        if col not in top_tokens:
            top_tokens[col] = random.choice(colors)
    return {
        "kind": "blocked",
        "removed": set(),
        "survive": {gate_id},
        "expect_chain": False,
    }


def _pattern_single_fuse(
    grid: np.ndarray,
    shaft: list[int],
    colors: list[int],
    top_tokens: dict[int, int],
) -> dict:
    span_len = random.randint(2, len(shaft))
    start = random.randint(0, len(shaft) - span_len)
    span = shaft[start : start + span_len]
    gate_color = random.choice(colors)
    gate_row = random.randint(4, grid.shape[0] - 5)
    gate_id = _place_gate(grid, gate_row, span, gate_color)

    pre_count = random.randint(1, span_len - 1)
    pre_cols = set(random.sample(span, k=pre_count))
    fill_cols = [col for col in span if col not in pre_cols]
    for col in pre_cols:
        _place_token(grid, gate_row - 1, col, gate_color)
    for col in fill_cols:
        top_tokens[col] = gate_color
    stack_cols = random.sample(list(pre_cols), k=random.randint(0, len(pre_cols)))
    for col in stack_cols:
        if col not in top_tokens:
            top_tokens[col] = random.choice([c for c in colors if c != gate_color])

    return {
        "kind": "single",
        "removed": {gate_id},
        "survive": set(),
        "expect_chain": False,
    }


def _pattern_idle(
    grid: np.ndarray,
    shaft: list[int],
    colors: list[int],
    top_tokens: dict[int, int],
) -> dict:
    span_len = random.randint(2, len(shaft))
    start = random.randint(0, len(shaft) - span_len)
    span = shaft[start : start + span_len]
    gate_color = random.choice(colors)
    gate_row = random.randint(4, grid.shape[0] - 5)
    gate_id = _place_gate(grid, gate_row, span, gate_color)

    covered_cols = random.sample(span, k=random.randint(0, span_len - 1))
    for col in covered_cols:
        _place_token(grid, gate_row - 1, col, gate_color)
        if random.random() < 0.5 and col not in top_tokens:
            top_tokens[col] = random.choice([c for c in colors if c != gate_color])

    return {
        "kind": "idle",
        "removed": set(),
        "survive": {gate_id},
        "expect_chain": False,
    }


def _build_example() -> tuple[np.ndarray, dict]:
    height, width, shafts, wall_cols = _make_layout()
    grid = np.full((height, width), BG_DEFAULT, dtype=int)
    _draw_walls(grid, wall_cols, WALL_DEFAULT)
    colors = _random_colors(k=random.randint(4, 6))
    top_tokens: dict[int, int] = {}

    infos: list[dict] = []
    order = list(range(len(shafts)))
    random.shuffle(order)

    cascade_idx = None
    for idx in order:
        if len(shafts[idx]) >= 3:
            cascade_idx = idx
            break
    if cascade_idx is None:
        cascade_idx = order[0]
    blocked_idx = next(idx for idx in order if idx != cascade_idx)

    use_cascade = len(shafts[cascade_idx]) >= 3 and random.random() < 0.8
    if use_cascade:
        infos.append(_pattern_cascade(grid, shafts[cascade_idx], colors, top_tokens))
    else:
        infos.append(_pattern_single_fuse(grid, shafts[cascade_idx], colors, top_tokens))
    infos.append(_pattern_blocked(grid, shafts[blocked_idx], colors, top_tokens))

    for idx in range(len(shafts)):
        if idx in (cascade_idx, blocked_idx):
            continue
        if random.random() < 0.5:
            infos.append(_pattern_single_fuse(grid, shafts[idx], colors, top_tokens))
        else:
            infos.append(_pattern_idle(grid, shafts[idx], colors, top_tokens))

    for col, color in sorted(top_tokens.items()):
        _place_token(grid, 0, col, color)

    meta = {
        "expect_chain": any(info["expect_chain"] for info in infos),
        "removed": set().union(*(info["removed"] for info in infos)),
        "survive": set().union(*(info["survive"] for info in infos)),
        "top_token_count": len(top_tokens),
        "gate_count": sum(len(info["removed"]) + len(info["survive"]) for info in infos),
    }
    return grid, meta


def _semantic_checks(inp: np.ndarray, meta: dict) -> None:
    out, trace = _simulate(inp, trace=True)
    reparsed = _parse_input(inp)

    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert meta["top_token_count"] >= 2
    assert len(reparsed["gates"]) == meta["gate_count"]
    assert not trace["initial_rounds"], "preloaded tokens must not trigger fusion before any top token drops"
    assert meta["removed"], "at least one gate must fuse in generated inputs"
    assert meta["survive"], "at least one gate must survive in generated inputs"

    removed = set(trace["removed_gates"])
    remaining = set(trace["remaining_gates"])
    assert meta["removed"] <= removed
    assert meta["survive"] <= remaining
    assert not (meta["removed"] & remaining)
    assert not np.array_equal(inp, out)

    if meta["expect_chain"]:
        assert any(len(drop["rounds"]) >= 2 for drop in trace["drops"]), "expected a chain collapse round"

    gate_pixels_in = int(np.sum((inp != BG_DEFAULT) & (inp != WALL_DEFAULT))) - meta["top_token_count"]
    gate_pixels_out = int(np.sum((out != trace["bg"]) & (out != trace["wall"])))
    assert gate_pixels_out < gate_pixels_in + meta["top_token_count"]

    main_out = main(inp)
    assert np.array_equal(main_out, out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp, meta = _build_example()
        try:
            _semantic_checks(inp, meta)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid fuse-gate input")
