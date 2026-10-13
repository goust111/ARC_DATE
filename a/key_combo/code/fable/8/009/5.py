from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
GREY = 5
BODY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _parse_emitters(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bodies = [tuple(map(int, rc)) for rc in np.argwhere((g != BG) & (g != GREY))]
    assert bodies, "expected at least one emitter body"

    emitters: list[dict] = []
    body_to_idx: dict[tuple[int, int], int] = {}
    tail_to_idx: dict[tuple[int, int], int] = {}
    seen_colors: set[int] = set()

    for body in sorted(bodies):
        r, c = body
        color = int(g[r, c])
        assert color in BODY_COLORS, f"invalid body color: {color}"
        assert color not in seen_colors, "expected unique emitter body colors"
        seen_colors.add(color)

        tails = []
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == GREY:
                tails.append((nr, nc))
        assert len(tails) == 1, "each emitter body must have exactly one grey tail"
        tail = tails[0]
        assert tail not in tail_to_idx, "grey tail cannot belong to multiple emitters"

        tr, tc = tail
        dr, dc = r - tr, c - tc
        assert (dr, dc) in DIRS4, "tail must be 4-adjacent to the body"

        body_to_idx[body] = len(emitters)
        tail_to_idx[tail] = len(emitters)
        emitters.append(
            {
                "body": body,
                "tail": tail,
                "color": color,
                "direction": (dr, dc),
            }
        )

    assert RED in seen_colors and list(sorted(col for col in seen_colors if col == RED)) == [RED], (
        "expected exactly one red emitter"
    )

    for tail, idx in tail_to_idx.items():
        adjacent_bodies = []
        for nr, nc in _neighbors4(*tail):
            if _in_bounds(g.shape, nr, nc) and (nr, nc) in body_to_idx:
                adjacent_bodies.append((nr, nc))
        assert adjacent_bodies == [emitters[idx]["body"]], "grey tail must touch only its own body"

    return emitters, body_to_idx, tail_to_idx


def _simulate(
    grid: np.ndarray,
    *,
    ignite_on_tail: bool = False,
    light_all_initially: bool = False,
    return_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    emitters, body_to_idx, tail_to_idx = _parse_emitters(g)
    color_to_idx = {int(em["color"]): i for i, em in enumerate(emitters)}
    start_idx = color_to_idx[RED]

    out = g.copy()
    if light_all_initially:
        initial_order = list(range(len(emitters)))
    else:
        initial_order = [start_idx]

    lit: set[int] = set(initial_order)
    queue: deque[int] = deque(initial_order)
    ignited_order = list(initial_order)
    events: list[dict] = []

    while queue:
        idx = queue.popleft()
        em = emitters[idx]
        body_r, body_c = em["body"]
        dr, dc = em["direction"]
        color = int(em["color"])

        painted: list[tuple[int, int]] = []
        r, c = body_r + dr, body_c + dc
        stop_reason = "border"
        hit_coord: tuple[int, int] | None = None
        ignited_idx: int | None = None

        while _in_bounds(out.shape, r, c):
            if int(out[r, c]) == BG:
                out[r, c] = color
                painted.append((r, c))
                r += dr
                c += dc
                continue

            hit_coord = (r, c)
            if hit_coord in body_to_idx:
                target_idx = body_to_idx[hit_coord]
                if target_idx not in lit:
                    lit.add(target_idx)
                    queue.append(target_idx)
                    ignited_order.append(target_idx)
                    ignited_idx = target_idx
                    stop_reason = "body"
                else:
                    stop_reason = "lit_body"
                break

            if hit_coord in tail_to_idx:
                stop_reason = "tail"
                if ignite_on_tail:
                    target_idx = tail_to_idx[hit_coord]
                    if target_idx not in lit:
                        lit.add(target_idx)
                        queue.append(target_idx)
                        ignited_order.append(target_idx)
                        ignited_idx = target_idx
                        stop_reason = "tail_ignite"
                break

            stop_reason = "ray"
            break

        events.append(
            {
                "emitter_idx": idx,
                "stop_reason": stop_reason,
                "painted": painted,
                "painted_count": len(painted),
                "hit_coord": hit_coord,
                "ignited_idx": ignited_idx,
            }
        )

    if not return_trace:
        return out

    trace = {
        "emitters": emitters,
        "lit": lit,
        "ignited_order": ignited_order,
        "events": events,
    }
    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Ignite the red emitter and accumulate only the rays reached through frontal body hits.

    Input structure:
    - A black background contains several emitters, each made of one unique-color body pixel
      and one grey tail pixel touching it by a 4-neighbor side.
    - Exactly one body is red, and every other non-background pixel belongs to some emitter.
    - The tail-to-body side determines the emitter's firing direction.

    Rule:
    - Start with only the red emitter lit.
    - A lit emitter shoots from the first cell just outside its body, moving in the
      tail-to-body direction and painting background cells with its own color.
    - The ray stops at the grid edge or at the first non-background cell it reaches.
    - If that blocking cell is the body of an unlit emitter, that emitter ignites and fires later.
      Hitting a grey tail, an already lit body, or an earlier ray only stops the ray.
    - Continue until no new emitter is ignited; return the original grid plus all painted rays.

    Key insight:
    - Grey tails are directional backplates, not secondary targets: only a frontal hit on a
      colored body propagates the chain, so some emitters stay permanently silent.
    """
    return _simulate(grid)


def _rotate_coord(coord: tuple[int, int], shape: tuple[int, int], k: int) -> tuple[int, int]:
    r, c = coord
    h, w = shape
    k %= 4
    if k == 0:
        return r, c
    if k == 1:
        return w - 1 - c, r
    if k == 2:
        return h - 1 - r, w - 1 - c
    return c, h - 1 - r


def _rotated_shape(shape: tuple[int, int], k: int) -> tuple[int, int]:
    h, w = shape
    return (h, w) if k % 2 == 0 else (w, h)


def _make_border_stop_motif() -> tuple[list[dict], tuple[int, int], str]:
    a = random.randint(4, 8)
    b = random.randint(5, 10)
    emitters = [
        {"body": (1, 1), "tail": (1, 0), "forced_color": RED},
        {"body": (1, 1 + a), "tail": (0, 1 + a)},
    ]
    return emitters, (b + 3, a + 3), "border"


def _make_ray_stop_motif() -> tuple[list[dict], tuple[int, int], str]:
    a = random.randint(4, 8)
    b = random.randint(3, 7)
    emitters = [
        {"body": (1, 1), "tail": (1, 0), "forced_color": RED},
        {"body": (1, 1 + a), "tail": (0, 1 + a)},
        {"body": (1 + b, 1 + a), "tail": (2 + b, 1 + a)},
    ]
    return emitters, (b + 4, a + 3), "ray"


def _make_tail_stop_motif() -> tuple[list[dict], tuple[int, int], str]:
    a = random.randint(6, 10)
    b = random.randint(4, 8)
    c = random.randint(3, a - 2)
    tail_col = 1 + a - c
    emitters = [
        {"body": (b + 2, 1), "tail": (b + 2, 0), "forced_color": RED},
        {"body": (b + 2, 1 + a), "tail": (b + 3, 1 + a)},
        {"body": (1, 1 + a), "tail": (1, 2 + a)},
        {"body": (2, tail_col), "tail": (1, tail_col)},
    ]
    return emitters, (b + 4, a + 4), "tail"


def _build_base_layout() -> tuple[np.ndarray, str]:
    motif_fn = random.choices(
        [_make_tail_stop_motif, _make_ray_stop_motif, _make_border_stop_motif],
        weights=[4, 3, 3],
        k=1,
    )[0]
    motif_emitters, motif_shape, terminal_reason = motif_fn()

    k = random.randint(0, 3)
    local_shape = _rotated_shape(motif_shape, k)
    transformed = []
    for item in motif_emitters:
        transformed.append(
            {
                "body": _rotate_coord(item["body"], motif_shape, k),
                "tail": _rotate_coord(item["tail"], motif_shape, k),
                "forced_color": item.get("forced_color"),
            }
        )

    h = random.randint(max(local_shape[0] + 3, 12), min(30, local_shape[0] + 12))
    w = random.randint(max(local_shape[1] + 3, 12), min(30, local_shape[1] + 12))
    r0 = random.randint(1, h - local_shape[0] - 1)
    c0 = random.randint(1, w - local_shape[1] - 1)

    n_emitters = len(transformed)
    free_colors = [color for color in BODY_COLORS if color != RED]
    random.shuffle(free_colors)
    assigned = iter(free_colors[: n_emitters - 1])

    grid = np.zeros((h, w), dtype=int)
    for item in transformed:
        body = (item["body"][0] + r0, item["body"][1] + c0)
        tail = (item["tail"][0] + r0, item["tail"][1] + c0)
        color = RED if item.get("forced_color") == RED else next(assigned)
        grid[body] = color
        grid[tail] = GREY

    return grid, terminal_reason


def _clear_zone(mask: np.ndarray, cells: list[tuple[int, int]], margin: int) -> bool:
    h, w = mask.shape
    for r, c in cells:
        r0 = max(0, r - margin)
        r1 = min(h, r + margin + 1)
        c0 = max(0, c - margin)
        c1 = min(w, c + margin + 1)
        if np.any(mask[r0:r1, c0:c1]):
            return False
    return True


def _place_extra_emitters(grid: np.ndarray, base_output: np.ndarray, count: int) -> bool:
    g = grid.copy()
    used_colors = {int(v) for v in g.flatten() if v not in {BG, GREY}}
    available = [color for color in BODY_COLORS if color not in used_colors]
    random.shuffle(available)
    if len(available) < count:
        return False

    h, w = g.shape
    placed = 0
    attempts = 0
    while placed < count and attempts < 800:
        attempts += 1
        body = (random.randint(1, h - 2), random.randint(1, w - 2))
        tail = random.choice(_neighbors4(*body))
        cells = [body, tail]
        if not all(_in_bounds(g.shape, r, c) for r, c in cells):
            continue
        if any(int(base_output[r, c]) != BG for r, c in cells):
            continue
        if any(int(g[r, c]) != BG for r, c in cells):
            continue
        if not _clear_zone(g != BG, cells, margin=1):
            continue
        if not _clear_zone(base_output != BG, cells, margin=1):
            continue

        g[body] = available[placed]
        g[tail] = GREY
        placed += 1

    if placed != count:
        return False

    grid[:] = g
    return True


def _simulate_all_fired(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, light_all_initially=True)


def _simulate_tail_wrong(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, ignite_on_tail=True)


def _validate_generated(grid: np.ndarray, terminal_reason: str) -> bool:
    out, trace = _simulate(grid, return_trace=True)
    emitters = trace["emitters"]
    lit = trace["lit"]
    order = trace["ignited_order"]
    events = trace["events"]

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if len(emitters) < 3:
        return False
    if len(lit) >= len(emitters):
        return False
    if len(order) < 2:
        return False
    if not np.any(out != grid):
        return False
    if events[-1]["stop_reason"] != terminal_reason:
        return False
    if terminal_reason == "tail":
        hit = events[-1]["hit_coord"]
        if hit is None or int(grid[hit]) != GREY:
            return False
    if terminal_reason == "ray" and events[-1]["stop_reason"] != "ray":
        return False
    if terminal_reason == "border" and events[-1]["painted_count"] < 2:
        return False

    all_fired = _simulate_all_fired(grid)
    if np.array_equal(all_fired, out):
        return False
    if terminal_reason == "tail":
        tail_wrong = _simulate_tail_wrong(grid)
        if np.array_equal(tail_wrong, out):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    grid, terminal_reason = _build_base_layout()
    base_output = main(grid)

    extra_target = random.randint(1, 2)
    if terminal_reason == "border":
        extra_target = random.randint(2, 3)
    if not _place_extra_emitters(grid, base_output, extra_target):
        return None

    try:
        if _validate_generated(grid, terminal_reason):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-reaction ray puzzle input")
