from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import get_bg, in_bounds


Coord = tuple[int, int]

GREY = 5
SIDES = ("top", "bottom", "left", "right")
STEPS = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
GENERATION_ATTEMPTS = 500
PLACE_ATTEMPTS = 300
DECOR_ATTEMPTS = 90
GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
EXTRA_TOKEN_RANGE = (0, 3)
BG_CHOICES = (0, 7, 8, 9)
COLOR_COUNT = 10

_GEN_CALL_COUNT = 0


def _add(p: Coord, step: Coord, k: int = 1) -> Coord:
    return int(p[0] + step[0] * k), int(p[1] + step[1] * k)


def _internal(shape: tuple[int, int], p: Coord) -> bool:
    h, w = shape
    r, c = p
    return 0 < r < h - 1 and 0 < c < w - 1


def _edge_side(shape: tuple[int, int], r: int, c: int) -> str:
    h, w = shape
    sides = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    assert len(sides) == 1, "falling pixels must sit on exactly one non-corner edge"
    return sides[0]


def _edge_start(shape: tuple[int, int], side: str, lane: int) -> Coord:
    h, w = shape
    if side == "top":
        return 0, int(lane)
    if side == "bottom":
        return h - 1, int(lane)
    if side == "left":
        return int(lane), 0
    if side == "right":
        return int(lane), w - 1
    raise ValueError(f"unknown side {side!r}")


def _lane_values(shape: tuple[int, int], side: str) -> list[int]:
    h, w = shape
    hi = w - 1 if side in ("top", "bottom") else h - 1
    return list(range(1, hi))


def _track_to_edge(shape: tuple[int, int], start: Coord, step: Coord) -> list[Coord]:
    cells = [start]
    p = start
    while True:
        p = _add(p, step)
        if not in_bounds(shape, *p):
            break
        cells.append(p)
    return cells


def _distance_to_initial_grey(track: list[Coord], grey: set[Coord]) -> tuple[int, bool]:
    for dist, p in enumerate(track[1:], start=1):
        if p in grey:
            return int(dist), True
    return int(len(track) - 1), False


def _make_token(
    shape: tuple[int, int],
    side: str,
    start: Coord,
    color: int,
    idx: int,
    *,
    must_final: Coord | None = None,
    must_be_blocked_by: Coord | None = None,
) -> dict:
    step = STEPS[side]
    return {
        "idx": int(idx),
        "side": side,
        "start": start,
        "color": int(color),
        "step": step,
        "track": _track_to_edge(shape, start, step),
        "must_final": must_final,
        "must_be_blocked_by": must_be_blocked_by,
    }


def _with_distances(tokens: list[dict], grey: set[Coord]) -> list[dict]:
    out = []
    distances = []
    for token in tokens:
        new_token = dict(token)
        dist, has_grey = _distance_to_initial_grey(new_token["track"], grey)
        new_token["distance"] = dist
        new_token["has_initial_grey"] = bool(has_grey)
        out.append(new_token)
        distances.append(dist)
    assert len(distances) == len(set(distances)), "initial grey distances must be unique"
    return out


def _falling_tokens(g: np.ndarray, bg: int) -> list[dict]:
    assert int(bg) != GREY, "grey is reserved for fixed obstacles"
    h, w = g.shape
    grey = {tuple(map(int, p)) for p in np.argwhere(g == GREY)}
    tokens: list[dict] = []
    interior_colored = []

    for r, c in np.argwhere((g != bg) & (g != GREY)):
        r = int(r)
        c = int(c)
        if r in (0, h - 1) or c in (0, w - 1):
            side = _edge_side(g.shape, r, c)
            tokens.append(_make_token(g.shape, side, (r, c), int(g[r, c]), len(tokens)))
        else:
            interior_colored.append((r, c))

    assert tokens, "expected at least one colored edge pixel"
    assert not interior_colored, "colored input pixels must all be edge fallers"
    return _with_distances(tokens, grey)


def _simulate_tokens(
    g: np.ndarray,
    tokens: list[dict],
    *,
    order: list[dict] | None = None,
    settled_blocks: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    bg = int(get_bg(g))
    initial_grey = {tuple(map(int, p)) for p in np.argwhere(g == GREY)}
    blockers = set(initial_grey)
    starts = {token["start"] for token in tokens}
    out = g.copy()
    for start in starts:
        out[start] = bg

    if order is None:
        order = sorted(tokens, key=lambda token: int(token["distance"]))

    records = []
    for token in order:
        start = token["start"]
        assert start not in blockers, "a settled pixel may not occupy another faller's start"
        final = start
        blocked_by = None
        for p in token["track"][1:]:
            if p in blockers:
                blocked_by = p
                break
            final = p
        out[final] = int(token["color"])
        if settled_blocks:
            blockers.add(final)
        records.append(
            {
                "idx": int(token["idx"]),
                "start": start,
                "final": final,
                "blocked_by": blocked_by,
                "blocked_by_grey": blocked_by in initial_grey if blocked_by is not None else False,
            }
        )
    return out, records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop colored edge pixels inward under four-way gravity, letting earlier settled pixels block later ones.

    Input structure:
    - The background is the dominant color and grey pixels are fixed interior obstacles.
    - Every non-background, non-grey input pixel is a single colored pixel on exactly one outer edge.
    - Top, bottom, left, and right edge pixels fall respectively down, up, right, and left.

    Rule:
    - For each edge pixel, compute its initial distance along its own track to the first grey obstacle; if there is no grey obstacle, use its distance to the opposite edge.
    - Process pixels in ascending order of those unique initial distances.
    - A pixel slides along its straight track until the next cell is a grey obstacle or a previously settled colored pixel, then it stops in the previous cell; with no blocker it stops on the opposite edge.
    - Clear the original edge pixels and preserve all grey obstacles.

    Key insight:
    The crossing tracks are not simultaneous: the nearest-to-grey faller settles first, and that settled colored pixel becomes a real obstacle for later fallers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = int(get_bg(g))
    tokens = _falling_tokens(g, bg)
    out, _records = _simulate_tokens(g, tokens)
    return out


# --------------- generator ---------------


def _color_stream(bg: int) -> list[int]:
    colors = [c for c in range(COLOR_COUNT) if c not in (int(bg), GREY)]
    random.shuffle(colors)
    return colors


def _start_through_point(shape: tuple[int, int], side: str, point: Coord) -> Coord:
    r, c = point
    return _edge_start(shape, side, c if side in ("top", "bottom") else r)


def _core_candidates(shape: tuple[int, int], early_side: str) -> list[tuple[Coord, str, int, Coord, Coord]]:
    h, w = shape
    late_sides = ("left", "right") if early_side in ("top", "bottom") else ("top", "bottom")
    candidates = []
    for late_side in late_sides:
        early_step = STEPS[early_side]
        late_step = STEPS[late_side]
        for r in range(2, h - 2):
            for c in range(2, w - 2):
                cross = (r, c)
                grey_early = _add(cross, early_step)
                if not _internal(shape, grey_early):
                    continue
                max_k = 0
                while _internal(shape, _add(cross, late_step, max_k + 1)):
                    max_k += 1
                for k in range(2, max_k + 1):
                    grey_late = _add(cross, late_step, k)
                    start_early = _start_through_point(shape, early_side, cross)
                    start_late = _start_through_point(shape, late_side, cross)
                    dist_early = abs(start_early[0] - grey_early[0]) + abs(start_early[1] - grey_early[1])
                    dist_late = abs(start_late[0] - grey_late[0]) + abs(start_late[1] - grey_late[1])
                    if dist_early < dist_late and grey_early != grey_late:
                        candidates.append((cross, late_side, k, grey_early, grey_late))
    random.shuffle(candidates)
    return candidates


def _build_grid(shape: tuple[int, int], bg: int, grey: set[Coord], tokens: list[dict]) -> np.ndarray | None:
    g = np.full(shape, int(bg), dtype=int)
    for r, c in grey:
        if not _internal(shape, (r, c)):
            return None
        g[r, c] = GREY
    for token in tokens:
        r, c = token["start"]
        if g[r, c] != bg:
            return None
        g[r, c] = int(token["color"])
    return g


def _state_valid(
    shape: tuple[int, int],
    bg: int,
    grey: set[Coord],
    tokens: list[dict],
    *,
    complete: bool = False,
    require_unobstructed: bool = False,
) -> bool:
    g = _build_grid(shape, bg, grey, tokens)
    if g is None or int(get_bg(g)) != int(bg):
        return False
    try:
        solved_tokens = _with_distances(tokens, grey)
        out, records = _simulate_tokens(g, solved_tokens)
        independent, independent_records = _simulate_tokens(g, solved_tokens, settled_blocks=False)
        reverse_order = sorted(solved_tokens, key=lambda token: int(token["distance"]), reverse=True)
        reversed_out, _ = _simulate_tokens(g, solved_tokens, order=reverse_order)
    except AssertionError:
        return False

    by_idx = {rec["idx"]: rec for rec in records}
    independent_by_idx = {rec["idx"]: rec for rec in independent_records}
    for token in solved_tokens:
        idx = int(token["idx"])
        if token.get("must_final") is not None and by_idx[idx]["final"] != token["must_final"]:
            return False
        if token.get("must_be_blocked_by") is not None:
            if by_idx[idx]["blocked_by"] != token["must_be_blocked_by"]:
                return False
            if by_idx[idx]["final"] == independent_by_idx[idx]["final"]:
                return False

    if not np.array_equal(out, main(g)):
        return False
    if np.array_equal(out, independent):
        return False
    if np.array_equal(out, reversed_out):
        return False
    if len({rec["final"] for rec in records}) != len(records):
        return False

    if complete:
        if len(tokens) < 4:
            return False
        if {token["side"] for token in solved_tokens} != set(SIDES):
            return False
        if require_unobstructed and not any(not token["has_initial_grey"] for token in solved_tokens):
            return False
        if max(g.shape) > 30 or max(out.shape) > 30:
            return False
    return True


def _make_core(shape: tuple[int, int], bg: int, early_side: str) -> tuple[set[Coord], list[dict]] | None:
    colors = _color_stream(bg)
    for cross, late_side, _k, grey_early, grey_late in _core_candidates(shape, early_side):
        early = _make_token(
            shape,
            early_side,
            _start_through_point(shape, early_side, cross),
            colors[0],
            0,
            must_final=cross,
        )
        late = _make_token(
            shape,
            late_side,
            _start_through_point(shape, late_side, cross),
            colors[1],
            1,
            must_be_blocked_by=cross,
        )
        grey = {grey_early, grey_late}
        tokens = [early, late]
        if _state_valid(shape, bg, grey, tokens):
            return grey, tokens
    return None


def _candidate_grey_cells(
    shape: tuple[int, int],
    grey: set[Coord],
    tokens: list[dict],
    new_token: dict,
) -> list[Coord | None]:
    starts = {token["start"] for token in tokens}
    starts.add(new_token["start"])
    candidates: list[Coord | None] = [None]
    for p in new_token["track"][1:-1]:
        if p in grey or p in starts or not _internal(shape, p):
            continue
        candidates.append(p)
    random.shuffle(candidates)
    return candidates


def _add_extra_token(
    shape: tuple[int, int],
    bg: int,
    grey: set[Coord],
    tokens: list[dict],
    side: str,
    *,
    require_no_grey: bool = False,
) -> tuple[set[Coord], list[dict]] | None:
    starts = {token["start"] for token in tokens}
    colors = _color_stream(bg)
    lanes = _lane_values(shape, side)
    random.shuffle(lanes)

    for lane in lanes:
        start = _edge_start(shape, side, lane)
        if start in starts or start in grey:
            continue
        color = colors[len(tokens) % len(colors)]
        token = _make_token(shape, side, start, color, len(tokens))
        current_dist, current_has_grey = _distance_to_initial_grey(token["track"], grey)
        if require_no_grey and current_has_grey:
            continue
        for cell in _candidate_grey_cells(shape, grey, tokens, token):
            if require_no_grey and cell is not None:
                continue
            new_grey = set(grey)
            if cell is not None:
                new_grey.add(cell)
            elif not require_no_grey and not current_has_grey and random.random() < 0.65:
                continue
            proposal = tokens + [token]
            try:
                proposal_with_dist = _with_distances(proposal, new_grey)
            except AssertionError:
                continue
            if proposal_with_dist[-1]["distance"] == current_dist and current_has_grey != proposal_with_dist[-1]["has_initial_grey"]:
                continue
            if _state_valid(shape, bg, new_grey, proposal):
                return new_grey, proposal
    return None


def _cluster_from_seed(shape: tuple[int, int], seed: Coord, size: int) -> set[Coord] | None:
    cells = {seed}
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for _ in range(size - 1):
        base = random.choice(tuple(cells))
        random.shuffle(deltas)
        added = False
        for dr, dc in deltas:
            p = (base[0] + dr, base[1] + dc)
            if _internal(shape, p) and p not in cells:
                cells.add(p)
                added = True
                break
        if not added:
            return None
    return cells


def _add_decorative_grey(
    shape: tuple[int, int],
    bg: int,
    grey: set[Coord],
    tokens: list[dict],
    *,
    require_unobstructed: bool,
) -> set[Coord]:
    starts = {token["start"] for token in tokens}
    target_blocks = random.randint(1, 4)
    out = set(grey)
    for _ in range(target_blocks):
        for _attempt in range(DECOR_ATTEMPTS):
            seed = (random.randint(1, shape[0] - 2), random.randint(1, shape[1] - 2))
            size = random.choice((1, 1, 2, 2, 3))
            cluster = _cluster_from_seed(shape, seed, size)
            if cluster is None or cluster & out or cluster & starts:
                continue
            proposal = out | cluster
            if _state_valid(
                shape,
                bg,
                proposal,
                tokens,
                complete=True,
                require_unobstructed=require_unobstructed,
            ):
                out = proposal
                break
    return out


def _gen_attempt(call_index: int, require_unobstructed: bool) -> np.ndarray | None:
    shape = (random.randint(*GRID_H_RANGE), random.randint(*GRID_W_RANGE))
    bg = random.choice(BG_CHOICES)
    early_side = SIDES[call_index % len(SIDES)] if random.random() < 0.65 else random.choice(SIDES)

    core = _make_core(shape, bg, early_side)
    if core is None:
        return None
    grey, tokens = core

    missing = [side for side in SIDES if side not in {token["side"] for token in tokens}]
    random.shuffle(missing)
    for side in missing:
        placed = None
        for _ in range(PLACE_ATTEMPTS):
            placed = _add_extra_token(shape, bg, grey, tokens, side)
            if placed is not None:
                break
        if placed is None:
            return None
        grey, tokens = placed

    if require_unobstructed:
        try:
            have_unobstructed = any(not token["has_initial_grey"] for token in _with_distances(tokens, grey))
        except AssertionError:
            return None
        if not have_unobstructed:
            placed = None
            sides = list(SIDES)
            random.shuffle(sides)
            for side in sides:
                for _ in range(PLACE_ATTEMPTS):
                    placed = _add_extra_token(shape, bg, grey, tokens, side, require_no_grey=True)
                    if placed is not None:
                        break
                if placed is not None:
                    break
            if placed is None:
                return None
            grey, tokens = placed

    for _ in range(random.randint(*EXTRA_TOKEN_RANGE)):
        side = random.choice(SIDES)
        placed = _add_extra_token(shape, bg, grey, tokens, side)
        if placed is not None:
            grey, tokens = placed

    if not _state_valid(shape, bg, grey, tokens, complete=True, require_unobstructed=require_unobstructed):
        return None
    grey = _add_decorative_grey(shape, bg, grey, tokens, require_unobstructed=require_unobstructed)
    if not _state_valid(shape, bg, grey, tokens, complete=True, require_unobstructed=require_unobstructed):
        return None

    return _build_grid(shape, bg, grey, tokens)


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    call_index = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    require_unobstructed = call_index % 3 == 0

    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(call_index, require_unobstructed=require_unobstructed)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid four-way gravity input")
