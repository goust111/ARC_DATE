from __future__ import annotations

import itertools
import math
import random
from collections import Counter

import numpy as np


DIRS_DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
GEN_ATTEMPTS = 800
SPRITE_SHAPES = [(2, 3), (3, 2), (3, 3)]
ALL_COLORS = list(range(10))


def _most_common(values) -> int:
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _connected(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not cells:
        return False
    seen = {cells[0]}
    stack = [cells[0]]
    cell_set = set(cells)
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if nxt in cell_set and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _sprite_catalog() -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for h, w in SPRITE_SHAPES:
        cells = [(r, c) for r in range(h) for c in range(w)]
        max_pixels = min(h * w - 1, 6)
        for count in range(3, max_pixels + 1):
            for combo in itertools.combinations(cells, count):
                mask = np.zeros((h, w), dtype=bool)
                for r, c in combo:
                    mask[r, c] = True
                if _bbox(mask) != (0, 0, h - 1, w - 1):
                    continue
                if not _connected(mask):
                    continue
                variants = [
                    mask,
                    np.fliplr(mask),
                    np.flipud(mask),
                    np.flipud(np.fliplr(mask)),
                ]
                keys = {tuple(map(int, variant.ravel())) for variant in variants}
                if len(keys) != 4:
                    continue
                out.append(mask.copy())
    return out


SPRITE_MASKS = _sprite_catalog()


def _extract_sprite(panel: np.ndarray, fill: int) -> tuple[np.ndarray, int, int]:
    interior = panel[1:-1, 1:-1]
    box = _bbox(interior != int(fill))
    assert box is not None
    r0, c0, r1, c1 = box
    patch = interior[r0 : r1 + 1, c0 : c1 + 1].copy()
    return patch, int(r0 + 1), int(c0 + 1)


def _split_panels(grid: np.ndarray) -> tuple[int, int, list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frame = int(g[0, 0])
    assert h >= 5 and w >= 5
    assert np.all(g[0, :] == frame)
    assert np.all(g[-1, :] == frame)

    separators = [c for c in range(w) if np.all(g[:, c] == frame)]
    assert len(separators) >= 4
    assert separators[0] == 0 and separators[-1] == w - 1
    strides = [b - a for a, b in zip(separators, separators[1:])]
    assert len(set(strides)) == 1
    stride = int(strides[0])
    panel_w = stride + 1

    panels = []
    interior_values = []
    for c0, c1 in zip(separators, separators[1:]):
        panel = g[:, c0 : c1 + 1]
        assert panel.shape == (h, panel_w)
        assert np.all(panel[:, 0] == frame)
        assert np.all(panel[:, -1] == frame)
        panels.append(panel)
        interior_values.extend(panel[1:-1, 1:-1].ravel())

    fill = _most_common(interior_values)
    return frame, int(fill), panels, panel_w


def _clone_state(state: tuple[int, int, int, int, np.ndarray]) -> tuple[int, int, int, int, np.ndarray]:
    r, c, dr, dc, patch = state
    return int(r), int(c), int(dr), int(dc), patch.copy()


def _step_state(
    state: tuple[int, int, int, int, np.ndarray], panel_h: int, panel_w: int
) -> tuple[tuple[int, int, int, int, np.ndarray], dict[str, int]]:
    r, c, dr, dc, patch = _clone_state(state)
    sprite_h, sprite_w = patch.shape
    min_r, min_c = 1, 1
    max_r = panel_h - sprite_h - 1
    max_c = panel_w - sprite_w - 1

    next_r = r + dr
    next_c = c + dc
    hits = {"horizontal_wall": 0, "vertical_wall": 0}

    if next_r < min_r or next_r > max_r:
        dr *= -1
        next_r = r + dr
        patch = np.flipud(patch).copy()
        hits["horizontal_wall"] = 1
    if next_c < min_c or next_c > max_c:
        dc *= -1
        next_c = c + dc
        patch = np.fliplr(patch).copy()
        hits["vertical_wall"] = 1

    return (int(next_r), int(next_c), int(dr), int(dc), patch), hits


def _advance_state(
    state: tuple[int, int, int, int, np.ndarray], steps: int, panel_h: int, panel_w: int
) -> tuple[tuple[int, int, int, int, np.ndarray], dict[str, int]]:
    cur = _clone_state(state)
    totals = {"horizontal_wall": 0, "vertical_wall": 0}
    for _ in range(int(steps)):
        cur, hits = _step_state(cur, panel_h, panel_w)
        for key in totals:
            totals[key] += hits[key]
    return cur, totals


def _state_period(panel_h: int, panel_w: int, sprite_h: int, sprite_w: int) -> int:
    span_r = panel_h - sprite_h - 2
    span_c = panel_w - sprite_w - 2
    period_r = 1 if span_r <= 0 else 2 * span_r
    period_c = 1 if span_c <= 0 else 2 * span_c
    return int(math.lcm(period_r, period_c))


def _render_panel(
    panel_h: int, panel_w: int, frame: int, fill: int, patch: np.ndarray, top: int, left: int
) -> np.ndarray:
    out = np.full((panel_h, panel_w), int(fill), dtype=int)
    out[0, :] = frame
    out[-1, :] = frame
    out[:, 0] = frame
    out[:, -1] = frame
    _paste(out, patch, top, left, transparent=fill)
    return out


def _assemble_panels(panels: list[np.ndarray]) -> np.ndarray:
    out = panels[0].copy()
    for panel in panels[1:]:
        out = np.hstack([out, panel[:, 1:]])
    return out


def _observed_states(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int, np.ndarray]]]:
    frame, fill, panels, panel_w = _split_panels(grid)
    observed = []
    for panel in panels:
        patch, top, left = _extract_sprite(panel, fill)
        observed.append((top, left, patch))
    return frame, fill, panel_w, observed


def _candidate_signature(candidate: dict, panel_h: int, panel_w: int, frame: int, fill: int) -> bytes:
    next_state = candidate["next_state"]
    panel = _render_panel(panel_h, panel_w, frame, fill, next_state[4], next_state[0], next_state[1])
    return panel.tobytes()


def _enumerate_candidates(
    observed: list[tuple[int, int, np.ndarray]],
    panel_h: int,
    panel_w: int,
    compare_pos: bool,
    compare_patch: bool,
) -> list[dict]:
    top0, left0, patch0 = observed[0]
    sprite_h, sprite_w = patch0.shape
    period = _state_period(panel_h, panel_w, sprite_h, sprite_w)
    candidates = []
    for gap in range(2, period + 1):
        for dr, dc in DIRS_DIAG:
            state = (top0, left0, dr, dc, patch0.copy())
            interval_hits = []
            ok = True
            for top, left, patch in observed[1:]:
                state, hits = _advance_state(state, gap, panel_h, panel_w)
                interval_hits.append(hits)
                if compare_pos and (state[0] != top or state[1] != left):
                    ok = False
                    break
                if compare_patch and not np.array_equal(state[4], patch):
                    ok = False
                    break
            if not ok:
                continue
            next_state, next_hits = _advance_state(state, gap, panel_h, panel_w)
            candidates.append(
                {
                    "gap": gap,
                    "velocity": (dr, dc),
                    "final_state": _clone_state(state),
                    "next_state": _clone_state(next_state),
                    "interval_hits": interval_hits,
                    "next_hits": next_hits,
                }
            )
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """Predict the next bounced sprite frame from a row of shared-border panels.

    Input structure:
    - The grid is a horizontal sequence of equal framed panels sharing border columns.
    - Every panel contains the same non-symmetric single-color sprite on the same fill color.
    - The sprite appears once per panel, but hidden diagonal bounce steps happened between visible frames.

    Rule:
    - Treat each panel as one snapshot of a sprite whose row and column velocities are each either +1 or -1.
    - Between snapshots the sprite advances by one fixed hidden step count; hitting a left/right wall reverses the horizontal velocity and mirrors the sprite left-right, while hitting a top/bottom wall reverses the vertical velocity and mirrors it up-down.
    - Infer the unique hidden step count and velocity from all shown positions and mirror orientations, then advance the last panel by the same count and output that single next framed panel.

    Key insight:
    - Sparse sampled positions alone are ambiguous after unseen bounces, and the sprite's mirrored orientation records exactly which wall reflections happened in between.
    """
    g = np.asarray(grid, dtype=int)
    frame, fill, panels, panel_w = _split_panels(g)
    observed = []
    for panel in panels:
        patch, top, left = _extract_sprite(panel, fill)
        observed.append((top, left, patch))

    candidates = _enumerate_candidates(
        observed,
        panel_h=g.shape[0],
        panel_w=panel_w,
        compare_pos=True,
        compare_patch=True,
    )
    assert len(candidates) == 1, "expected a unique trajectory"
    candidate = candidates[0]
    next_state = candidate["next_state"]
    return _render_panel(g.shape[0], panel_w, frame, fill, next_state[4], next_state[0], next_state[1])


def _random_sprite(fill: int, color: int) -> np.ndarray:
    mask = random.choice(SPRITE_MASKS)
    sprite = np.full(mask.shape, int(fill), dtype=int)
    sprite[mask] = int(color)
    if random.random() < 0.5:
        sprite = np.flipud(sprite).copy()
    if random.random() < 0.5:
        sprite = np.fliplr(sprite).copy()
    return sprite


def _has_hidden_bounce(hits: dict[str, int]) -> bool:
    return hits["horizontal_wall"] + hits["vertical_wall"] > 0


def _attempt_case() -> tuple[np.ndarray, np.ndarray] | None:
    panel_h = random.randint(8, 11)
    panel_w = random.randint(6, 8)
    n_panels = random.choice([3, 4])
    if n_panels * (panel_w - 1) + 1 > 30:
        return None

    frame, fill, sprite_color = random.sample(ALL_COLORS, 3)
    sprite = _random_sprite(fill, sprite_color)
    sprite_h, sprite_w = sprite.shape
    if sprite_h > panel_h - 4 or sprite_w > panel_w - 4:
        return None

    span_r = panel_h - sprite_h - 2
    span_c = panel_w - sprite_w - 2
    if min(span_r, span_c) < 2:
        return None
    period = _state_period(panel_h, panel_w, sprite_h, sprite_w)
    if period < 4:
        return None

    gap = random.randint(2, period)
    start_top = random.randint(1, panel_h - sprite_h - 1)
    start_left = random.randint(1, panel_w - sprite_w - 1)
    start_dr, start_dc = random.choice(DIRS_DIAG)
    state = (start_top, start_left, start_dr, start_dc, sprite.copy())

    states = [_clone_state(state)]
    interval_hits = []
    for _ in range(n_panels - 1):
        state, hits = _advance_state(state, gap, panel_h, panel_w)
        states.append(_clone_state(state))
        interval_hits.append(hits)
    next_state, next_hits = _advance_state(state, gap, panel_h, panel_w)

    if not all(_has_hidden_bounce(hits) for hits in interval_hits):
        return None
    if not _has_hidden_bounce(next_hits):
        return None

    all_hits = interval_hits + [next_hits]
    if not any(h["horizontal_wall"] for h in all_hits):
        return None
    if not any(h["vertical_wall"] for h in all_hits):
        return None

    observed_panels = [
        _render_panel(panel_h, panel_w, frame, fill, st[4], st[0], st[1]) for st in states
    ]
    expected = _render_panel(panel_h, panel_w, frame, fill, next_state[4], next_state[0], next_state[1])
    if any(np.array_equal(expected, panel) for panel in observed_panels):
        return None

    inp = _assemble_panels(observed_panels)
    observed = [(st[0], st[1], st[4].copy()) for st in states]

    full_candidates = _enumerate_candidates(
        observed, panel_h=panel_h, panel_w=panel_w, compare_pos=True, compare_patch=True
    )
    if len(full_candidates) != 1:
        return None
    if full_candidates[0]["gap"] != gap or full_candidates[0]["velocity"] != (start_dr, start_dc):
        return None

    pos_only = _enumerate_candidates(
        observed, panel_h=panel_h, panel_w=panel_w, compare_pos=True, compare_patch=False
    )
    pos_outputs = {
        _candidate_signature(candidate, panel_h, panel_w, frame, fill) for candidate in pos_only
    }
    if len(pos_outputs) < 2:
        return None

    orient_only = _enumerate_candidates(
        observed, panel_h=panel_h, panel_w=panel_w, compare_pos=False, compare_patch=True
    )
    orient_outputs = {
        _candidate_signature(candidate, panel_h, panel_w, frame, fill) for candidate in orient_only
    }
    if len(orient_outputs) < 2:
        return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _attempt_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
