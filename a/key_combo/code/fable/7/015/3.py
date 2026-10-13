from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background(grid: np.ndarray) -> int:
    """Return the unique most frequent (background) color."""
    counts = Counter(map(int, grid.ravel()))
    ordered = counts.most_common()
    assert ordered and (len(ordered) == 1 or ordered[0][1] > ordered[1][1])
    return int(ordered[0][0])


def _components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    """4-connected non-background components, as arrays of (row, column)."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in _DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if (0 <= nr < h and 0 <= nc < w and not seen[nr, nc]
                            and int(grid[nr, nc]) != bg):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            result.append(np.asarray(cells, dtype=int))
    return result


def _l_bend(body: np.ndarray) -> tuple[int, int]:
    """Find the unique bbox corner with both a vertical and horizontal body neighbor."""
    assert body.shape[0] >= 3
    cells = {tuple(map(int, rc)) for rc in body}
    r0, c0 = map(int, body.min(axis=0))
    r1, c1 = map(int, body.max(axis=0))
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    candidates = []
    for r, c in cells & corners:
        vertical = (r - 1, c) in cells or (r + 1, c) in cells
        horizontal = (r, c - 1) in cells or (r, c + 1) in cells
        if vertical and horizontal:
            candidates.append((r, c))
    assert len(candidates) == 1, "body must be a one-cell-wide L with one bend"
    return candidates[0]


def _read_scene(grid: np.ndarray) -> tuple[int, tuple[int, int, int], list[dict]]:
    """Parse the isolated control and all marked L sprites."""
    bg = _background(grid)
    components = _components(grid, bg)
    singles = [comp for comp in components if len(comp) == 1]
    assert len(singles) == 1, "expected exactly one isolated control point"
    cr, cc = map(int, singles[0][0])
    control = (cr, cc, int(grid[cr, cc]))

    sprites = []
    for coords in components:
        if len(coords) == 1:
            continue
        counts = Counter(int(grid[r, c]) for r, c in coords)
        repeated = [color for color, count in counts.items() if count > 1]
        singleton = [color for color, count in counts.items() if count == 1]
        assert len(counts) == 2 and len(repeated) == 1 and len(singleton) == 1, (
            "each sprite must have one repeated body color and one marker pixel"
        )
        body_color, marker_color = repeated[0], singleton[0]
        body = np.asarray(
            [(int(r), int(c)) for r, c in coords if int(grid[r, c]) == body_color],
            dtype=int,
        )
        bend = _l_bend(body)
        marker_cells = [(int(r), int(c)) for r, c in coords if int(grid[r, c]) == marker_color]
        marker = marker_cells[0]
        assert abs(marker[0] - bend[0]) == 1 and abs(marker[1] - bend[1]) == 1, (
            "the marker must be embedded diagonally inside the L"
        )
        sprites.append({
            "coords": coords,
            "body_color": body_color,
            "marker_color": marker_color,
            "marker": marker,
            "bend": bend,
        })

    assert len(sprites) >= 2, "expected several sprites"
    body_colors = [sprite["body_color"] for sprite in sprites]
    assert len(body_colors) == len(set(body_colors)), "body colors must be unique"
    return bg, control, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """Move a chain of marked L sprites onto the control and occupied marker-cell anchors.

    Input structure:
    - The dominant background separates several 4-connected multicolor L sprites and
      exactly one isolated control pixel.
    - Every sprite has a unique repeated body color and one differently colored marker
      embedded diagonally beside its L bend; body colors are pairwise distinct.

    Rule:
    - A sprite marked with the control color moves so its L bend lands on the control.
    - A sprite marked with another sprite's body color instead lands on that other
      sprite's marker cell as it appeared in the input. Erase all original sprites and
      the control before drawing all translated sprites; unmatched sprites stay put.

    Key insight:
    - Except for the lone control, every destination is already occupied: color pairing
      reveals that each predecessor's embedded marker pixel is also the next anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg, control, sprites = _read_scene(g)
    control_pos = control[:2]
    control_color = control[2]
    by_body = {sprite["body_color"]: sprite for sprite in sprites}

    targets = []
    matched_targets = []
    for sprite in sprites:
        marker_color = sprite["marker_color"]
        if marker_color == control_color:
            target = control_pos
        elif marker_color in by_body:
            target = by_body[marker_color]["marker"]
        else:
            target = sprite["bend"]
        targets.append(target)
        if target != sprite["bend"]:
            matched_targets.append(target)

    assert matched_targets, "expected at least one linked sprite"
    assert len(matched_targets) == len(set(matched_targets)), "matched anchors must be one-to-one"

    # Simultaneous erase is essential because some destinations are occupied in the input.
    out = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
    out[control_pos] = bg

    occupied: set[tuple[int, int]] = set()
    h, w = g.shape
    for sprite, target in zip(sprites, targets):
        dr = target[0] - sprite["bend"][0]
        dc = target[1] - sprite["bend"][1]
        translated = []
        for r, c in sprite["coords"]:
            nr, nc = int(r) + dr, int(c) + dc
            assert 0 <= nr < h and 0 <= nc < w, "translated sprite must stay in bounds"
            translated.append((nr, nc, int(g[r, c])))
        for r, c, _color in translated:
            assert (r, c) not in occupied and int(out[r, c]) == bg, (
                "final sprites must not overlap stationary or translated sprites"
            )
            occupied.add((r, c))
        for r, c, color in translated:
            out[r, c] = color
    return out


# ------------------------------- generator -------------------------------

_COLORS = tuple(range(10))
_MAX_ATTEMPTS = 500
_PLACE_TRIES = 500


def _make_sprite(body_color: int, marker_color: int) -> dict:
    v_len = random.randint(2, 4)
    h_len = random.randint(2, 4)
    vertical_sign = random.choice((-1, 1))
    horizontal_sign = random.choice((-1, 1))
    bend = (0 if vertical_sign == 1 else v_len - 1,
            0 if horizontal_sign == 1 else h_len - 1)
    cells: list[tuple[int, int, int]] = []
    br, bc = bend
    for r in range(v_len):
        cells.append((r, bc, body_color))
    for c in range(h_len):
        if c != bc:
            cells.append((br, c, body_color))
    marker = (br + vertical_sign, bc + horizontal_sign)
    cells.append((marker[0], marker[1], marker_color))
    return {"shape": (v_len, h_len), "bend_local": bend,
            "marker_local": marker, "cells_local": cells,
            "body_color": body_color, "marker_color": marker_color}


def _expanded(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells
            for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)
            if 0 <= r + dr < h and 0 <= c + dc < w}


def _place_sprite(grid: np.ndarray, sprite: dict, bg: int) -> bool:
    h, w = grid.shape
    sh, sw = sprite["shape"]
    occupied = set(map(tuple, np.argwhere(grid != bg)))
    blocked = _expanded(occupied, h, w, 1)
    candidates = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            cells = {(r0 + r, c0 + c) for r, c, _ in sprite["cells_local"]}
            if cells.isdisjoint(blocked):
                candidates.append((r0, c0))
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    sprite["origin"] = (r0, c0)
    sprite["bend"] = (r0 + sprite["bend_local"][0], c0 + sprite["bend_local"][1])
    sprite["marker"] = (r0 + sprite["marker_local"][0], c0 + sprite["marker_local"][1])
    sprite["cells"] = [(r0 + r, c0 + c, color) for r, c, color in sprite["cells_local"]]
    for r, c, color in sprite["cells"]:
        grid[r, c] = color
    return True


def _place_control(grid: np.ndarray, color: int, bg: int) -> tuple[int, int] | None:
    h, w = grid.shape
    blocked = _expanded(set(map(tuple, np.argwhere(grid != bg))), h, w, 1)
    choices = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
               if (r, c) not in blocked]
    if not choices:
        return None
    r, c = random.choice(choices)
    grid[r, c] = color
    return r, c


def _render_expected(inp: np.ndarray, sprites: list[dict], control: tuple[int, int], bg: int) -> np.ndarray | None:
    """Independent metadata oracle; it does not call or parse through main()."""
    out = inp.copy()
    for sprite in sprites:
        for r, c, _ in sprite["cells"]:
            out[r, c] = bg
    out[control] = bg

    occupied: set[tuple[int, int]] = set()
    h, w = inp.shape
    for i, sprite in enumerate(sprites):
        if i == 0:
            target = control
        elif i < len(sprites) - 1:
            target = sprites[i - 1]["marker"]
        else:  # deliberately unmatched counterexample
            target = sprite["bend"]
        dr, dc = target[0] - sprite["bend"][0], target[1] - sprite["bend"][1]
        translated = [(r + dr, c + dc, color) for r, c, color in sprite["cells"]]
        if any(not (0 <= r < h and 0 <= c < w) for r, c, _ in translated):
            return None
        if any((r, c) in occupied or int(out[r, c]) != bg for r, c, _ in translated):
            return None
        for r, c, color in translated:
            occupied.add((r, c))
            out[r, c] = color
    return out


def _semantically_valid(inp: np.ndarray, expected: np.ndarray, sprites: list[dict],
                        control: tuple[int, int], bg: int) -> bool:
    """Check uniqueness, chain dependencies, non-degeneracy, and solver agreement."""
    try:
        parsed_bg, parsed_control, parsed = _read_scene(inp)
        actual = main(inp)
    except Exception:
        return False
    if parsed_bg != bg or parsed_control[:2] != control or len(parsed) != len(sprites):
        return False
    if not np.array_equal(actual, expected) or np.array_equal(inp, expected):
        return False

    chain_len = len(sprites) - 1
    if chain_len < 2:
        return False
    body_colors = {sprite["body_color"] for sprite in sprites}
    unmatched = sprites[-1]
    if unmatched["marker_color"] in body_colors or unmatched["marker_color"] == parsed_control[2]:
        return False

    # Every occupied anchor must matter and must differ from both the visible control
    # and the tempting but wrong predecessor-bend anchor.
    for i in range(1, chain_len):
        anchor = sprites[i - 1]["marker"]
        if anchor == control or anchor == sprites[i - 1]["bend"] or anchor == sprites[i]["bend"]:
            return False
    if any(sprites[i]["bend"] == (control if i == 0 else sprites[i - 1]["marker"])
           for i in range(chain_len)):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h, w = random.randint(18, 27), random.randint(18, 27)
    bg = random.choice(_COLORS)
    palette = [color for color in _COLORS if color != bg]
    chain_len = random.randint(2, 4)
    total = chain_len + 1  # one unmatched sprite is always an explicit negative case

    body_colors = random.sample(palette, total)
    remaining = [color for color in palette if color not in body_colors]
    if len(remaining) < 2:
        return None
    control_color, unmatched_marker = random.sample(remaining, 2)
    marker_colors = [control_color] + body_colors[:chain_len - 1] + [unmatched_marker]
    sprites = [_make_sprite(body, marker)
               for body, marker in zip(body_colors, marker_colors)]

    inp = np.full((h, w), bg, dtype=int)
    # Larger sprites first makes varied 2--4 leg lengths easier to place.
    order = sorted(range(total), key=lambda i: len(sprites[i]["cells_local"]), reverse=True)
    for i in order:
        if not _place_sprite(inp, sprites[i], bg):
            return None
    control = _place_control(inp, control_color, bg)
    if control is None:
        return None

    expected = _render_expected(inp, sprites, control, bg)
    if expected is None or not _semantically_valid(inp, expected, sprites, control, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked chain with an unmatched counterexample."""
    for _ in range(_MAX_ATTEMPTS):
        result = _generate_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid chain-migration input")
