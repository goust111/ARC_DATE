from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ARROW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR_TO_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_RETRIES = 500
CHAIN_GAP_MIN = 7
CHAIN_GAP_MAX = 10
EXTRA_ARROW_TRIES = 120


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _arrow_cells(body_r0: int, body_c0: int, direction: str) -> np.ndarray:
    cells = [(r, c) for r in range(body_r0, body_r0 + 3) for c in range(body_c0, body_c0 + 3)]
    if direction == "right":
        cells.extend((body_r0 + 1, body_c0 - offset) for offset in range(1, 4))
    elif direction == "left":
        cells.extend((body_r0 + 1, body_c0 + 2 + offset) for offset in range(1, 4))
    elif direction == "down":
        cells.extend((body_r0 - offset, body_c0 + 1) for offset in range(1, 4))
    elif direction == "up":
        cells.extend((body_r0 + 2 + offset, body_c0 + 1) for offset in range(1, 4))
    else:
        raise AssertionError(f"unknown direction: {direction}")
    return np.array(sorted(cells), dtype=int)


def _ray_start(body: tuple[int, int], direction: str) -> tuple[int, int]:
    body_r0, body_c0 = body
    if direction == "right":
        return body_r0 + 1, body_c0 + 3
    if direction == "left":
        return body_r0 + 1, body_c0 - 1
    if direction == "down":
        return body_r0 + 3, body_c0 + 1
    if direction == "up":
        return body_r0 - 1, body_c0 + 1
    raise AssertionError(f"unknown direction: {direction}")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue
            color = int(grid[r0, c0])
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (
                        int(arr[:, 0].min()),
                        int(arr[:, 1].min()),
                        int(arr[:, 0].max()),
                        int(arr[:, 1].max()),
                    ),
                }
            )
    return comps


def _tail_direction(tail: set[tuple[int, int]], body_r0: int, body_c0: int) -> str | None:
    rows = sorted(r for r, _ in tail)
    cols = sorted(c for _, c in tail)
    if len(set(rows)) == 1 and cols == list(range(cols[0], cols[0] + 3)):
        if rows[0] == body_r0 + 1 and cols[-1] == body_c0 - 1:
            return "right"
        if rows[0] == body_r0 + 1 and cols[0] == body_c0 + 3:
            return "left"
    if len(set(cols)) == 1 and rows == list(range(rows[0], rows[0] + 3)):
        if cols[0] == body_c0 + 1 and rows[-1] == body_r0 - 1:
            return "down"
        if cols[0] == body_c0 + 1 and rows[0] == body_r0 + 3:
            return "up"
    return None


def _parse_arrow(comp: dict) -> dict:
    coords = {tuple(map(int, cell)) for cell in comp["coords"].tolist()}
    assert len(coords) == 12, "each arrow must contain a 3x3 body and a 3-cell tail"
    min_r, min_c, max_r, max_c = comp["bbox"]
    candidates: list[tuple[int, int, str]] = []
    for body_r0 in range(min_r, max_r - 1):
        for body_c0 in range(min_c, max_c - 1):
            square = {(r, c) for r in range(body_r0, body_r0 + 3) for c in range(body_c0, body_c0 + 3)}
            if not square.issubset(coords):
                continue
            tail = coords - square
            if len(tail) != 3:
                continue
            direction = _tail_direction(tail, body_r0, body_c0)
            if direction is not None:
                candidates.append((body_r0, body_c0, direction))
    assert len(candidates) == 1, "each component must be one arrow with a unique 3x3 body"
    body_r0, body_c0, direction = candidates[0]
    return {
        "body": (int(body_r0), int(body_c0)),
        "direction": direction,
        "coords": np.array(sorted(coords), dtype=int),
        "color": int(comp["color"]),
        "initial_color": int(comp["color"]),
    }


def _solve_from_arrows(arrows: list[dict], shape: tuple[int, int], bg: int) -> tuple[np.ndarray, list[dict], list[dict]]:
    h, w = shape
    ordered = sorted(
        [
            {
                "body": tuple(map(int, arrow["body"])),
                "direction": str(arrow["direction"]),
                "coords": np.array(arrow["coords"], dtype=int),
                "color": int(arrow["color"]),
                "initial_color": int(arrow.get("initial_color", arrow["color"])),
            }
            for arrow in arrows
        ],
        key=lambda arrow: (arrow["body"][0], arrow["body"][1]),
    )

    occupancy: dict[tuple[int, int], int] = {}
    current_colors: list[int] = []
    out = np.full(shape, bg, dtype=int)
    for idx, arrow in enumerate(ordered):
        coords = arrow["coords"]
        assert coords.shape == (12, 2)
        assert np.all((0 <= coords[:, 0]) & (coords[:, 0] < h) & (0 <= coords[:, 1]) & (coords[:, 1] < w))
        for rr, cc in coords.tolist():
            loc = (int(rr), int(cc))
            assert loc not in occupancy, "arrows must not overlap"
            occupancy[loc] = idx
        current_colors.append(int(arrow["color"]))
        out[coords[:, 0], coords[:, 1]] = int(arrow["color"])

    events: list[dict] = []
    for idx, arrow in enumerate(ordered):
        emit_color = int(current_colors[idx])
        changed_before_turn = emit_color != int(arrow["initial_color"])
        r, c = _ray_start(arrow["body"], arrow["direction"])
        dr, dc = DIR_TO_STEP[arrow["direction"]]
        hit_idx: int | None = None
        hit_changed = False
        ray_len = 0
        while 0 <= r < h and 0 <= c < w:
            target = occupancy.get((r, c))
            if target is not None and target != idx:
                hit_idx = int(target)
                hit_changed = int(current_colors[target]) != emit_color
                current_colors[target] = emit_color
                target_coords = ordered[target]["coords"]
                out[target_coords[:, 0], target_coords[:, 1]] = emit_color
                break
            out[r, c] = emit_color
            ray_len += 1
            r += dr
            c += dc
        events.append(
            {
                "emitter": idx,
                "hit": hit_idx,
                "miss": hit_idx is None,
                "ray_len": int(ray_len),
                "changed_before_turn": bool(changed_before_turn),
                "hit_changed": bool(hit_changed),
            }
        )
    return out, ordered, events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire one-pixel rays from arrow bodies so hit arrows are recolored and later fire in their new color.

    Input structure:
    - The background is black.
    - Each foreground object is one monochrome arrow made from a solid 3x3 square plus a straight 3-cell tail protruding from the middle of one side.
    - The tail side determines the arrow's pointing direction, and the arrows are separated from one another.

    Rule:
    - Sort the arrows by the top-left corner of their 3x3 bodies, from top to bottom then left to right.
    - Each arrow fires a 1-pixel-wide ray from the middle cell just beyond the body edge opposite its tail, using the arrow's current color.
    - A ray keeps painting until it reaches the grid boundary or the first pixel of another arrow; on a hit, the ray stops before the target and recolors the entire target arrow.
    - Arrow shapes and positions stay fixed, but an arrow that was recolored before its turn later fires using its updated color.

    Key insight:
    - The 3x3 body reveals where to sort and where to launch, while the tail reveals direction; because hits recolor intact targets instead of removing them, the processing order creates visible color-propagation chains.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    arrows = [_parse_arrow(comp) for comp in _find_components(g, bg)]
    assert arrows, "expected at least one arrow"
    out, _ordered, _events = _solve_from_arrows(arrows, g.shape, int(bg))
    return out


def _body_ranges(shape: tuple[int, int], direction: str) -> tuple[range, range]:
    h, w = shape
    if direction == "right":
        return range(1, h - 3), range(3, w - 3)
    if direction == "left":
        return range(1, h - 3), range(1, w - 5)
    if direction == "down":
        return range(3, h - 3), range(1, w - 3)
    if direction == "up":
        return range(1, h - 5), range(1, w - 3)
    raise AssertionError(f"unknown direction: {direction}")


def _expanded_cells(coords: np.ndarray) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in coords.tolist():
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded.add((rr + dr, cc + dc))
    return expanded


def _make_arrow(body: tuple[int, int], direction: str, color: int) -> dict:
    return {
        "body": tuple(map(int, body)),
        "direction": direction,
        "coords": _arrow_cells(int(body[0]), int(body[1]), direction),
        "color": int(color),
        "initial_color": int(color),
    }


def _can_place_arrow(
    shape: tuple[int, int],
    body: tuple[int, int],
    direction: str,
    blocked: set[tuple[int, int]],
    protected_rows: set[int],
    protected_cols: set[int],
) -> bool:
    row_range, col_range = _body_ranges(shape, direction)
    if body[0] not in row_range or body[1] not in col_range:
        return False
    coords = _arrow_cells(int(body[0]), int(body[1]), direction)
    h, w = shape
    if not np.all((0 <= coords[:, 0]) & (coords[:, 0] < h) & (0 <= coords[:, 1]) & (coords[:, 1] < w)):
        return False
    if any(int(r) in protected_rows or int(c) in protected_cols for r, c in coords.tolist()):
        return False
    return not any((int(r), int(c)) in blocked for r, c in coords.tolist())


def _register_arrow(
    arrows: list[dict],
    blocked: set[tuple[int, int]],
    body: tuple[int, int],
    direction: str,
    color: int,
) -> dict:
    arrow = _make_arrow(body, direction, color)
    arrows.append(arrow)
    blocked.update(_expanded_cells(arrow["coords"]))
    return arrow


def _build_chain(shape: tuple[int, int], arrows: list[dict], blocked: set[tuple[int, int]]) -> tuple[str, set[int], set[int]]:
    h, w = shape
    chain_colors = random.sample(ARROW_COLORS, 3)
    if random.random() < 0.5:
        body_r0 = random.randint(1, h - 4)
        c1 = random.randint(3, w - 4 - 2 * CHAIN_GAP_MIN)
        g1_max = min(CHAIN_GAP_MAX, w - 4 - c1 - CHAIN_GAP_MIN)
        gap1 = random.randint(CHAIN_GAP_MIN, g1_max)
        g2_max = min(CHAIN_GAP_MAX, w - 4 - c1 - gap1)
        gap2 = random.randint(CHAIN_GAP_MIN, g2_max)
        c2 = c1 + gap1
        c3 = c2 + gap2
        _register_arrow(arrows, blocked, (body_r0, c1), "right", chain_colors[0])
        _register_arrow(arrows, blocked, (body_r0, c2), "right", chain_colors[1])
        _register_arrow(arrows, blocked, (body_r0, c3), "right", chain_colors[2])
        return "horizontal", {body_r0 + 1}, set()

    body_c0 = random.randint(1, w - 4)
    r1 = random.randint(3, h - 4 - 2 * CHAIN_GAP_MIN)
    g1_max = min(CHAIN_GAP_MAX, h - 4 - r1 - CHAIN_GAP_MIN)
    gap1 = random.randint(CHAIN_GAP_MIN, g1_max)
    g2_max = min(CHAIN_GAP_MAX, h - 4 - r1 - gap1)
    gap2 = random.randint(CHAIN_GAP_MIN, g2_max)
    r2 = r1 + gap1
    r3 = r2 + gap2
    _register_arrow(arrows, blocked, (r1, body_c0), "down", chain_colors[0])
    _register_arrow(arrows, blocked, (r2, body_c0), "down", chain_colors[1])
    _register_arrow(arrows, blocked, (r3, body_c0), "down", chain_colors[2])
    return "vertical", set(), {body_c0 + 1}


def _place_miss_arrow(
    shape: tuple[int, int],
    axis: str,
    arrows: list[dict],
    blocked: set[tuple[int, int]],
    protected_rows: set[int],
    protected_cols: set[int],
) -> bool:
    h, w = shape
    color = random.choice(ARROW_COLORS)
    if axis == "horizontal":
        direction = random.choice(["left", "right"])
        candidates = [r for r in _body_ranges(shape, direction)[0] if r + 1 not in protected_rows]
        random.shuffle(candidates)
        for body_r0 in candidates:
            cols = list(_body_ranges(shape, direction)[1])
            random.shuffle(cols)
            for body_c0 in cols:
                body = (body_r0, body_c0)
                if _can_place_arrow(shape, body, direction, blocked, protected_rows, protected_cols):
                    _register_arrow(arrows, blocked, body, direction, color)
                    protected_rows.add(body_r0 + 1)
                    return True
        return False

    direction = random.choice(["up", "down"])
    candidates = [c for c in _body_ranges(shape, direction)[1] if c + 1 not in protected_cols]
    random.shuffle(candidates)
    for body_c0 in candidates:
        rows = list(_body_ranges(shape, direction)[0])
        random.shuffle(rows)
        for body_r0 in rows:
            body = (body_r0, body_c0)
            if _can_place_arrow(shape, body, direction, blocked, protected_rows, protected_cols):
                _register_arrow(arrows, blocked, body, direction, color)
                protected_cols.add(body_c0 + 1)
                return True
    return False


def _place_extra_arrows(
    shape: tuple[int, int],
    arrows: list[dict],
    blocked: set[tuple[int, int]],
    protected_rows: set[int],
    protected_cols: set[int],
    count: int,
) -> None:
    for _ in range(count):
        for _ in range(EXTRA_ARROW_TRIES):
            direction = random.choice(tuple(DIR_TO_STEP))
            rows = list(_body_ranges(shape, direction)[0])
            cols = list(_body_ranges(shape, direction)[1])
            body = (random.choice(rows), random.choice(cols))
            if not _can_place_arrow(shape, body, direction, blocked, protected_rows, protected_cols):
                continue
            _register_arrow(arrows, blocked, body, direction, random.choice(ARROW_COLORS))
            break


def _render_input(shape: tuple[int, int], arrows: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for arrow in arrows:
        coords = arrow["coords"]
        grid[coords[:, 0], coords[:, 1]] = int(arrow["color"])
    return grid


def _valid_generated_sample(grid: np.ndarray, arrows: list[dict]) -> bool:
    expected, ordered, events = _solve_from_arrows(arrows, grid.shape, BG)
    if sum(event["hit"] is not None for event in events) < 2:
        return False
    if not any(event["miss"] for event in events):
        return False
    if not any(event["changed_before_turn"] for event in events):
        return False
    if len({arrow["direction"] for arrow in ordered}) < 2:
        return False
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        shape = (random.randint(21, 29), random.randint(21, 29))
        arrows: list[dict] = []
        blocked: set[tuple[int, int]] = set()
        axis, protected_rows, protected_cols = _build_chain(shape, arrows, blocked)
        if not _place_miss_arrow(shape, axis, arrows, blocked, protected_rows, protected_cols):
            continue
        extra_count = random.randint(0, 2)
        _place_extra_arrows(shape, arrows, blocked, protected_rows, protected_cols, extra_count)
        grid = _render_input(shape, arrows)
        if _valid_generated_sample(grid, arrows):
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
