from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
BG_CHOICES = [4, 6, 7, 8]

MAX_GENERATION_ATTEMPTS = 400
MIN_SPRITES = 2
MAX_SPRITES = 4
LEFT_W_RANGE = (7, 14)
RIGHT_W_RANGE = (6, 10)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, f"expected one full-height grey separator, got {cols}"
    return int(cols[0])


def _background_color(grid: np.ndarray) -> int:
    colors = [int(x) for x in grid.flatten() if int(x) != GREY]
    return int(Counter(colors).most_common(1)[0][0])


def _slide_left_half(grid: np.ndarray, sep: int, bg: int) -> np.ndarray:
    left = grid[:, :sep]
    shifted = np.full_like(left, bg)
    for comp in _components4(left == BLACK):
        max_c = max(c for _, c in comp)
        delta = (sep - 1) - max_c
        assert delta >= 0
        for r, c in comp:
            shifted[r, c + delta] = BLACK
    return shifted


def _trigger_rows_from_slid_left(left: np.ndarray, bg: int) -> list[int]:
    h, w = left.shape
    trigger_col = w - 2
    rows: set[int] = set()
    for comp in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp):
            continue
        if any(c == trigger_col for _, c in comp):
            rows.update(r for r, _ in comp if _ == trigger_col)
    return sorted(rows)


def _control_map(grid: np.ndarray, sep: int) -> dict[int, tuple[int, int]]:
    coords = np.argwhere((grid == BLUE) | (grid == GREEN))
    row_map: dict[int, tuple[int, int]] = {}
    used_cols: set[int] = set()
    for r, c in coords:
        r = int(r)
        c = int(c)
        assert c > sep, "controls must be on the right half"
        assert r not in row_map, "controls must occupy unique rows"
        assert c not in used_cols, "controls must occupy unique columns"
        row_map[r] = (c, int(grid[r, c]))
        used_cols.add(c)
    return row_map


def _render_rays(base: np.ndarray, sep: int, bg: int, trigger_rows: list[int]) -> np.ndarray:
    out = base.copy()
    h, w = out.shape
    controls = _control_map(out, sep)
    blue_turns: list[tuple[int, int]] = []

    for r in trigger_rows:
        if r in controls:
            ctrl_c, ctrl_color = controls[r]
            for c in range(sep + 1, ctrl_c):
                if out[r, c] == bg:
                    out[r, c] = RED
            if ctrl_color == GREEN:
                for c in range(ctrl_c + 1, w):
                    if out[r, c] == bg:
                        out[r, c] = RED
            else:
                blue_turns.append((r, ctrl_c))
        else:
            for c in range(sep + 1, w):
                if out[r, c] == bg:
                    out[r, c] = RED

    for r, c in blue_turns:
        rr = r + 1
        while rr < h and out[rr, c] == bg:
            out[rr, c] = RED
            rr += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black shapes rightward to a grey separator, then route hole-fired red rays by control color.

    Input structure:
    - A full-height grey separator column splits the grid into a left shape pane and a right control pane.
    - The left pane contains several black 4-connected shapes on a uniform background color.
    - The right pane contains a few green or blue single-pixel controls, with no two sharing a row or a column.

    Rule:
    - Translate each black connected component horizontally until its rightmost cell is flush with the separator.
    - In the slid left pane, find enclosed background holes that touch the column immediately left of the separator-facing black wall.
    - Each row crossed by such a hole emits a red ray into the right pane: with no control on that row it goes straight to the right edge; with a green control it passes through and continues rightward; with a blue control it stops horizontally there and turns downward from the next cell, continuing until the bottom edge or the first non-background cell.
    - Controls keep their original colors, and red paint only fills background cells.

    Key insight:
    - The output depends on three linked decisions in order: sliding creates or repositions the hole, the hole chooses the emitting rows, and the encountered control color rewrites the ray path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator_col(g)
    bg = _background_color(g)

    out = g.copy()
    out[:, :sep] = _slide_left_half(g, sep, bg)
    trigger_rows = _trigger_rows_from_slid_left(out[:, :sep], bg)
    return _render_rays(out, sep, bg, trigger_rows)


def _scatter(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[random.randrange(slots)] += 1
    return parts


def _make_emitter_patch(height: int, width: int, bg: int) -> np.ndarray:
    patch = np.full((height, width), bg, dtype=int)
    left_wall = random.randint(0, max(0, width - 3))
    patch[0, left_wall:] = BLACK
    patch[-1, left_wall:] = BLACK
    patch[:, left_wall] = BLACK
    patch[:, -1] = BLACK

    if left_wall > 0:
        arm_rows = random.sample(range(1, height - 1), k=random.randint(1, min(2, height - 2)))
        for r in arm_rows:
            patch[r, : left_wall + 1] = BLACK
        if random.random() < 0.5:
            r0 = random.randint(1, height - 2)
            r1 = random.randint(r0, height - 2)
            patch[r0 : r1 + 1, 0] = BLACK
    return patch


def _make_distractor_patch(kind: str, height: int, width: int, bg: int) -> np.ndarray:
    patch = np.full((height, width), bg, dtype=int)
    if kind == "solid":
        patch[:, :] = BLACK
    elif kind == "c_open_left":
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[:, -1] = BLACK
    elif kind == "c_open_right":
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[:, 0] = BLACK
    else:
        patch[-1, :] = BLACK
        patch[:, -1] = BLACK
        if random.random() < 0.5:
            patch[0, -1] = BLACK
    return patch


def _sample_sprite_specs(left_w: int) -> list[dict]:
    n_sprites = random.randint(MIN_SPRITES, MAX_SPRITES)
    emitter_count = 2 if n_sprites >= 2 and random.random() < 0.6 else 1
    emitter_count = min(emitter_count, n_sprites)

    specs: list[dict] = []
    for idx in range(emitter_count):
        trigger_rows = random.randint(3, 5) if idx == 0 else random.randint(2, 4)
        height = trigger_rows + 2
        width = random.randint(3, min(6, left_w - 1))
        specs.append({"kind": "emitter", "height": height, "width": width})

    distractor_kinds = ["solid", "c_open_left", "c_open_right", "l_shape"]
    for _ in range(n_sprites - emitter_count):
        kind = random.choice(distractor_kinds)
        if kind == "solid":
            height = random.randint(2, 4)
            width = random.randint(2, min(5, left_w - 1))
        else:
            height = random.randint(3, 5)
            width = random.randint(3, min(5, left_w - 1))
        specs.append({"kind": kind, "height": height, "width": width})

    random.shuffle(specs)
    return specs


def _place_output_sprites(height: int, sep: int, bg: int, specs: list[dict]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((height, sep), bg, dtype=int)
    hs = [spec["height"] for spec in specs]
    base_gaps = [0] + [1] * (len(specs) - 1) + [0]
    min_height = sum(hs) + sum(base_gaps)
    extra = height - min_height
    gaps = [a + b for a, b in zip(base_gaps, _scatter(extra, len(base_gaps)))]

    sprites: list[dict] = []
    row = gaps[0]
    for idx, spec in enumerate(specs):
        sh = spec["height"]
        sw = spec["width"]
        if spec["kind"] == "emitter":
            patch = _make_emitter_patch(sh, sw, bg)
        else:
            patch = _make_distractor_patch(spec["kind"], sh, sw, bg)
        c0 = sep - sw
        grid[row : row + sh, c0:sep][patch == BLACK] = BLACK
        trigger_rows = [row + rr for rr in range(1, sh - 1)] if spec["kind"] == "emitter" else []
        sprites.append(
            {
                "kind": spec["kind"],
                "patch": patch,
                "row": row,
                "col": c0,
                "trigger_rows": trigger_rows,
            }
        )
        row += sh + gaps[idx + 1]
    return grid, sprites


def _choose_row_roles(sprites: list[dict]) -> tuple[dict[int, str], bool]:
    emitter_groups = [sprite["trigger_rows"] for sprite in sprites if sprite["kind"] == "emitter"]
    emitter_groups = [rows for rows in emitter_groups if rows]
    trigger_rows = sorted(r for rows in emitter_groups for r in rows)
    assert len(trigger_rows) >= 3

    block_vertical = len(emitter_groups) >= 2 and random.random() < 0.5
    roles: dict[int, str] = {r: "none" for r in trigger_rows}

    if block_vertical:
        upper_rows = emitter_groups[0]
        lower_rows = emitter_groups[1]
        blue_row = upper_rows[-1]
        roles[blue_row] = "blue"
        blocked_target = lower_rows[0]
        assert blocked_target - blue_row >= 3
    else:
        blue_row = trigger_rows[-1]
        roles[blue_row] = "blue"

    green_candidates = [r for r in trigger_rows if r != blue_row]
    green_row = random.choice(green_candidates)
    roles[green_row] = "green"

    extra_green_candidates = [r for r in trigger_rows if roles[r] == "none"]
    if len(extra_green_candidates) >= 2 and random.random() < 0.5:
        roles[random.choice(extra_green_candidates)] = "green"

    assert any(role == "none" for role in roles.values())
    return roles, block_vertical


def _build_controls(
    height: int,
    width: int,
    sep: int,
    roles: dict[int, str],
) -> dict[int, tuple[int, int]]:
    trigger_rows = sorted(roles)
    controlled_rows = [r for r in trigger_rows if roles[r] in {"blue", "green"}]
    non_trigger_rows = [r for r in range(height) if r not in roles]

    distractor_count = min(random.randint(1, 2), max(0, (width - sep - 1) - len(controlled_rows)))
    distractor_rows = random.sample(non_trigger_rows, k=distractor_count) if distractor_count else []

    available_cols = list(range(sep + 1, width))
    random.shuffle(available_cols)
    controls: dict[int, tuple[int, int]] = {}

    def take_column(prefer_not_edge: bool) -> int:
        if prefer_not_edge:
            for i, col in enumerate(available_cols):
                if col < width - 1:
                    return available_cols.pop(i)
        return available_cols.pop()

    for r in controlled_rows:
        role = roles[r]
        col = take_column(prefer_not_edge=True)
        controls[r] = (col, BLUE if role == "blue" else GREEN)

    for r in distractor_rows:
        col = take_column(prefer_not_edge=False)
        controls[r] = (col, random.choice([BLUE, GREEN]))

    assert len({r for r in controls}) == len(controls)
    assert len({c for c, _ in controls.values()}) == len(controls)
    return controls


def _compose_input(
    left_output: np.ndarray,
    sprites: list[dict],
    sep: int,
    bg: int,
    controls: dict[int, tuple[int, int]],
) -> np.ndarray:
    h, left_w = left_output.shape
    w = left_w + 1 + (max(c for c, _ in controls.values()) - sep if controls else 6)
    w = max(w, sep + 1 + 6)
    grid = np.full((h, w), bg, dtype=int)
    grid[:, sep] = GREY

    zero_delta_budget = 1 if random.random() < 0.35 else 0
    for sprite in sprites:
        patch = sprite["patch"]
        r0 = sprite["row"]
        c0 = sprite["col"]
        max_delta = c0
        assert max_delta >= 1
        delta = random.randint(1, max_delta)
        if zero_delta_budget and random.random() < 0.5:
            delta = 0
            zero_delta_budget -= 1
        target_c0 = c0 - delta
        sub = grid[r0 : r0 + patch.shape[0], target_c0 : target_c0 + patch.shape[1]]
        sub[patch == BLACK] = BLACK

    for r, (c, color) in controls.items():
        grid[r, c] = color
    return grid


def _build_attempt() -> tuple[np.ndarray, dict] | None:
    bg = random.choice(BG_CHOICES)
    left_w = random.randint(*LEFT_W_RANGE)
    right_w = random.randint(*RIGHT_W_RANGE)
    width = left_w + 1 + right_w
    if width > 30:
        return None

    sep = left_w
    specs = _sample_sprite_specs(left_w)
    min_height = sum(spec["height"] for spec in specs) + max(0, len(specs) - 1)
    if min_height > 26:
        return None
    height = random.randint(min_height, min(26, min_height + 6))

    left_output, sprites = _place_output_sprites(height, sep, bg, specs)
    roles, block_vertical = _choose_row_roles(sprites)
    controls = _build_controls(height, width, sep, roles)

    grid = np.full((height, width), bg, dtype=int)
    grid[:, sep] = GREY
    grid[:, :sep] = left_output
    for r, (c, color) in controls.items():
        grid[r, c] = color

    input_grid = np.full((height, width), bg, dtype=int)
    input_grid[:, sep] = GREY
    zero_delta_budget = 1 if random.random() < 0.35 else 0
    shifted = False
    for sprite in sprites:
        patch = sprite["patch"]
        r0 = sprite["row"]
        c0 = sprite["col"]
        max_delta = c0
        assert max_delta >= 1
        delta = random.randint(1, max_delta)
        if zero_delta_budget and random.random() < 0.5:
            delta = 0
            zero_delta_budget -= 1
        shifted |= delta > 0
        target_c0 = c0 - delta
        sub = input_grid[r0 : r0 + patch.shape[0], target_c0 : target_c0 + patch.shape[1]]
        sub[patch == BLACK] = BLACK

    if not shifted:
        return None
    for r, (c, color) in controls.items():
        input_grid[r, c] = color

    meta = {
        "bg": bg,
        "sep": sep,
        "roles": roles,
        "controls": controls,
        "block_vertical": block_vertical,
    }
    return input_grid, meta


def _validate_attempt(inp: np.ndarray, meta: dict) -> bool:
    sep = meta["sep"]
    bg = meta["bg"]
    roles = meta["roles"]
    controls = meta["controls"]
    block_vertical = meta["block_vertical"]

    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if not np.all(inp[:, sep] == GREY):
        return False
    if Counter(int(x) for x in inp.flatten() if int(x) != GREY).most_common(1)[0][0] != bg:
        return False
    if len({r for r in controls}) != len(controls) or len({c for c, _ in controls.values()}) != len(controls):
        return False

    slid_left = _slide_left_half(inp, sep, bg)
    trigger_rows = _trigger_rows_from_slid_left(slid_left, bg)
    if sorted(trigger_rows) != sorted(roles):
        return False
    if not any(role == "blue" for role in roles.values()):
        return False
    if not any(role == "green" for role in roles.values()):
        return False
    if not any(role == "none" for role in roles.values()):
        return False

    out = main(inp)
    if np.array_equal(out, inp):
        return False
    if np.count_nonzero(out == RED) == 0:
        return False

    for row, role in roles.items():
        if role == "none":
            if not np.all(out[row, sep + 1 :] == RED):
                return False
        elif role == "green":
            col, _ = controls[row]
            if out[row, col] != GREEN:
                return False
            if col == sep + 1 or col == out.shape[1] - 1:
                return False
            if not np.all(out[row, sep + 1 : col] == RED):
                return False
            if not np.all(out[row, col + 1 :] == RED):
                return False
        else:
            col, _ = controls[row]
            if out[row, col] != BLUE:
                return False
            if not np.all(out[row, sep + 1 : col] == RED):
                return False
            if np.any(out[row, col + 1 :] == RED):
                return False
            below = out[row + 1 :, col] if row + 1 < out.shape[0] else np.array([], dtype=int)
            if below.size == 0 or np.count_nonzero(below == RED) == 0:
                return False
            if block_vertical:
                if below[-1] == RED:
                    return False
            else:
                if out[-1, col] != RED:
                    return False

    distractor_rows = [r for r in controls if r not in roles]
    for row in distractor_rows:
        if np.count_nonzero(out[row] == controls[row][1]) != 1:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        inp, meta = built
        if _validate_attempt(inp, meta):
            return inp
    raise RuntimeError("failed to generate a valid valve-and-turn input")
