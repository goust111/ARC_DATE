from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
TEMPLATE_SIZE = 3
STENCIL_SIZE = 4
FRAME_PAD = 1
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_MIN = 17
GRID_MAX = 22
COPY_COUNT_RANGE = (3, 5)
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _copy_frame_box(r: int, c: int) -> tuple[int, int, int, int]:
    return (r - 1, c - 1, r + TEMPLATE_SIZE, c + TEMPLATE_SIZE)


def _draw_frame(grid: np.ndarray, r: int, c: int, color: int) -> None:
    assert r > 0 and c > 0
    assert r + TEMPLATE_SIZE < grid.shape[0] and c + TEMPLATE_SIZE < grid.shape[1]
    grid[r - 1 : r + TEMPLATE_SIZE + 1, c - 1] = int(color)
    grid[r - 1 : r + TEMPLATE_SIZE + 1, c + TEMPLATE_SIZE] = int(color)
    grid[r - 1, c - 1 : c + TEMPLATE_SIZE + 1] = int(color)
    grid[r + TEMPLATE_SIZE, c - 1 : c + TEMPLATE_SIZE + 1] = int(color)


def _find_template_stencil(grid: np.ndarray) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(h - TEMPLATE_SIZE):
        for c in range(w - TEMPLATE_SIZE):
            if np.all(grid[r : r + TEMPLATE_SIZE, c + TEMPLATE_SIZE] == GREY) and np.all(
                grid[r + TEMPLATE_SIZE, c : c + STENCIL_SIZE] == GREY
            ):
                candidates.append((r, c))
    assert len(candidates) == 1, "expected exactly one grey L-stencil"
    return candidates[0]


def _template_colors(template: np.ndarray) -> list[int]:
    colors = [int(v) for v in np.unique(template) if int(v) not in (BG, GREY)]
    assert len(colors) == 2, "template must use exactly two non-background colors"
    assert not np.any(template == BG), "template must be fully colored"
    return colors


def _ordered_template_colors(template: np.ndarray) -> list[int]:
    colors = _template_colors(template)
    flat = template.ravel()
    counts = {color: int(np.count_nonzero(template == color)) for color in colors}
    first_idx = {color: int(np.flatnonzero(flat == color)[0]) for color in colors}
    return sorted(colors, key=lambda color: (counts[color], first_idx[color]))


def _classify_window(window: np.ndarray, template: np.ndarray, bg: int) -> dict | None:
    if np.any(window == GREY):
        return None

    mismatches = window != template
    mismatch_count = int(mismatches.sum())
    if mismatch_count == 0:
        return {"kind": "exact"}
    if not 1 <= mismatch_count <= 3:
        return None

    intrusion_values = [int(v) for v in window[mismatches]]
    intrusion = intrusion_values[0]
    if intrusion in _template_colors(template) or intrusion in (bg, GREY):
        return None
    if any(value != intrusion for value in intrusion_values):
        return None

    return {
        "kind": "damaged",
        "intrusion": intrusion,
        "damage_mask": mismatches.copy(),
    }


def _scan_copies(
    grid: np.ndarray, template: np.ndarray, source: tuple[int, int]
) -> tuple[list[tuple[int, int]], list[dict]]:
    exact: list[tuple[int, int]] = []
    damaged: list[dict] = []
    bg = _get_bg(grid)
    h, w = grid.shape
    for r in range(h - TEMPLATE_SIZE + 1):
        for c in range(w - TEMPLATE_SIZE + 1):
            if (r, c) == source:
                continue
            info = _classify_window(grid[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE], template, bg)
            if info is None:
                continue
            if info["kind"] == "exact":
                exact.append((r, c))
            else:
                damaged.append(
                    {
                        "pos": (r, c),
                        "intrusion": int(info["intrusion"]),
                        "damage_mask": np.asarray(info["damage_mask"], dtype=bool),
                    }
                )
    return exact, damaged


def _swap_and_repair(
    patch: np.ndarray, template: np.ndarray, damage_mask: np.ndarray, intrusion: int
) -> np.ndarray:
    out = np.array(patch, dtype=int, copy=True)
    for color in _ordered_template_colors(template):
        if not np.any(damage_mask & (template == color)):
            continue
        intrusion_mask = out == int(intrusion)
        color_mask = out == int(color)
        out[intrusion_mask] = -1
        out[color_mask] = int(intrusion)
        out[out == -1] = int(color)

    out[damage_mask] = template[damage_mask]
    return out


def _apply_transform(grid: np.ndarray, template: np.ndarray, damaged: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for item in damaged:
        r, c = item["pos"]
        patch = out[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE]
        out[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE] = _swap_and_repair(
            patch, template, item["damage_mask"], int(item["intrusion"])
        )
        _draw_frame(out, r, c, int(item["intrusion"]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair every partially matched copy of the marked 3x3 template by swapping the intrusion color with the affected template colors, then frame only the repaired copies.

    Input structure:
    - The background is black, and a fully colored 3x3 template sits next to a grey L-stencil:
      a grey column immediately on its right and a grey row immediately below it.
    - Elsewhere in the grid are several separated 3x3 copies of that template with at least
      one background cell of clearance for a possible outer frame.
    - Some copies are exact, while damaged copies differ from the template in 1-3 cells, and
      all mismatching cells inside one damaged copy share the same non-template intrusion color.

    Rule:
    - Read the template from the 3x3 region at the upper-left of the unique grey L-stencil.
    - Leave every exact copy unchanged.
    - For each damaged copy, collect the template colors that were overwritten by the intrusion color.
      Process those affected template colors from rarer to more common within the template; for each
      color, swap that color and the intrusion color across the whole 3x3 copy.
    - After the swaps, restore the originally damaged cells to their template colors, then draw a
      one-pixel-thick outer rectangle around that 3x3 copy using the intrusion color.

    Key insight:
    The damage is repaired by a bidirectional color exchange, not by a one-way overwrite, so repaired
    copies keep a visible intrusion-colored trace and only those repaired copies receive a matching frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert _get_bg(g) == BG, "expected black background"

    source = _find_template_stencil(g)
    sr, sc = source
    template = g[sr : sr + TEMPLATE_SIZE, sc : sc + TEMPLATE_SIZE]
    exact, damaged = _scan_copies(g, template, source)

    assert exact, "expected at least one intact copy"
    assert damaged, "expected at least one damaged copy"
    intrusion_colors = {int(item["intrusion"]) for item in damaged}
    assert len(intrusion_colors) == 1, "all damaged copies must share one intrusion color"

    for item in damaged:
        r, c = item["pos"]
        assert r > 0 and c > 0
        assert r + TEMPLATE_SIZE < g.shape[0] and c + TEMPLATE_SIZE < g.shape[1]

    return _apply_transform(g, template, damaged)


def _random_template() -> np.ndarray:
    while True:
        minority_color, majority_color = random.sample(PALETTE, 2)
        minority_count = random.randint(2, 4)
        minority_cells = random.sample(range(TEMPLATE_SIZE * TEMPLATE_SIZE), minority_count)
        rows = {cell // TEMPLATE_SIZE for cell in minority_cells}
        cols = {cell % TEMPLATE_SIZE for cell in minority_cells}
        if len(rows) < 2 or len(cols) < 2:
            continue
        flat = np.full(TEMPLATE_SIZE * TEMPLATE_SIZE, majority_color, dtype=int)
        for cell in minority_cells:
            flat[cell] = minority_color
        return flat.reshape((TEMPLATE_SIZE, TEMPLATE_SIZE))


def _pick_positions(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [
        (r, c)
        for r in range(1, h - TEMPLATE_SIZE)
        for c in range(1, w - TEMPLATE_SIZE)
        if not _rects_overlap(_copy_frame_box(r, c), (0, 0, 4, 4))
    ]
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = [(0, 0, 4, 4)]
    for r, c in candidates:
        box = _copy_frame_box(r, c)
        if any(_rects_overlap(box, other) for other in boxes):
            continue
        chosen.append((r, c))
        boxes.append(box)
        if len(chosen) == count:
            return chosen
    return None


def _damage_mask_for_mode(template: np.ndarray, mode: str) -> np.ndarray:
    colors = _ordered_template_colors(template)
    coords_by_color = {
        color: [tuple(map(int, rc)) for rc in np.argwhere(template == color)] for color in colors
    }
    damage = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=bool)

    if mode == "both":
        primary = [random.choice(coords_by_color[colors[0]]), random.choice(coords_by_color[colors[1]])]
        chosen = set(primary)
        all_cells = [cell for cells in coords_by_color.values() for cell in cells if cell not in chosen]
        extra_count = random.randint(0, min(1, len(all_cells)))
        if extra_count:
            chosen.update(random.sample(all_cells, extra_count))
    else:
        color = random.choice(colors)
        max_count = min(3, len(coords_by_color[color]))
        chosen = set(random.sample(coords_by_color[color], random.randint(1, max_count)))

    for r, c in chosen:
        damage[r, c] = True
    assert 1 <= int(damage.sum()) <= 3
    return damage


def _build_damaged_patch(template: np.ndarray, damage_mask: np.ndarray, intrusion: int) -> np.ndarray:
    patch = np.array(template, dtype=int, copy=True)
    patch[damage_mask] = int(intrusion)
    return patch


def _expected_from_spec(
    grid: np.ndarray, template: np.ndarray, damaged_specs: list[dict]
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for item in damaged_specs:
        r, c = item["pos"]
        out[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE] = _swap_and_repair(
            out[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE],
            template,
            item["damage_mask"],
            int(item["intrusion"]),
        )
        _draw_frame(out, r, c, int(item["intrusion"]))
    return out


def _validate_generated_case(
    grid: np.ndarray,
    template: np.ndarray,
    intact_positions: list[tuple[int, int]],
    damaged_specs: list[dict],
    intrusion: int,
) -> bool:
    source = _find_template_stencil(grid)
    if source != (0, 0):
        return False

    found_template = grid[:TEMPLATE_SIZE, :TEMPLATE_SIZE]
    if not np.array_equal(found_template, template):
        return False

    exact, damaged = _scan_copies(grid, template, source)
    if sorted(exact) != sorted(intact_positions):
        return False

    found_by_pos = {tuple(item["pos"]): item for item in damaged}
    expected_by_pos = {tuple(item["pos"]): item for item in damaged_specs}
    if set(found_by_pos) != set(expected_by_pos):
        return False

    for pos, expected in expected_by_pos.items():
        found = found_by_pos[pos]
        if int(found["intrusion"]) != int(intrusion):
            return False
        if not np.array_equal(found["damage_mask"], expected["damage_mask"]):
            return False

    intrusion_mask = np.zeros(grid.shape, dtype=bool)
    for item in damaged_specs:
        r, c = item["pos"]
        intrusion_mask[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE] |= item["damage_mask"]
    if not np.array_equal(grid == int(intrusion), intrusion_mask):
        return False

    expected = _expected_from_spec(grid, template, damaged_specs)
    actual = main(grid)
    return np.array_equal(actual, expected)


def _generate_case() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    template = _random_template()
    template_colors = _template_colors(template)
    intrusion = random.choice([color for color in PALETTE if color not in template_colors])

    copy_count = random.randint(*COPY_COUNT_RANGE)
    intact_count = random.randint(1, copy_count - 1)
    damaged_count = copy_count - intact_count
    positions = _pick_positions(h, w, copy_count)
    if positions is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[:TEMPLATE_SIZE, :TEMPLATE_SIZE] = template
    grid[:TEMPLATE_SIZE, TEMPLATE_SIZE] = GREY
    grid[TEMPLATE_SIZE, :STENCIL_SIZE] = GREY

    random.shuffle(positions)
    damaged_positions = positions[:damaged_count]
    intact_positions = positions[damaged_count:]

    force_both = random.random() < 0.6
    damaged_specs: list[dict] = []
    for index, (r, c) in enumerate(damaged_positions):
        if force_both and index == 0:
            mode = "both"
        else:
            mode = "both" if random.random() < 0.35 else "single"
        damage_mask = _damage_mask_for_mode(template, mode)
        grid[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE] = _build_damaged_patch(
            template, damage_mask, intrusion
        )
        damaged_specs.append(
            {
                "pos": (r, c),
                "intrusion": int(intrusion),
                "damage_mask": damage_mask,
            }
        )

    for r, c in intact_positions:
        grid[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE] = template

    if not _validate_generated_case(grid, template, intact_positions, damaged_specs, intrusion):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
