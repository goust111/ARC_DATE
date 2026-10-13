from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SLOT_COUNT = 8
STEP_NAMES = ("id", "rot90", "rot180", "rot270", "flip_h", "flip_v")
CHAIN_TEMPLATES = {
    3: (
        {
            "steps": (0, 1, 2, 0, 1, 2, 0),
            "filled": (True, True, True, True, False, False, True, False),
        },
        {
            "steps": (0, 1, 2, 1, 0, 2, 1),
            "filled": (True, True, True, True, False, True, False, False),
        },
    ),
    4: (
        {
            "steps": (0, 1, 2, 3, 0, 1, 2),
            "filled": (True, True, True, True, True, False, False, True),
        },
        {
            "steps": (0, 1, 2, 3, 1, 0, 2),
            "filled": (True, True, True, True, True, False, False, True),
        },
    ),
}
MAX_GENERATION_ATTEMPTS = 400


def _apply_step(tile: np.ndarray, name: str) -> np.ndarray:
    if name == "id":
        return tile.copy()
    if name == "rot90":
        return np.rot90(tile, -1).copy()
    if name == "rot180":
        return np.rot90(tile, 2).copy()
    if name == "rot270":
        return np.rot90(tile, 1).copy()
    if name == "flip_h":
        return np.fliplr(tile).copy()
    if name == "flip_v":
        return np.flipud(tile).copy()
    raise ValueError(f"unknown transform: {name}")


def _all_d4_images(tile: np.ndarray) -> list[np.ndarray]:
    return [
        tile.copy(),
        np.rot90(tile, -1).copy(),
        np.rot90(tile, 2).copy(),
        np.rot90(tile, 1).copy(),
        np.fliplr(tile).copy(),
        np.flipud(tile).copy(),
        tile.T.copy(),
        np.rot90(tile.T, 2).copy(),
    ]


def _sig(tile: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, tile.shape)), tuple(map(int, tile.ravel()))


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _components_by_color(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((color, cells))
    return comps


def _find_slots(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    slots: list[dict[str, object]] = []
    for color, cells in _components_by_color(grid, bg):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w or side_h < 4:
            continue
        if len(cells) != 4 * side_h - 4:
            continue

        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(region[0, :] == color)
            and np.all(region[-1, :] == color)
            and np.all(region[:, 0] == color)
            and np.all(region[:, -1] == color)
        ):
            continue
        if np.any(region[1:-1, 1:-1] == color):
            continue

        inner = region[1:-1, 1:-1].copy()
        slots.append(
            {
                "bbox": (r0, c0, r1, c1),
                "border": int(color),
                "tile": inner,
                "filled": bool(np.any(inner != bg)),
            }
        )

    assert slots, "expected at least one framed slot"
    slots.sort(key=lambda slot: (slot["bbox"][0], slot["bbox"][1]))
    side = slots[0]["tile"].shape
    assert all(slot["tile"].shape == side for slot in slots), "all slots must share one tile size"
    return slots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a row-major chain of framed slots by bootstrapping each border color's transform from filled neighbors.

    Input structure:
    - The canvas contains several equal square slots with one-cell colored frames, ordered left-to-right then top-to-bottom.
    - The first slot contains a non-symmetric interior tile; later slots have border colors drawn from a small codebook and are either filled with a tile or left blank.
    - A blank slot is an all-background interior, while filled interiors keep their colors and the frame geometry is unchanged.

    Rule:
    - Treat slot 1 as the chain seed. For every later border color, infer its fixed transform by finding an adjacent pair of already filled slots whose second slot has that border color.
    - The allowed per-step transforms are identity, 90/180/270 degree rotation, horizontal mirror, and vertical mirror.
    - Walk the chain in order: each slot must equal the previous slot's interior after applying the transform named by that slot's border color; fill blank slots with that result and keep filled slots as consistency checks.

    Key insight:
    - The border colors do not reference the first tile independently; they define a chained transform codebook, so consecutive blanks must be propagated step by step after the codebook is bootstrapped from filled neighboring pairs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    slots = _find_slots(g, bg)
    assert len(slots) >= 2, "expected a chain with at least two slots"
    assert slots[0]["filled"], "the chain head must be filled"

    color_to_step: dict[int, str] = {}
    for idx in range(1, len(slots)):
        if not (slots[idx - 1]["filled"] and slots[idx]["filled"]):
            continue
        prev_tile = slots[idx - 1]["tile"]
        next_tile = slots[idx]["tile"]
        matches = [
            name
            for name in STEP_NAMES
            if np.array_equal(_apply_step(prev_tile, name), next_tile)
        ]
        assert len(matches) == 1, "each filled neighboring pair must identify one transform"
        border = int(slots[idx]["border"])
        if border in color_to_step:
            assert color_to_step[border] == matches[0], "border color transform must stay consistent"
        else:
            color_to_step[border] = matches[0]

    required_colors = {int(slot["border"]) for slot in slots[1:]}
    assert required_colors <= color_to_step.keys(), "every later border color must be inferable"

    out = g.copy()
    current = slots[0]["tile"]
    for idx in range(1, len(slots)):
        border = int(slots[idx]["border"])
        expected = _apply_step(current, color_to_step[border])
        r0, c0, r1, c1 = slots[idx]["bbox"]
        if slots[idx]["filled"]:
            assert np.array_equal(slots[idx]["tile"], expected), "filled slot breaks the transform chain"
        else:
            out[r0 + 1 : r1, c0 + 1 : c1] = expected
        current = expected
    return out


def _random_base_tile(side: int, colors: list[int]) -> np.ndarray | None:
    area = side * side
    target_non_bg = random.randint(area // 3, (2 * area) // 3)

    tile = np.full((side, side), BG, dtype=int)
    all_cells = [(r, c) for r in range(side) for c in range(side)]
    forced = random.sample(all_cells, len(colors))
    used = set(forced)
    for (r, c), color in zip(forced, colors):
        tile[r, c] = color

    remaining = [cell for cell in all_cells if cell not in used]
    extra_count = max(0, target_non_bg - len(colors))
    for r, c in random.sample(remaining, extra_count):
        tile[r, c] = random.choice(colors)

    if not np.any(tile == BG) or np.count_nonzero(tile != BG) < len(colors) + 2:
        return None
    if len({_sig(image) for image in _all_d4_images(tile)}) != 8:
        return None
    return tile


def _simulate_states(seed: np.ndarray, step_names: list[str], step_refs: tuple[int, ...]) -> list[np.ndarray]:
    states = [seed.copy()]
    current = seed.copy()
    for ref in step_refs:
        current = _apply_step(current, step_names[ref])
        states.append(current)
    return states


def _has_multi_step_requirement(seed: np.ndarray, states: list[np.ndarray], filled: tuple[bool, ...]) -> bool:
    single_step_sigs = {_sig(_apply_step(seed, name)) for name in STEP_NAMES}
    empty_indices = [idx for idx, is_filled in enumerate(filled) if not is_filled]
    return any(_sig(states[idx]) not in single_step_sigs for idx in empty_indices)


def _has_consecutive_empty_run(filled: tuple[bool, ...]) -> bool:
    return any((not filled[idx]) and (not filled[idx + 1]) for idx in range(len(filled) - 1))


def _choose_chain(
    seed: np.ndarray, code_count: int
) -> tuple[list[str], tuple[int, ...], tuple[bool, ...], list[np.ndarray]] | None:
    template = random.choice(CHAIN_TEMPLATES[code_count])
    step_refs = template["steps"]
    filled = template["filled"]

    for _ in range(300):
        step_names = random.sample(list(STEP_NAMES), code_count)
        states = _simulate_states(seed, step_names, step_refs)
        if len({_sig(state) for state in states}) != len(states):
            continue
        if not _has_consecutive_empty_run(filled):
            continue
        if not _has_multi_step_requirement(seed, states, filled):
            continue
        first_empty = min(idx for idx, is_filled in enumerate(filled) if not is_filled)
        if not any(filled[idx] for idx in range(first_empty + 1, len(filled))):
            continue
        return step_names, step_refs, filled, states
    return None


def _slot_positions(outer: int, gap: int) -> list[tuple[int, int]]:
    positions = []
    for idx in range(SLOT_COUNT):
        row, col = divmod(idx, 3)
        positions.append((row * (outer + gap), col * (outer + gap)))
    return positions


def _render_grid(
    states: list[np.ndarray],
    filled: tuple[bool, ...],
    border_colors: list[int],
    gap: int,
    margins: tuple[int, int, int, int],
) -> np.ndarray:
    inner = int(states[0].shape[0])
    outer = inner + 2
    top_margin, left_margin, bottom_margin, right_margin = margins
    height = top_margin + 3 * outer + 2 * gap + bottom_margin
    width = left_margin + 3 * outer + 2 * gap + right_margin
    grid = np.full((height, width), BG, dtype=int)

    for idx, (r, c) in enumerate(_slot_positions(outer, gap)):
        r0 = top_margin + r
        c0 = left_margin + c
        color = border_colors[idx]
        grid[r0 : r0 + outer, c0] = color
        grid[r0 : r0 + outer, c0 + outer - 1] = color
        grid[r0, c0 : c0 + outer] = color
        grid[r0 + outer - 1, c0 : c0 + outer] = color
        if filled[idx]:
            grid[r0 + 1 : r0 + outer - 1, c0 + 1 : c0 + outer - 1] = states[idx]
    return grid


def _expected_border_map(code_colors: list[int], step_names: list[str]) -> dict[int, str]:
    return {int(color): step_names[idx] for idx, color in enumerate(code_colors)}


def _validate_instance(inp: np.ndarray, out: np.ndarray, expected_map: dict[int, str], filled: tuple[bool, ...]) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, out):
        return False

    bg = _get_bg(inp)
    slots = _find_slots(inp, bg)
    if len(slots) != SLOT_COUNT or not slots[0]["filled"]:
        return False

    empties = [idx for idx, slot in enumerate(slots) if not slot["filled"]]
    if sorted(empties) != [idx for idx, is_filled in enumerate(filled) if not is_filled]:
        return False

    inferred: dict[int, str] = {}
    for idx in range(1, len(slots)):
        if not (slots[idx - 1]["filled"] and slots[idx]["filled"]):
            continue
        prev_tile = slots[idx - 1]["tile"]
        next_tile = slots[idx]["tile"]
        names = [name for name in STEP_NAMES if np.array_equal(_apply_step(prev_tile, name), next_tile)]
        if len(names) != 1:
            return False
        inferred[int(slots[idx]["border"])] = names[0]

    for color, step in expected_map.items():
        if inferred.get(color) != step:
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    code_count = random.choice((3, 4))
    palette = random.sample(range(1, 10), code_count + 1 + random.choice((2, 3)))
    head_color = palette[0]
    code_colors = palette[1 : 1 + code_count]
    tile_colors = palette[1 + code_count :]
    side = random.randint(4, 6)
    seed = _random_base_tile(side, tile_colors)
    if seed is None:
        return None

    chain = _choose_chain(seed, code_count)
    if chain is None:
        return None
    step_names, step_refs, filled, states = chain
    if len(step_names) != code_count:
        return None

    gap = random.randint(1, 2)
    margins = tuple(random.randint(0, 1) for _ in range(4))
    border_colors = [head_color] + [code_colors[ref] for ref in step_refs]

    inp = _render_grid(states, filled, border_colors, gap, margins)
    out = _render_grid(states, tuple(True for _ in range(SLOT_COUNT)), border_colors, gap, margins)
    expected_map = _expected_border_map(code_colors, step_names)

    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    if not _validate_instance(inp, out, expected_map, filled):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid transform-chain puzzle")
