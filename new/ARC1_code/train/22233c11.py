from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_h, bbox_w, colors_present, find_components, get_bg, paste
import random
from itertools import product


def _paste_teal_block(out: np.ndarray, r: int, c: int, size: int, bg: int) -> None:
    H, W = out.shape
    rr0 = max(0, int(r))
    cc0 = max(0, int(c))
    rr1 = min(H, int(r) + int(size))
    cc1 = min(W, int(c) + int(size))
    if rr0 < rr1 and cc0 < cc1:
        assert np.all(out[rr0:rr1, cc0:cc1] == int(bg))
    patch = np.full((int(size), int(size)), Color.TEAL, dtype=int)
    paste(out, patch, int(r), int(c), clip=True)


def main(grid: np.ndarray) -> np.ndarray:
    """Paint teal square blocks at the two outward opposite corners of each diagonal foreground motif.

    Input structure
    - The canvas has one background color and one foreground color.
    - Each 8-connected foreground object has a square bounding box split into a 2-by-2 grid of equal square blocks.
    - Exactly two diagonal blocks of that footprint are filled with the foreground color; the other two are background.

    Rule
    - Preserve the input.
    - If the filled blocks are northwest/southeast, paint teal blocks just outside the northeast and southwest corners.
    - If the filled blocks are northeast/southwest, paint teal blocks just outside the northwest and southeast corners.
    - Clip any painted block at the grid boundary.

    Key insight
    - The green pixels are scaled square blocks; the task completes the perpendicular diagonal of the expanded block layout.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    source_colors = colors_present(g, exclude={bg})
    assert len(source_colors) == 1
    source_color = int(source_colors[0])
    assert source_color != Color.TEAL

    out = g.copy()
    components = find_components(
        g,
        bg=bg,
        colors=[source_color],
        connectivity=8,
        monochromatic=False,
    )
    assert components

    for comp in components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = bbox_h(comp["bbox"])
        w = bbox_w(comp["bbox"])
        assert h == w and h % 2 == 0
        size = h // 2

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        occupied: set[tuple[int, int]] = set()
        for br in range(2):
            for bc in range(2):
                block = sub[br * size : (br + 1) * size, bc * size : (bc + 1) * size]
                if np.all(block == source_color):
                    occupied.add((br, bc))
                else:
                    assert np.all(block == bg)

        if occupied == {(0, 0), (1, 1)}:
            targets = [(r0 - size, c1 + 1), (r1 + 1, c0 - size)]
        else:
            assert occupied == {(0, 1), (1, 0)}
            targets = [(r0 - size, c0 - size), (r1 + 1, c1 + 1)]

        for rr, cc in targets:
            _paste_teal_block(out, rr, cc, size, bg)

    return out

# --------------- generator (o5x) ---------------
# Canvas and colors.
GEN_GRID_SHAPE = (10, 10)
GEN_BACKGROUND = Color.BLACK
GEN_TARGET_COLOR = Color.TEAL
GEN_FOREGROUND_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)

# Motif structure.
GEN_MOTIF_COUNT_RANGE = (1, 2)
GEN_BLOCK_SIZES = (1, 2, 3)
GEN_DIAGONALS = (0, 1)
GEN_DIAGONAL_NW_SE = 0
GEN_MOTIF_BLOCK_SPAN = 2

# Placement and validation.
GEN_PAD_OFFSETS = (-1, 0, 1)
GEN_RETRY_LIMIT = 300


def _gen_rect_cells(shape: tuple[int, int], r: int, c: int, size: int) -> set[tuple[int, int]]:
    H, W = shape
    rr0 = max(0, int(r))
    cc0 = max(0, int(c))
    rr1 = min(H, int(r) + int(size))
    cc1 = min(W, int(c) + int(size))
    return {(rr, cc) for rr in range(rr0, rr1) for cc in range(cc0, cc1)}


def _gen_padded_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    H, W = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in GEN_PAD_OFFSETS:
            for dc in GEN_PAD_OFFSETS:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    out.add((rr, cc))
    return out


def _gen_block_origins(
    r0: int, c0: int, size: int, diagonal: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return foreground and outward target block top-lefts before clipping."""
    s = int(size)
    span = GEN_MOTIF_BLOCK_SPAN * s
    if diagonal == GEN_DIAGONAL_NW_SE:
        fg_blocks = [(r0, c0), (r0 + s, c0 + s)]
        target_blocks = [(r0 - s, c0 + span), (r0 + span, c0 - s)]
    else:
        fg_blocks = [(r0, c0 + s), (r0 + s, c0)]
        target_blocks = [(r0 - s, c0 - s), (r0 + span, c0 + span)]
    return fg_blocks, target_blocks


def _gen_motif_parts(
    shape: tuple[int, int], r0: int, c0: int, size: int, diagonal: int
) -> dict | None:
    s = int(size)
    fg_blocks, teal_blocks = _gen_block_origins(r0, c0, s, int(diagonal))

    fg = set()
    for rr, cc in fg_blocks:
        fg |= _gen_rect_cells(shape, rr, cc, s)

    teal_parts = [_gen_rect_cells(shape, rr, cc, s) for rr, cc in teal_blocks]
    if any(len(part) == 0 for part in teal_parts):
        return None
    if teal_parts[0] & teal_parts[1]:
        return None
    teal = teal_parts[0] | teal_parts[1]
    if fg & teal:
        return None

    return {
        "size": s,
        "diagonal": int(diagonal),
        "fg_blocks": fg_blocks,
        "teal_blocks": teal_blocks,
        "fg": fg,
        "teal": teal,
        "bbox": _gen_rect_cells(shape, r0, c0, GEN_MOTIF_BLOCK_SPAN * s),
    }


def _gen_candidate_placements(shape: tuple[int, int], size: int, diagonal: int) -> list[dict]:
    H, W = shape
    options = []
    span = GEN_MOTIF_BLOCK_SPAN * int(size)
    for r0 in range(H - span + 1):
        for c0 in range(W - span + 1):
            parts = _gen_motif_parts(shape, r0, c0, int(size), int(diagonal))
            if parts is not None:
                options.append(parts)
    random.shuffle(options)
    return options


def _gen_fits(parts: dict, state: dict, shape: tuple[int, int]) -> bool:
    if parts["bbox"] & state["bbox"]:
        return False
    if parts["teal"] & state["teal"]:
        return False
    if parts["teal"] & state["bbox"]:
        return False
    if parts["bbox"] & state["teal"]:
        return False
    return not (parts["fg"] & state["fg_padded"])


def _gen_add_state(parts: dict, state: dict, shape: tuple[int, int]) -> dict:
    return {
        "bbox": state["bbox"] | parts["bbox"],
        "teal": state["teal"] | parts["teal"],
        "fg_padded": state["fg_padded"] | _gen_padded_cells(parts["fg"], shape),
    }


def _gen_choose_placements(specs: tuple[tuple[int, int], ...], shape: tuple[int, int]) -> list[dict] | None:
    options = [_gen_candidate_placements(shape, size, diagonal) for size, diagonal in specs]
    if any(len(opts) == 0 for opts in options):
        return None

    order = sorted(range(len(specs)), key=lambda i: (len(options[i]), -specs[i][0]))

    def rec(k: int, state: dict, chosen: list[tuple[int, dict]]) -> list[tuple[int, dict]] | None:
        if k == len(order):
            return chosen
        idx = order[k]
        for parts in options[idx]:
            if _gen_fits(parts, state, shape):
                placed = rec(k + 1, _gen_add_state(parts, state, shape), chosen + [(idx, parts)])
                if placed is not None:
                    return placed
        return None

    initial_state = {"bbox": set(), "teal": set(), "fg_padded": set()}
    placed = rec(0, initial_state, [])
    if placed is None:
        return None

    result: list[dict] = [None] * len(specs)  # type: ignore[list-item]
    for idx, parts in placed:
        result[idx] = parts
    return result


def _gen_feasible_specs(count: int, shape: tuple[int, int]) -> list[tuple[tuple[int, int], ...]]:
    feasible = []
    for sizes in product(GEN_BLOCK_SIZES, repeat=int(count)):
        for diagonals in product(GEN_DIAGONALS, repeat=int(count)):
            specs = tuple((int(sizes[i]), int(diagonals[i])) for i in range(int(count)))
            if _gen_choose_placements(specs, shape) is not None:
                feasible.append(specs)
    random.shuffle(feasible)
    return feasible


def _gen_paste_square(grid: np.ndarray, r: int, c: int, size: int, color: int, *, clip: bool) -> None:
    patch = np.full((int(size), int(size)), int(color), dtype=int)
    paste(grid, patch, int(r), int(c), clip=clip)


def _gen_sample_placements(count: int, shape: tuple[int, int]) -> list[dict] | None:
    feasible = _gen_feasible_specs(count, shape)
    if not feasible:
        return None
    specs = random.choice(feasible)
    return _gen_choose_placements(specs, shape)


def _gen_render_output(shape: tuple[int, int], bg: int, fg: int, placements: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for parts in placements:
        for rr, cc in parts["fg_blocks"]:
            _gen_paste_square(out, rr, cc, parts["size"], fg, clip=False)
    for parts in placements:
        for rr, cc in parts["teal_blocks"]:
            _gen_paste_square(out, rr, cc, parts["size"], GEN_TARGET_COLOR, clip=True)
    return out


def _gen_attempt() -> np.ndarray | None:
    shape = GEN_GRID_SHAPE
    bg = GEN_BACKGROUND
    fg = random.choice(GEN_FOREGROUND_COLORS)
    count = random.randint(*GEN_MOTIF_COUNT_RANGE)

    placements = _gen_sample_placements(count, shape)
    if placements is None:
        return None

    out = _gen_render_output(shape, bg, fg, placements)
    inp = out.copy()
    inp[inp == GEN_TARGET_COLOR] = bg
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 22233c11 input")
