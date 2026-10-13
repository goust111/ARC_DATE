import numpy as np


BG = 0
FRAME = 5
PALETTE = tuple(c for c in range(10) if c not in (BG, FRAME))
INSTR_H = 3
INSTR_W = 7
MAX_GEN_ATTEMPTS = 400

_SRC_PIXELS = ((1, 0), (1, 1), (1, 2), (1, 3), (0, 4), (2, 4))
_TARGET_PIXEL = (1, 6)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _find_canvas_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    box = _bbox(grid == FRAME)
    assert box is not None, "expected a gray frame"
    r0, c0, r1, c1 = box
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"
    assert np.all(grid[r0, c0 : c1 + 1] == FRAME)
    assert np.all(grid[r1, c0 : c1 + 1] == FRAME)
    assert np.all(grid[r0 : r1 + 1, c0] == FRAME)
    assert np.all(grid[r0 : r1 + 1, c1] == FRAME)
    return r0, c0, r1, c1


def _parse_instruction_at(grid: np.ndarray, top: int, left: int) -> tuple[int, int] | None:
    patch = grid[top : top + INSTR_H, left : left + INSTR_W]
    if patch.shape != (INSTR_H, INSTR_W):
        return None

    src = int(patch[1, 0])
    if src in (BG, FRAME):
        return None

    for rr, cc in _SRC_PIXELS:
        if int(patch[rr, cc]) != src:
            return None

    dst = int(patch[_TARGET_PIXEL])
    if dst in (BG, FRAME, src):
        return None

    for rr in range(INSTR_H):
        for cc in range(INSTR_W):
            if (rr, cc) in _SRC_PIXELS or (rr, cc) == _TARGET_PIXEL:
                continue
            if int(patch[rr, cc]) != BG:
                return None

    return src, dst


def _parse_instructions(grid: np.ndarray, canvas_box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    _, _, _, canvas_c1 = canvas_box
    h, w = grid.shape
    found: list[tuple[int, int, int, int]] = []

    for top in range(h - INSTR_H + 1):
        for left in range(canvas_c1 + 1, w - INSTR_W + 1):
            instr = _parse_instruction_at(grid, top, left)
            if instr is not None:
                src, dst = instr
                found.append((top, left, src, dst))

    found.sort(key=lambda item: (item[0], item[1]))
    assert found, "expected at least one instruction"
    return [(src, dst) for _top, _left, src, dst in found]


def _apply_instructions(canvas: np.ndarray, instructions: list[tuple[int, int]]) -> tuple[np.ndarray, list[int]]:
    out = canvas.copy()
    change_counts: list[int] = []
    for src, dst in instructions:
        mask = out == src
        change_counts.append(int(mask.sum()))
        out[mask] = dst
    return out, change_counts


def main(grid: np.ndarray) -> np.ndarray:
    """Apply the right-side arrow instruction chain to the framed canvas and crop the interior."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == FRAME), "gray frame is required"

    canvas_box = _find_canvas_bbox(g)
    r0, c0, r1, c1 = canvas_box
    canvas = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    instructions = _parse_instructions(g, canvas_box)
    out, _ = _apply_instructions(canvas, instructions)
    return out


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    assert 1 <= parts <= total
    if parts == 1:
        return np.array([total], dtype=int)
    cuts = np.sort(rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    return np.diff(np.concatenate(([0], cuts, [total]))).astype(int)


def _build_instruction_chain(
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[int], bool]:
    num_instructions = int(rng.integers(2, 5))
    chain_mode = bool(rng.integers(0, 2))

    if not chain_mode:
        sources = list(map(int, rng.choice(np.array(PALETTE), size=num_instructions, replace=False)))
        source_set = set(sources)
        instructions = []
        for src in sources:
            dst_pool = [c for c in PALETTE if c != src and c not in source_set]
            dst = int(dst_pool[int(rng.integers(len(dst_pool)))])
            instructions.append((int(src), dst))
        return instructions, sources, False

    chain_len = int(rng.integers(2, min(4, num_instructions) + 1))
    chain_colors = list(map(int, rng.choice(np.array(PALETTE), size=chain_len + 1, replace=False)))
    chain_instructions = [(chain_colors[i], chain_colors[i + 1]) for i in range(chain_len)]
    required_initial = [chain_colors[0]]

    remaining = num_instructions - chain_len
    independent_instructions: list[tuple[int, int]] = []
    if remaining:
        excluded = set(chain_colors)
        indep_sources = list(
            map(int, rng.choice(np.array([c for c in PALETTE if c not in excluded]), size=remaining, replace=False))
        )
        all_sources = set(chain_colors[:-1]) | set(indep_sources)
        for src in indep_sources:
            dst_pool = [c for c in PALETTE if c != src and c not in all_sources]
            dst = int(dst_pool[int(rng.integers(len(dst_pool)))])
            independent_instructions.append((int(src), dst))
            required_initial.append(int(src))

    if independent_instructions and bool(rng.integers(0, 2)):
        instructions = independent_instructions + chain_instructions
    else:
        instructions = chain_instructions + independent_instructions

    return instructions, required_initial, True


def _make_mosaic(
    h: int,
    w: int,
    rng: np.random.Generator,
    required_colors: list[int],
) -> np.ndarray:
    required = list(dict.fromkeys(int(c) for c in required_colors))
    total_colors = int(rng.integers(max(2, len(required)), min(6, len(PALETTE)) + 1))
    extras_pool = [c for c in PALETTE if c not in required]
    extras_needed = max(0, total_colors - len(required))
    extras = []
    if extras_needed:
        extras = list(map(int, rng.choice(np.array(extras_pool), size=extras_needed, replace=False)))
    colors = required + extras

    min_cells = len(colors)
    coarse_h = 2
    coarse_w = 2
    for _ in range(50):
        coarse_h = int(rng.integers(2, min(h, 5) + 1))
        coarse_w = int(rng.integers(2, min(w, 6) + 1))
        if coarse_h * coarse_w >= min_cells:
            break
    coarse = rng.choice(np.array(colors), size=(coarse_h, coarse_w))

    positions = [(r, c) for r in range(coarse_h) for c in range(coarse_w)]
    rng.shuffle(positions)
    for color, (r, c) in zip(colors, positions):
        coarse[r, c] = int(color)

    row_sizes = _random_composition(h, coarse_h, rng)
    col_sizes = _random_composition(w, coarse_w, rng)
    mosaic = np.repeat(np.repeat(coarse, row_sizes, axis=0), col_sizes, axis=1).astype(int)

    forced_positions = rng.permutation(h * w)
    idx = 0
    for color in required:
        copies = int(rng.integers(2, 4))
        for _ in range(copies):
            if idx >= len(forced_positions):
                idx = 0
            pos = int(forced_positions[idx])
            idx += 1
            rr, cc = divmod(pos, w)
            mosaic[rr, cc] = int(color)

    return mosaic


def _instruction_rows(num_instructions: int, rng: np.random.Generator) -> tuple[list[int], int]:
    gaps = [int(rng.integers(1, 4)) for _ in range(max(0, num_instructions - 1))]
    tops = [0]
    for gap in gaps:
        tops.append(tops[-1] + INSTR_H + gap)
    total_h = tops[-1] + INSTR_H
    return tops, total_h


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    instructions, required_initial, _chain_mode = _build_instruction_chain(rng)
    inner_h = int(rng.integers(4, 10))
    inner_w = int(rng.integers(4, 11))
    mosaic = _make_mosaic(inner_h, inner_w, rng, required_initial)

    expected, change_counts = _apply_instructions(mosaic, instructions)
    if any(count == 0 for count in change_counts):
        return None
    if np.array_equal(expected, mosaic):
        return None

    frame_h = inner_h + 2
    frame_w = inner_w + 2
    inst_tops, inst_h = _instruction_rows(len(instructions), rng)

    top_margin = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))
    extra_h = int(rng.integers(0, 3))
    grid_h = max(frame_h, inst_h) + top_margin + bottom_margin + extra_h

    left_margin = int(rng.integers(1, 4))
    gap = int(rng.integers(2, 6))
    right_margin = int(rng.integers(1, 4))
    extra_w = int(rng.integers(0, 3))
    grid_w = left_margin + frame_w + gap + INSTR_W + right_margin + extra_w

    frame_top = int(rng.integers(top_margin, grid_h - bottom_margin - frame_h + 1))
    inst_top = int(rng.integers(top_margin, grid_h - bottom_margin - inst_h + 1))
    frame_left = left_margin
    inst_left = frame_left + frame_w + gap

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    grid[frame_top, frame_left : frame_left + frame_w] = FRAME
    grid[frame_top + frame_h - 1, frame_left : frame_left + frame_w] = FRAME
    grid[frame_top : frame_top + frame_h, frame_left] = FRAME
    grid[frame_top : frame_top + frame_h, frame_left + frame_w - 1] = FRAME
    grid[frame_top + 1 : frame_top + 1 + inner_h, frame_left + 1 : frame_left + 1 + inner_w] = mosaic

    for rel_top, (src, dst) in zip(inst_tops, instructions):
        top = inst_top + rel_top
        grid[top + 1, inst_left : inst_left + 4] = int(src)
        grid[top, inst_left + 4] = int(src)
        grid[top + 2, inst_left + 4] = int(src)
        grid[top + 1, inst_left + 6] = int(dst)

    try:
        parsed = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(parsed, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
