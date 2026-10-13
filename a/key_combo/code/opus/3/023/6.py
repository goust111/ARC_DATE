from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, paste


LAYER_COUNTS = (4, 6, 8)
PALETTE = tuple(range(1, 10))
PALETTE_SIZE_RANGE = (4, 6)
CUE_COUNT_RANGE = (1, 3)
CANVAS_SIDE_RANGE = (16, 24)
EDGE_MARGIN = 1
MAX_ATTEMPTS = 500

_SHAPES_BY_AREA = {
    1: (
        np.array([[1]], dtype=int),
    ),
    2: (
        np.array([[1, 1]], dtype=int),
        np.array([[1], [1]], dtype=int),
    ),
    3: (
        np.array([[1, 1, 1]], dtype=int),
        np.array([[1], [1], [1]], dtype=int),
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[1, 1], [0, 1]], dtype=int),
        np.array([[1, 0], [1, 1]], dtype=int),
        np.array([[0, 1], [1, 1]], dtype=int),
    ),
    4: (
        np.array([[1, 1, 1, 1]], dtype=int),
        np.array([[1], [1], [1], [1]], dtype=int),
        np.array([[1, 1], [1, 1]], dtype=int),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
        np.array([[0, 1], [0, 1], [1, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
        np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
    ),
}


def _solid_strip_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    candidates = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if min(height, width) != 4:
            continue
        if height % 2 or width % 2:
            continue
        if int(comp["area"]) != height * width:
            continue
        candidates.append((height * width, (r0, c0, r1, c1)))
    assert candidates, "expected one solid 4-wide strip"
    best_area = max(area for area, _bbox in candidates)
    winners = [bbox for area, bbox in candidates if area == best_area]
    assert len(winners) == 1, "expected a unique largest strip"
    return winners[0]


def _reverse_row_major_blocks(strip: np.ndarray) -> list[np.ndarray]:
    h, w = strip.shape
    assert h % 2 == 0 and w % 2 == 0
    blocks: list[np.ndarray] = []
    for br in range(h // 2 - 1, -1, -1):
        for bc in range(w // 2 - 1, -1, -1):
            blocks.append(strip[2 * br : 2 * br + 2, 2 * bc : 2 * bc + 2].copy())
    return blocks


def _forward_row_major_blocks(strip: np.ndarray) -> list[np.ndarray]:
    h, w = strip.shape
    assert h % 2 == 0 and w % 2 == 0
    blocks: list[np.ndarray] = []
    for br in range(h // 2):
        for bc in range(w // 2):
            blocks.append(strip[2 * br : 2 * br + 2, 2 * bc : 2 * bc + 2].copy())
    return blocks


def _expand_blocks_to_rings(blocks: list[np.ndarray]) -> np.ndarray:
    layers = np.asarray([block.reshape(4) for block in blocks], dtype=int)
    side_half = len(blocks)
    assert layers.shape == (side_half, 4)

    out = np.empty((2 * side_half, 2 * side_half), dtype=int)
    ul_seq, ur_seq, ll_seq, lr_seq = (layers[:, i] for i in range(4))
    for r in range(side_half):
        for c in range(side_half):
            out[r, c] = ul_seq[min(r, c)]
            out[r, side_half + c] = ur_seq[min(r, side_half - 1 - c)]
            out[side_half + r, c] = ll_seq[min(side_half - 1 - r, c)]
            out[side_half + r, side_half + c] = lr_seq[min(side_half - 1 - r, side_half - 1 - c)]
    return out


def _block_major_color(block: np.ndarray) -> int:
    flat = [int(v) for v in block.reshape(-1)]
    counts = Counter(flat)
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    top_left = int(block[0, 0])
    return top_left if top_left in winners else winners[0]


def _cue_sizes(grid: np.ndarray, bg: int, strip_bbox: tuple[int, int, int, int]) -> dict[int, int]:
    outside = np.array(grid, copy=True, dtype=int)
    r0, c0, r1, c1 = strip_bbox
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    cues: dict[int, int] = {}
    for comp in find_components(
        outside,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        color = int(comp["color"])
        cues[color] = cues.get(color, 0) + int(comp["area"])
    return cues


def _rotate_ring_border(board: np.ndarray, layer: int, turns: int) -> None:
    turns %= 4
    if turns == 0:
        return
    end = board.shape[0] - layer
    sub = board[layer:end, layer:end]
    rotated = np.rot90(sub, -turns)
    board[layer, layer:end] = rotated[0, :]
    board[end - 1, layer:end] = rotated[-1, :]
    board[layer:end, layer] = rotated[:, 0]
    board[layer:end, end - 1] = rotated[:, -1]


def _solve_from_blocks(blocks: list[np.ndarray], cues: dict[int, int]) -> np.ndarray:
    base = _expand_blocks_to_rings(blocks)
    out = base.copy()
    for layer, block in enumerate(blocks):
        main_color = _block_major_color(block)
        turns = int(cues.get(main_color, 0)) % 4
        _rotate_ring_border(out, layer, turns)
    return out


def _parse_strip_and_cues(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = int(get_bg(g))
    strip_bbox = _solid_strip_bbox(g, bg)
    r0, c0, r1, c1 = strip_bbox
    strip = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    blocks = _reverse_row_major_blocks(strip)
    cues = _cue_sizes(g, bg, strip_bbox)
    return strip, blocks, cues


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 4-wide 2x2-block strip into nested square rings and rotate each ring from same-color cue sprites.

    Input structure:
    - The background is the most common color.
    - One foreground object is a solid rectangle whose shorter side is 4 and whose cells partition exactly into 2x2 blocks.
    - Outside that strip are 1..3 small monochrome 4-connected cue sprites whose colors also occur inside the strip.

    Rule:
    - Identify the strip as the unique solid 4-wide rectangle and read its 2x2 blocks in reverse row-major order: bottom-right to top-left.
    - Expand those blocks into the `8fff9e47` nested-square layout, where each block supplies one concentric ring and its TL/TR/BL/BR pixels color the four ring quadrants.
    - For each layer, compute the source block's majority color, breaking ties by the block's top-left pixel, then find the outside cue of the same color.
    - Rotate only that ring clockwise by `(cue area mod 4) * 90`; if no same-color cue exists, leave the ring unchanged.

    Key insight:
    - The outside sprites do not rotate colors globally; the strip must be unfolded first so each ring can be matched by its own source block's majority color.
    """
    _strip, blocks, cues = _parse_strip_and_cues(grid)
    return _solve_from_blocks(blocks, cues)


def _color_layer_rotation(board: np.ndarray, cues: dict[int, int]) -> np.ndarray:
    out = np.array(board, copy=True, dtype=int)
    for color, area in cues.items():
        turns = int(area) % 4
        if turns == 0:
            continue
        mask = (board == int(color)).astype(int)
        rotated = np.rot90(mask, -turns)
        out[out == int(color)] = 0
        out[rotated == 1] = int(color)
    return out


def _random_block(
    main_color: int,
    palette: list[int],
    turns: int,
    *,
    force_include: int | None = None,
) -> np.ndarray:
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    pair_patterns = (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
        ((0, 0), (1, 1)),
        ((0, 1), (1, 0)),
    )
    others = [color for color in palette if color != main_color]
    assert others

    for _ in range(200):
        block = np.empty((2, 2), dtype=int)
        if random.random() < 0.45:
            block[:, :] = int(main_color)
            minor_pos = random.choice(positions)
            minor_color = int(force_include) if force_include is not None else random.choice(others)
            block[minor_pos] = minor_color
        else:
            main_positions = set(random.choice(pair_patterns))
            remaining = [pos for pos in positions if pos not in main_positions]
            if force_include is None:
                extra_colors = random.sample(others, 2)
            else:
                spare = [color for color in others if color != int(force_include)]
                extra_colors = [int(force_include), random.choice(spare)]
                random.shuffle(extra_colors)
            for pos in main_positions:
                block[pos] = int(main_color)
            for pos, color in zip(remaining, extra_colors):
                block[pos] = int(color)

        if _block_major_color(block) != int(main_color):
            continue
        if turns % 4 and np.array_equal(block, np.rot90(block, -turns)):
            continue
        return block

    raise RuntimeError("failed to build a non-degenerate block")


def _layers_to_strip(blocks: list[np.ndarray]) -> np.ndarray:
    layer_count = len(blocks)
    assert layer_count in LAYER_COUNTS
    block_rows = layer_count // 2
    strip = np.zeros((2 * block_rows, 4), dtype=int)
    row_major = list(reversed(blocks))
    idx = 0
    for br in range(block_rows):
        for bc in range(2):
            strip[2 * br : 2 * br + 2, 2 * bc : 2 * bc + 2] = row_major[idx]
            idx += 1
    return strip


def _random_shape(area: int, color: int) -> np.ndarray:
    mask = random.choice(_SHAPES_BY_AREA[int(area)])
    return mask.astype(int) * int(color)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _placement_options(
    canvas_shape: tuple[int, int],
    occupied: np.ndarray,
    patch: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = canvas_shape
    ph, pw = patch.shape
    forbidden = _dilate8(occupied)
    options: list[tuple[int, int]] = []
    for r0 in range(EDGE_MARGIN, h - ph - EDGE_MARGIN + 1):
        for c0 in range(EDGE_MARGIN, w - pw - EDGE_MARGIN + 1):
            cells = np.argwhere(patch != 0)
            if all(not forbidden[r0 + int(r), c0 + int(c)] for r, c in cells):
                options.append((r0, c0))
    return options


def _build_instance() -> tuple[np.ndarray, dict]:
    layer_count = random.choice(LAYER_COUNTS)
    palette_size = random.randint(*PALETTE_SIZE_RANGE)
    palette = random.sample(PALETTE, palette_size)
    cue_count = random.randint(CUE_COUNT_RANGE[0], min(CUE_COUNT_RANGE[1], palette_size - 1))
    cue_colors = random.sample(palette, cue_count)
    non_cue_colors = [color for color in palette if color not in cue_colors]
    assert non_cue_colors

    if cue_count == 1:
        cue_areas = {cue_colors[0]: random.choice((1, 2, 3))}
    else:
        sizes = random.sample((1, 2, 3, 4), cue_count)
        cue_areas = {color: area for color, area in zip(cue_colors, sizes)}

    rotating_cues = [color for color, area in cue_areas.items() if area % 4]
    assert rotating_cues
    witness_cue = random.choice(rotating_cues)
    witness_plain = random.choice(non_cue_colors)

    specs = [
        {"main": witness_cue, "force_include": random.choice(non_cue_colors)},
        {"main": witness_plain, "force_include": witness_cue},
    ]
    for color in cue_colors:
        if color != witness_cue:
            specs.append({"main": color, "force_include": random.choice(non_cue_colors)})
    specs.append({"main": random.choice(non_cue_colors), "force_include": random.choice(cue_colors)})

    weight_map = {color: (4 if color in cue_colors else 3) for color in palette}
    while len(specs) < layer_count:
        main_color = random.choices(palette, weights=[weight_map[color] for color in palette], k=1)[0]
        include_choices = [color for color in cue_colors if color != main_color]
        force_include = random.choice(include_choices) if include_choices and random.random() < 0.65 else None
        specs.append({"main": main_color, "force_include": force_include})
    random.shuffle(specs)

    blocks = []
    for spec in specs:
        main_color = int(spec["main"])
        force_include = spec["force_include"]
        turns = int(cue_areas.get(main_color, 0)) % 4
        blocks.append(_random_block(main_color, palette, turns, force_include=force_include))

    strip = _layers_to_strip(blocks)

    canvas_h = random.randint(max(CANVAS_SIDE_RANGE[0], strip.shape[0] + 6), CANVAS_SIDE_RANGE[1])
    canvas_w = random.randint(max(CANVAS_SIDE_RANGE[0], strip.shape[1] + 8), CANVAS_SIDE_RANGE[1])
    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    strip_options = _placement_options(grid.shape, occupied, strip)
    assert strip_options
    strip_r, strip_c = random.choice(strip_options)
    paste(grid, strip, strip_r, strip_c)
    occupied[strip_r : strip_r + strip.shape[0], strip_c : strip_c + strip.shape[1]] = True

    cue_patches = []
    cue_boxes = []
    for color, area in cue_areas.items():
        patch = _random_shape(area, color)
        options = _placement_options(grid.shape, occupied, patch)
        if not options:
            raise RuntimeError("failed to place cue")
        r0, c0 = random.choice(options)
        paste(grid, patch, r0, c0, transparent=0)
        occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] |= patch != 0
        cue_patches.append(patch)
        cue_boxes.append((r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1))

    meta = {
        "blocks": blocks,
        "cue_areas": cue_areas,
        "palette": palette,
        "cue_colors": cue_colors,
        "witness_cue": witness_cue,
        "witness_plain": witness_plain,
        "strip_bbox": (strip_r, strip_c, strip_r + strip.shape[0] - 1, strip_c + strip.shape[1] - 1),
        "cue_boxes": cue_boxes,
    }
    return grid, meta


def _validate_instance(grid: np.ndarray, meta: dict) -> bool:
    blocks = meta["blocks"]
    cue_areas = meta["cue_areas"]
    witness_cue = int(meta["witness_cue"])
    witness_plain = int(meta["witness_plain"])

    correct = _solve_from_blocks(blocks, cue_areas)
    if not np.array_equal(main(grid), correct):
        return False

    if correct.shape[0] != correct.shape[1] or correct.shape[0] > 30:
        return False

    layer_turns = [int(cue_areas.get(_block_major_color(block), 0)) % 4 for block in blocks]
    if not any(turns != 0 for turns in layer_turns):
        return False
    if not any(cue_areas.get(_block_major_color(block), 0) == 0 for block in blocks):
        return False

    changed_layers = 0
    for block, turns in zip(blocks, layer_turns):
        if turns and not np.array_equal(block, np.rot90(block, -turns)):
            changed_layers += 1
    if changed_layers == 0:
        return False

    witness_ok = False
    for block in blocks:
        main_color = _block_major_color(block)
        if main_color != witness_plain:
            continue
        if witness_cue not in set(map(int, block.reshape(-1))):
            continue
        if int(cue_areas.get(main_color, 0)) % 4 == int(cue_areas[witness_cue]) % 4:
            continue
        witness_ok = True
        break
    if not witness_ok:
        return False

    no_rotation = _expand_blocks_to_rings(blocks)
    if np.array_equal(correct, no_rotation):
        return False

    forward = _solve_from_blocks(list(reversed(blocks)), cue_areas)
    if np.array_equal(correct, forward):
        return False

    color_layer = _color_layer_rotation(no_rotation, cue_areas)
    if np.array_equal(correct, color_layer):
        return False

    strip, parsed_blocks, parsed_cues = _parse_strip_and_cues(grid)
    if strip.shape[1] != 4 and strip.shape[0] != 4:
        return False
    if len(parsed_blocks) != len(blocks):
        return False
    if parsed_cues != cue_areas:
        return False
    for a, b in zip(parsed_blocks, blocks):
        if not np.array_equal(a, b):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid, meta = _build_instance()
        except Exception:
            continue
        if _validate_instance(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid nested-ring rotation instance")
