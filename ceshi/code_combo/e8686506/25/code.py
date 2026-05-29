# Combo variant of e8686506 (index 25)
# combo_spec: D5/1+D18/0
# Name: 同色必须相邻+棋盘格排列

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the D5/1 + D18/0 combo
# -----------------------------------------------------------------------------
# a) Concrete behavior / trace
#    The source task assembled a dense rectangle by locking a symmetric outline and
#    then placing scattered pieces under a "same colors must not 4-touch" rule.
#    D5/1 flips that relation: repeated colors are now supposed to merge into one
#    connected block. D18/0 replaces the old nested frame-with-holes interior by a
#    checkerboard-style layout. Combining them literally at the single-cell level is
#    contradictory, because a true 1x1 checkerboard never lets same-colored cells
#    4-touch. This combo therefore lifts the checkerboard one level up: the output
#    rectangle contains a fixed cross-shaped scaffold that partitions the board into
#    four square rooms arranged as a 2x2 checkerboard. Each room is monochrome, and
#    each room color arrives in the input as multiple disconnected fragments that
#    must fuse into one 2x2 cluster. Example trace:
#      - old task: two red shards would be kept apart by a no-touch constraint.
#      - fused task: the same red shards are split as a room-specific template,
#        detected as belonging to the top-left checker square, and merged into a
#        solid 2x2 red block.
# b) Complexity calibration
#    The scaffold keeps the puzzle in the same "assemble into a dense rectangle"
#    family instead of degenerating into a crop. Four rooms are enough to make the
#    checkerboard visible, while four distinct fragment templates prevent ambiguity:
#    two horizontal dominoes, two vertical dominoes, an L+singleton, and the
#    mirrored L+singleton. This is small enough to infer from 2-3 examples, but not
#    so small that chamber assignment becomes arbitrary.
# c) Implicit-assumption audit against the original solver
#    - The original propagation engine relied on exact left-right color symmetry.
#      That assumption conflicts with connected same-color checker squares, because
#      mirrored same-color rooms would become disconnected copies. The combo solver
#      therefore drops the old mirror-based packing logic and uses the scaffold plus
#      fragment-template matching instead.
#    - The original D5 exclusion test (`same colors may not 4-touch`) is invalid and
#      must be removed. The new target explicitly requires the opposite behavior.
#    - The old singleton-border heuristic is also invalid here: singletons are now
#      legitimate interior fragments because they are only intermediate shards that
#      later merge into a room cluster.
#    - The new main() assumes one scaffold component with the canonical 7x7 cross
#      mask and exactly four non-frame colors whose component-signature multisets are
#      the four expected templates. The generator guarantees each assumption.
# d) Placement legality
#    Generation is output-first. A reserved 7x7 bounding box is carved out on the
#    input canvas, all room fragments are scattered outside it with Chebyshev padding
#    1, and only then is the scaffold pasted back into the reserved box. This keeps
#    every object in-bounds, non-overlapping, and disconnected in the input.
# e) Visual observability
#    A human sees a large grouped cross scaffold already sitting inside a square
#    bounding box, plus several detached same-color shards elsewhere. Across samples,
#    the output always turns those shards into four solid room-filling squares in a
#    checkerboard arrangement. The "same colors now attract" rule is directly visible
#    because every non-frame color appears multiple times in the input but only once
#    as a connected cluster in the output.
# f) Visual cleanliness
#    The scene has only two layers: one anchored scaffold and four scattered room
#    colors. No extra marker system, numeric color ordering, or overlapping clues are
#    introduced. The resulting images stay sparse in the input and blocky in the
#    output.
# g) Color inference
#    Colors are globally permuted and never interpreted numerically. Chamber identity
#    is determined only by visible fragment structure: each room color has a stable
#    multiset of piece shapes, and that template tells the solver which checkerboard
#    room the color must fill.
# -----------------------------------------------------------------------------


_OUT_H = 7
_OUT_W = 7
_CANVAS_MIN = 13
_CANVAS_MAX = 16
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 250


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _scaffold_mask() -> np.ndarray:
    mask = np.zeros((_OUT_H, _OUT_W), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    mask[3, :] = True
    mask[:, 3] = True
    return mask


_SCAFFOLD_MASK = _scaffold_mask()

_ROOM_SLICES = {
    "tl": (slice(1, 3), slice(1, 3)),
    "tr": (slice(1, 3), slice(4, 6)),
    "bl": (slice(4, 6), slice(1, 3)),
    "br": (slice(4, 6), slice(4, 6)),
}

_ROOM_TEMPLATE_COORDS = {
    "tl": [[(0, 0), (0, 1)], [(1, 0), (1, 1)]],
    "tr": [[(0, 0), (1, 0)], [(0, 1), (1, 1)]],
    "bl": [[(0, 0), (0, 1), (1, 0)], [(1, 1)]],
    "br": [[(0, 1), (1, 0), (1, 1)], [(0, 0)]],
}


def _template_key(parts: list[list[tuple[int, int]]]) -> tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...]:
    sigs = []
    for coords in parts:
        arr = np.array(coords, dtype=int)
        sigs.append(_piece_sig(arr))
    return tuple(sorted(sigs))


_ROOM_KEYS = {room: _template_key(parts) for room, parts in _ROOM_TEMPLATE_COORDS.items()}
_KEY_TO_ROOM = {key: room for room, key in _ROOM_KEYS.items()}


def _patch_from_coords(coords: list[tuple[int, int]], color: int, bg: int) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(color)
    return patch


def _find_scaffold(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], np.ndarray]:
    candidates: list[tuple[int, tuple[int, int], np.ndarray]] = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if (r1 - r0 + 1, c1 - c0 + 1) != (_OUT_H, _OUT_W):
                continue
            mask = np.zeros((_OUT_H, _OUT_W), dtype=bool)
            for r, c in pts:
                mask[int(r) - r0, int(c) - c0] = True
            if np.array_equal(mask, _SCAFFOLD_MASK):
                candidates.append((int(col), (int(r0), int(c0)), mask))
    assert len(candidates) == 1, "expected exactly one canonical scaffold component"
    return candidates[0]


def _room_key_from_color(g: np.ndarray, color: int) -> tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...]:
    comps = components_mask(g == color, connectivity=4)
    assert len(comps) >= 2, "each room color must arrive as multiple fragments"
    return tuple(sorted(_piece_sig(pts) for pts in comps))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered same-color shards into a dense rectangle whose interior is a
    four-room checkerboard.

    Input structure:
    - The most common color is background.
    - One non-background color already forms a fixed 7x7 cross scaffold whose bbox is
      the whole output rectangle.
    - Every other color appears as several disconnected fragments scattered away from
      that scaffold, with one fragment-template family per checkerboard room.

    Rule:
    - Normalize the scaffold to the output frame and recover the four 2x2 rooms that
      it encloses.
    - For each remaining color, inspect the multiset of its fragment shapes.
    - Place that color into the unique room whose template matches those fragments,
      filling the entire room so the scattered shards become one 4-connected cluster.

    Key insight:
    - The checkerboard is expressed by the scaffold's four room positions, while the
      D5 inversion is expressed by the required merge: repeated colors are not meant
      to stay apart anymore. The fragment-template family tells you which room each
      color belongs to, and once a room is identified its final 2x2 cluster is forced.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    total_area = int(np.sum(g != bg))
    assert total_area == (_OUT_H * _OUT_W), "this combo uses a fixed dense 7x7 output"

    frame_color, _, _ = _find_scaffold(g, bg)
    room_colors = sorted(set(map(int, np.unique(g))) - {bg, frame_color})
    assert len(room_colors) == 4, "expected exactly four non-frame room colors"

    out = np.full((_OUT_H, _OUT_W), frame_color, dtype=int)
    seen_rooms: set[str] = set()
    for col in room_colors:
        room_key = _room_key_from_color(g, int(col))
        assert room_key in _KEY_TO_ROOM, "encountered unknown room fragment template"
        room = _KEY_TO_ROOM[room_key]
        assert room not in seen_rooms, "each room template must be used by exactly one color"
        rs, cs = _ROOM_SLICES[room]
        out[rs, cs] = int(col)
        seen_rooms.add(room)

    assert seen_rooms == set(_ROOM_SLICES), "all four checkerboard rooms must be assigned"
    for col in room_colors:
        assert len(components_mask(out == col, connectivity=4)) == 1, "room colors must cluster in the output"
    assert len(components_mask(out == frame_color, connectivity=4)) == 1, "the scaffold must stay connected"
    return out.astype(int)


def _build_output() -> tuple[np.ndarray, int, dict[str, int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    frame_color = palette[0]
    room_colors = {
        "tl": int(palette[1]),
        "tr": int(palette[2]),
        "bl": int(palette[3]),
        "br": int(palette[4]),
    }

    out = np.full((_OUT_H, _OUT_W), frame_color, dtype=int)
    for room, (rs, cs) in _ROOM_SLICES.items():
        out[rs, cs] = room_colors[room]
    return out, bg, room_colors


def _frame_patch(frame_color: int, bg: int) -> np.ndarray:
    patch = np.full((_OUT_H, _OUT_W), bg, dtype=int)
    patch[_SCAFFOLD_MASK] = int(frame_color)
    return patch


def _room_patches(room_colors: dict[str, int], bg: int) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for room, color in room_colors.items():
        for coords in _ROOM_TEMPLATE_COORDS[room]:
            patches.append(_patch_from_coords(coords, int(color), bg))
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _scatter(bg: int, frame_color: int, room_colors: dict[str, int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)

    max_r = rows - _OUT_H - 2
    max_c = cols - _OUT_W - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("reserved scaffold box does not fit the canvas")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)
    canvas[off_r : off_r + _OUT_H, off_c : off_c + _OUT_W] = -1

    for patch in _room_patches(room_colors, bg):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    bbox_region = canvas[off_r : off_r + _OUT_H, off_c : off_c + _OUT_W]
    bbox_region[:] = bg
    paste(canvas, _frame_patch(frame_color, bg), off_r, off_c, transparent=bg)
    return canvas


def _variant_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, object]:
    bg_in = get_bg(inp)
    colors_out = sorted(map(int, np.unique(out)))
    counts_in = {}
    counts_out = {}
    for col in colors_out:
        if col == get_bg(out):
            continue
        counts_in[col] = len(components_mask(inp == col, connectivity=4))
        counts_out[col] = len(components_mask(out == col, connectivity=4))

    frame_color = None
    for col in sorted(set(map(int, np.unique(inp))) - {bg_in}):
        comps = components_mask(inp == col, connectivity=4)
        if len(comps) == 1:
            pts = comps[0]
            r0, c0, r1, c1 = bbox_coords(pts)
            if (r1 - r0 + 1, c1 - c0 + 1) == (_OUT_H, _OUT_W):
                frame_color = int(col)
                break

    room_uniform = 0
    for rs, cs in _ROOM_SLICES.values():
        room = out[rs, cs]
        if len(set(map(int, room.flatten()))) == 1:
            room_uniform += 1

    merged_colors = 0
    for col, nin in counts_in.items():
        if col == frame_color:
            continue
        if nin >= 2 and counts_out.get(col, 0) == 1:
            merged_colors += 1

    return {
        "room_uniform": room_uniform,
        "merged_colors": merged_colors,
        "input_component_counts": tuple(sorted(v for k, v in counts_in.items() if k != frame_color)),
    }


def _gen_attempt() -> np.ndarray | None:
    out, bg, room_colors = _build_output()
    frame_color = int(out[0, 0])
    try:
        inp = _scatter(bg, frame_color, room_colors)
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, out):
        return None

    metrics = _variant_metrics(inp, out)
    if metrics["room_uniform"] != 4:
        return None
    if metrics["merged_colors"] != 4:
        return None
    if metrics["input_component_counts"] != (2, 2, 2, 2):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid D5/1 + D18/0 combo input for e8686506")
