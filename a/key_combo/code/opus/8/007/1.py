from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, find_components, neighbors4


PALETTE = (Color.BLUE, Color.RED, Color.GREEN)
GATE = Color.GREY
FRAME = Color.YELLOW
BG = Color.BLACK
GEN_MAX_ATTEMPTS = 400


def _identify_tokens_and_objects(grid: np.ndarray) -> tuple[list[tuple[int, int, int]], list[dict]]:
    comps = find_components(
        grid,
        bg=BG,
        colors=PALETTE,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    tokens: list[tuple[int, int, int]] = []
    objects: list[dict] = []
    for comp in comps:
        area = int(comp["area"])
        if area == 1:
            r, c = map(int, comp["coords"][0])
            tokens.append((r, c, int(comp["color"])))
        else:
            objects.append(comp)
    return tokens, objects


def _gate_components(grid: np.ndarray) -> list[dict]:
    comps = find_components(
        grid,
        bg=BG,
        colors=[GATE],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        assert int(comp["area"]) == 3
        assert r0 == r1 and c1 - c0 == 2
    return comps


def _rotate_gates(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for gate in _gate_components(out):
        r0, c0, _r1, _c1 = gate["bbox"]
        center_c = c0 + 1
        out[r0, c0 : c0 + 3] = BG
        for rr in range(r0 - 1, r0 + 2):
            if 0 <= rr < out.shape[0]:
                out[rr, center_c] = GATE
    return out


def _drop_tokens(rotated: np.ndarray, tokens: list[tuple[int, int, int]]) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    final = np.array(rotated, dtype=int, copy=True)
    for r, c, _color in tokens:
        final[r, c] = BG

    landed: list[tuple[int, int, int]] = []
    for r, c, color in sorted(tokens, key=lambda t: (-t[0], t[1], t[2])):
        rr = int(r)
        while rr + 1 < final.shape[0] and int(final[rr + 1, c]) == BG:
            rr += 1
        final[rr, c] = int(color)
        landed.append((rr, c, int(color)))
    return final, landed


def _paint_expanded_frame(out: np.ndarray, box: tuple[int, int, int, int]) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = box
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(h - 1, r1 + 1)
    ec1 = min(w - 1, c1 + 1)
    for r in range(er0, er1 + 1):
        for c in range(ec0, ec1 + 1):
            if r not in (er0, er1) and c not in (ec0, ec1):
                continue
            if int(out[r, c]) == BG:
                out[r, c] = FRAME


def _component_boxes(final: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    mask = (final != BG) & (final != GATE)
    comps = components_mask(mask, connectivity=4)
    infos = [(bbox_coords(coords), coords) for coords in comps]
    infos.sort(key=lambda item: (item[0][1], item[0][0]))
    return infos


def _final_without_frames(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    tokens, _objects = _identify_tokens_and_objects(grid)
    rotated = _rotate_gates(grid)
    return _drop_tokens(rotated, tokens)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each grey gate open, let the colored tokens fall, then frame every large merged component.

    Input structure:
    - A black background contains several grey 1x3 horizontal gate bars in the upper half.
    - Blue, red, or green single-pixel tokens sit above those bars.
    - The lower half contains separated 4-connected single-color foreground objects of the same blue/red/green palette.

    Rule:
    - Rotate every grey horizontal bar in place into a vertical 3x1 bar centered on the middle cell, clipping out-of-bounds cells.
    - Treat the rotated grid as the new scene and let all tokens fall straight downward simultaneously, stopping above the first non-black cell or the bottom edge; tokens in the same column stack.
    - Ignore grey cells and build 4-connected components over all remaining non-black cells without separating by color.
    - For each component of area at least 2, expand its bounding box by one cell in every direction, clip to the grid, and paint the perimeter yellow wherever the current cell is black.

    Key insight:
    - The important objects are the post-fall multicolor connected components created by the gate opening, not the original tokens or colors viewed separately.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    final, _landed = _final_without_frames(g)
    out = np.array(final, dtype=int, copy=True)

    for box, coords in _component_boxes(final):
        if int(coords.shape[0]) >= 2:
            _paint_expanded_frame(out, box)
    return out


def _coord_neighbors(coords: set[tuple[int, int]], r: int, c: int) -> bool:
    return any((nr, nc) in coords for nr, nc in neighbors4(r, c))


def _can_place_object(grid: np.ndarray, coords: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
    for r, c in coords:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords and int(grid[nr, nc]) != BG:
                return False
    return True


def _place_object(grid: np.ndarray, coords: set[tuple[int, int]], color: int) -> None:
    assert _can_place_object(grid, coords)
    for r, c in coords:
        grid[r, c] = int(color)


def _object_color_pair() -> tuple[int, int]:
    if random.random() < 0.7:
        a, b = random.sample(PALETTE, 2)
        return int(a), int(b)
    color = int(random.choice(PALETTE))
    return color, color


def _choose_shape(kind: str, lr: int, tc: int) -> list[set[tuple[int, int]]]:
    if kind == "extend":
        variants = [
            [{(lr + 1, tc), (lr + 2, tc)}],
            [{(lr + 1, tc), (lr + 2, tc), (lr + 2, tc + 1)}],
            [{(lr + 1, tc), (lr + 2, tc), (lr + 2, tc - 1)}],
        ]
    elif kind == "bridge_r":
        variants = [
            [{(lr + 1, tc), (lr + 2, tc)}, {(lr, tc + 1), (lr, tc + 2)}],
            [{(lr + 1, tc), (lr + 2, tc), (lr + 2, tc + 1)}, {(lr, tc + 1), (lr, tc + 2)}],
            [{(lr + 1, tc), (lr + 2, tc)}, {(lr, tc + 1), (lr, tc + 2), (lr - 1, tc + 2)}],
        ]
    elif kind == "bridge_l":
        variants = [
            [{(lr + 1, tc), (lr + 2, tc)}, {(lr, tc - 1), (lr, tc - 2)}],
            [{(lr + 1, tc), (lr + 2, tc), (lr + 2, tc - 1)}, {(lr, tc - 1), (lr, tc - 2)}],
            [{(lr + 1, tc), (lr + 2, tc)}, {(lr, tc - 1), (lr, tc - 2), (lr - 1, tc - 2)}],
        ]
    elif kind == "stack_r":
        variants = [
            [{(lr + 1, tc), (lr + 2, tc)}, {(lr, tc + 1), (lr - 1, tc + 1)}],
            [{(lr + 1, tc), (lr + 2, tc), (lr + 2, tc + 1)}, {(lr, tc + 1), (lr - 1, tc + 1)}],
            [{(lr + 1, tc), (lr + 2, tc)}, {(lr, tc + 1), (lr - 1, tc + 1), (lr - 1, tc + 2)}],
        ]
    elif kind == "stack_l":
        variants = [
            [{(lr + 1, tc), (lr + 2, tc)}, {(lr, tc - 1), (lr - 1, tc - 1)}],
            [{(lr + 1, tc), (lr + 2, tc), (lr + 2, tc - 1)}, {(lr, tc - 1), (lr - 1, tc - 1)}],
            [{(lr + 1, tc), (lr + 2, tc)}, {(lr, tc - 1), (lr - 1, tc - 1), (lr - 1, tc - 2)}],
        ]
    else:
        raise ValueError(f"unknown kind: {kind}")
    return random.choice(variants)


def _sample_group_columns(width: int, count: int) -> list[int] | None:
    cols = list(range(2, width - 2))
    for _ in range(80):
        pick = sorted(random.sample(cols, count))
        if all(b - a >= 4 for a, b in zip(pick, pick[1:])):
            return pick
    return None


def _compatible_kinds(tc: int, width: int, force_bridge: bool) -> list[str]:
    kinds = ["extend"]
    if tc + 2 < width:
        kinds.extend(["bridge_r", "stack_r"])
    if tc - 2 >= 0:
        kinds.extend(["bridge_l", "stack_l"])
    if force_bridge:
        kinds = [kind for kind in kinds if kind != "extend"]
    return kinds


def _random_object_color(count: int) -> list[int]:
    if count == 1:
        return [int(random.choice(PALETTE))]
    a, b = _object_color_pair()
    return [a, b]


def _place_groups(h: int, w: int) -> tuple[np.ndarray, list[dict]] | None:
    obj_grid = np.zeros((h, w), dtype=int)
    n_groups = random.randint(2, 4)
    cols = _sample_group_columns(w, n_groups)
    if cols is None:
        return None

    specs: list[dict] = []
    forced_bridge_done = False
    for idx, tc in enumerate(cols):
        force_bridge = not forced_bridge_done and idx == n_groups - 1
        kinds = _compatible_kinds(tc, w, force_bridge)
        random.shuffle(kinds)
        placed = False
        for kind in kinds:
            token_count = 2 if kind.startswith("stack") else 1
            low = h // 2 + (1 if token_count == 2 else 0)
            high = h - 3
            if low > high:
                continue
            rows = list(range(low, high + 1))
            random.shuffle(rows)
            for lr in rows:
                objects = _choose_shape(kind, lr, tc)
                colors = _random_object_color(len(objects))
                if any(not _can_place_object(obj_grid, obj) for obj in objects):
                    continue
                for obj, color in zip(objects, colors):
                    _place_object(obj_grid, obj, color)
                specs.append({"tc": tc, "lr": lr, "kind": kind, "token_count": token_count})
                placed = True
                forced_bridge_done = forced_bridge_done or kind != "extend"
                break
            if placed:
                break
        if not placed:
            return None

    return obj_grid, specs


def _standalone_templates(anchor_r: int, anchor_c: int) -> list[set[tuple[int, int]]]:
    return [
        {(anchor_r, anchor_c), (anchor_r + 1, anchor_c)},
        {(anchor_r, anchor_c), (anchor_r, anchor_c + 1)},
        {(anchor_r, anchor_c), (anchor_r + 1, anchor_c), (anchor_r + 1, anchor_c + 1)},
        {(anchor_r, anchor_c), (anchor_r + 1, anchor_c), (anchor_r + 2, anchor_c)},
    ]


def _add_standalone_objects(grid: np.ndarray) -> None:
    h, w = grid.shape
    for _ in range(random.randint(0, 2)):
        anchors = [(r, c) for r in range(h // 2, h - 2) for c in range(0, w - 2)]
        random.shuffle(anchors)
        for ar, ac in anchors:
            templates = _standalone_templates(ar, ac)
            random.shuffle(templates)
            for coords in templates:
                if _can_place_object(grid, coords):
                    _place_object(grid, coords, int(random.choice(PALETTE)))
                    anchors = []
                    break
            if not anchors:
                break


def _place_gates_and_tokens(obj_grid: np.ndarray, specs: list[dict]) -> np.ndarray | None:
    h, w = obj_grid.shape
    grid = np.array(obj_grid, dtype=int, copy=True)
    occupied_gate_cells: set[tuple[int, int]] = set()

    for spec in specs:
        tc = int(spec["tc"])
        token_count = int(spec["token_count"])
        centers = [gc for gc in (tc - 1, tc + 1) if 1 <= gc < w - 1]
        random.shuffle(centers)
        gate_rows = list(range(2, h // 2))
        random.shuffle(gate_rows)
        placed = False
        for gc in centers:
            for gr in gate_rows:
                if gr >= spec["lr"]:
                    continue
                start_rows = list(range(0, gr, 2))
                if len(start_rows) < token_count:
                    continue
                bar_cells = {(gr, gc - 1), (gr, gc), (gr, gc + 1)}
                post_cells = {(rr, gc) for rr in range(gr - 1, gr + 2) if 0 <= rr < h}
                if any(cell in occupied_gate_cells for cell in bar_cells):
                    continue
                if any(int(grid[r, c]) != BG for r, c in bar_cells):
                    continue
                if any(int(obj_grid[r, c]) != BG for r, c in post_cells):
                    continue
                token_rows = sorted(random.sample(start_rows, token_count))
                if any(int(grid[tr, tc]) != BG for tr in token_rows):
                    continue
                for r, c in bar_cells:
                    grid[r, c] = GATE
                for tr in token_rows:
                    grid[tr, tc] = int(random.choice(PALETTE))
                occupied_gate_cells |= bar_cells
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return grid


def _original_object_mask(grid: np.ndarray) -> np.ndarray:
    mask = np.zeros(grid.shape, dtype=bool)
    _tokens, objects = _identify_tokens_and_objects(grid)
    for obj in objects:
        coords = obj["coords"]
        mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _validate_generator_grid(grid: np.ndarray) -> bool:
    tokens, objects = _identify_tokens_and_objects(grid)
    if len(tokens) < 2 or not objects:
        return False

    try:
        gates = _gate_components(grid)
    except AssertionError:
        return False
    if len(gates) < 2:
        return False

    for r, c, _color in tokens:
        below = int(np.sum(grid[r + 1 :, c] == GATE))
        if below != 1:
            return False

    try:
        rotated = _rotate_gates(grid)
        final, landed = _drop_tokens(rotated, tokens)
    except AssertionError:
        return False

    object_mask = _original_object_mask(grid)
    for rr, cc, _color in landed:
        if rr == final.shape[0] - 1:
            return False
        if not any(
            0 <= nr < final.shape[0] and 0 <= nc < final.shape[1] and object_mask[nr, nc]
            for nr, nc in neighbors4(rr, cc)
        ):
            return False

    if np.any((rotated == GATE) & object_mask):
        return False

    comps = _component_boxes(final)
    if not any(int(coords.shape[0]) >= 4 for _box, coords in comps):
        return False
    if not any(len(set(int(final[r, c]) for r, c in coords)) >= 2 for _box, coords in comps):
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape and np.any(out == FRAME)


def _generate_attempt() -> np.ndarray | None:
    n_groups = random.randint(2, 4)
    w_min = max(13, 5 * n_groups + 2)
    h = random.randint(13, 18)
    w = random.randint(w_min, 24)

    placed = _place_groups(h, w)
    if placed is None:
        return None
    obj_grid, specs = placed
    _add_standalone_objects(obj_grid)
    grid = _place_gates_and_tokens(obj_grid, specs)
    if grid is None:
        return None
    if _validate_generator_grid(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gate-release input")
