import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GENERATION_RETRIES = 400
_GEN_CALL_COUNT = 0


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_non_bg_components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _depth_map(side: int) -> np.ndarray:
    idx = np.arange(side)
    edge_dist = np.minimum(idx, side - 1 - idx)
    return np.minimum(edge_dist[:, None], edge_dist[None, :])


def _render_complete_onion(side: int, layer_colors: list[int]) -> np.ndarray:
    return np.asarray(layer_colors, dtype=int)[_depth_map(side)]


def _rotate_layer_colors(layer_colors: list[int], steps: int, gap_side: str) -> list[int]:
    count = len(layer_colors)
    shift = int(steps) % count
    if gap_side == "top":
        return [layer_colors[(i - shift) % count] for i in range(count)]
    return [layer_colors[(i + shift) % count] for i in range(count)]


def _analyze_onion(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = bbox_coords(coords)
    side = int(r1 - r0 + 1)
    assert side == int(c1 - c0 + 1), "each onion must occupy a square bbox"
    assert side >= 5, "each onion must have at least three layers"

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes = np.argwhere(patch == bg)
    assert holes.shape[0] == 1, "each onion must have exactly one gap pixel"
    gap_r, gap_c = map(int, holes[0])

    gap_depth = min(gap_r, gap_c, side - 1 - gap_r, side - 1 - gap_c)
    on_top = gap_r == gap_depth and gap_depth < gap_c < side - 1 - gap_depth
    on_bottom = gap_r == side - 1 - gap_depth and gap_depth < gap_c < side - 1 - gap_depth
    assert on_top ^ on_bottom, "the gap must lie on a non-corner top or bottom edge"

    layers = (side + 1) // 2
    depth = _depth_map(side)
    layer_colors = []
    for k in range(layers):
        vals = sorted(set(map(int, patch[(depth == k) & (patch != bg)])))
        assert len(vals) == 1, "each visible layer must have exactly one color"
        layer_colors.append(vals[0])
    assert len(set(layer_colors)) == len(layer_colors), "layer colors must be distinct"

    return {
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "layers": layers,
        "gap_side": "top" if on_top else "bottom",
        "step": gap_depth + 1,
        "layer_colors": layer_colors,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each concentric onion's layer colors using its gap side and gap depth, then fill the gap.

    Input structure:
    - Black background with one or two multicolor square "onions", each formed by concentric square layers.
    - Within each onion, every layer has a distinct color and the innermost layer is the center pixel or smallest square.
    - Exactly one pixel is missing from one layer of each onion, and that gap lies on a non-corner top or bottom edge.

    Rule:
    - Treat each onion independently and count the gapped layer from the outside inward, starting at 1.
    - If the gap is on the top edge, cycle the onion's layer colors inward by that many steps; if the gap is on the bottom edge, cycle them outward by that many steps.
    - Repaint every layer with its rotated color and restore the missing pixel so the onion becomes complete.

    Key insight:
    - The gap is a two-part control marker: its edge gives the rotation direction and its layer depth gives the rotation step count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == BG, "expected black background"

    out = g.copy()
    components = _find_non_bg_components(g, bg)
    assert 1 <= len(components) <= 2, "expected one or two onions"

    for coords in components:
        info = _analyze_onion(g, coords, bg)
        rotated = _rotate_layer_colors(info["layer_colors"], info["step"], info["gap_side"])
        r0, c0, r1, c1 = info["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = _render_complete_onion(info["side"], rotated)

    return out


def _valid_steps(side: int) -> list[int]:
    layers = (side + 1) // 2
    max_step = (side - 1) // 2
    return [step for step in range(1, max_step + 1) if not (layers % 2 == 0 and 2 * step == layers)]


def _choose_side(max_side: int, force_parity: str | None, prefer_step_kind: str | None) -> int | None:
    options = []
    for side in range(5, max_side + 1):
        if force_parity == "odd" and side % 2 == 0:
            continue
        if force_parity == "even" and side % 2 == 1:
            continue
        steps = _valid_steps(side)
        if prefer_step_kind == "eq1" and 1 not in steps:
            continue
        if prefer_step_kind == "gt1" and not any(step > 1 for step in steps):
            continue
        options.append(side)
    if not options:
        return None
    return random.choice(options)


def _sample_step(side: int, prefer_step_kind: str | None) -> int:
    steps = _valid_steps(side)
    if prefer_step_kind == "eq1":
        steps = [step for step in steps if step == 1]
    elif prefer_step_kind == "gt1":
        gt1 = [step for step in steps if step > 1]
        if gt1:
            steps = gt1
    return random.choice(steps)


def _build_onion_spec(max_side: int, constraint: dict) -> dict | None:
    side = _choose_side(max_side, constraint.get("force_parity"), constraint.get("prefer_step_kind"))
    if side is None:
        return None
    layers = (side + 1) // 2
    gap_side = constraint.get("prefer_side") or random.choice(["top", "bottom"])
    step = _sample_step(side, constraint.get("prefer_step_kind"))
    layer_colors = random.sample(PALETTE, layers)
    patch = _render_complete_onion(side, layer_colors)
    depth = step - 1
    gap_col = random.randint(depth + 1, side - depth - 2)
    gap_row = depth if gap_side == "top" else side - 1 - depth
    patch = patch.copy()
    patch[gap_row, gap_col] = BG
    return {
        "side": side,
        "layers": layers,
        "gap_side": gap_side,
        "step": step,
        "layer_colors": layer_colors,
        "input_patch": patch,
    }


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _place_onions(shape: tuple[int, int], specs: list[dict]) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []
    ordered_specs = sorted(specs, key=lambda spec: spec["side"], reverse=True)
    for spec in ordered_specs:
        side = spec["side"]
        if side > h - 2 or side > w - 2:
            return None
        positions = []
        for _ in range(160):
            r0 = random.randint(1, h - side - 1)
            c0 = random.randint(1, w - side - 1)
            box = (r0, c0, r0 + side - 1, c0 + side - 1)
            if all(_boxes_separated(box, other, 2) for other in placed_boxes):
                positions.append((r0, c0))
                if len(positions) >= 12:
                    break
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        grid[r0 : r0 + side, c0 : c0 + side] = spec["input_patch"]
        spec["bbox"] = (r0, c0, r0 + side - 1, c0 + side - 1)
        placed_boxes.append(spec["bbox"])
    return grid


def _mode_constraints(mode: int) -> list[dict]:
    if mode == 0:
        return [{"prefer_side": "top", "prefer_step_kind": "eq1", "force_parity": "odd"}]
    if mode == 1:
        return [{"prefer_side": "bottom", "prefer_step_kind": "gt1", "force_parity": "even"}]
    if mode == 2:
        return [
            {"prefer_side": "top", "prefer_step_kind": "eq1", "force_parity": "odd"},
            {"prefer_side": "bottom", "prefer_step_kind": "gt1", "force_parity": "even"},
        ]
    if mode == 3:
        return [
            {"prefer_side": "top", "prefer_step_kind": "gt1", "force_parity": "even"},
            {"prefer_side": "bottom", "prefer_step_kind": "eq1", "force_parity": "odd"},
        ]
    if mode == 4:
        return [{"prefer_side": random.choice(["top", "bottom"]), "prefer_step_kind": "gt1", "force_parity": None}]
    return [
        {"prefer_side": "top", "prefer_step_kind": None, "force_parity": random.choice(["odd", "even"])},
        {"prefer_side": "bottom", "prefer_step_kind": None, "force_parity": random.choice(["odd", "even"])},
    ]


def _paint_transformed_onion(out: np.ndarray, info: dict, gap_side: str, step: int) -> None:
    r0, c0, r1, c1 = info["bbox"]
    rotated = _rotate_layer_colors(info["layer_colors"], step, gap_side)
    out[r0 : r1 + 1, c0 : c1 + 1] = _render_complete_onion(info["side"], rotated)


def _semantic_checks(grid: np.ndarray) -> bool:
    components = sorted(_find_non_bg_components(grid, BG), key=lambda coords: bbox_coords(coords))
    if not (1 <= len(components) <= 2):
        return False

    infos = [_analyze_onion(grid, coords, BG) for coords in components]
    if len(infos) == 2 and (infos[0]["gap_side"], infos[0]["step"]) == (infos[1]["gap_side"], infos[1]["step"]):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    fill_only = grid.copy()
    for info in infos:
        r0, c0, r1, c1 = info["bbox"]
        fill_only[r0 : r1 + 1, c0 : c1 + 1] = _render_complete_onion(info["side"], info["layer_colors"])
    if np.array_equal(out, fill_only):
        return False

    opposite = grid.copy()
    for info in infos:
        other_side = "bottom" if info["gap_side"] == "top" else "top"
        _paint_transformed_onion(opposite, info, other_side, info["step"])
    if np.array_equal(out, opposite):
        return False

    if any(info["step"] > 1 for info in infos):
        one_step = grid.copy()
        for info in infos:
            _paint_transformed_onion(one_step, info, info["gap_side"], 1)
        if np.array_equal(out, one_step):
            return False

    if len(infos) == 2:
        global_rule = grid.copy()
        shared_side = infos[0]["gap_side"]
        shared_step = infos[0]["step"]
        for info in infos:
            _paint_transformed_onion(global_rule, info, shared_side, shared_step)
        if np.array_equal(out, global_rule):
            return False

    return True


def _generate_attempt(mode: int) -> np.ndarray | None:
    constraints = _mode_constraints(mode)
    n_onions = len(constraints)
    if n_onions == 1:
        h = random.randint(12, 30)
        w = random.randint(12, 30)
        max_side = min(h - 2, w - 2, 16)
    else:
        h = random.randint(18, 30)
        w = random.randint(18, 30)
        max_side = min(h - 2, w - 2, 12)
    if max_side < 5:
        return None

    specs = []
    for constraint in constraints:
        spec = _build_onion_spec(max_side, constraint)
        if spec is None:
            return None
        specs.append(spec)

    grid = _place_onions((h, w), specs)
    if grid is None:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    mode = _GEN_CALL_COUNT % 6
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt(mode)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid onion-gap input")
