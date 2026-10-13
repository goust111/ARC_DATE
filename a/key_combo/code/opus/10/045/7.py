from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


TEMPLATE_SIZE = 4
TEMPLATE_OFFSETS = tuple(
    (r, c)
    for r in range(TEMPLATE_SIZE)
    for c in range(TEMPLATE_SIZE)
    if r in (0, TEMPLATE_SIZE - 1) or c in (0, TEMPLATE_SIZE - 1)
)
TEMPLATE_SET = set(TEMPLATE_OFFSETS)
SIDE_NAMES = ("top", "bottom", "left", "right")
REL_PATTERNS = (
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
)
STEP_CHOICES = (2, 3, 4)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(g: np.ndarray, color: int) -> list[np.ndarray]:
    coords = np.argwhere(g == color)
    if coords.size == 0:
        return []
    cells = {tuple(map(int, rc)) for rc in coords}
    seen: set[tuple[int, int]] = set()
    comps: list[np.ndarray] = []
    for cell in sorted(cells):
        if cell in seen:
            continue
        q = deque([cell])
        seen.add(cell)
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in cells and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        comps.append(np.array(comp, dtype=int))
    return comps


def _template_cells(anchor: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0 = anchor
    return tuple((r0 + dr, c0 + dc) for dr, dc in TEMPLATE_OFFSETS)


def _template_in_bounds(anchor: tuple[int, int], shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0 = anchor
    return 0 <= r0 and 0 <= c0 and r0 + TEMPLATE_SIZE <= h and c0 + TEMPLATE_SIZE <= w


def _fit_unique_anchor(visible_coords: np.ndarray, shape: tuple[int, int]) -> tuple[int, int]:
    coords = [tuple(map(int, rc)) for rc in visible_coords]
    candidates: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr, dc in TEMPLATE_OFFSETS:
            anchor = (r - dr, c - dc)
            if not _template_in_bounds(anchor, shape):
                continue
            if all((rr - anchor[0], cc - anchor[1]) in TEMPLATE_SET for rr, cc in coords):
                candidates.add(anchor)
    assert len(candidates) == 1, "visible fragment must fit a unique 4x4 hollow-square anchor"
    return next(iter(candidates))


def _is_bar_component(comp: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    if len(comp) != TEMPLATE_SIZE:
        return False
    r0, c0, r1, c1 = _bbox(comp)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if (height, width) not in ((1, TEMPLATE_SIZE), (TEMPLATE_SIZE, 1)):
        return False
    return r0 in (0, h - TEMPLATE_SIZE) or c0 in (0, w - TEMPLATE_SIZE)


def _match_bar(
    line_candidates: list[np.ndarray],
    anchor: tuple[int, int],
    shape: tuple[int, int],
) -> dict[str, object]:
    h, w = shape
    ar, ac = anchor
    matches = []
    for comp in line_candidates:
        r0, c0, r1, c1 = _bbox(comp)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if (bh, bw) == (1, TEMPLATE_SIZE) and c0 == ac and c1 == ac + TEMPLATE_SIZE - 1:
            step = (-TEMPLATE_SIZE, 0) if r0 < ar else (TEMPLATE_SIZE, 0)
            distance = abs(ar - r0)
            if distance % TEMPLATE_SIZE == 0 and distance >= 2 * TEMPLATE_SIZE:
                matches.append(
                    {
                        "bbox": (r0, c0, r1, c1),
                        "bar_anchor": (r0, c0),
                        "step": step,
                        "side": "top" if r0 < ar else "bottom",
                    }
                )
        elif (bh, bw) == (TEMPLATE_SIZE, 1) and r0 == ar and r1 == ar + TEMPLATE_SIZE - 1:
            step = (0, -TEMPLATE_SIZE) if c0 < ac else (0, TEMPLATE_SIZE)
            distance = abs(ac - c0)
            if distance % TEMPLATE_SIZE == 0 and distance >= 2 * TEMPLATE_SIZE:
                matches.append(
                    {
                        "bbox": (r0, c0, r1, c1),
                        "bar_anchor": (r0, c0),
                        "step": step,
                        "side": "left" if c0 < ac else "right",
                    }
                )
    assert len(matches) == 1, "each sprite must match exactly one aligned control bar"
    bar = matches[0]
    br, bc = bar["bar_anchor"]
    assert _template_in_bounds((br, bc), (h, w))
    return bar


def _parse_scene(g: np.ndarray) -> dict[int, dict[str, object]]:
    h, w = g.shape
    colors = [int(color) for color in np.unique(g) if int(color) != 0]
    assert 3 <= len(colors) <= 4
    sprites: dict[int, dict[str, object]] = {}
    for color in colors:
        comps = _components_of_color(g, color)
        line_candidates = [comp for comp in comps if _is_bar_component(comp, (h, w))]
        visible_parts = [comp for comp in comps if comp.tolist() not in [lc.tolist() for lc in line_candidates]]
        if visible_parts:
            visible_coords = np.concatenate(visible_parts, axis=0)
        else:
            visible_coords = np.zeros((0, 2), dtype=int)
        assert len(visible_coords) > 0, "each color needs a visible sprite fragment"
        anchor = _fit_unique_anchor(visible_coords, (h, w))
        bar = _match_bar(line_candidates, anchor, (h, w))
        sprites[color] = {
            "anchor": anchor,
            "visible": {tuple(map(int, rc)) for rc in visible_coords},
            "expected": set(_template_cells(anchor)),
            "bar_anchor": bar["bar_anchor"],
            "step": bar["step"],
            "side": bar["side"],
            "bar_bbox": bar["bbox"],
        }
    return sprites


def _occlusion_order(g: np.ndarray, sprites: dict[int, dict[str, object]]) -> tuple[list[int], dict[tuple[int, int], int]]:
    colors = set(sprites)
    above_to_below = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    pair_cover_count: dict[tuple[int, int], int] = defaultdict(int)

    for color, info in sprites.items():
        for r, c in info["expected"]:
            seen = int(g[r, c])
            if seen == color:
                continue
            assert seen in colors, "missing template pixels must be covered by another sprite color"
            pair_cover_count[(seen, color)] += 1
            if color not in above_to_below[seen]:
                above_to_below[seen].add(color)
                indegree[color] += 1

    order: list[int] = []
    remaining = set(colors)
    while remaining:
        tops = sorted(color for color in remaining if indegree[color] == 0)
        assert len(tops) == 1, "occlusion relations must determine a unique top-to-bottom order"
        top = tops[0]
        order.append(top)
        remaining.remove(top)
        for below in above_to_below[top]:
            indegree[below] -= 1
    return order, pair_cover_count


def _paint_template(out: np.ndarray, anchor: tuple[int, int], color: int) -> tuple[int, int]:
    painted = 0
    skipped = 0
    for dr, dc in TEMPLATE_OFFSETS:
        r = anchor[0] + dr
        c = anchor[1] + dc
        if out[r, c] == 0:
            out[r, c] = color
            painted += 1
        else:
            skipped += 1
    return painted, skipped


def _render_with_order(
    shape: tuple[int, int],
    sprites: dict[int, dict[str, object]],
    order: list[int],
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.zeros(shape, dtype=int)
    stats = {"skipped": 0, "skipped_nonorigin": 0}
    for color in order:
        info = sprites[color]
        current = tuple(info["anchor"])
        step = tuple(info["step"])
        target = tuple(info["bar_anchor"])
        while True:
            assert _template_in_bounds(current, shape)
            _painted, skipped = _paint_template(out, current, color)
            stats["skipped"] += skipped
            if current != tuple(info["anchor"]):
                stats["skipped_nonorigin"] += skipped
            if current == target:
                break
            nxt = (current[0] + step[0], current[1] + step[1])
            assert _template_in_bounds(nxt, shape)
            current = nxt
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each overlapped 4x4 hollow-square sprite, infer the top-to-bottom occlusion order, and tile complete copies toward its matched control bar.

    Input structure:
    - The background is black.
    - Three or four different colors show visible fragments of the same 4x4 hollow-square border template, with partial overlaps hiding lower-layer border pixels.
    - Each sprite color also has one matching 1-cell-thick control bar aligned with the sprite's row span or column span near a grid edge.

    Rule:
    - For each color, fit its visible fragment to a unique translated 4x4 hollow-square template.
    - Whenever another sprite color occupies a template pixel that should belong to the current color, that other color is above it; topologically sort these relations to get a unique front-to-back order.
    - Erase the control bars, then process sprites from top to bottom: redraw the full template at the original anchor and every 4 cells toward the matched bar until the bar position is reached, skipping any pixel already occupied by an earlier sprite.

    Key insight:
    - The hidden layering order is reused as the tiling priority, so the same occlusion analysis that reconstructs the original overlap also determines which repeated copies are allowed to claim shared space.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sprites = _parse_scene(g)
    order, _ = _occlusion_order(g, sprites)
    out, _ = _render_with_order(g.shape, sprites, order)
    return out


def _choose_side_assignment(k: int) -> tuple[str, ...]:
    if k == 4:
        return tuple(random.sample(SIDE_NAMES, 4))
    subset = random.sample(SIDE_NAMES, 3)
    return tuple(random.sample(subset, 3))


def _solve_axis(
    rel_values: tuple[int, ...],
    side_to_idx: dict[str, int],
    neg_name: str,
    pos_name: str,
    neg_steps: int | None,
    pos_steps: int | None,
) -> tuple[int, int]:
    rel_max = max(rel_values)
    neg_idx = side_to_idx.get(neg_name)
    pos_idx = side_to_idx.get(pos_name)

    if neg_idx is not None:
        base_from_neg = TEMPLATE_SIZE * neg_steps - rel_values[neg_idx]
    else:
        base_from_neg = None

    if neg_idx is not None and pos_idx is not None:
        base = base_from_neg
        size = base + rel_values[pos_idx] + TEMPLATE_SIZE * (pos_steps + 1)
        if base < 0 or size > 30 or base + rel_max + TEMPLATE_SIZE > size:
            raise ValueError("axis constraints out of bounds")
        return base, size

    if neg_idx is not None:
        base = base_from_neg
        min_size = base + rel_max + TEMPLATE_SIZE
        choices = [size for size in range(min_size + 2, 31) if size <= 30]
        if not choices:
            raise ValueError("no slack on singly constrained axis")
        return base, random.choice(choices)

    if pos_idx is not None:
        candidates = []
        for base in range(0, 20):
            size = base + rel_values[pos_idx] + TEMPLATE_SIZE * (pos_steps + 1)
            if size <= 30 and base + rel_max + TEMPLATE_SIZE <= size - 2:
                candidates.append((base, size))
        if not candidates:
            raise ValueError("no placement for positive-side axis")
        return random.choice(candidates)

    raise ValueError("axis must be constrained by at least one side")


def _anchor_and_bar_for_side(
    anchor: tuple[int, int],
    side: str,
    shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int]]:
    h, w = shape
    r0, c0 = anchor
    if side == "top":
        return anchor, (0, c0), (0, c0, 0, c0 + TEMPLATE_SIZE - 1)
    if side == "bottom":
        br = h - TEMPLATE_SIZE
        return anchor, (br, c0), (br, c0, br, c0 + TEMPLATE_SIZE - 1)
    if side == "left":
        return anchor, (r0, 0), (r0, 0, r0 + TEMPLATE_SIZE - 1, 0)
    bc = w - TEMPLATE_SIZE
    return anchor, (r0, bc), (r0, bc, r0 + TEMPLATE_SIZE - 1, bc)


def _construct_candidate() -> tuple[np.ndarray, list[int], dict[int, dict[str, object]]]:
    k = random.choice((3, 4))
    rels = REL_PATTERNS[random.randrange(len(REL_PATTERNS))][:k]
    colors = random.sample([1, 2, 3, 4], k)
    sides = _choose_side_assignment(k)
    side_to_idx = {side: i for i, side in enumerate(sides)}
    steps = {side: random.choice(STEP_CHOICES) for side in side_to_idx}

    rel_rows = tuple(r for r, _ in rels)
    rel_cols = tuple(c for _, c in rels)
    base_r, h = _solve_axis(
        rel_rows,
        side_to_idx,
        "top",
        "bottom",
        steps.get("top"),
        steps.get("bottom"),
    )
    base_c, w = _solve_axis(
        rel_cols,
        side_to_idx,
        "left",
        "right",
        steps.get("left"),
        steps.get("right"),
    )

    anchors = [(base_r + dr, base_c + dc) for dr, dc in rels]
    assert all(_template_in_bounds(anchor, (h, w)) for anchor in anchors)

    sprite_meta: dict[int, dict[str, object]] = {}
    bar_rects: list[tuple[int, int, int, int]] = []
    for color, anchor, side in zip(colors, anchors, sides):
        _anchor, bar_anchor, bar_bbox = _anchor_and_bar_for_side(anchor, side, (h, w))
        sprite_meta[color] = {
            "anchor": anchor,
            "bar_anchor": bar_anchor,
            "side": side,
            "step": (bar_anchor[0] - anchor[0], bar_anchor[1] - anchor[1]),
            "bar_bbox": bar_bbox,
            "expected": set(_template_cells(anchor)),
        }
        bar_rects.append(bar_bbox)

    grid = np.zeros((h, w), dtype=int)
    for color in reversed(colors):
        for r, c in sprite_meta[color]["expected"]:
            grid[r, c] = color
    for color in colors:
        r0, c0, r1, c1 = sprite_meta[color]["bar_bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid, colors, sprite_meta


def _validate_generated_grid(
    grid: np.ndarray,
    expected_order: list[int],
    designed: dict[int, dict[str, object]],
) -> bool:
    sprites = _parse_scene(grid)
    order, pair_cover_count = _occlusion_order(grid, sprites)
    if order != expected_order:
        return False

    for color, info in designed.items():
        if tuple(sprites[color]["anchor"]) != tuple(info["anchor"]):
            return False
        if tuple(sprites[color]["bar_anchor"]) != tuple(info["bar_anchor"]):
            return False
        if tuple(sprites[color]["step"]) != tuple(
            np.sign(np.array(info["bar_anchor"]) - np.array(info["anchor"])) * TEMPLATE_SIZE
        ):
            return False

    for above, below in zip(order, order[1:]):
        if pair_cover_count[(above, below)] < 2:
            return False

    axes = {"vertical" if sprites[color]["step"][0] else "horizontal" for color in order}
    if len(axes) < 2:
        return False

    out, stats = _render_with_order(grid.shape, sprites, order)
    if stats["skipped"] <= 0:
        return False
    reverse_out, _ = _render_with_order(grid.shape, sprites, list(reversed(order)))
    if np.array_equal(out, reverse_out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, order, designed = _construct_candidate()
            if _validate_generated_grid(grid, order, designed):
                return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid new puzzle input")
