from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = list(range(1, 10))

CHAIN_LEN_RANGE = (3, 5)
CHAIN_HEAD_OUTER_RANGE = (9, 14)
TAIL_OUTER_RANGE = (3, 8)
THICKNESS_CHOICES = (1, 1, 2, 2, 3)

GRID_RANGE = (22, 30)
PLACEMENT_GAP = 1
GEN_RETRIES = 500
PLACE_RETRIES = 40
POSITION_SAMPLES = 240


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    components: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        seen = np.zeros((h, w), dtype=bool)
        mask = grid == int(color)
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue
                queue: deque[tuple[int, int]] = deque([(r, c)])
                seen[r, c] = True
                coords: list[tuple[int, int]] = []
                while queue:
                    rr, cc = queue.popleft()
                    coords.append((rr, cc))
                    for nr, nc in _neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            queue.append((nr, nc))
                rows = [rr for rr, _ in coords]
                cols = [cc for _, cc in coords]
                components.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": (min(rows), min(cols), max(rows), max(cols)),
                        "area": len(coords),
                    }
                )
    return components


def _make_ring(outer: int, thickness: int, color: int, bg: int = BG) -> np.ndarray:
    patch = np.full((outer, outer), int(color), dtype=int)
    hole = outer - 2 * thickness
    patch[thickness : thickness + hole, thickness : thickness + hole] = int(bg)
    return patch


def _ring_signature(grid: np.ndarray, component: dict, bg: int) -> dict | None:
    r0, c0, r1, c1 = component["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    color = int(component["color"])
    h, w = patch.shape
    if h != w or h < 3:
        return None
    if not np.all((patch == bg) | (patch == color)):
        return None

    mask = patch == color
    outer = int(h)
    for thickness in range(1, outer // 2 + 1):
        hole = outer - 2 * thickness
        if hole < 1:
            continue
        expected = np.ones((outer, outer), dtype=bool)
        expected[thickness : thickness + hole, thickness : thickness + hole] = False
        if np.array_equal(mask, expected):
            return {
                "color": color,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "outer": outer,
                "thickness": int(thickness),
                "hole": int(hole),
            }
    return None


def _extract_rings(grid: np.ndarray, bg: int) -> list[dict]:
    rings: list[dict] = []
    for component in _find_monochrome_components(grid, bg):
        ring = _ring_signature(grid, component, bg)
        if ring is not None:
            rings.append(ring)
    return rings


def _find_chain_indices(rings: list[dict]) -> list[int]:
    n = len(rings)
    children = [[] for _ in range(n)]
    parents = [[] for _ in range(n)]
    for i, outer_ring in enumerate(rings):
        for j, inner_ring in enumerate(rings):
            if i == j:
                continue
            if int(outer_ring["hole"]) == int(inner_ring["outer"]):
                children[i].append(j)
                parents[j].append(i)

    active = [i for i in range(n) if children[i] or parents[i]]
    assert active, "expected one nontrivial exact-fit chain"
    assert all(len(children[i]) <= 1 and len(parents[i]) <= 1 for i in active), "chain must be unique"

    heads = [i for i in active if not parents[i]]
    tails = [i for i in active if not children[i]]
    assert len(heads) == 1 and len(tails) == 1, "expected one path-shaped assembly chain"

    chain: list[int] = []
    seen: set[int] = set()
    cur = heads[0]
    while True:
        assert cur not in seen, "chain must not cycle"
        chain.append(cur)
        seen.add(cur)
        if not children[cur]:
            break
        cur = children[cur][0]

    assert seen == set(active), "all matched rings must belong to the same chain"
    assert len(chain) >= 2, "expected at least two rings in the chain"
    return chain


def _render_chain(rings: list[dict], chain: list[int], bg: int) -> np.ndarray:
    head_outer = int(rings[chain[0]]["outer"])
    out = np.full((head_outer, head_outer), int(bg), dtype=int)
    offset = 0
    for idx in chain:
        ring = rings[idx]
        patch = _make_ring(int(ring["outer"]), int(ring["thickness"]), int(ring["color"]), bg)
        size = patch.shape[0]
        view = out[offset : offset + size, offset : offset + size]
        mask = patch != int(bg)
        view[mask] = patch[mask]
        offset += int(ring["thickness"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the exact-fit square-ring chain and discard the non-matching distractors.

    Input structure:
    - A black background contains several separated monochrome hollow square rings, each with its own color.
    - Every ring has a square outer bbox, a centered square cavity, and wall thickness at least one cell.
    - Exactly one subset of rings forms an exact-fit chain where one ring's cavity side length equals the next ring's outer side length.

    Rule:
    - Detect every square ring and compute its outer size and cavity size.
    - Keep only the unique chain defined by `hole size == next outer size`; drop all isolated distractor rings.
    - Output a square canvas as large as the chain head's outer size, then place the kept rings concentrically so each inner ring exactly fills the previous cavity. The innermost cavity stays black.

    Key insight:
    - The assembly order is determined by exact cavity-to-outer equality, not by simple size sorting, because distractor rings can be size-interleaved yet fit nowhere.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _bg_color(g)
    rings = _extract_rings(g, bg)
    assert rings, "expected at least one square ring"
    chain = _find_chain_indices(rings)
    return _render_chain(rings, chain, bg)


def _sample_chain_specs() -> list[dict] | None:
    length = random.randint(*CHAIN_LEN_RANGE)
    thicknesses = [0] * length
    outers = [0] * length
    holes = [0] * length

    for idx in range(length - 1):
        thicknesses[idx] = int(random.choice(THICKNESS_CHOICES))
    if max(thicknesses[:-1]) < 2:
        bump_idx = random.randrange(length - 1)
        thicknesses[bump_idx] = random.choice((2, 3))

    delta = 2 * sum(thicknesses[:-1])
    tail_lo = max(TAIL_OUTER_RANGE[0], CHAIN_HEAD_OUTER_RANGE[0] - delta)
    tail_hi = min(TAIL_OUTER_RANGE[1], CHAIN_HEAD_OUTER_RANGE[1] - delta)
    if tail_hi < tail_lo:
        return None

    tail_outer = random.randint(tail_lo, tail_hi)
    tail_thickness_max = min(3, (tail_outer - 1) // 2)
    if tail_thickness_max < 1:
        return None
    thicknesses[-1] = random.randint(1, tail_thickness_max)

    outers[-1] = tail_outer
    holes[-1] = outers[-1] - 2 * thicknesses[-1]
    for idx in range(length - 2, -1, -1):
        outers[idx] = outers[idx + 1] + 2 * thicknesses[idx]
        holes[idx] = outers[idx + 1]

    return [
        {"outer": int(outers[i]), "thickness": int(thicknesses[i]), "hole": int(holes[i]), "role": "chain"}
        for i in range(length)
    ]


def _valid_distractor_specs(chain_specs: list[dict], selected: list[dict]) -> list[dict]:
    outers = {int(spec["outer"]) for spec in chain_specs + selected}
    holes = {int(spec["hole"]) for spec in chain_specs + selected}
    candidates: list[dict] = []
    for outer in range(3, 19):
        for thickness in range(1, min(3, (outer - 1) // 2) + 1):
            hole = outer - 2 * thickness
            if hole < 1:
                continue
            if outer in outers or outer in holes or hole in outers or hole in holes:
                continue
            candidates.append(
                {"outer": int(outer), "thickness": int(thickness), "hole": int(hole), "role": "distractor"}
            )
    return candidates


def _between_links(spec: dict, chain_specs: list[dict]) -> bool:
    outer = int(spec["outer"])
    for idx in range(len(chain_specs) - 1):
        if int(chain_specs[idx]["outer"]) > outer > int(chain_specs[idx + 1]["outer"]):
            return True
    return False


def _pick_distractors(chain_specs: list[dict]) -> list[dict] | None:
    head_outer = int(chain_specs[0]["outer"])
    chain_parity = int(chain_specs[0]["outer"]) % 2
    selected: list[dict] = []

    candidates = _valid_distractor_specs(chain_specs, selected)
    between = [spec for spec in candidates if _between_links(spec, chain_specs)]
    if not between:
        return None
    selected.append(random.choice(between))

    candidates = _valid_distractor_specs(chain_specs, selected)
    larger = [spec for spec in candidates if head_outer < int(spec["outer"]) <= min(18, head_outer + 2)]
    if not larger:
        return None
    same_parity = [spec for spec in larger if int(spec["outer"]) % 2 == chain_parity]
    selected.append(random.choice(same_parity or larger))

    return selected


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_boxes(height: int, width: int, sizes: list[int]) -> list[tuple[int, int, int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx], reverse=True)
    for _ in range(PLACE_RETRIES):
        placed: dict[int, tuple[int, int, int, int]] = {}
        ok = True
        for obj_idx in order:
            size = sizes[obj_idx]
            row_lo = 1 if height - size >= 2 else 0
            row_hi = height - size - 1 if height - size >= 2 else height - size
            col_lo = 1 if width - size >= 2 else 0
            col_hi = width - size - 1 if width - size >= 2 else width - size
            if row_hi < row_lo or col_hi < col_lo:
                return None

            box = None
            for _ in range(POSITION_SAMPLES):
                r = random.randint(row_lo, row_hi)
                c = random.randint(col_lo, col_hi)
                candidate = (r, c, r + size - 1, c + size - 1)
                if all(_boxes_separated(candidate, other, PLACEMENT_GAP) for other in placed.values()):
                    box = candidate
                    break
            if box is None:
                ok = False
                break
            placed[obj_idx] = box

        if ok:
            return [placed[idx] for idx in range(len(sizes))]
    return None


def _render_input(specs: list[dict], boxes: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec, box in zip(specs, boxes):
        r0, c0, r1, c1 = box
        size = r1 - r0 + 1
        patch = _make_ring(size, int(spec["thickness"]), int(spec["color"]), BG)
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch
    return grid


def _render_expected_from_specs(chain_specs: list[dict]) -> np.ndarray:
    return _render_chain(chain_specs, list(range(len(chain_specs))), BG)


def _validate_generated(grid: np.ndarray, chain_specs: list[dict], distractor_specs: list[dict]) -> bool:
    rings = _extract_rings(grid, BG)
    if len(rings) != len(chain_specs) + len(distractor_specs):
        return False

    try:
        chain = _find_chain_indices(rings)
        out = main(grid)
    except AssertionError:
        return False

    expected = _render_expected_from_specs(chain_specs)
    if not np.array_equal(out, expected):
        return False
    if len(chain) != len(chain_specs):
        return False

    output_colors = set(map(int, np.unique(out))) - {BG}
    chain_colors = {int(spec["color"]) for spec in chain_specs}
    distractor_colors = {int(spec["color"]) for spec in distractor_specs}
    if output_colors != chain_colors:
        return False
    if output_colors & distractor_colors:
        return False

    has_between = any(_between_links(spec, chain_specs) for spec in distractor_specs)
    has_larger = any(int(spec["outer"]) > int(chain_specs[0]["outer"]) for spec in distractor_specs)
    if not (has_between and has_larger):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        chain_specs = _sample_chain_specs()
        if chain_specs is None:
            continue

        distractor_specs = _pick_distractors(chain_specs)
        if distractor_specs is None:
            continue

        all_specs = chain_specs + distractor_specs
        if len(all_specs) > len(PALETTE):
            continue

        colors = random.sample(PALETTE, len(all_specs))
        for spec, color in zip(all_specs, colors):
            spec["color"] = int(color)

        max_outer = max(int(spec["outer"]) for spec in all_specs)
        bbox_area = sum(int(spec["outer"]) ** 2 for spec in all_specs)
        density_side = int(np.ceil(np.sqrt(bbox_area) * 1.2))
        min_side = max(GRID_RANGE[0], max_outer + 2, density_side)
        if min_side > GRID_RANGE[1]:
            continue
        height = random.randint(min_side, GRID_RANGE[1])
        width = random.randint(min_side, GRID_RANGE[1])

        shuffled = all_specs[:]
        random.shuffle(shuffled)
        sizes = [int(spec["outer"]) for spec in shuffled]
        boxes = _place_boxes(height, width, sizes)
        if boxes is None:
            continue

        grid = _render_input(shuffled, boxes, (height, width))
        if _validate_generated(grid, chain_specs, distractor_specs):
            return grid

    raise RuntimeError("failed to generate a valid exact-fit square-ring chain puzzle")
