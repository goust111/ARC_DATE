import hashlib
import itertools
import math
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
_RNG = np.random.default_rng()
_SEEN_OUTPUTS: set[str] = set()
_GEN_ATTEMPTS = 2000


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _reduce_vector(dr: int, dc: int) -> tuple[int, int]:
    g = math.gcd(abs(int(dr)), abs(int(dc)))
    assert g > 0
    return int(dr // g), int(dc // g)


def _hash_grid(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _chain_steps(
    support: list[tuple[int, int]],
    top_left: tuple[int, int],
    vector: tuple[int, int],
    shape: tuple[int, int],
) -> list[set[tuple[int, int]]]:
    h, w = shape
    base_r, base_c = top_left
    dr, dc = vector
    steps: list[set[tuple[int, int]]] = []
    k = 1
    while True:
        visible: set[tuple[int, int]] = set()
        for sr, sc in support:
            rr = base_r + sr + k * dr
            cc = base_c + sc + k * dc
            if 0 <= rr < h and 0 <= cc < w:
                visible.add((rr, cc))
        if not visible:
            break
        steps.append(visible)
        k += 1
    return steps


def _render_expected(
    shape: tuple[int, int],
    body_color: int,
    output_occupied: set[tuple[int, int]],
    decoys: list[tuple[tuple[int, int], int]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for (r, c), color in decoys:
        out[r, c] = int(color)
    for r, c in output_occupied:
        out[r, c] = int(body_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Copy the anchored monochrome sprite repeatedly along each same-color offset chain.

    Input structure:
    - A mostly black grid contains one multi-cell monochrome sprite.
    - One to three differently colored anchor pixels are embedded inside that sprite.
    - Far away, each anchor color appears once more as an isolated control pixel; extra
      isolated pixels of unrelated colors may also appear as decoys.

    Rule:
    - For each anchor color, pair the embedded anchor with the unique external isolated
      pixel of the same color and read the translation vector from anchor to isolate.
    - Repaint every anchor position back to the sprite body color, then paste the whole
      sprite at offsets v, 2v, 3v, ... until a translated copy has no cell left in bounds.
    - Keep only in-bounds cells from partially clipped copies, and leave unrelated decoy
      single pixels unchanged.

    Key insight:
    - The same-color external single pixel is not a one-time destination marker; it gives
      the step vector of an entire independent replication chain, and all control pixels
      disappear into ordinary body-color cells in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    components = _find_components(g != bg)
    sprite_components = [comp for comp in components if len(comp) > 1]
    assert len(sprite_components) == 1
    sprite_coords = sprite_components[0]

    body_color = int(Counter(int(g[r, c]) for r, c in sprite_coords).most_common(1)[0][0])
    r0, c0, r1, c1 = _bbox(sprite_coords)
    sprite_crop = g[r0 : r1 + 1, c0 : c1 + 1]
    sprite_mask = sprite_crop != bg
    support = [tuple(map(int, rc)) for rc in np.argwhere(sprite_mask)]

    anchors: list[tuple[tuple[int, int], int]] = []
    for rr, cc in support:
        color = int(sprite_crop[rr, cc])
        if color != body_color:
            anchors.append(((rr, cc), color))
    assert 1 <= len(anchors) <= 3
    assert len({color for _, color in anchors}) == len(anchors)

    out = g.copy()
    for (rr, cc), _color in anchors:
        out[r0 + rr, c0 + cc] = body_color

    for (rr, cc), color in anchors:
        abs_anchor = (r0 + rr, c0 + cc)
        color_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        assert len(color_cells) == 2
        external = [cell for cell in color_cells if cell != abs_anchor]
        assert len(external) == 1
        dr = external[0][0] - abs_anchor[0]
        dc = external[0][1] - abs_anchor[1]

        for cells in _chain_steps(support, (r0, c0), (dr, dc), g.shape):
            for sr, sc in cells:
                out[sr, sc] = body_color

    return out


def _sprite_mask(anchor_count: int) -> np.ndarray | None:
    for _ in range(200):
        h = int(_RNG.integers(3, 6))
        w = int(_RNG.integers(3, 6))
        mask = np.ones((h, w), dtype=bool)

        removable = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        for r in range(h):
            for c in range(w):
                if (r in {0, h - 1} or c in {0, w - 1}) and (r, c) not in removable:
                    removable.append((r, c))
        _RNG.shuffle(removable)
        for cell in removable[: int(_RNG.integers(0, min(4, len(removable)) + 1))]:
            mask[cell] = False
            interiors = [
                (r, c)
                for r in range(1, h - 1)
                for c in range(1, w - 1)
                if mask[r, c]
            ]
            if (
                len(_find_components(mask)) != 1
                or len(interiors) < anchor_count
                or int(mask.sum()) < anchor_count + 4
            ):
                mask[cell] = True

        interiors = [
            (r, c)
            for r in range(1, h - 1)
            for c in range(1, w - 1)
            if mask[r, c]
        ]
        if len(interiors) >= anchor_count and int(mask.sum()) >= anchor_count + 4:
            return mask
    return None


def _vector_pool(sprite_h: int, sprite_w: int) -> list[tuple[int, int]]:
    row_steps = [sprite_h + d for d in range(1, 4)]
    col_steps = [sprite_w + d for d in range(1, 4)]
    pool: set[tuple[int, int]] = set()
    for dr in row_steps:
        for dc in col_steps:
            pool.update(
                {
                    (0, dc),
                    (0, -dc),
                    (dr, 0),
                    (-dr, 0),
                    (dr, 1),
                    (dr, -1),
                    (-dr, 1),
                    (-dr, -1),
                    (1, dc),
                    (-1, dc),
                    (1, -dc),
                    (-1, -dc),
                    (dr, dc),
                    (dr, -dc),
                    (-dr, dc),
                    (-dr, -dc),
                }
            )
    return list(pool)


def _pick_vectors(
    anchor_count: int,
    support: list[tuple[int, int]],
    top_left: tuple[int, int],
    shape: tuple[int, int],
) -> list[dict] | None:
    original = {(top_left[0] + r, top_left[1] + c) for r, c in support}
    sprite_h = max(r for r, _ in support) + 1
    sprite_w = max(c for _, c in support) + 1
    infos = []
    for vector in _vector_pool(sprite_h, sprite_w):
        steps = _chain_steps(support, top_left, vector, shape)
        if not steps:
            continue
        flat = set().union(*steps)
        if flat & original:
            continue
        if sum(len(step) for step in steps) != len(flat):
            continue
        infos.append(
            {
                "vector": vector,
                "direction": _reduce_vector(*vector),
                "steps": steps,
                "flat": flat,
                "chain_len": len(steps),
                "has_partial": any(len(step) < len(support) for step in steps),
            }
        )
    if not infos:
        return None

    multi = [info for info in infos if info["chain_len"] >= 2]
    preferred = [info for info in multi if info["has_partial"]] or multi
    if not preferred:
        return None

    for _ in range(100):
        first = preferred[int(_RNG.integers(len(preferred)))]
        chosen = [first]
        occupied = set(original) | set(first["flat"])
        directions = {first["direction"]}

        others = infos[:]
        _RNG.shuffle(others)
        for info in others:
            if info is first:
                continue
            if info["direction"] in directions:
                continue
            if occupied & info["flat"]:
                continue
            chosen.append(info)
            occupied |= set(info["flat"])
            directions.add(info["direction"])
            if len(chosen) == anchor_count:
                return chosen

        if anchor_count == 1:
            return chosen
    return None


def _isolated_from(
    pos: tuple[int, int],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    r, c = pos
    h, w = shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) in occupied:
                return False
    return True


def _assign_anchors(
    interiors: list[tuple[int, int]],
    vectors: list[dict],
    top_left: tuple[int, int],
    original_support: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    for rels in itertools.permutations(interiors, len(vectors)):
        singles: list[tuple[int, int]] = []
        occupied = set(original_support)
        ok = True
        for rel, info in zip(rels, vectors):
            pos = (top_left[0] + rel[0] + info["vector"][0], top_left[1] + rel[1] + info["vector"][1])
            if not _isolated_from(pos, occupied, shape):
                ok = False
                break
            singles.append(pos)
            occupied.add(pos)
        if ok:
            return list(rels), singles
    return None


def _validate_case(
    grid: np.ndarray,
    expected: np.ndarray,
    body_color: int,
    anchor_colors: list[int],
    matched_singles: list[tuple[int, int]],
    decoys: list[tuple[tuple[int, int], int]],
    selected_vectors: list[dict],
    support_size: int,
) -> bool:
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(grid, expected):
        return False
    if np.count_nonzero(expected == body_color) <= support_size:
        return False
    if not any(info["chain_len"] >= 2 for info in selected_vectors):
        return False
    if not any(info["has_partial"] for info in selected_vectors):
        return False
    for color in anchor_colors:
        if np.count_nonzero(grid == color) != 2:
            return False
    for r, c in matched_singles:
        if expected[r, c] != body_color:
            return False
    for (r, c), color in decoys:
        if expected[r, c] != color:
            return False
    comps = _find_components(grid != BG)
    if sum(len(comp) > 1 for comp in comps) != 1:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(18, 29))
    w = int(_RNG.integers(18, 29))
    anchor_count = int(_RNG.choice([1, 2, 3], p=[0.25, 0.45, 0.30]))

    mask = _sprite_mask(anchor_count)
    if mask is None:
        return None
    sprite_h, sprite_w = mask.shape
    support = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    interiors = [
        (r, c)
        for r in range(1, sprite_h - 1)
        for c in range(1, sprite_w - 1)
        if mask[r, c]
    ]

    row_center = max(2, min(h - sprite_h - 2, h // 2 - sprite_h // 2 + int(_RNG.integers(-2, 3))))
    col_center = max(2, min(w - sprite_w - 2, w // 2 - sprite_w // 2 + int(_RNG.integers(-2, 3))))
    top_left = (row_center, col_center)
    original_support = {(top_left[0] + r, top_left[1] + c) for r, c in support}

    selected_vectors = _pick_vectors(anchor_count, support, top_left, (h, w))
    if selected_vectors is None:
        return None

    assigned = _assign_anchors(interiors, selected_vectors, top_left, original_support, (h, w))
    if assigned is None:
        return None
    anchor_rels, matched_singles = assigned

    body_color = int(_RNG.choice(PALETTE))
    remaining = [color for color in PALETTE if color != body_color]
    _RNG.shuffle(remaining)
    anchor_colors = remaining[:anchor_count]
    remaining = remaining[anchor_count:]
    decoy_count = int(min(len(remaining), _RNG.integers(1, 4)))
    decoy_colors = remaining[:decoy_count]

    output_occupied = set(original_support)
    for info in selected_vectors:
        output_occupied |= set(info["flat"])

    decoy_candidates = []
    for r in range(h):
        for c in range(w):
            pos = (r, c)
            if pos in output_occupied or pos in matched_singles:
                continue
            if _isolated_from(pos, output_occupied | set(matched_singles), (h, w)):
                decoy_candidates.append(pos)
    if len(decoy_candidates) < decoy_count:
        return None
    _RNG.shuffle(decoy_candidates)
    decoy_positions: list[tuple[int, int]] = []
    decoy_occupied = output_occupied | set(matched_singles)
    for pos in decoy_candidates:
        if _isolated_from(pos, decoy_occupied, (h, w)):
            decoy_positions.append(pos)
            decoy_occupied.add(pos)
            if len(decoy_positions) == decoy_count:
                break
    if len(decoy_positions) < decoy_count:
        return None
    decoys = list(zip(decoy_positions, decoy_colors))

    grid = np.full((h, w), BG, dtype=int)
    for r, c in original_support:
        grid[r, c] = body_color
    for rel, color in zip(anchor_rels, anchor_colors):
        grid[top_left[0] + rel[0], top_left[1] + rel[1]] = int(color)
    for pos, color in zip(matched_singles, anchor_colors):
        grid[pos] = int(color)
    for (r, c), color in decoys:
        grid[r, c] = int(color)

    expected = _render_expected((h, w), body_color, output_occupied, decoys)
    if not _validate_case(
        grid,
        expected,
        body_color,
        anchor_colors,
        matched_singles,
        decoys,
        selected_vectors,
        len(original_support),
    ):
        return None

    output_sig = _hash_grid(expected)
    if output_sig in _SEEN_OUTPUTS:
        return None
    _SEEN_OUTPUTS.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 8")
