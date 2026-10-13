import random
from collections import defaultdict

import numpy as np


BG = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _extract_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == BG or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in NEIGHBORS4:
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if seen[rr, cc] or int(g[rr, cc]) != color:
                        continue
                    seen[rr, cc] = True
                    stack.append((rr, cc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            r1 = min(rows)
            r2 = max(rows)
            c1 = min(cols)
            c2 = max(cols)
            mask = np.zeros((r2 - r1 + 1, c2 - c1 + 1), dtype=bool)
            rel_coords: list[tuple[int, int]] = []
            for r, c in coords:
                rr = r - r1
                cc = c - c1
                mask[rr, cc] = True
                rel_coords.append((rr, cc))

            components.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "bbox": (r1, c1, r2, c2),
                    "mask": mask,
                    "rel_coords": tuple(rel_coords),
                }
            )

    return components


def _horizontal_pair_solutions(mask_left: np.ndarray, mask_right: np.ndarray) -> list[int]:
    h, w = mask_left.shape
    assert mask_right.shape == (h, w)
    solutions: list[int] = []

    for shift in range(0, w + 1):
        overlap = w - shift
        if not bool(mask_left[:, :shift].all()):
            continue
        if not bool(mask_right[:, overlap:].all()):
            continue
        if overlap > 0 and not np.array_equal(mask_left[:, shift:], ~mask_right[:, :overlap]):
            continue
        solutions.append(shift)

    return solutions


def _pairable(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return bool(_horizontal_pair_solutions(mask_a, mask_b) or _horizontal_pair_solutions(mask_b, mask_a))


def _unique_matching(group: list[dict]) -> list[tuple[dict, dict]]:
    n = len(group)
    assert n % 2 == 0 and n > 0

    pairable = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            ok = _pairable(group[i]["mask"], group[j]["mask"])
            pairable[i][j] = ok
            pairable[j][i] = ok

    solutions: list[list[tuple[int, int]]] = []
    used = [False] * n
    current: list[tuple[int, int]] = []

    def backtrack() -> None:
        if len(solutions) > 1:
            return

        try:
            i = next(idx for idx, taken in enumerate(used) if not taken)
        except StopIteration:
            solutions.append(current.copy())
            return

        used[i] = True
        for j in range(i + 1, n):
            if used[j] or not pairable[i][j]:
                continue
            used[j] = True
            current.append((i, j))
            backtrack()
            current.pop()
            used[j] = False
        used[i] = False

    backtrack()
    assert len(solutions) == 1, f"ambiguous or missing pairing for group size {n}"
    return [(group[i], group[j]) for i, j in solutions[0]]


def _is_left(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    cols_a = np.argwhere(mask_a)[:, 1]
    cols_b = np.argwhere(mask_b)[:, 1]
    sum_a = int(cols_a.sum())
    sum_b = int(cols_b.sum())
    area_a = int(cols_a.size)
    area_b = int(cols_b.size)
    lhs = sum_a * area_b
    rhs = sum_b * area_a
    assert lhs != rhs, "left/right tie is not allowed"
    return lhs < rhs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered monochrome fragments by translational shape complement, then swap colors by left/right role.

    Input structure:
    - bg = black.
    - The grid contains an even number of monochrome 4-connected fragments.
    - Within each hidden pair, the two fragments have the same bbox size.
    - Each pair can be translated horizontally to tile a solid rectangle without overlap or gaps.

    Rule:
    - Group fragments by bbox size.
    - Within each size-group, pair fragments whose shapes are complementary under a horizontal translation.
    - Compare each paired fragment's mean relative column inside its own bbox.
    - Recolor the left-leaning fragment with the right fragment's color, and vice versa.
    - Keep all fragment positions fixed.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    components = _extract_components(g)
    assert components and len(components) % 2 == 0

    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for comp in components:
        groups[comp["mask"].shape].append(comp)

    for group in groups.values():
        for comp_a, comp_b in _unique_matching(group):
            if _is_left(comp_a["mask"], comp_b["mask"]):
                left, right = comp_a, comp_b
            else:
                left, right = comp_b, comp_a

            for r, c in left["coords"]:
                out[r, c] = int(right["color"])
            for r, c in right["coords"]:
                out[r, c] = int(left["color"])

    return out


# --------------- generator ---------------

GEN_PAIR_COUNT_CHOICES = (2, 3, 4)
GEN_BBOX_HEIGHTS = (3, 4, 5, 6)
GEN_BBOX_WIDTHS = (3, 4, 5, 6)
GEN_REPEAT_SIZE_PROB = 0.35
GEN_GRID_MIN = 14
GEN_GRID_MAX = 34
GEN_PLACEMENT_PAD = 1
GEN_RETRIES = 400


def _mask_signature(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes()


def _build_pair_masks(height: int, width: int, shift: int, cuts: list[int]) -> tuple[np.ndarray, np.ndarray]:
    overlap = width - shift
    left = np.zeros((height, width), dtype=bool)
    right = np.zeros((height, width), dtype=bool)

    left[:, :shift] = True
    right[:, overlap:] = True

    for r, cut in enumerate(cuts):
        if cut > 0:
            left[r, shift : shift + cut] = True
        if cut < overlap:
            right[r, cut:overlap] = True

    return left, right


def _random_pair_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    if width < 3:
        return None

    shift = random.randint(1, width - 1)
    overlap = width - shift

    cuts = [random.randint(0, overlap) for _ in range(height)]
    zero_row = random.randrange(height)
    full_row_choices = [r for r in range(height) if r != zero_row]
    if not full_row_choices:
        return None
    full_row = random.choice(full_row_choices)
    cuts[zero_row] = 0
    cuts[full_row] = overlap

    left, right = _build_pair_masks(height, width, shift, cuts)

    if random.random() < 0.5:
        left = np.flipud(left)
        right = np.flipud(right)
    if random.random() < 0.5:
        left = np.fliplr(left)
        right = np.fliplr(right)

    try:
        _ = _is_left(left, right)
    except AssertionError:
        return None
    if not _pairable(left, right):
        return None
    if not (_bbox_is_full(left) and _bbox_is_full(right)):
        return None
    return left, right


def _bbox_is_full(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return (int(r0), int(c0), int(r1), int(c1)) == (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)


def _pick_pair_shapes(pair_count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    shapes: list[tuple[np.ndarray, np.ndarray]] = []
    seen_by_size: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    used_sizes: list[tuple[int, int]] = []

    for _ in range(pair_count):
        placed = False
        for _attempt in range(200):
            if used_sizes and random.random() < GEN_REPEAT_SIZE_PROB:
                height, width = random.choice(used_sizes)
            else:
                height = random.choice(GEN_BBOX_HEIGHTS)
                width = random.choice(GEN_BBOX_WIDTHS)

            pair = _random_pair_masks(height, width)
            if pair is None:
                continue
            left, right = pair

            size = (height, width)
            existing = seen_by_size[size]
            if any(
                _pairable(left, mask) or _pairable(right, mask) or _mask_signature(left) == _mask_signature(mask) or _mask_signature(right) == _mask_signature(mask)
                for mask in existing
            ):
                continue

            seen_by_size[size].extend([left, right])
            if size not in used_sizes:
                used_sizes.append(size)
            shapes.append((left, right))
            placed = True
            break

        if not placed:
            raise RuntimeError("failed to construct unambiguous pair shapes")

    return shapes


def _place_piece(grid: np.ndarray, reserve: np.ndarray, mask: np.ndarray, color: int) -> tuple[int, int] | None:
    h, w = mask.shape
    gh, gw = grid.shape
    for _ in range(200):
        r0 = random.randint(0, gh - h)
        c0 = random.randint(0, gw - w)
        rr0 = max(0, r0 - GEN_PLACEMENT_PAD)
        cc0 = max(0, c0 - GEN_PLACEMENT_PAD)
        rr1 = min(gh, r0 + h + GEN_PLACEMENT_PAD)
        cc1 = min(gw, c0 + w + GEN_PLACEMENT_PAD)
        if bool(reserve[rr0:rr1, cc0:cc1].any()):
            continue
        reserve[rr0:rr1, cc0:cc1] = True
        sub = grid[r0 : r0 + h, c0 : c0 + w]
        sub[mask] = color
        return r0, c0
    return None


def _render_pair_grid(shapes: list[tuple[np.ndarray, np.ndarray]], colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    total_boxes = sum((mask.shape[0] + 2) * (mask.shape[1] + 2) for pair in shapes for mask in pair)
    side = max(
        GEN_GRID_MIN,
        int(np.ceil(np.sqrt(total_boxes * random.uniform(1.35, 1.8)))),
        max(max(mask.shape) for pair in shapes for mask in pair) + 2,
    )
    gh = random.randint(side, min(GEN_GRID_MAX, side + 5))
    gw = random.randint(side, min(GEN_GRID_MAX, side + 5))
    if gh < max(mask.shape[0] for pair in shapes for mask in pair) or gw < max(mask.shape[1] for pair in shapes for mask in pair):
        return None

    grid = np.full((gh, gw), BG, dtype=int)
    out = np.full((gh, gw), BG, dtype=int)
    reserve = np.zeros((gh, gw), dtype=bool)

    pieces = []
    color_iter = iter(colors)
    for left_mask, right_mask in shapes:
        left_color = next(color_iter)
        right_color = next(color_iter)
        pieces.append((left_mask, left_color, right_color, True))
        pieces.append((right_mask, right_color, left_color, False))

    random.shuffle(pieces)

    for mask, input_color, output_color, _ in pieces:
        placed = _place_piece(grid, reserve, mask, input_color)
        if placed is None:
            return None
        r0, c0 = placed
        out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = output_color

    return grid, out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(GEN_PAIR_COUNT_CHOICES)
    shapes = _pick_pair_shapes(pair_count)
    colors = random.sample(range(1, 10), 2 * pair_count)
    rendered = _render_pair_grid(shapes, colors)
    if rendered is None:
        return None
    inp, expected = rendered

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
