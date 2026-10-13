from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_GEN_RETRIES = 300
_GEN_COLORS = tuple(range(1, 10))
_SHAPE_LIBRARY = (
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1], [1, 0], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=int),
)


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_signature(coords: list[tuple[int, int]]) -> tuple[bool, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False, (r0, c0, r1, c1)
    expected = 2 * h + 2 * w - 4
    if len(coords) != expected:
        return False, (r0, c0, r1, c1)
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return False, (r0, c0, r1, c1)
    return True, (r0, c0, r1, c1)


def _normalize_shape(coords: list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    rel = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return rel, r1 - r0 + 1, c1 - c0 + 1


def _inside_box(point: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the marker-color chain to choose a source sprite, then stamp that sprite's
    shape inside each frame using the frame's own border color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    frames: list[dict[str, object]] = []
    non_frame_components: list[tuple[int, list[tuple[int, int]]]] = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for coords in _components(g == color):
            is_frame, box = _frame_signature(coords)
            if is_frame:
                frames.append({"color": color, "box": box})
            else:
                non_frame_components.append((color, coords))

    assert frames, "expected at least one frame"

    marker_positions: set[tuple[int, int]] = set()
    for frame in frames:
        r0, c0, r1, c1 = frame["box"]  # type: ignore[index]
        marker_pos = (r0 + 1, c0 + 1)
        marker_positions.add(marker_pos)
        frame["marker_color"] = int(g[marker_pos])
        frame["inner_h"] = r1 - r0 - 1
        frame["inner_w"] = c1 - c0 - 1

    frame_boxes = [frame["box"] for frame in frames]
    sprites: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for color, coords in non_frame_components:
        if len(coords) == 1 and coords[0] in marker_positions:
            continue
        if any(_inside_box(pt, box) for pt in coords for box in frame_boxes):
            continue
        assert color not in sprites, f"expected one sprite for color {color}"
        sprites[color] = _normalize_shape(coords)

    out = np.full_like(g, bg)

    for frame in frames:
        color = int(frame["color"])  # type: ignore[index]
        r0, c0, r1, c1 = frame["box"]  # type: ignore[index]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color

        marker_color = int(frame["marker_color"])  # type: ignore[index]
        assert marker_color in sprites, f"missing source sprite for color {marker_color}"
        rel, sprite_h, sprite_w = sprites[marker_color]
        inner_h = int(frame["inner_h"])  # type: ignore[index]
        inner_w = int(frame["inner_w"])  # type: ignore[index]
        assert sprite_h < inner_h and sprite_w < inner_w, "sprite must fit strictly inside frame"

        top = r0 + 1 + (inner_h - sprite_h) // 2
        left = c0 + 1 + (inner_w - sprite_w) // 2
        for dr, dc in rel:
            out[top + dr, left + dc] = color

    return out


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _random_shape(rng: random.Random) -> np.ndarray:
    if rng.random() < 0.7:
        mask = rng.choice(_SHAPE_LIBRARY).astype(bool)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        k = rng.randrange(4)
        mask = np.rot90(mask, k)
        return _trim(mask)

    h = rng.randint(2, 4)
    w = rng.randint(2, 4)
    total = h * w
    target = rng.randint(max(3, total // 2), total - 1)
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < target:
        r, c = rng.choice(tuple(cells))
        steps = list(_NEIGHBORS4)
        rng.shuffle(steps)
        for dr, dc in steps:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                cells.add((nr, nc))
                break
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _trim(mask)


def _sample_unique_shapes(rng: random.Random, count: int) -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    while len(shapes) < count:
        shape = _random_shape(rng)
        if shape.shape[0] < 2 or shape.shape[1] < 2:
            continue
        sig = _shape_signature(shape)
        if sig in seen:
            continue
        seen.add(sig)
        shapes.append(shape)
    return shapes


def _random_derangement(items: list[int], rng: random.Random) -> list[int]:
    perm = items[:]
    while True:
        rng.shuffle(perm)
        if all(a != b for a, b in zip(items, perm)):
            return perm[:]


def _single_cycle_mapping(items: list[int], rng: random.Random) -> list[int]:
    order = items[:]
    rng.shuffle(order)
    mapping = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
    return [mapping[item] for item in items]


def _can_place(occ: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occ.shape
    if top < 1 or left < 1 or top + height >= h or left + width >= w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not occ[r0:r1, c0:c1].any()


def _mark_place(occ: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    occ[top : top + height, left : left + width] = True


def _place_box(
    occ: np.ndarray,
    height: int,
    width: int,
    rng: random.Random,
    trials: int = 200,
) -> tuple[int, int] | None:
    h, w = occ.shape
    if height + 2 > h or width + 2 > w:
        return None
    for _ in range(trials):
        top = rng.randint(1, h - height - 1)
        left = rng.randint(1, w - width - 1)
        if _can_place(occ, top, left, height, width):
            _mark_place(occ, top, left, height, width)
            return top, left
    return None


def _paint_shape(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_frames = rng.randint(2, 4)
    colors = rng.sample(list(_GEN_COLORS), num_frames)
    shapes = _sample_unique_shapes(rng, num_frames)
    max_shape_h = max(shape.shape[0] for shape in shapes)
    max_shape_w = max(shape.shape[1] for shape in shapes)

    if num_frames > 2 and rng.random() < 0.5:
        marker_colors = _single_cycle_mapping(colors, rng)
    else:
        marker_colors = _random_derangement(colors, rng)

    frame_specs = []
    total_frame_area = 0
    for color, marker_color in zip(colors, marker_colors):
        inner_h = rng.randint(max_shape_h + 2, max_shape_h + 5)
        inner_w = rng.randint(max_shape_w + 2, max_shape_w + 5)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        frame_specs.append((color, marker_color, inner_h, inner_w, outer_h, outer_w))
        total_frame_area += outer_h * outer_w

    total_sprite_area = sum(int(shape.size) for shape in shapes)
    side_need = int(np.ceil(np.sqrt(total_frame_area + total_sprite_area))) + 7
    side_need = max(
        side_need,
        max(spec[4] for spec in frame_specs) + max(shape.shape[0] for shape in shapes) + 5,
        max(spec[5] for spec in frame_specs) + max(shape.shape[1] for shape in shapes) + 5,
    )
    base_dim = max(22, side_need)
    upper_dim = min(30, max(base_dim + 3, 24))
    if base_dim > upper_dim:
        return None
    height = rng.randint(base_dim, upper_dim)
    width = rng.randint(base_dim, upper_dim)

    occ = np.zeros((height, width), dtype=bool)
    grid = np.zeros((height, width), dtype=int)

    frame_positions: list[tuple[int, int]] = []
    ordered_frames = sorted(
        enumerate(frame_specs),
        key=lambda item: item[1][4] * item[1][5],
        reverse=True,
    )
    frame_tops_lefts: dict[int, tuple[int, int]] = {}
    for idx, (_, _, _, _, outer_h, outer_w) in ordered_frames:
        placed = _place_box(occ, outer_h, outer_w, rng)
        if placed is None:
            return None
        frame_tops_lefts[idx] = placed

    for idx, (color, marker_color, inner_h, inner_w, outer_h, outer_w) in enumerate(frame_specs):
        top, left = frame_tops_lefts[idx]
        frame_positions.append((top, left))
        grid[top, left : left + outer_w] = color
        grid[top + outer_h - 1, left : left + outer_w] = color
        grid[top : top + outer_h, left] = color
        grid[top : top + outer_h, left + outer_w - 1] = color
        grid[top + 1, left + 1] = marker_color

    sprite_positions: list[tuple[int, int]] = []
    ordered_sprites = sorted(
        enumerate(shapes),
        key=lambda item: item[1].shape[0] * item[1].shape[1],
        reverse=True,
    )
    sprite_tops_lefts: dict[int, tuple[int, int]] = {}
    for idx, shape in ordered_sprites:
        placed = _place_box(occ, shape.shape[0], shape.shape[1], rng)
        if placed is None:
            return None
        sprite_tops_lefts[idx] = placed

    for idx, (color, shape) in enumerate(zip(colors, shapes)):
        top, left = sprite_tops_lefts[idx]
        sprite_positions.append((top, left))
        _paint_shape(grid, top, left, shape.astype(bool), color)

    if len(set(colors)) != num_frames or len(set(marker_colors)) != num_frames:
        return None

    counts = Counter(marker_colors)
    if any(counts[color] != 1 for color in colors):
        return None

    solved = main(grid)
    if solved.shape != grid.shape:
        return None

    for (frame_top, frame_left), (_, _, inner_h, inner_w, outer_h, outer_w) in zip(frame_positions, frame_specs):
        inner = solved[frame_top + 1 : frame_top + outer_h - 1, frame_left + 1 : frame_left + outer_w - 1]
        if np.count_nonzero(inner) == 0:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_GEN_RETRIES} attempts")
