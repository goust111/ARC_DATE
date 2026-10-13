from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
AZURE = 8
MAROON = 9
PALETTE_ORDER = (GREEN, YELLOW, ORANGE, MAROON)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _touches_border(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _legend_mapping(g: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = g.shape
    allowed = set(PALETTE_ORDER)

    bottom = [(h - 1, c, int(g[h - 1, c])) for c in range(w) if int(g[h - 1, c]) in allowed]
    right = [(r, w - 1, int(g[r, w - 1])) for r in range(h) if int(g[r, w - 1]) in allowed]

    mask = np.zeros((h, w), dtype=bool)
    mapping = {idx + 1: col for idx, col in enumerate(PALETTE_ORDER)}

    if len(bottom) == 4 and {col for _, _, col in bottom} == allowed:
        ordered = bottom
    elif len(right) == 4 and {col for _, _, col in right} == allowed:
        ordered = right
    else:
        return mapping, mask

    mapping = {}
    for idx, (r, c, col) in enumerate(ordered):
        mapping[idx + 1] = int(col)
        mask[int(r), int(c)] = True
    return mapping, mask


def _detect_shapes(g: np.ndarray) -> list[dict[str, object]]:
    h, w = g.shape
    azure_comps = _components(g == AZURE)
    labels = np.full((h, w), -1, dtype=int)
    for cid, comp in enumerate(azure_comps):
        for r, c in comp:
            labels[r, c] = int(cid)

    interior_cells: dict[int, list[tuple[int, int]]] = defaultdict(list)
    eye_counts: dict[int, int] = defaultdict(int)

    for region in _components(g != AZURE):
        if _touches_border(region, h, w):
            continue
        colors = {int(g[r, c]) for r, c in region}
        if not colors.issubset({BLACK, RED}):
            continue
        adjacent = set()
        for r, c in region:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] >= 0:
                    adjacent.add(int(labels[nr, nc]))
        if len(adjacent) != 1:
            continue
        cid = next(iter(adjacent))
        interior_cells[cid].extend(region)
        eye_counts[cid] += sum(1 for r, c in region if int(g[r, c]) == RED)

    shapes: list[dict[str, object]] = []
    for cid, comp in enumerate(azure_comps):
        if cid not in interior_cells:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        shapes.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "center": ((int(r0 + r1) // 2), (int(c0 + c1) // 2)),
                "eyes": int(eye_counts[cid]),
            }
        )
    return shapes


def _draw_l_path(out: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    r0, c0 = start
    r1, c1 = end

    step_c = 1 if c1 >= c0 else -1
    for c in range(c0, c1 + step_c, step_c):
        if int(out[r0, c]) == BLACK:
            out[r0, c] = int(color)

    step_r = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + step_r, step_r):
        if int(out[r, c1]) == BLACK:
            out[r, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group hollow azure figures by their red-eye count and connect each group with a legend-colored path.

    Input structure:
    - Background is black.
    - Several closed azure outlines appear on the grid; each outline is a 1-pixel-thick border around a black interior.
    - Every figure contains 1 to 4 red single-pixel "eyes" inside its enclosed interior.
    - The bottom row or the rightmost column contains four color samples, ordered as green, yellow, orange, maroon.

    Rule:
    - Detect each azure figure and count how many red eyes lie in its enclosed interior.
    - Use the edge legend to map eye counts 1, 2, 3, 4 to the corresponding connection colors.
    - For each eye-count group, sort the figures by center column, breaking ties by center row, and connect consecutive centers with a 1-pixel Manhattan path that goes horizontal first and then vertical.
    - Draw only onto black cells, leaving azure borders, red eyes, and the legend unchanged.

    Key insight:
    - The red dots define grouping, while the edge legend converts those groups into output colors; the actual figure size only affects center placement for the connecting paths.
    """
    g = np.asarray(grid, dtype=int)
    legend, _legend_mask = _legend_mapping(g)
    shapes = _detect_shapes(g)

    out = g.copy()
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for shape in shapes:
        eyes = int(shape["eyes"])
        if 1 <= eyes <= 4:
            groups[eyes].append(shape)

    for eyes, members in groups.items():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda item: (int(item["center"][1]), int(item["center"][0])))
        color = int(legend.get(eyes, PALETTE_ORDER[eyes - 1]))
        for left, right in zip(members, members[1:], strict=False):
            _draw_l_path(out, tuple(left["center"]), tuple(right["center"]), color)

    return out


def _filled_rectangle(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _filled_l_shape(h: int, w: int, thickness: int = 3) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :thickness] = True
    mask[h - thickness :, :] = True
    return mask


def _filled_t_shape(h: int, w: int, thickness: int = 3) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    stem0 = (w - thickness) // 2
    mask[:thickness, :] = True
    mask[:, stem0 : stem0 + thickness] = True
    return mask


def _shape_boundary_and_interior(filled: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h, w = filled.shape
    boundary = np.zeros((h, w), dtype=bool)
    interior: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not bool(filled[r, c]):
                continue
            edge = r == 0 or c == 0 or r == h - 1 or c == w - 1
            outside = False
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not bool(filled[nr, nc]):
                    outside = True
                    break
            if edge or outside:
                boundary[r, c] = True
            else:
                interior.append((int(r), int(c)))
    return boundary, interior


def _candidate_shapes(family: str, rotation: int) -> list[dict[str, object]]:
    raw_masks: list[tuple[tuple[int, ...], np.ndarray]] = []
    if family == "rect":
        for h in range(5, 10):
            for w in range(5, 11):
                raw_masks.append(((h, w), _filled_rectangle(h, w)))
    elif family == "l":
        for h in range(5, 10):
            for w in range(5, 10):
                raw_masks.append(((h, w, 3), _filled_l_shape(h, w, 3)))
    else:
        for h in range(6, 10):
            for w in range(7, 12, 2):
                raw_masks.append(((h, w, 3), _filled_t_shape(h, w, 3)))

    out: list[dict[str, object]] = []
    for raw_sig, raw_mask in raw_masks:
        mask = np.rot90(raw_mask, rotation)
        boundary, interior = _shape_boundary_and_interior(mask)
        if len(interior) < 4:
            continue
        sig = (family, rotation) + raw_sig + mask.shape
        out.append(
            {
                "signature": sig,
                "boundary": boundary,
                "interior": interior,
                "shape": tuple(int(x) for x in mask.shape),
            }
        )
    return out


def _best_eye_sample(interior: list[tuple[int, int]], count: int, rng: random.Random) -> list[tuple[int, int]]:
    if count == 1:
        return [rng.choice(interior)]

    best = None
    best_score = (-1, -1)
    for _ in range(40):
        sample = rng.sample(interior, count)
        min_dist = min(
            abs(r0 - r1) + abs(c0 - c1)
            for i, (r0, c0) in enumerate(sample)
            for r1, c1 in sample[i + 1 :]
        )
        spread = len({r for r, _ in sample}) + len({c for _, c in sample})
        score = (min_dist, spread)
        if score > best_score:
            best = sample
            best_score = score
    assert best is not None
    return [(int(r), int(c)) for r, c in best]


def _group_sizes(rng: random.Random) -> dict[int, int]:
    counts = rng.sample([1, 2, 3, 4], rng.choice([2, 3, 4]))
    sizes = {count: 1 for count in counts}

    boosted = rng.choice(counts)
    sizes[boosted] = rng.choice([2, 3]) if len(counts) == 2 and rng.random() < 0.4 else 2

    while sum(sizes.values()) < 6 and rng.random() < 0.35:
        choices = [count for count in counts if sizes[count] < 3]
        if not choices:
            break
        sizes[rng.choice(choices)] += 1

    return sizes


def _layout_grid(objects: list[dict[str, object]], legend_side: str, rng: random.Random) -> np.ndarray | None:
    margin_top = rng.randint(1, 3)
    margin_left = rng.randint(1, 3)
    band_gap = rng.randint(2, 3)
    reserve_rows = 2 if legend_side == "bottom" else 1
    reserve_cols = 2 if legend_side == "right" else 1

    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for obj in objects:
        groups[int(obj["eyes"])].append(obj)

    order = list(groups.keys())
    rng.shuffle(order)

    band_specs = []
    content_width = 0
    for eyes in order:
        items = groups[eyes]
        gap_choices = [rng.randint(3, 5) for _ in range(max(0, len(items) - 1))]
        width = sum(int(item["shape"][1]) for item in items) + sum(gap_choices)
        height = max(int(item["shape"][0]) for item in items) + rng.randint(0, 2)
        band_specs.append((eyes, items, gap_choices, width, height))
        content_width = max(content_width, width)

    content_height = sum(spec[4] for spec in band_specs) + band_gap * max(0, len(band_specs) - 1)
    h = margin_top + content_height + reserve_rows
    w = margin_left + content_width + reserve_cols
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    cursor_r = margin_top
    placed: list[dict[str, object]] = []
    for eyes, items, gap_choices, width, height in band_specs:
        max_shift = w - reserve_cols - margin_left - width
        start_c = margin_left + (0 if max_shift <= 0 else rng.randint(0, max_shift))
        cursor_c = start_c
        for idx, item in enumerate(items):
            shape_h, shape_w = item["shape"]
            dy = 0 if height == shape_h else rng.randint(0, height - shape_h)
            placed.append(
                {
                    **item,
                    "top": int(cursor_r + dy),
                    "left": int(cursor_c),
                }
            )
            cursor_c += int(shape_w)
            if idx < len(gap_choices):
                cursor_c += int(gap_choices[idx])
        cursor_r += int(height + band_gap)

    for item in placed:
        top = int(item["top"])
        left = int(item["left"])
        boundary = np.asarray(item["boundary"], dtype=bool)
        for r, c in np.argwhere(boundary):
            grid[top + int(r), left + int(c)] = AZURE
        for r, c in item["eyes_pos"]:
            grid[top + int(r), left + int(c)] = RED

    if legend_side == "bottom":
        cols = sorted(rng.sample(list(range(1, w - 1)), 4))
        for col, color in zip(cols, PALETTE_ORDER, strict=False):
            grid[h - 1, int(col)] = int(color)
    else:
        rows = sorted(rng.sample(list(range(1, h - 1)), 4))
        for row, color in zip(rows, PALETTE_ORDER, strict=False):
            grid[int(row), w - 1] = int(color)

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    family = rng.choice(["rect", "l", "t"])
    rotation = 0 if family == "rect" else rng.randrange(4)
    candidates = _candidate_shapes(family, rotation)
    group_sizes = _group_sizes(rng)
    total_objects = sum(group_sizes.values())
    if len(candidates) < total_objects:
        return None

    chosen = rng.sample(candidates, total_objects)
    rng.shuffle(chosen)

    objects: list[dict[str, object]] = []
    idx = 0
    for eyes, count in group_sizes.items():
        for _ in range(count):
            base = chosen[idx]
            idx += 1
            eyes_pos = _best_eye_sample(base["interior"], eyes, rng)
            objects.append(
                {
                    "eyes": int(eyes),
                    "boundary": np.asarray(base["boundary"], dtype=bool),
                    "shape": tuple(int(x) for x in base["shape"]),
                    "eyes_pos": eyes_pos,
                }
            )

    legend_side = rng.choice(["bottom", "right"])
    grid = _layout_grid(objects, legend_side, rng)
    if grid is None:
        return None

    detected = sorted(shape["eyes"] for shape in _detect_shapes(grid))
    expected = sorted(obj["eyes"] for obj in objects)
    if detected != expected:
        return None

    out = main(grid)
    painted = np.argwhere(out != grid)
    if len(painted) < 4:
        return None
    if any(int(grid[r, c]) != BLACK for r, c in painted):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
