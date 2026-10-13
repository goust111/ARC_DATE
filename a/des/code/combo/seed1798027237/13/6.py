from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
BODY_COLORS = (1, 2, 3, 4, 6, 7, 8)

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
FRAME_H_RANGE = (12, 18)  # outer size, includes the grey border
FRAME_W_RANGE = (12, 18)
TEMPLATE_COUNT_RANGE = (2, 4)
TEMPLATE_AREA_RANGE = (3, 8)
TEMPLATE_SPAN_MAX = 5
GEN_ATTEMPTS = 400
SEED_PLACE_RETRIES = 500
TEMPLATE_PLACE_RETRIES = 500


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def normalize_cells(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def transform_cells(cells: list[tuple[int, int]], variant: int) -> list[tuple[int, int]]:
    transformed = []
    for r, c in cells:
        if variant == 0:
            nr, nc = r, c
        elif variant == 1:
            nr, nc = r, -c
        elif variant == 2:
            nr, nc = -r, c
        elif variant == 3:
            nr, nc = -r, -c
        elif variant == 4:
            nr, nc = c, r
        elif variant == 5:
            nr, nc = c, -r
        elif variant == 6:
            nr, nc = -c, r
        else:
            nr, nc = -c, -r
        transformed.append((nr, nc))
    return normalize_cells(transformed)


def find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey = np.argwhere(grid == GREY)
    assert grey.size > 0, "expected a grey rectangular fence"
    r0, c0 = map(int, grey.min(axis=0))
    r1, c1 = map(int, grey.max(axis=0))
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"

    border = np.zeros_like(grid, dtype=bool)
    border[r0, c0 : c1 + 1] = True
    border[r1, c0 : c1 + 1] = True
    border[r0 : r1 + 1, c0] = True
    border[r0 : r1 + 1, c1] = True

    assert np.all(grid[border] == GREY), "frame border must be a complete grey rectangle"
    assert np.count_nonzero(grid == GREY) == int(border.sum()), "grey pixels must belong only to the fence"
    assert not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == GREY), "grey may not appear inside the fence"
    return r0, c0, r1, c1


def extract_seeds(grid: np.ndarray, frame: tuple[int, int, int, int]) -> dict[int, tuple[int, int]]:
    r0, c0, r1, c1 = frame
    seed_mask = np.zeros_like(grid, dtype=bool)
    seed_mask[r0 + 1 : r1, c0 + 1 : c1] = grid[r0 + 1 : r1, c0 + 1 : c1] != BLACK
    comps = components_mask(seed_mask)
    seeds: dict[int, tuple[int, int]] = {}
    for comp in comps:
        assert len(comp) == 1, "each seed must be a singleton pixel"
        sr, sc = comp[0]
        color = int(grid[sr, sc])
        assert color not in (BLACK, GREY, MAROON), "seed colors must be ordinary body colors"
        assert color not in seeds, "seed colors must be unique"
        seeds[color] = (sr, sc)
    assert seeds, "expected at least one seed inside the fence"
    return seeds


def extract_templates(grid: np.ndarray, frame: tuple[int, int, int, int]) -> dict[int, list[tuple[int, int]]]:
    r0, c0, r1, c1 = frame
    mask = grid != BLACK
    mask[r0 : r1 + 1, c0 : c1 + 1] = False

    templates: dict[int, list[tuple[int, int]]] = {}
    for comp in components_mask(mask):
        colors = {int(grid[r, c]) for r, c in comp}
        assert MAROON in colors, "each external template must contain a maroon anchor"
        body_colors = colors - {MAROON}
        assert len(body_colors) == 1, "each external template must have exactly one body color"
        body_color = int(next(iter(body_colors)))
        assert body_color not in (BLACK, GREY, MAROON)
        assert body_color not in templates, "template colors must be unique"

        anchors = [(r, c) for r, c in comp if int(grid[r, c]) == MAROON]
        assert len(anchors) == 1, "each template must have exactly one anchor"
        ar, ac = anchors[0]
        offsets = sorted((int(r - ar), int(c - ac)) for r, c in comp)
        assert len(offsets) >= 2, "template must contain anchor plus at least one more pixel"
        templates[body_color] = offsets

    assert templates, "expected at least one external template"
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each external template into the fenced region by aligning its maroon anchor to the matching seed color.
    """
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = find_frame(g)
    seeds = extract_seeds(g, (r0, c0, r1, c1))
    templates = extract_templates(g, (r0, c0, r1, c1))
    assert set(seeds) == set(templates), "seed colors and template colors must match one-to-one"

    paint: dict[tuple[int, int], int] = {}
    for color, (sr, sc) in seeds.items():
        for dr, dc in templates[color]:
            tr = sr + dr
            tc = sc + dc
            if r0 < tr < r1 and c0 < tc < c1:
                prev = paint.get((tr, tc))
                assert prev in (None, color), "grown templates may not overlap with conflicting colors"
                paint[(tr, tc)] = color

    out = g.copy()
    for (r, c), color in paint.items():
        out[r, c] = color
    return out


def random_polyomino(rng: random.Random) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    for _ in range(200):
        target_area = rng.randint(*TEMPLATE_AREA_RANGE)
        cells: set[tuple[int, int]] = {(0, 0)}
        while len(cells) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            frontier_list = list(frontier)
            rng.shuffle(frontier_list)

            added = False
            for nr, nc in frontier_list:
                trial = set(cells)
                trial.add((nr, nc))
                norm = normalize_cells(trial)
                height = max(r for r, _ in norm) + 1
                width = max(c for _, c in norm) + 1
                if height <= TEMPLATE_SPAN_MAX and width <= TEMPLATE_SPAN_MAX:
                    cells = trial
                    added = True
                    break
            if not added:
                break
        if len(cells) != target_area:
            continue

        shape = normalize_cells(cells)
        shape = transform_cells(shape, rng.randrange(8))
        if len({r for r, _ in shape}) == 1 and len({c for _, c in shape}) == 1:
            continue
        anchor = rng.choice(shape)
        return shape, anchor
    raise RuntimeError("failed to sample a template shape")


def offset_shape(shape: list[tuple[int, int]], anchor: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = anchor
    return [(r - ar, c - ac) for r, c in shape]


def choose_seed_positions(
    rng: random.Random,
    templates: list[dict[str, object]],
    frame: tuple[int, int, int, int],
) -> dict[int, tuple[int, int]]:
    r0, c0, r1, c1 = frame
    used_output: set[tuple[int, int]] = set()
    seed_positions: dict[int, tuple[int, int]] = {}

    clip_count = min(rng.choices([0, 1, 2], weights=[2, 3, 1])[0], max(0, len(templates) - 1))
    clipped_indices = set(rng.sample(range(len(templates)), clip_count))

    templates_order = sorted(
        range(len(templates)),
        key=lambda idx: -len(templates[idx]["offsets"]),  # type: ignore[index]
    )

    placed_seeds: list[tuple[int, int]] = []
    for idx in templates_order:
        template = templates[idx]
        color = int(template["color"])
        offsets = list(template["offsets"])  # type: ignore[arg-type]
        require_clip = idx in clipped_indices

        placed = False
        for _ in range(SEED_PLACE_RETRIES):
            if require_clip and rng.random() < 0.8:
                mode = rng.randrange(4)
                if mode == 0:
                    sr = r0 + 1
                    sc = rng.randint(c0 + 1, c1 - 1)
                elif mode == 1:
                    sr = r1 - 1
                    sc = rng.randint(c0 + 1, c1 - 1)
                elif mode == 2:
                    sr = rng.randint(r0 + 1, r1 - 1)
                    sc = c0 + 1
                else:
                    sr = rng.randint(r0 + 1, r1 - 1)
                    sc = c1 - 1
            else:
                sr = rng.randint(r0 + 1, r1 - 1)
                sc = rng.randint(c0 + 1, c1 - 1)

            if any(abs(sr - pr) + abs(sc - pc) <= 1 for pr, pc in placed_seeds):
                continue

            painted = set()
            fully_inside = True
            for dr, dc in offsets:
                tr = sr + int(dr)
                tc = sc + int(dc)
                if r0 < tr < r1 and c0 < tc < c1:
                    painted.add((tr, tc))
                else:
                    fully_inside = False

            if require_clip and fully_inside:
                continue
            if not require_clip and not fully_inside:
                continue
            if len(painted) < 2:
                continue
            if painted & used_output:
                continue

            used_output |= painted
            seed_positions[color] = (sr, sc)
            placed_seeds.append((sr, sc))
            placed = True
            break

        if not placed:
            raise RuntimeError("failed to place seeds without overlap")

    return seed_positions


def mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                blocked[nr, nc] = True


def place_external_templates(
    rng: random.Random,
    grid: np.ndarray,
    templates: list[dict[str, object]],
    frame: tuple[int, int, int, int],
) -> None:
    h, w = grid.shape
    r0, c0, r1, c1 = frame

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True

    order = list(range(len(templates)))
    rng.shuffle(order)
    for idx in order:
        template = templates[idx]
        color = int(template["color"])
        shape = list(template["shape"])  # type: ignore[arg-type]
        anchor = tuple(template["anchor"])  # type: ignore[arg-type]
        box_h = max(r for r, _ in shape) + 1
        box_w = max(c for _, c in shape) + 1

        placed = False
        for _ in range(TEMPLATE_PLACE_RETRIES):
            top = rng.randint(0, h - box_h)
            left = rng.randint(0, w - box_w)
            abs_cells = [(top + r, left + c) for r, c in shape]
            if any(blocked[r, c] for r, c in abs_cells):
                continue

            for r, c in abs_cells:
                grid[r, c] = MAROON if (r - top, c - left) == anchor else color
            mark_blocked(blocked, abs_cells)
            placed = True
            break

        if not placed:
            raise RuntimeError("failed to place templates outside the fence")


def validate_instance(grid: np.ndarray) -> bool:
    try:
        frame = find_frame(grid)
        seeds = extract_seeds(grid, frame)
        templates = extract_templates(grid, frame)
        if set(seeds) != set(templates):
            return False
        out = main(grid)
    except Exception:
        return False

    r0, c0, r1, c1 = frame
    preserve = np.ones_like(grid, dtype=bool)
    preserve[r0 + 1 : r1, c0 + 1 : c1] = False
    if not np.array_equal(out[preserve], grid[preserve]):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)

    if h < FRAME_H_RANGE[0] + 4 or w < FRAME_W_RANGE[0] + 4:
        return None

    frame_h = rng.randint(FRAME_H_RANGE[0], min(FRAME_H_RANGE[1], h - 4))
    frame_w = rng.randint(FRAME_W_RANGE[0], min(FRAME_W_RANGE[1], w - 4))
    top = rng.randint(2, h - frame_h - 2)
    left = rng.randint(2, w - frame_w - 2)
    frame = (top, left, top + frame_h - 1, left + frame_w - 1)

    template_count = rng.randint(*TEMPLATE_COUNT_RANGE)
    colors = rng.sample(BODY_COLORS, template_count)
    templates: list[dict[str, object]] = []
    for color in colors:
        shape, anchor = random_polyomino(rng)
        offsets = offset_shape(shape, anchor)
        templates.append(
            {
                "color": color,
                "shape": shape,
                "anchor": anchor,
                "offsets": offsets,
            }
        )

    try:
        seeds = choose_seed_positions(rng, templates, frame)
    except RuntimeError:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    r0, c0, r1, c1 = frame
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    for color, (sr, sc) in seeds.items():
        grid[sr, sc] = color

    try:
        place_external_templates(rng, grid, templates, frame)
    except RuntimeError:
        return None

    if not validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fenced-template puzzle")
