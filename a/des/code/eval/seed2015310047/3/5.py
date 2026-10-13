from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
TEMPLATE_COLORS = (1, 2, 3, 4, 6, 8, 9)

GRID_WIDTH_RANGE = (16, 28)
SCENE_HEIGHT_RANGE = (10, 16)
TEMPLATE_COUNT_RANGE = (3, 5)
OBSTACLE_COUNT_RANGE = (1, 3)
SEED_EXTRA_RANGE = (0, 2)
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 250


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    if connectivity == 8:
        deltas = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    else:
        deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _find_divider_row(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates: list[int] = []
    for r in range(1, h - 1):
        row = grid[r]
        if int(row[0]) == bg or not np.all(row == row[0]):
            continue
        if not np.any(grid[:r] != bg):
            continue
        if not np.any(grid[r + 1 :] != bg):
            continue
        candidates.append(r)

    assert candidates, "expected a full-width divider row"
    for r in candidates:
        if int(grid[r, 0]) == GREY:
            return int(r)
    return int(candidates[0])


def _extract_templates(zone: np.ndarray, bg: int) -> dict[int, list[tuple[int, int]]]:
    templates: dict[int, list[tuple[int, int]]] = {}
    for color in sorted(int(v) for v in np.unique(zone) if int(v) != bg):
        comps = _components(zone == color, connectivity=8)
        assert len(comps) == 1, "each template color should form exactly one component"
        coords = comps[0]
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        center_r = h // 2
        center_c = w // 2
        patch = np.zeros((h, w), dtype=bool)
        patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
        assert patch[center_r, center_c], "template center cell must be occupied"
        rels = [
            (int(rr - center_r), int(cc - center_c))
            for rr, cc in np.argwhere(patch)
        ]
        templates[color] = rels
    assert templates, "template area may not be empty"
    return templates


def _scan_seeds(scene: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    seeds: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((scene != bg) & (scene != ORANGE)):
        seeds.append((int(r), int(c), int(scene[int(r), int(c)])))
    seeds.sort()
    return seeds


def _apply_growth(
    scene: np.ndarray,
    templates: dict[int, list[tuple[int, int]]],
    bg: int = BLACK,
    seeds: list[tuple[int, int, int]] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    h, w = scene.shape
    if seeds is None:
        seeds = _scan_seeds(scene, bg)
    else:
        seeds = sorted((int(r), int(c), int(color)) for r, c, color in seeds)

    out = np.array(scene, dtype=int, copy=True)
    obstacle_mask = out == ORANGE
    for r, c, _color in seeds:
        if 0 <= r < h and 0 <= c < w and not obstacle_mask[r, c]:
            out[r, c] = bg

    occupied = out != bg
    metrics = {"edge": 0, "obstacle": 0, "overlap": 0, "placed": 0}

    for sr, sc, color in seeds:
        if color not in templates:
            continue
        for dr, dc in templates[color]:
            rr = sr + dr
            cc = sc + dc
            if not (0 <= rr < h and 0 <= cc < w):
                metrics["edge"] += 1
                continue
            if obstacle_mask[rr, cc]:
                metrics["obstacle"] += 1
                continue
            if occupied[rr, cc]:
                metrics["overlap"] += 1
                continue
            out[rr, cc] = color
            occupied[rr, cc] = True
            metrics["placed"] += 1

    return out, metrics


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    divider = _find_divider_row(g, bg)
    templates = _extract_templates(g[:divider], bg)
    scene = g[divider + 1 :].copy()
    out, _metrics = _apply_growth(scene, templates, bg=bg)
    return out


def _manhattan_path(
    start: tuple[int, int],
    end: tuple[int, int],
    horizontal_first: bool,
) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    cells = [(sr, sc)]
    rr, cc = sr, sc

    axes = ("h", "v") if horizontal_first else ("v", "h")
    for axis in axes:
        if axis == "h":
            step = 1 if ec >= cc else -1
            while cc != ec:
                cc += step
                cells.append((rr, cc))
        else:
            step = 1 if er >= rr else -1
            while rr != er:
                rr += step
                cells.append((rr, cc))
    return cells


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    stack = [tuple(map(int, cells[0]))]
    seen[stack[0]] = True
    count = 0
    while stack:
        r, c = stack.pop()
        count += 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return count == int(mask.sum())


def _generate_template_mask(rng: random.Random) -> np.ndarray:
    dims = ((3, 3), (3, 5), (5, 3), (5, 5))
    for _ in range(200):
        h, w = dims[rng.randrange(len(dims))]
        center = (h // 2, w // 2)
        mask = np.zeros((h, w), dtype=bool)
        mask[center] = True

        anchors = [
            (0, rng.randrange(w)),
            (h - 1, rng.randrange(w)),
            (rng.randrange(h), 0),
            (rng.randrange(h), w - 1),
        ]
        for anchor in anchors:
            for cell in _manhattan_path(center, anchor, horizontal_first=bool(rng.randrange(2))):
                mask[cell] = True

        budget = rng.randint(0, max(2, (h * w) // 4))
        for _ in range(budget):
            filled = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
            rr, cc = filled[rng.randrange(len(filled))]
            dirs = list(_neighbors4(rr, cc))
            rng.shuffle(dirs)
            for nr, nc in dirs:
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    mask[nr, nc] = True
                    break

        if mask.all():
            continue
        if int(mask.sum()) < 5:
            continue
        if not (
            mask[0].any()
            and mask[-1].any()
            and mask[:, 0].any()
            and mask[:, -1].any()
            and mask[center]
        ):
            continue
        if not _is_connected(mask):
            continue
        return mask.astype(int)
    raise RuntimeError("failed to build a template mask")


def _mask_to_rels(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    center_r = h // 2
    center_c = w // 2
    return [
        (int(r - center_r), int(c - center_c))
        for r, c in np.argwhere(mask.astype(bool))
    ]


def _layout_templates(
    rng: random.Random,
    items: list[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]], int]:
    indexed = [
        {"color": int(color), "mask": np.asarray(mask, dtype=int), "h": int(mask.shape[0]), "w": int(mask.shape[1])}
        for color, mask in items
    ]
    rng.shuffle(indexed)
    max_width = GRID_WIDTH_RANGE[1]

    for row_count in (1, 2) if rng.random() < 0.5 else (2, 1):
        rows: list[list[dict[str, object]]] = [[] for _ in range(row_count)]
        row_widths = [0] * row_count
        row_heights = [0] * row_count

        order = sorted(indexed, key=lambda item: (int(item["w"]), int(item["h"])), reverse=True)
        for item in order:
            idx = min(range(row_count), key=lambda ridx: row_widths[ridx])
            gap = 0 if not rows[idx] else 1
            row_widths[idx] += gap + int(item["w"])
            row_heights[idx] = max(row_heights[idx], int(item["h"]))
            rows[idx].append(item)

        used_rows = [idx for idx in range(row_count) if rows[idx]]
        width_needed = max(row_widths[idx] for idx in used_rows) + 2
        if width_needed > max_width:
            continue

        width = max(GRID_WIDTH_RANGE[0], width_needed + rng.randint(0, max_width - width_needed))
        top_h = 2 + sum(row_heights[idx] for idx in used_rows) + max(0, len(used_rows) - 1)
        top = np.full((top_h, width), BLACK, dtype=int)
        templates: dict[int, list[tuple[int, int]]] = {}

        row_top = 1
        for idx in used_rows:
            free = width - 2 - row_widths[idx]
            left = 1 + rng.randint(0, free)
            cursor = left
            for item in rows[idx]:
                mask = np.asarray(item["mask"], dtype=int)
                h = int(item["h"])
                w = int(item["w"])
                r0 = row_top + (row_heights[idx] - h) // 2
                c0 = cursor
                color = int(item["color"])
                top[r0 : r0 + h, c0 : c0 + w][mask.astype(bool)] = color
                templates[color] = _mask_to_rels(mask)
                cursor += w + 1
            row_top += row_heights[idx] + 1

        return top, templates, width

    raise RuntimeError("failed to layout templates")


def _place_obstacles(rng: random.Random, scene_h: int, scene_w: int) -> np.ndarray | None:
    scene = np.full((scene_h, scene_w), BLACK, dtype=int)
    occupied = np.zeros((scene_h, scene_w), dtype=bool)
    obstacle_count = rng.randint(*OBSTACLE_COUNT_RANGE)

    for _ in range(obstacle_count):
        placed = False
        for _attempt in range(PLACEMENT_ATTEMPTS):
            oh = rng.randint(2, min(4, scene_h))
            ow = rng.randint(2, min(5, scene_w))
            r0 = rng.randint(0, scene_h - oh)
            c0 = rng.randint(0, scene_w - ow)
            rr0 = max(0, r0 - 1)
            rr1 = min(scene_h, r0 + oh + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(scene_w, c0 + ow + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + oh, c0 : c0 + ow] = True
            scene[r0 : r0 + oh, c0 : c0 + ow] = ORANGE
            placed = True
            break
        if not placed:
            return None
    return scene


def _valid_seed_spot(
    r: int,
    c: int,
    base_scene: np.ndarray,
    seeds: list[tuple[int, int, int]],
) -> bool:
    if int(base_scene[r, c]) == ORANGE:
        return False
    for sr, sc, _color in seeds:
        if max(abs(sr - r), abs(sc - c)) <= 1:
            return False
    return True


def _candidate_pools(
    base_scene: np.ndarray,
    seeds: list[tuple[int, int, int]],
    kind: str,
) -> list[list[tuple[int, int]]]:
    h, w = base_scene.shape
    obstacle_mask = base_scene == ORANGE
    open_cells = [(r, c) for r in range(h) for c in range(w) if not obstacle_mask[r, c]]

    edge_cells = [
        (r, c)
        for r, c in open_cells
        if r in (0, h - 1) or c in (0, w - 1) or r in (1, h - 2) or c in (1, w - 2)
    ]

    near_obstacle: list[tuple[int, int]] = []
    far_interior: list[tuple[int, int]] = []
    for r, c in open_cells:
        close = False
        for rr in range(max(0, r - 2), min(h, r + 3)):
            for cc in range(max(0, c - 2), min(w, c + 3)):
                if obstacle_mask[rr, cc]:
                    close = True
                    break
            if close:
                break
        if close:
            near_obstacle.append((r, c))
        elif 1 <= r < h - 1 and 1 <= c < w - 1:
            far_interior.append((r, c))

    near_seed: list[tuple[int, int]] = []
    if seeds:
        seed_set = {(sr, sc) for sr, sc, _color in seeds}
        for sr, sc, _color in seeds:
            for r in range(max(0, sr - 4), min(h, sr + 5)):
                for c in range(max(0, sc - 4), min(w, sc + 5)):
                    if (r, c) not in seed_set and not obstacle_mask[r, c]:
                        near_seed.append((r, c))
        near_seed = list(dict.fromkeys(near_seed))

    if kind == "edge":
        return [edge_cells, open_cells]
    if kind == "obstacle":
        return [near_obstacle, open_cells]
    if kind == "overlap":
        return [near_seed, near_obstacle, open_cells]
    if kind == "normal":
        return [far_interior, open_cells]
    return [open_cells]


def _matches_kind(delta: dict[str, int], kind: str) -> bool:
    if delta["placed"] <= 0:
        return False
    if kind == "edge":
        return delta["edge"] > 0
    if kind == "obstacle":
        return delta["obstacle"] > 0
    if kind == "overlap":
        return delta["overlap"] > 0
    if kind == "normal":
        return delta["edge"] == 0 and delta["obstacle"] == 0 and delta["overlap"] == 0
    return True


def _try_add_seed(
    rng: random.Random,
    base_scene: np.ndarray,
    templates: dict[int, list[tuple[int, int]]],
    seeds: list[tuple[int, int, int]],
    kind: str,
    preferred_colors: list[int] | None = None,
) -> bool:
    _out, before = _apply_growth(base_scene, templates, seeds=seeds)
    colors = list(preferred_colors if preferred_colors is not None else templates.keys())
    rng.shuffle(colors)
    pools = _candidate_pools(base_scene, seeds, kind)

    for pool in pools:
        if not pool:
            continue
        trial_positions = pool[:]
        rng.shuffle(trial_positions)
        for r, c in trial_positions[: max(60, min(len(trial_positions), 180))]:
            if not _valid_seed_spot(r, c, base_scene, seeds):
                continue
            for color in colors:
                candidate = (int(r), int(c), int(color))
                _new_out, after = _apply_growth(base_scene, templates, seeds=seeds + [candidate])
                delta = {key: int(after[key] - before[key]) for key in before}
                if _matches_kind(delta, kind):
                    seeds.append(candidate)
                    return True

    for color in colors:
        for r in range(base_scene.shape[0]):
            for c in range(base_scene.shape[1]):
                if not _valid_seed_spot(r, c, base_scene, seeds):
                    continue
                candidate = (int(r), int(c), int(color))
                _new_out, after = _apply_growth(base_scene, templates, seeds=seeds + [candidate])
                delta = {key: int(after[key] - before[key]) for key in before}
                if _matches_kind(delta, kind):
                    seeds.append(candidate)
                    return True
    return False


def _build_input(
    top: np.ndarray,
    width: int,
    base_scene: np.ndarray,
    seeds: list[tuple[int, int, int]],
) -> np.ndarray:
    full = np.full((top.shape[0] + 1 + base_scene.shape[0], width), BLACK, dtype=int)
    full[: top.shape[0]] = top
    full[top.shape[0]] = GREY
    full[top.shape[0] + 1 :] = base_scene
    for r, c, color in seeds:
        full[top.shape[0] + 1 + r, c] = color
    return full


def _focus_ok(focus: str, metrics: dict[str, int]) -> bool:
    kinds_hit = sum(int(metrics[name] > 0) for name in ("edge", "obstacle", "overlap"))
    if focus == "plain":
        return kinds_hit == 0
    if focus == "edge":
        return metrics["edge"] > 0
    if focus == "obstacle":
        return metrics["obstacle"] > 0
    if focus == "overlap":
        return metrics["overlap"] > 0
    return kinds_hit >= 2


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    template_count = rng.randint(*TEMPLATE_COUNT_RANGE)
    colors = rng.sample(TEMPLATE_COLORS, template_count)

    masks: list[tuple[int, np.ndarray]] = []
    seen_masks: set[tuple[int, ...]] = set()
    for color in colors:
        for _ in range(100):
            mask = _generate_template_mask(rng)
            sig = tuple(int(v) for v in mask.ravel()) + mask.shape
            if sig in seen_masks:
                continue
            seen_masks.add(sig)
            masks.append((color, mask))
            break
        else:
            return None

    try:
        top, templates, width = _layout_templates(rng, masks)
    except RuntimeError:
        return None

    scene_h = rng.randint(*SCENE_HEIGHT_RANGE)
    base_scene = _place_obstacles(rng, scene_h, width)
    if base_scene is None:
        return None

    focus = rng.choice(("plain", "edge", "obstacle", "overlap", "mixed"))
    seeds: list[tuple[int, int, int]] = []

    if focus == "edge":
        if not _try_add_seed(rng, base_scene, templates, seeds, "edge"):
            return None
    elif focus == "obstacle":
        if not _try_add_seed(rng, base_scene, templates, seeds, "obstacle"):
            return None
    elif focus == "overlap":
        if not _try_add_seed(rng, base_scene, templates, seeds, "normal"):
            return None
        if not _try_add_seed(rng, base_scene, templates, seeds, "overlap"):
            return None
    elif focus == "mixed":
        pair = rng.choice((("edge", "overlap"), ("obstacle", "overlap"), ("edge", "obstacle")))
        for kind in pair:
            if kind == "overlap" and not seeds:
                if not _try_add_seed(rng, base_scene, templates, seeds, "normal"):
                    return None
            if not _try_add_seed(rng, base_scene, templates, seeds, kind):
                return None

    used_colors = {color for _r, _c, color in seeds}
    shuffled_colors = colors[:]
    rng.shuffle(shuffled_colors)
    for color in shuffled_colors:
        if color in used_colors:
            continue
        target_kind = "normal" if focus == "plain" else "any"
        if not _try_add_seed(rng, base_scene, templates, seeds, target_kind, preferred_colors=[color]):
            return None
        used_colors.add(color)

    target_seed_count = min(6, template_count + rng.randint(*SEED_EXTRA_RANGE))
    extra_kinds = {
        "plain": ("normal",),
        "edge": ("normal", "edge"),
        "obstacle": ("normal", "obstacle"),
        "overlap": ("normal", "overlap"),
        "mixed": ("normal", "edge", "obstacle", "overlap"),
    }[focus]

    while len(seeds) < target_seed_count:
        kind = rng.choice(extra_kinds)
        if not _try_add_seed(rng, base_scene, templates, seeds, kind):
            if kind != "normal" and _try_add_seed(rng, base_scene, templates, seeds, "normal"):
                continue
            break

    out, metrics = _apply_growth(base_scene, templates, seeds=seeds)
    if not _focus_ok(focus, metrics):
        return None
    if metrics["placed"] <= len(seeds) + 2:
        return None
    if len({color for _r, _c, color in seeds}) < min(3, template_count):
        return None

    full = _build_input(top, width, base_scene, seeds)
    if not np.array_equal(main(full), out):
        return None
    return full


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
