import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME_COLOR = 5
KINDS = ("horizontal", "vertical", "rot180", "quad", "asymmetric")
USABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 30)
LEGEND_COUNT_OPTIONS = (3, 4, 5)
CONTAINER_COUNT_RANGE = (3, 6)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200
PATTERN_ATTEMPTS = 400

DEMO_VARIANTS = {
    "horizontal": [
        np.array([[1, 0], [1, 0]], dtype=bool),
        np.array([[0, 1], [0, 1]], dtype=bool),
    ],
    "vertical": [
        np.array([[1, 1], [0, 0]], dtype=bool),
        np.array([[0, 0], [1, 1]], dtype=bool),
    ],
    "rot180": [
        np.array([[1, 0], [0, 1]], dtype=bool),
        np.array([[0, 1], [1, 0]], dtype=bool),
    ],
    "quad": [
        np.array([[1, 1], [1, 1]], dtype=bool),
    ],
    "asymmetric": [
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1], [1, 1]], dtype=bool),
    ],
}

QUAD_TEMPLATES = {
    3: [
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    ],
    4: [
        np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=bool),
        np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=bool),
        np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=bool),
        np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]], dtype=bool),
        np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool),
    ],
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                if rr > 0 and bool(mask[rr - 1, cc]) and not bool(seen[rr - 1, cc]):
                    seen[rr - 1, cc] = True
                    q.append((rr - 1, cc))
                if rr + 1 < h and bool(mask[rr + 1, cc]) and not bool(seen[rr + 1, cc]):
                    seen[rr + 1, cc] = True
                    q.append((rr + 1, cc))
                if cc > 0 and bool(mask[rr, cc - 1]) and not bool(seen[rr, cc - 1]):
                    seen[rr, cc - 1] = True
                    q.append((rr, cc - 1))
                if cc + 1 < w and bool(mask[rr, cc + 1]) and not bool(seen[rr, cc + 1]):
                    seen[rr, cc + 1] = True
                    q.append((rr, cc + 1))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _symmetry_flags(mask: np.ndarray) -> tuple[bool, bool, bool, bool, bool]:
    horiz = bool(np.array_equal(mask, np.flipud(mask)))
    vert = bool(np.array_equal(mask, np.fliplr(mask)))
    rot180 = bool(np.array_equal(mask, np.rot90(mask, 2)))
    diag_main = bool(np.array_equal(mask, mask.T))
    diag_anti = bool(np.array_equal(mask, np.fliplr(np.flipud(mask)).T))
    return horiz, vert, rot180, diag_main, diag_anti


def _pattern_kind(mask: np.ndarray | None) -> str | None:
    if mask is None or mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        return None
    horiz, vert, rot180, diag_main, diag_anti = _symmetry_flags(mask)
    if horiz and vert and diag_main and diag_anti:
        return "quad"
    if horiz and not (vert or rot180 or diag_main or diag_anti):
        return "horizontal"
    if vert and not (horiz or rot180 or diag_main or diag_anti):
        return "vertical"
    if rot180 and not (horiz or vert or diag_main or diag_anti):
        return "rot180"
    if not (horiz or vert or rot180 or diag_main or diag_anti):
        return "asymmetric"
    return None


def _legend_kind(mask: np.ndarray) -> str | None:
    coords = np.argwhere(mask)
    count = int(coords.shape[0])
    if count == 4:
        return "quad"
    if count == 3:
        return "asymmetric"
    if count != 2:
        return None
    rows = {int(v) for v in coords[:, 0]}
    cols = {int(v) for v in coords[:, 1]}
    if len(cols) == 1:
        return "horizontal"
    if len(rows) == 1:
        return "vertical"
    return "rot180"


def _legend_mapping(grid: np.ndarray) -> dict[str, int]:
    h, w = grid.shape
    if h < 3:
        return {}
    mapping: dict[str, int] = {}
    for c in range(w - 1):
        color = int(grid[h - 1, c])
        if color in (BG, FRAME_COLOR):
            continue
        demo = grid[h - 3 : h - 1, c : c + 2]
        mask = demo != BG
        if demo.shape != (2, 2) or not bool(mask.any()):
            continue
        vals = {int(v) for v in demo[mask]}
        if len(vals) != 1:
            continue
        kind = _legend_kind(mask)
        if kind is not None:
            mapping[kind] = color
    return mapping


def _find_containers(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    grey_mask = grid == FRAME_COLOR
    boxes: list[tuple[int, int, int, int]] = []
    for comp in _components(grey_mask):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue
        patch = grey_mask[r0 : r1 + 1, c0 : c1 + 1]
        frame = np.zeros_like(patch, dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        if np.array_equal(patch, frame):
            boxes.append((r0, c0, r1, c1))
    boxes.sort()
    return boxes


def _extract_inner_pattern_mask(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    r0, c0, r1, c1 = box
    inside = grid[r0 + 1 : r1, c0 + 1 : c1]
    coords = np.argwhere(inside != BG)
    if coords.size == 0:
        return None
    vals = {int(v) for v in inside[inside != BG]}
    if len(vals) != 1:
        return None
    rr0, cc0 = coords.min(axis=0)
    rr1, cc1 = coords.max(axis=0)
    patch = inside[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1]
    return patch != BG


def _paint_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    mapping = _legend_mapping(g)
    for box in _find_containers(g):
        kind = _pattern_kind(_extract_inner_pattern_mask(g, box))
        if kind in mapping:
            _paint_frame(out, box, int(mapping[kind]))
    return out


def _composition(total: int, parts: int) -> list[int]:
    if total <= 0:
        return [0] * parts
    return [int(v) for v in np.random.multinomial(int(total), [1.0 / parts] * parts)]


def _pack_intervals(length: int, widths: list[int], min_gap: int) -> list[tuple[int, int]] | None:
    extra = int(length) - sum(int(v) for v in widths) - int(min_gap) * (len(widths) - 1)
    if extra < 0:
        return None
    gaps = _composition(extra, len(widths) + 1)
    pos = int(gaps[0])
    intervals: list[tuple[int, int]] = []
    for i, width in enumerate(widths):
        width = int(width)
        intervals.append((pos, pos + width - 1))
        pos += width
        if i + 1 < len(widths):
            pos += int(min_gap) + int(gaps[i + 1])
    return intervals


def _full_span(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_full_span_mask(size: int) -> np.ndarray:
    for _ in range(PATTERN_ATTEMPTS):
        p = random.uniform(0.28, 0.72)
        mask = np.random.rand(size, size) < p
        if 3 <= int(mask.sum()) <= size * size - 1 and _full_span(mask):
            return mask
    raise RuntimeError("failed to sample full-span mask")


def _random_pattern_mask(kind: str, size: int) -> np.ndarray:
    for _ in range(PATTERN_ATTEMPTS):
        if kind == "horizontal":
            p = random.uniform(0.25, 0.75)
            mask = np.zeros((size, size), dtype=bool)
            for r in range((size + 1) // 2):
                row = np.random.rand(size) < p
                mask[r, :] = row
                mask[size - 1 - r, :] = row
        elif kind == "vertical":
            p = random.uniform(0.25, 0.75)
            mask = np.zeros((size, size), dtype=bool)
            for c in range((size + 1) // 2):
                col = np.random.rand(size) < p
                mask[:, c] = col
                mask[:, size - 1 - c] = col
        elif kind == "rot180":
            p = random.uniform(0.25, 0.75)
            mask = np.zeros((size, size), dtype=bool)
            for r in range(size):
                for c in range(size):
                    rr, cc = size - 1 - r, size - 1 - c
                    if r > rr or (r == rr and c > cc):
                        continue
                    val = bool(random.random() < p)
                    mask[r, c] = val
                    mask[rr, cc] = val
        elif kind == "quad":
            mask = random.choice(QUAD_TEMPLATES[size]).copy()
        else:
            mask = _random_full_span_mask(size)
        if _full_span(mask) and _pattern_kind(mask) == kind:
            return mask
    raise RuntimeError(f"failed to sample pattern kind={kind} size={size}")


def _draw_legend(grid: np.ndarray, legend_pairs: list[tuple[str, int]]) -> bool:
    starts = _pack_intervals(grid.shape[1], [2] * len(legend_pairs), min_gap=1)
    if starts is None:
        return False
    top = grid.shape[0] - 3
    shuffled = legend_pairs[:]
    random.shuffle(shuffled)
    for (kind, color), (c0, _c1) in zip(shuffled, starts):
        demo = random.choice(DEMO_VARIANTS[kind])
        block = grid[top : top + 2, c0 : c0 + 2]
        block[demo] = int(color)
        grid[top + 2, c0] = int(color)
    return True


def _choose_container_kinds(legend_kinds: list[str], count: int) -> list[str]:
    present = list(legend_kinds)
    missing = [kind for kind in KINDS if kind not in present]
    kinds = [random.choice(present)]
    if missing:
        kinds.append(random.choice(missing))
    while len(kinds) < count:
        if missing and random.random() < 0.5:
            kinds.append(random.choice(missing))
        else:
            kinds.append(random.choice(present))
    random.shuffle(kinds)
    return kinds


def _place_containers(
    grid: np.ndarray,
    container_kinds: list[str],
    mapping: dict[str, int],
) -> list[dict] | None:
    h, w = grid.shape
    limit_row = h - 5
    reserved = np.zeros((h, w), dtype=bool)
    placed_specs = []
    for kind in container_kinds:
        size = random.choice([3, 4])
        mask = _random_pattern_mask(kind, size)
        inner_h = size + random.randint(0, 2)
        inner_w = size + random.randint(0, 2)
        pr = random.randint(0, inner_h - size)
        pc = random.randint(0, inner_w - size)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        placed_specs.append(
            {
                "kind": kind,
                "mask": mask,
                "size": size,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "pr": pr,
                "pc": pc,
                "outer_h": outer_h,
                "outer_w": outer_w,
            }
        )
    order = sorted(range(len(placed_specs)), key=lambda i: placed_specs[i]["outer_h"] * placed_specs[i]["outer_w"], reverse=True)
    placements: list[dict | None] = [None] * len(placed_specs)
    for idx in order:
        spec = placed_specs[idx]
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        max_r0 = limit_row - outer_h + 1
        max_c0 = w - outer_w - 1
        if max_r0 < 1 or max_c0 < 1:
            return None
        success = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            rr0 = r0 - 1
            cc0 = c0 - 1
            rr1 = min(h - 1, r1 + 1)
            cc1 = min(w - 1, c1 + 1)
            if bool(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                continue
            placements[idx] = {"r0": r0, "c0": c0, "r1": r1, "c1": c1, **spec}
            reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            success = True
            break
        if not success:
            return None
    finalized: list[dict] = []
    for item in placements:
        assert item is not None
        finalized.append(item)
    for item in finalized:
        _paint_frame(grid, (item["r0"], item["c0"], item["r1"], item["c1"]), FRAME_COLOR)
        rr = item["r0"] + 1 + item["pr"]
        cc = item["c0"] + 1 + item["pc"]
        color_choices = [c for c in USABLE_COLORS if c != mapping.get(item["kind"], -1)]
        pattern_color = random.choice(color_choices or USABLE_COLORS)
        patch = grid[rr : rr + item["size"], cc : cc + item["size"]]
        patch[item["mask"]] = int(pattern_color)
        item["pattern_color"] = int(pattern_color)
    return finalized


def _validate_generated_grid(
    grid: np.ndarray,
    legend_mapping: dict[str, int],
    placements: list[dict],
) -> bool:
    if _legend_mapping(grid) != legend_mapping:
        return False

    found = _find_containers(grid)
    expected_boxes = sorted((p["r0"], p["c0"], p["r1"], p["c1"]) for p in placements)
    if found != expected_boxes:
        return False

    intended = Counter(p["kind"] for p in placements)
    observed = Counter()
    for box in found:
        kind = _pattern_kind(_extract_inner_pattern_mask(grid, box))
        if kind is None:
            return False
        observed[kind] += 1
    if observed != intended:
        return False

    out = main(grid)
    for p in placements:
        box = (p["r0"], p["c0"], p["r1"], p["c1"])
        expected = int(legend_mapping.get(p["kind"], FRAME_COLOR))
        r0, c0, r1, c1 = box
        if not bool(np.all(out[r0, c0 : c1 + 1] == expected)):
            return False
        if not bool(np.all(out[r1, c0 : c1 + 1] == expected)):
            return False
        if not bool(np.all(out[r0 : r1 + 1, c0] == expected)):
            return False
        if not bool(np.all(out[r0 : r1 + 1, c1] == expected)):
            return False
        if not np.array_equal(out[r0 + 1 : r1, c0 + 1 : c1], grid[r0 + 1 : r1, c0 + 1 : c1]):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)

    legend_count = random.choice(LEGEND_COUNT_OPTIONS)
    legend_kinds = random.sample(list(KINDS), legend_count)
    legend_colors = random.sample(USABLE_COLORS, legend_count)
    legend_mapping = {kind: color for kind, color in zip(legend_kinds, legend_colors)}

    if not _draw_legend(grid, list(legend_mapping.items())):
        return None

    container_count = random.randint(*CONTAINER_COUNT_RANGE)
    container_kinds = _choose_container_kinds(legend_kinds, container_count)
    placements = _place_containers(grid, container_kinds, legend_mapping)
    if placements is None:
        return None

    if not _validate_generated_grid(grid, legend_mapping, placements):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
