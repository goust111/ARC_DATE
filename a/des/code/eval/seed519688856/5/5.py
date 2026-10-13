from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
GREY = 5
ORANGE = 7
MAROON = 9
VOTE_COLORS = (BLUE, GREEN, ORANGE)


def _mask(lines: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in lines], dtype=bool)


TEMPLATE_LIBRARY = (
    _mask(
        [
            "11111",
            "10001",
            "10001",
            "10001",
            "11111",
        ]
    ),
    _mask(
        [
            "00100",
            "01110",
            "11011",
            "01110",
            "00100",
        ]
    ),
    _mask(
        [
            "0011100",
            "0111110",
            "1100011",
            "1000001",
            "1100011",
            "0111110",
            "0011100",
        ]
    ),
    _mask(
        [
            "0001000",
            "0011100",
            "0110110",
            "1100011",
            "0110110",
            "0011100",
            "0001000",
        ]
    ),
)


_RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _adjacent_label_candidates(
    grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    out: list[tuple[int, int, int]] = []
    for r in range(max(0, r0 - 1), min(h, r1 + 2)):
        for c in range(max(0, c0 - 1), min(w, c1 + 2)):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            on_side = ((r == r0 - 1 or r == r1 + 1) and c0 <= c <= c1) or (
                (c == c0 - 1 or c == c1 + 1) and r0 <= r <= r1
            )
            if not on_side:
                continue
            color = int(grid[r, c])
            if color in VOTE_COLORS:
                out.append((r, c, color))
    return out


def _extract_templates(grid: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    grey_components = _components_mask(grid == GREY, connectivity=8)
    templates: dict[int, np.ndarray] = {}
    protected = np.zeros(grid.shape, dtype=bool)
    for comp in grey_components:
        bbox = _bbox(comp)
        candidates = _adjacent_label_candidates(grid, bbox)
        assert len(candidates) == 1, "each grey template must have exactly one adjacent label"
        label_r, label_c, label_color = candidates[0]
        assert label_color not in templates, "label colors must be unique"
        templates[label_color] = _mask_from_coords(comp)
        for r, c in comp:
            protected[r, c] = True
        protected[label_r, label_c] = True
    assert 2 <= len(templates) <= 3, "expected two or three templates"
    return templates, protected


def _extract_objects(
    grid: np.ndarray, bg: int, protected: np.ndarray, label_colors: set[int]
) -> list[list[tuple[int, int]]]:
    mask = (grid != bg) & (~protected)
    comps = _components_mask(mask, connectivity=4)
    assert comps, "expected at least one object"
    objects: list[list[tuple[int, int]]] = []
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        assert MAROON in colors, "object must contain maroon body pixels"
        assert colors <= ({MAROON} | label_colors), "unexpected object color"
        objects.append(comp)
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each maroon object by the template selected by its internal vote majority."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    assert bg == BLACK, "background must be black"

    templates, protected = _extract_templates(g)
    objects = _extract_objects(g, bg, protected, set(templates))

    placements: list[tuple[int, int, np.ndarray]] = []
    clear_cells: list[tuple[int, int]] = []
    occupied = protected.copy()

    for comp in objects:
        votes = Counter(int(g[r, c]) for r, c in comp if int(g[r, c]) in templates)
        assert votes, "each object must contain vote pixels"
        top_count = max(votes.values())
        winners = [color for color, count in votes.items() if count == top_count]
        assert len(winners) == 1, "vote majority must be unique"
        winner = winners[0]

        r0, c0, r1, c1 = _bbox(comp)
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        tpl = templates[winner]
        h, w = tpl.shape
        top = center_r - h // 2
        left = center_c - w // 2
        assert 0 <= top and 0 <= left and top + h <= g.shape[0] and left + w <= g.shape[1]
        rr, cc = np.where(tpl)
        for dr, dc in zip(rr.tolist(), cc.tolist()):
            r = top + dr
            c = left + dc
            assert not occupied[r, c], "repainted template must stay clear of the legend and other outputs"
        for dr, dc in zip(rr.tolist(), cc.tolist()):
            occupied[top + dr, left + dc] = True
        placements.append((top, left, tpl))
        clear_cells.extend(comp)
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True

    out = g.copy()
    for r, c in clear_cells:
        out[r, c] = bg
    for top, left, tpl in placements:
        patch = out[top : top + tpl.shape[0], left : left + tpl.shape[1]]
        patch[tpl] = GREY
    return out


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    h, w = mask.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                cells.append((r, c))
    return cells


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    comp = _components_mask(mask, connectivity=4)
    return len(comp) == 1


def _make_object_mask(required_votes: int) -> np.ndarray:
    for _ in range(100):
        mask = np.ones((5, 5), dtype=bool)
        corner_specs = {
            "nw": [(0, 0), (0, 1), (1, 0)],
            "ne": [(0, 4), (0, 3), (1, 4)],
            "sw": [(4, 0), (4, 1), (3, 0)],
            "se": [(4, 4), (4, 3), (3, 4)],
        }
        for key in ("nw", "ne", "sw", "se"):
            mode = int(_RNG.integers(0, 3))
            cells = corner_specs[key]
            if mode >= 1:
                mask[cells[0]] = False
            if mode == 2:
                mask[cells[1]] = False
                mask[cells[2]] = False
        if not _connected(mask):
            continue
        if any(not np.any(mask[i, :]) for i in range(5)) or any(not np.any(mask[:, j]) for j in range(5)):
            continue
        if len(_interior_cells(mask)) < required_votes:
            continue
        return mask
    raise RuntimeError("failed to build object mask")


def _vote_plan(colors: list[int], winner: int) -> list[int]:
    others = [color for color in colors if color != winner]
    _RNG.shuffle(others)
    if len(colors) == 2:
        return [winner] * 3 + [others[0]]
    return [winner] * 3 + [others[0]] * 2 + [others[1]]


def _paint_template(
    grid: np.ndarray, top: int, left: int, mask: np.ndarray, label_color: int, orientation: str
) -> tuple[int, int]:
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = GREY
    if orientation == "top":
        label_pos = (top + mask.shape[0], left + mask.shape[1] // 2)
    else:
        label_pos = (top + mask.shape[0] // 2, left + mask.shape[1])
    grid[label_pos] = label_color
    return label_pos


def _reserve_rectangle(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(blocked.shape[0] - 1, r1)
    cc1 = min(blocked.shape[1] - 1, c1)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_centers(
    blocked: np.ndarray,
    field: tuple[int, int, int, int],
    specs: list[dict[str, object]],
) -> list[tuple[int, int]] | None:
    r_min, c_min, r_max, c_max = field
    reserve = blocked.copy()
    ordered = sorted(
        enumerate(specs),
        key=lambda item: max(item[1]["tpl_mask"].shape),  # type: ignore[index]
        reverse=True,
    )
    centers: list[tuple[int, int] | None] = [None] * len(specs)
    for idx, spec in ordered:
        obj_h, obj_w = spec["obj_mask"].shape  # type: ignore[index]
        tpl_h, tpl_w = spec["tpl_mask"].shape  # type: ignore[index]
        half_h = max(obj_h, tpl_h) // 2
        half_w = max(obj_w, tpl_w) // 2
        candidates: list[tuple[int, int]] = []
        for r in range(r_min + half_h, r_max - half_h + 1):
            for c in range(c_min + half_w, c_max - half_w + 1):
                rr0 = r - half_h - 1
                cc0 = c - half_w - 1
                rr1 = r + half_h + 1
                cc1 = c + half_w + 1
                if np.any(reserve[max(0, rr0) : min(reserve.shape[0], rr1 + 1), max(0, cc0) : min(reserve.shape[1], cc1 + 1)]):
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        center = candidates[int(_RNG.integers(len(candidates)))]
        centers[idx] = center
        _reserve_rectangle(reserve, center[0] - half_h - 1, center[1] - half_w - 1, center[0] + half_h + 1, center[1] + half_w + 1)
    return [center for center in centers if center is not None]


def _paint_object(
    grid: np.ndarray,
    center: tuple[int, int],
    obj_mask: np.ndarray,
    vote_colors: list[int],
) -> None:
    h, w = obj_mask.shape
    top = center[0] - h // 2
    left = center[1] - w // 2
    patch = grid[top : top + h, left : left + w]
    patch[obj_mask] = MAROON
    interiors = _interior_cells(obj_mask)
    assert len(interiors) >= len(vote_colors)
    order = _RNG.permutation(len(interiors))[: len(vote_colors)]
    for idx, color in zip(order.tolist(), vote_colors):
        r, c = interiors[idx]
        patch[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    orientation = "top" if _RNG.random() < 0.5 else "side"
    template_count = 2 if _RNG.random() < 0.5 else 3
    chosen_colors = list(_RNG.choice(np.array(VOTE_COLORS), size=template_count, replace=False).tolist())
    template_indices = _RNG.choice(len(TEMPLATE_LIBRARY), size=template_count, replace=False)
    template_masks = [TEMPLATE_LIBRARY[int(i)] for i in template_indices.tolist()]

    gaps = [int(_RNG.integers(2, 5)) for _ in range(template_count - 1)]
    widths = [mask.shape[1] for mask in template_masks]
    heights = [mask.shape[0] for mask in template_masks]
    max_h = max(heights)
    max_w = max(widths)

    num_objects = template_count + int(_RNG.integers(0, 2))
    object_specs: list[dict[str, object]] = []
    winners = chosen_colors.copy()
    while len(winners) < num_objects:
        winners.append(int(_RNG.choice(chosen_colors)))
    _RNG.shuffle(winners)
    required_votes = 4 if template_count == 2 else 6
    for winner in winners:
        obj_mask = _make_object_mask(required_votes)
        object_specs.append(
            {
                "winner": winner,
                "obj_mask": obj_mask,
                "tpl_mask": template_masks[chosen_colors.index(winner)],
                "votes": _vote_plan(chosen_colors, winner),
            }
        )

    if orientation == "top":
        min_w = sum(widths) + sum(gaps) + 4
        w_low = max(24, min_w + 2)
        w_high = w_low + 7
        h_low = max(24, max_h + 18)
        h_high = h_low + 6
    else:
        min_h = sum(heights) + sum(gaps) + 4
        h_low = max(24, min_h + 2)
        h_high = h_low + 7
        w_low = max(24, max_w + 18)
        w_high = w_low + 6

    h = int(_RNG.integers(h_low, h_high + 1))
    w = int(_RNG.integers(w_low, w_high + 1))
    grid = np.full((h, w), BLACK, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    if orientation == "top":
        top = int(_RNG.integers(1, 3))
        total_w = sum(widths) + sum(gaps)
        left = int(_RNG.integers(1, w - total_w - 1))
        x = left
        bottom = 0
        for idx, mask in enumerate(template_masks):
            label = _paint_template(grid, top, x, mask, chosen_colors[idx], orientation)
            _reserve_rectangle(blocked, top - 1, x - 1, top + mask.shape[0], x + mask.shape[1],)
            blocked[label] = True
            bottom = max(bottom, top + mask.shape[0])
            if idx < len(gaps):
                x += mask.shape[1] + gaps[idx]
        field = (bottom + 2, 1, h - 2, w - 2)
    else:
        left = int(_RNG.integers(1, 3))
        total_h = sum(heights) + sum(gaps)
        top = int(_RNG.integers(1, h - total_h - 1))
        y = top
        right = 0
        for idx, mask in enumerate(template_masks):
            label = _paint_template(grid, y, left, mask, chosen_colors[idx], orientation)
            _reserve_rectangle(blocked, y - 1, left - 1, y + mask.shape[0], left + mask.shape[1],)
            blocked[label] = True
            right = max(right, left + mask.shape[1])
            if idx < len(gaps):
                y += mask.shape[0] + gaps[idx]
        field = (1, right + 2, h - 2, w - 2)

    centers = _place_centers(blocked, field, object_specs)
    if centers is None:
        return None

    for spec, center in zip(object_specs, centers):
        _paint_object(grid, center, spec["obj_mask"], spec["votes"])  # type: ignore[arg-type]

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
