import random
from collections import deque

import numpy as np


ORANGE = 7
GEN_RETRIES = 300
REGION_ROW_COUNT_RANGE = (2, 4)
REGION_COL_COUNT_RANGE = (2, 4)
REGION_HEIGHT_RANGE = (5, 8)
REGION_WIDTH_RANGE = (5, 8)
GRID_HEIGHT_RANGE = (14, 30)
GRID_WIDTH_RANGE = (14, 30)
PALETTE = tuple(c for c in range(10) if c not in {0, ORANGE})


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _diag_neighbors(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1))


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rect_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _fill_patch(kind: str, main_color: int, minor_color: int, height: int, width: int) -> np.ndarray:
    rows = np.arange(height, dtype=int)[:, None]
    cols = np.arange(width, dtype=int)[None, :]
    if kind == "horizontal":
        mask = rows % 2 == 0
    elif kind == "vertical":
        mask = cols % 2 == 0
    elif kind == "checker":
        mask = (rows + cols) % 2 == 0
    else:
        raise AssertionError("unknown fill kind")
    return np.where(mask, int(main_color), int(minor_color))


def _find_separator_color(g: np.ndarray, bg: int) -> int:
    candidates = []
    for color in sorted(set(map(int, np.unique(g))) - {int(bg)}):
        comps = _components_from_mask(g != int(color))
        if len(comps) < 2:
            continue
        if all(_is_rect_component(comp) for comp in comps):
            candidates.append(int(color))
    assert len(candidates) == 1, "expected a unique separator color"
    return candidates[0]


def _classify_region(
    g: np.ndarray,
    region_bbox: tuple[int, int, int, int],
    bg: int,
    sep: int,
) -> tuple[int, int, str]:
    r0, c0, r1, c1 = region_bbox
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    seed_coords = np.argwhere((patch != int(bg)) & (patch != int(sep)))
    assert seed_coords.size > 0, "each region must contain a seed pattern"

    counts: dict[int, int] = {}
    for color in patch[(patch != int(bg)) & (patch != int(sep))].tolist():
        counts[int(color)] = counts.get(int(color), 0) + 1
    assert len(counts) == 2, "each seed must use exactly two colors"

    ranked = sorted(counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))
    main_color, main_count = ranked[0]
    minor_color, minor_count = ranked[1]
    assert main_count > minor_count, "main color must be the majority color"

    contact_axes: set[str] = set()
    has_diagonal_contact = False
    for rr, cc in np.argwhere(patch == int(minor_color)):
        rr = int(rr)
        cc = int(cc)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1] and int(patch[nr, nc]) == int(main_color):
                contact_axes.add("vertical" if nr != rr else "horizontal")
        for nr, nc in _diag_neighbors(rr, cc):
            if 0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1] and int(patch[nr, nc]) == int(main_color):
                has_diagonal_contact = True

    if contact_axes:
        assert len(contact_axes) == 1, "stripe seed must imply a unique stripe axis"
        axis = next(iter(contact_axes))
        kind = "horizontal" if axis == "vertical" else "vertical"
        return int(main_color), int(minor_color), kind

    assert has_diagonal_contact, "checker seed must have diagonal main/minor contact"
    return int(main_color), int(minor_color), "checker"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the seed pattern's connectivity to decide whether each pane receives
    horizontal stripes, vertical stripes, or a checkerboard fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(np.bincount(g.ravel()).argmax())
    sep = _find_separator_color(g, bg)

    out = g.copy()
    for comp in _components_from_mask(g != int(sep)):
        assert _is_rect_component(comp), "each pane must be rectangular"
        r0, c0, r1, c1 = _bbox(comp)
        main_color, minor_color, kind = _classify_region(g, (r0, c0, r1, c1), bg, sep)
        out[r0 : r1 + 1, c0 : c1 + 1] = _fill_patch(
            kind,
            int(main_color),
            int(minor_color),
            r1 - r0 + 1,
            c1 - c0 + 1,
        )

    return out


H_STRIPE_TEMPLATES = (
    np.array([[0, 2, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 2, 0]], dtype=int),
    np.array([[0, 2, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 2, 0]], dtype=int),
    np.array([[2, 2, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 2, 2]], dtype=int),
)

V_STRIPE_TEMPLATES = (
    np.array([[0, 1], [2, 1], [0, 1]], dtype=int),
    np.array([[1, 0], [1, 2], [1, 0]], dtype=int),
    np.array([[0, 1, 0], [2, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 2], [0, 1, 0]], dtype=int),
    np.array([[2, 1], [2, 1], [0, 1]], dtype=int),
    np.array([[1, 2], [1, 2], [1, 0]], dtype=int),
)

CHECKER_TEMPLATES = (
    np.array([[2, 0, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 2], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 1], [0, 2, 0], [1, 0, 0]], dtype=int),
    np.array([[0, 0, 1], [0, 2, 0], [1, 0, 1]], dtype=int),
    np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=int),
    np.array([[0, 0, 1], [0, 2, 0], [1, 0, 0]], dtype=int),
)


def _validate_templates() -> None:
    for kind, templates in (
        ("horizontal", H_STRIPE_TEMPLATES),
        ("vertical", V_STRIPE_TEMPLATES),
        ("checker", CHECKER_TEMPLATES),
    ):
        for template in templates:
            counts = {1: int(np.count_nonzero(template == 1)), 2: int(np.count_nonzero(template == 2))}
            assert counts[1] > counts[2] > 0
            contact_axes: set[str] = set()
            diag = False
            for r, c in np.argwhere(template == 2):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < template.shape[0] and 0 <= nc < template.shape[1] and int(template[nr, nc]) == 1:
                        contact_axes.add("vertical" if nr != int(r) else "horizontal")
                for nr, nc in _diag_neighbors(int(r), int(c)):
                    if 0 <= nr < template.shape[0] and 0 <= nc < template.shape[1] and int(template[nr, nc]) == 1:
                        diag = True
            if kind == "checker":
                assert not contact_axes and diag
            elif kind == "horizontal":
                assert contact_axes == {"vertical"}
            else:
                assert contact_axes == {"horizontal"}


_validate_templates()


def _sample_region_sizes(count_range: tuple[int, int], size_range: tuple[int, int]) -> list[int]:
    count = random.randint(*count_range)
    return [random.randint(*size_range) for _ in range(count)]


def _region_bboxes(heights: list[int], widths: list[int]) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    r = 1
    for height in heights:
        c = 1
        for width in widths:
            boxes.append((r, c, r + height - 1, c + width - 1))
            c += width + 1
        r += height + 1
    return boxes


def _draw_frame_and_separators(grid: np.ndarray, heights: list[int], widths: list[int]) -> None:
    grid[0, :] = ORANGE
    grid[-1, :] = ORANGE
    grid[:, 0] = ORANGE
    grid[:, -1] = ORANGE

    r = 0
    for height in heights[:-1]:
        r += height + 1
        grid[r, :] = ORANGE

    c = 0
    for width in widths[:-1]:
        c += width + 1
        grid[:, c] = ORANGE


def _choose_region_kinds(n: int) -> list[str]:
    assert n >= 3
    kinds = ["horizontal", "vertical", "checker"]
    while len(kinds) < n:
        kinds.append(random.choice(["horizontal", "vertical", "checker"]))
    random.shuffle(kinds)
    return kinds


def _ordered_color_pairs(n: int) -> list[tuple[int, int]]:
    pairs = [(a, b) for a in PALETTE for b in PALETTE if b != a]
    random.shuffle(pairs)
    return pairs[:n]


def _choose_template(kind: str) -> np.ndarray:
    if kind == "horizontal":
        return random.choice(H_STRIPE_TEMPLATES)
    if kind == "vertical":
        return random.choice(V_STRIPE_TEMPLATES)
    return random.choice(CHECKER_TEMPLATES)


def _place_seed(
    grid: np.ndarray,
    region_bbox: tuple[int, int, int, int],
    kind: str,
    main_color: int,
    minor_color: int,
) -> tuple[int, int, str]:
    template = _choose_template(kind)
    ph, pw = template.shape
    r0, c0, r1, c1 = region_bbox
    region_h = r1 - r0 + 1
    region_w = c1 - c0 + 1
    assert region_h >= ph + 2 and region_w >= pw + 2, "pane must allow a 1-cell seed margin"

    top = random.randint(r0 + 1, r1 - ph)
    left = random.randint(c0 + 1, c1 - pw)
    patch = np.zeros_like(template)
    patch[template == 1] = int(main_color)
    patch[template == 2] = int(minor_color)
    window = grid[top : top + ph, left : left + pw]
    window[patch != 0] = patch[patch != 0]
    return ph, pw, kind


def _build_output_from_specs(
    shape: tuple[int, int],
    heights: list[int],
    widths: list[int],
    specs: list[tuple[tuple[int, int, int, int], int, int, str]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    _draw_frame_and_separators(out, heights, widths)
    for region_bbox, main_color, minor_color, kind in specs:
        r0, c0, r1, c1 = region_bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = _fill_patch(
            kind,
            int(main_color),
            int(minor_color),
            r1 - r0 + 1,
            c1 - c0 + 1,
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    heights = _sample_region_sizes(REGION_ROW_COUNT_RANGE, REGION_HEIGHT_RANGE)
    widths = _sample_region_sizes(REGION_COL_COUNT_RANGE, REGION_WIDTH_RANGE)
    h = sum(heights) + len(heights) + 1
    w = sum(widths) + len(widths) + 1
    if not (GRID_HEIGHT_RANGE[0] <= h <= GRID_HEIGHT_RANGE[1]):
        return None
    if not (GRID_WIDTH_RANGE[0] <= w <= GRID_WIDTH_RANGE[1]):
        return None

    inp = np.zeros((h, w), dtype=int)
    _draw_frame_and_separators(inp, heights, widths)
    boxes = _region_bboxes(heights, widths)
    if len(boxes) < 3:
        return None

    kinds = _choose_region_kinds(len(boxes))
    color_pairs = _ordered_color_pairs(len(boxes))
    if len(color_pairs) != len(boxes):
        return None

    specs: list[tuple[tuple[int, int, int, int], int, int, str]] = []
    for region_bbox, kind, (main_color, minor_color) in zip(boxes, kinds, color_pairs, strict=False):
        _place_seed(inp, region_bbox, kind, int(main_color), int(minor_color))
        specs.append((region_bbox, int(main_color), int(minor_color), kind))

    out = _build_output_from_specs(inp.shape, heights, widths, specs)
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
