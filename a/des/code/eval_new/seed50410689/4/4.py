from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAROON = 9

TEMPLATE_COLORS = (BLUE, RED, GREEN, YELLOW)
CORNERS = ("TL", "TR", "BL", "BR")

SHAPE_LIBRARY = (
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=int),
)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _inside(r: int, c: int, shape: tuple[int, int]) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _template_anchor_positions(r0: int, c0: int, r1: int, c1: int) -> dict[str, tuple[int, int]]:
    return {
        "TL": (r0 - 1, c0),
        "TR": (r0 - 1, c1),
        "BL": (r1 + 1, c0),
        "BR": (r1 + 1, c1),
    }


def _container_corner_positions(r0: int, c0: int, r1: int, c1: int) -> dict[str, tuple[int, int]]:
    return {
        "TL": (r0 + 1, c0 + 1),
        "TR": (r0 + 1, c1 - 1),
        "BL": (r1 - 1, c0 + 1),
        "BR": (r1 - 1, c1 - 1),
    }


def _find_containers(g: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for comp in _components(g == GRAY):
        r0, c0, r1, c1 = _bbox(comp)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w or side_h < 5:
            continue
        if not np.all(g[r0, c0 : c1 + 1] == GRAY):
            continue
        if not np.all(g[r1, c0 : c1 + 1] == GRAY):
            continue
        if not np.all(g[r0 : r1 + 1, c0] == GRAY):
            continue
        if not np.all(g[r0 : r1 + 1, c1] == GRAY):
            continue
        border_cells = 4 * side_h - 4
        if len(comp) != border_cells:
            continue
        inner_side = side_h - 2
        if inner_side % 3:
            continue
        corners = _container_corner_positions(r0, c0, r1, c1)
        anchor = [name for name, (rr, cc) in corners.items() if int(g[rr, cc]) == MAROON]
        if len(anchor) != 1:
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_side": inner_side,
                "anchor": anchor[0],
            }
        )
    assert frames, "expected at least one gray container"
    return frames


def _find_templates(g: np.ndarray) -> dict[str, np.ndarray]:
    template_map: dict[str, np.ndarray] = {}
    payload = (~np.isin(g, [BG, GRAY, MAROON]))
    for comp in _components(payload):
        color = int(g[comp[0][0], comp[0][1]])
        if any(int(g[r, c]) != color for r, c in comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        anchor_hits = []
        for name, (rr, cc) in _template_anchor_positions(r0, c0, r1, c1).items():
            if _inside(rr, cc, g.shape) and int(g[rr, cc]) == MAROON:
                anchor_hits.append(name)
        if len(anchor_hits) != 1:
            continue
        pattern = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        template_map[anchor_hits[0]] = pattern
    assert set(template_map) == set(CORNERS), "expected one 3x3 template for each corner anchor"
    return template_map


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each gray container with the enlarged legend template selected by its maroon corner marker.

    Input structure:
    - The background is black.
    - Several gray hollow square containers appear in the upper area; each container has a square interior whose side length is a multiple of 3, and exactly one maroon pixel sits in one interior corner.
    - A legend area contains four distinct 3x3 colored templates on black, each with a single maroon anchor pixel placed just outside one of the four template corners.

    Rule:
    - For each container, read which interior corner contains the maroon pixel: top-left, top-right, bottom-left, or bottom-right.
    - Find the legend template whose maroon anchor occupies the same corner.
    - Let k be the container's interior side length divided by 3, enlarge the chosen 3x3 template by factor k, and overwrite the whole container interior with that enlarged pattern.
    - Keep the gray borders and the legend area unchanged.

    Key insight:
    - The maroon pixels do not belong to the final drawing; they are control markers that link each container to one legend template and also determine the required scaling factor through the container size.
    """
    g = np.asarray(grid, dtype=int)
    templates = _find_templates(g)
    containers = _find_containers(g)

    out = g.copy()
    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        inner_side = int(container["inner_side"])
        scale = inner_side // 3
        anchor = str(container["anchor"])
        template = templates[anchor]
        enlarged = np.kron(template, np.ones((scale, scale), dtype=int))
        out[r0 + 1 : r1, c0 + 1 : c1] = enlarged
    return out


def _split_units(rng: np.random.Generator, total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.choice(np.arange(total + parts - 1), size=parts - 1, replace=False).tolist())
    out: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _sample_templates(rng: np.random.Generator) -> dict[str, np.ndarray]:
    shape_ids = rng.choice(len(SHAPE_LIBRARY), size=4, replace=False)
    shapes = [SHAPE_LIBRARY[int(idx)] for idx in shape_ids]
    shape_order = rng.permutation(4)
    color_order = rng.permutation(4)
    templates: dict[str, np.ndarray] = {}
    for i, corner in enumerate(CORNERS):
        color = TEMPLATE_COLORS[int(color_order[i])]
        shape = shapes[int(shape_order[i])]
        templates[corner] = (shape * color).astype(int)
    return templates


def _legend_positions(rng: np.random.Generator, width: int) -> list[int]:
    base = [1, 2, 2, 2, 1]
    min_width = 4 * 3 + sum(base)
    assert width >= min_width
    extra = width - min_width
    bonus = _split_units(rng, extra, len(base))
    left = base[0] + bonus[0]
    gaps = [base[i + 1] + bonus[i + 1] for i in range(3)]
    xs = [left]
    for gap in gaps:
        xs.append(xs[-1] + 3 + gap)
    return xs


def _draw_legends(
    grid: np.ndarray,
    templates: dict[str, np.ndarray],
    legend_top: int,
    xs: list[int],
    rng: np.random.Generator,
) -> None:
    legend_order = list(rng.permutation(4))
    corners = list(CORNERS)
    for slot, idx in enumerate(legend_order):
        corner = corners[int(idx)]
        x = xs[slot]
        grid[legend_top + 1 : legend_top + 4, x : x + 3] = templates[corner]
        mark_r, mark_c = {
            "TL": (legend_top, x),
            "TR": (legend_top, x + 2),
            "BL": (legend_top + 4, x),
            "BR": (legend_top + 4, x + 2),
        }[corner]
        grid[mark_r, mark_c] = MAROON


def _place_containers(
    rng: np.random.Generator,
    grid: np.ndarray,
    templates: dict[str, np.ndarray],
    top_limit: int,
) -> np.ndarray | None:
    h, w = grid.shape
    usable_area = top_limit * w
    if usable_area < 70:
        return None

    if usable_area < 170:
        max_containers = 2
    elif usable_area < 250:
        max_containers = 3
    else:
        max_containers = 4
    count = int(rng.integers(1, max_containers + 1))

    inner_choices = np.array([3, 6, 9], dtype=int)
    inner_sizes = rng.choice(inner_choices, size=count, replace=True).tolist()
    anchors = list(rng.choice(np.array(CORNERS), size=count, replace=False if count <= 4 else True).tolist())
    order = np.argsort([-size for size in inner_sizes])

    occupied = np.zeros_like(grid, dtype=bool)
    placements: list[tuple[int, int, int, str]] = []

    for idx in order:
        inner = int(inner_sizes[int(idx)])
        outer = inner + 2
        anchor = anchors[int(idx)]
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, top_limit - outer):
            for c0 in range(1, w - outer):
                if occupied[r0 - 1 : r0 + outer + 1, c0 - 1 : c0 + outer + 1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = candidates[int(rng.integers(len(candidates)))]
        occupied[r0 - 1 : r0 + outer + 1, c0 - 1 : c0 + outer + 1] = True
        placements.append((r0, c0, inner, anchor))

    out = grid.copy()
    expected = out.copy()
    for r0, c0, inner, anchor in placements:
        outer = inner + 2
        r1 = r0 + outer - 1
        c1 = c0 + outer - 1
        out[r0, c0 : c1 + 1] = GRAY
        out[r1, c0 : c1 + 1] = GRAY
        out[r0 : r1 + 1, c0] = GRAY
        out[r0 : r1 + 1, c1] = GRAY
        mr, mc = _container_corner_positions(r0, c0, r1, c1)[anchor]
        out[mr, mc] = MAROON

        scale = inner // 3
        enlarged = np.kron(templates[anchor], np.ones((scale, scale), dtype=int))
        expected[r0 + 1 : r1, c0 + 1 : c1] = enlarged
        expected[r0, c0 : c1 + 1] = GRAY
        expected[r1, c0 : c1 + 1] = GRAY
        expected[r0 : r1 + 1, c0] = GRAY
        expected[r0 : r1 + 1, c1] = GRAY

    solved = main(out)
    if not np.array_equal(solved, expected):
        return None
    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = np.random.default_rng()
        height = int(rng.integers(19, 31))
        width = int(rng.integers(20, 31))
        legend_top = height - 5
        top_limit = legend_top - 1
        if top_limit < 6:
            continue

        grid = np.zeros((height, width), dtype=int)
        templates = _sample_templates(rng)
        xs = _legend_positions(rng, width)
        _draw_legends(grid, templates, legend_top, xs, rng)

        placed = _place_containers(rng, grid, templates, top_limit)
        if placed is not None:
            return placed
    raise RuntimeError("failed to generate a valid input grid")
