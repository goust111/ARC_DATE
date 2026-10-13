from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

TEMPLATE_COLORS = (BLUE, YELLOW)
RECT_COLORS = (GREY, RED)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_template(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    mask = np.isin(grid, TEMPLATE_COLORS)
    components = _components(mask)
    assert len(components) == 1, "expected one blue-yellow template component"
    bbox = _bbox(components[0])
    r0, c0, r1, c1 = bbox
    template = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    colors = set(map(int, np.unique(template))) - {BLACK}
    assert colors == set(TEMPLATE_COLORS), "template must use blue and yellow"
    return bbox, template


def _find_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    mask = np.isin(grid, RECT_COLORS)
    components = _components(mask)
    rectangles: list[dict[str, object]] = []
    for coords in components:
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        assert h >= 3 and w >= 3, "expected grey rectangles with a non-empty interior"
        assert np.all(np.isin(patch, RECT_COLORS)), "rectangle bbox must contain only grey/red"
        assert np.all(patch[0, :] == GREY) and np.all(patch[-1, :] == GREY)
        assert np.all(patch[:, 0] == GREY) and np.all(patch[:, -1] == GREY)
        inner = patch[1:-1, 1:-1]
        red_count = int(np.count_nonzero(inner == RED))
        assert 1 <= red_count <= 4, "expected 1..4 red markers per rectangle"
        rectangles.append(
            {
                "bbox": bbox,
                "inner_shape": inner.shape,
                "red_count": red_count,
            }
        )
    rectangles.sort(key=lambda rect: tuple(rect["bbox"]))  # type: ignore[arg-type]
    assert rectangles, "expected at least one grey rectangle"
    return rectangles


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=int))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a counted-and-scaled template into only the grey rectangles that can contain it.

    Input structure:
    - A black background contains one small 4-connected blue/yellow template near the upper-left area.
    - Several disconnected grey solid rectangles appear elsewhere; each rectangle may contain 1..4 red markers strictly inside its grey border.
    - The template is the only blue/yellow component, while each grey target is a full rectangle whose border stays grey.

    Rule:
    - Detect the template and keep its bbox pattern.
    - For each grey rectangle, count its interior red markers and use that count as an integer scale factor for the template.
    - If the scaled template fits entirely inside the rectangle's strict interior, repaint the whole interior grey and stamp the scaled template at that interior's upper-left corner; otherwise leave the rectangle unchanged.

    Key insight:
    - The red count alone is not enough: the same scale factor only triggers a stamp when the rectangle's interior is geometrically large enough to hold the scaled template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _template_bbox, template = _extract_template(g)
    rectangles = _find_rectangles(g)

    out = g.copy()
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]  # type: ignore[misc]
        inner_h, inner_w = rect["inner_shape"]  # type: ignore[misc]
        k = int(rect["red_count"])
        scaled = _scale_sprite(template, k)
        stamp_h, stamp_w = scaled.shape
        if stamp_h > inner_h or stamp_w > inner_w:
            continue

        out[r0 + 1 : r1, c0 + 1 : c1] = GREY
        stamp_view = out[r0 + 1 : r0 + 1 + stamp_h, c0 + 1 : c0 + 1 + stamp_w]
        stamp_mask = scaled != BLACK
        stamp_view[stamp_mask] = scaled[stamp_mask]

    return out


def _random_template(rng: random.Random) -> np.ndarray:
    height, width = rng.choice([(2, 2), (2, 3)])
    while True:
        template = np.array(
            [[rng.choice(TEMPLATE_COLORS) for _c in range(width)] for _r in range(height)],
            dtype=int,
        )
        if BLUE in template and YELLOW in template:
            return template


def _fit_scales(template: np.ndarray) -> list[int]:
    th, tw = template.shape
    return [k for k in range(1, 5) if th * k <= 8 and tw * k <= 8]


def _make_fit_spec(template: np.ndarray, rng: random.Random, forced_k: int | None = None) -> dict[str, int | bool]:
    th, tw = template.shape
    choices = _fit_scales(template)
    assert choices, "template must admit at least one fit scale"
    k = forced_k if forced_k is not None else rng.choice(choices)
    min_inner_h = th * k
    min_inner_w = tw * k
    inner_h = rng.randint(min_inner_h, 8)
    inner_w = rng.randint(min_inner_w, 8)
    return {"h": inner_h + 2, "w": inner_w + 2, "k": k, "fits": True}


def _make_nonfit_spec(template: np.ndarray, rng: random.Random, forced_k: int | None = None) -> dict[str, int | bool] | None:
    th, tw = template.shape
    for _ in range(200):
        k = forced_k if forced_k is not None else rng.randint(1, 4)
        inner_h = rng.randint(1, 8)
        inner_w = rng.randint(1, 8)
        if inner_h * inner_w < k:
            continue
        if inner_h < th * k or inner_w < tw * k:
            return {"h": inner_h + 2, "w": inner_w + 2, "k": k, "fits": False}
    return None


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    height, width = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > height or c0 + w > width:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(height, r0 + h + pad)
    cc1 = min(width, c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _place_box(
    occupied: np.ndarray,
    h: int,
    w: int,
    row_range: tuple[int, int] | None,
    col_range: tuple[int, int] | None,
    rng: random.Random,
) -> tuple[int, int] | None:
    height, width = occupied.shape
    r_lo = 0 if row_range is None else row_range[0]
    r_hi = height - h if row_range is None else min(row_range[1], height - h)
    c_lo = 0 if col_range is None else col_range[0]
    c_hi = width - w if col_range is None else min(col_range[1], width - w)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    candidates = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
    rng.shuffle(candidates)
    for r0, c0 in candidates:
        if _can_place(occupied, r0, c0, h, w, pad=1):
            occupied[r0 : r0 + h, c0 : c0 + w] = True
            return r0, c0
    return None


def _sample_marker_cells(h: int, w: int, k: int, rng: random.Random) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    return rng.sample(cells, k)


def _render_input(
    grid_shape: tuple[int, int],
    template: np.ndarray,
    template_pos: tuple[int, int],
    rectangles: list[dict[str, object]],
    rng: random.Random,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    grid = np.full(grid_shape, BLACK, dtype=int)
    tr, tc = template_pos
    th, tw = template.shape
    grid[tr : tr + th, tc : tc + tw] = template

    rect_infos: list[dict[str, object]] = []
    for rect in rectangles:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        h = int(rect["h"])
        w = int(rect["w"])
        k = int(rect["k"])
        grid[r0 : r0 + h, c0 : c0 + w] = GREY
        markers = _sample_marker_cells(h, w, k, rng)
        for dr, dc in markers:
            grid[r0 + dr, c0 + dc] = RED
        rect_infos.append(
            {
                "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
                "k": k,
                "fits": bool(rect["fits"]),
                "markers": markers,
            }
        )
    return grid, rect_infos


def _expected_output(
    inp: np.ndarray,
    template: np.ndarray,
    rect_infos: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    for rect in rect_infos:
        if not bool(rect["fits"]):
            continue
        r0, c0, r1, c1 = rect["bbox"]  # type: ignore[misc]
        k = int(rect["k"])
        scaled = _scale_sprite(template, k)
        sh, sw = scaled.shape
        out[r0 + 1 : r1, c0 + 1 : c1] = GREY
        view = out[r0 + 1 : r0 + 1 + sh, c0 + 1 : c0 + 1 + sw]
        mask = scaled != BLACK
        view[mask] = scaled[mask]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(22, 30)
    grid_w = rng.randint(22, 30)
    template = _random_template(rng)

    shared_scale = rng.choice(_fit_scales(template))
    rect_count = rng.randint(4, 6)

    specs: list[dict[str, int | bool]] = [_make_fit_spec(template, rng, shared_scale)]
    same_count_nonfit = _make_nonfit_spec(template, rng, shared_scale)
    if same_count_nonfit is None:
        return None
    specs.append(same_count_nonfit)

    for _ in range(rect_count - 2):
        if rng.random() < 0.5:
            specs.append(_make_fit_spec(template, rng))
        else:
            spec = _make_nonfit_spec(template, rng)
            if spec is None:
                return None
            specs.append(spec)

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    template_pos = _place_box(
        occupied,
        template.shape[0],
        template.shape[1],
        row_range=(0, min(4, grid_h - template.shape[0])),
        col_range=(0, min(5, grid_w - template.shape[1])),
        rng=rng,
    )
    if template_pos is None:
        return None

    placed: list[dict[str, int | bool]] = []
    for spec in sorted(specs, key=lambda item: int(item["h"]) * int(item["w"]), reverse=True):
        pos = _place_box(occupied, int(spec["h"]), int(spec["w"]), None, None, rng)
        if pos is None:
            return None
        placed.append(
            {
                "r0": pos[0],
                "c0": pos[1],
                "h": int(spec["h"]),
                "w": int(spec["w"]),
                "k": int(spec["k"]),
                "fits": bool(spec["fits"]),
            }
        )

    inp, rect_infos = _render_input((grid_h, grid_w), template, template_pos, placed, rng)
    expected = _expected_output(inp, template, rect_infos)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None

    fit_count = sum(bool(rect["fits"]) for rect in rect_infos)
    nonfit_count = len(rect_infos) - fit_count
    if fit_count < 1 or nonfit_count < 1:
        return None
    if np.count_nonzero(inp == RED) != sum(int(rect["k"]) for rect in rect_infos):
        return None
    if inp.shape != produced.shape or max(inp.shape) > 30 or max(produced.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid task-6 input")
