from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists():
    sys.path.append(str(_UTILS_DIR))

try:
    from arc_utils import Color, bbox_h, bbox_w, crop_bbox, find_components, get_bg
except Exception:
    class Color:
        BLACK = 0
        GREY = 5
        NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])

    def bbox_h(bbox: tuple[int, int, int, int]) -> int:
        return int(bbox[2] - bbox[0] + 1)

    def bbox_w(bbox: tuple[int, int, int, int]) -> int:
        return int(bbox[3] - bbox[1] + 1)

    def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = bbox
        return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()

    def _components_from_mask(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
        h, w = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 8:
            steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        comps: list[np.ndarray] = []
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for dr, dc in steps:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(np.array(coords, dtype=int))
        return comps

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | None = None,
        min_area: int = 1,
        return_format: str = "dict",
    ) -> list[dict]:
        g = np.asarray(grid, dtype=int)
        if bg is None:
            bg = get_bg(g)
        bg = int(bg)
        palette = sorted(set(map(int, g.flatten())) - {bg}) if colors is None else [int(c) for c in colors if int(c) != bg]
        out = []
        if monochromatic:
            for color in palette:
                for coords in _components_from_mask(g == color, connectivity):
                    if coords.shape[0] < int(min_area):
                        continue
                    r0 = int(coords[:, 0].min())
                    c0 = int(coords[:, 1].min())
                    r1 = int(coords[:, 0].max())
                    c1 = int(coords[:, 1].max())
                    out.append({"color": int(color), "coords": coords, "bbox": (r0, c0, r1, c1), "area": int(coords.shape[0])})
            return out
        for coords in _components_from_mask(g != bg, connectivity):
            if coords.shape[0] < int(min_area):
                continue
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            color = int(g[coords[0, 0], coords[0, 1]])
            out.append({"color": color, "coords": coords, "bbox": (r0, c0, r1, c1), "area": int(coords.shape[0])})
        return out


FRAME_COLOR = Color.GREY
BG = Color.BLACK
SPRITE_COLORS = [c for c in Color.NOT_BLACK if c != FRAME_COLOR]
MAX_RETRIES = 500

SPRITE_LIBRARY = [
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[1, 1, 1, 1]], dtype=int),
    np.array([[1], [1], [1], [1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
]


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask != 0)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1] != 0, dtype=bool)


def _random_sprite_mask() -> np.ndarray:
    mask = np.array(random.choice(SPRITE_LIBRARY), dtype=int)
    for _ in range(random.randrange(4)):
        mask = np.rot90(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    return _normalize_mask(mask)


def _frame_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _component_patch(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.zeros((bbox_h(comp["bbox"]), bbox_w(comp["bbox"])), dtype=bool)
    coords = np.asarray(comp["coords"], dtype=int)
    patch[coords[:, 0] - int(r0), coords[:, 1] - int(c0)] = True
    return patch


def _is_frame_component(comp: dict) -> bool:
    if int(comp["color"]) != int(FRAME_COLOR):
        return False
    r0, c0, r1, c1 = comp["bbox"]
    h = bbox_h(comp["bbox"])
    w = bbox_w(comp["bbox"])
    if h < 4 or w < 4:
        return False
    coords = {tuple(map(int, rc)) for rc in np.asarray(comp["coords"], dtype=int)}
    if len(coords) != int(2 * (h + w) - 4):
        return False
    border = {
        (r, c)
        for r in range(int(r0), int(r1) + 1)
        for c in range(int(c0), int(c1) + 1)
        if r in (int(r0), int(r1)) or c in (int(c0), int(c1))
    }
    return coords == border


def _find_frame(grid: np.ndarray, bg: int) -> dict:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, colors=[FRAME_COLOR], return_format="dict")
    candidates = [comp for comp in comps if _is_frame_component(comp)]
    assert len(candidates) == 1, "expected exactly one rectangular grey frame"
    return candidates[0]


def _parse_scene(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame = _find_frame(g, bg)
    fr0, fc0, fr1, fc1 = map(int, frame["bbox"])
    inner_r0, inner_c0 = fr0 + 1, fc0 + 1
    inner_r1, inner_c1 = fr1 - 1, fc1 - 1
    assert inner_r0 <= inner_r1 and inner_c0 <= inner_c1
    control_row = fr0 - 1
    assert control_row >= 0

    control_specs: dict[int, dict] = {}
    row = g[control_row]
    for color in sorted(set(map(int, row.tolist())) - {bg, int(FRAME_COLOR)}):
        cols = sorted(int(c) for c in np.where(row == int(color))[0])
        assert len(cols) == 2, "each control color must appear exactly twice"
        c1, c2 = cols
        period = int(c2 - c1)
        ox = int(c1 - inner_c0)
        assert period > 0
        assert ox >= 0
        assert (inner_c1 - inner_c0 + 1) % period == 0
        control_specs[int(color)] = {"color": int(color), "c1": int(c1), "period": period, "ox": ox}

    assert 2 <= len(control_specs) <= 3
    all_comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    sprites = []
    for color, spec in control_specs.items():
        proto = [comp for comp in all_comps if int(comp["color"]) == int(color) and 3 <= int(comp["area"]) <= 5]
        assert len(proto) == 1, "each control color must match exactly one sprite prototype"
        comp = proto[0]
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        assert pr1 < fr0 - 1 or pr0 > fr1 or pc1 < fc0 or pc0 > fc1, "sprite prototypes must be outside the frame"
        patch = _component_patch(comp)
        sprites.append(
            {
                "color": int(color),
                "patch": patch,
                "period": int(spec["period"]),
                "ox": int(spec["ox"]),
                "c1": int(spec["c1"]),
            }
        )

    colors_seen = {sprite["color"] for sprite in sprites}
    assert colors_seen == set(control_specs)
    return int(bg), tuple(map(int, frame["bbox"])), sprites


def _tile_positions(inner_h: int, inner_w: int, patch: np.ndarray, period: int, ox: int) -> list[tuple[int, int]]:
    sh, sw = map(int, patch.shape)
    if ox < 0 or ox + sw > inner_w:
        return []
    rows = list(range(0, inner_h - sh + 1, int(period)))
    cols = list(range(int(ox), inner_w - sw + 1, int(period)))
    return [(int(r), int(c)) for r in rows for c in cols]


def _render_interior(inner_h: int, inner_w: int, sprites: list[dict], *, reverse_priority: bool = False, single_copy: bool = False, force_zero_offset: bool = False) -> np.ndarray:
    out = np.zeros((inner_h, inner_w), dtype=int)
    ordered = sorted(sprites, key=lambda spec: int(spec["c1"]), reverse=bool(reverse_priority))
    for spec in ordered:
        ox = 0 if force_zero_offset else int(spec["ox"])
        positions = _tile_positions(inner_h, inner_w, np.asarray(spec["patch"], dtype=bool), int(spec["period"]), ox)
        if single_copy and positions:
            positions = positions[:1]
        coords = np.argwhere(np.asarray(spec["patch"], dtype=bool))
        for r0, c0 in positions:
            for dr, dc in coords:
                rr = int(r0 + dr)
                cc = int(c0 + dc)
                if out[rr, cc] == 0:
                    out[rr, cc] = int(spec["color"])
    return out


def _render_output(shape: tuple[int, int], bg: int, frame_bbox: tuple[int, int, int, int], sprites: list[dict]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    out[fr0 : fr1 + 1, fc0] = int(FRAME_COLOR)
    out[fr0 : fr1 + 1, fc1] = int(FRAME_COLOR)
    out[fr0, fc0 : fc1 + 1] = int(FRAME_COLOR)
    out[fr1, fc0 : fc1 + 1] = int(FRAME_COLOR)
    inner_h = int(fr1 - fr0 - 1)
    inner_w = int(fc1 - fc0 - 1)
    interior = _render_interior(inner_h, inner_w, sprites)
    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = interior
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile each colored sprite inside the grey frame using the spacing encoded by its top control pixels.

    Input structure:
    - Black background surrounds one grey one-pixel-thick rectangular frame whose interior is empty.
    - Outside the frame sit 2-3 monochrome 4-connected sprite prototypes, each in a unique non-grey color and having 3-5 pixels.
    - On the row immediately above the frame, each sprite color appears as exactly two single-pixel control marks.

    Rule:
    - For each sprite color C, read the two control columns c1 < c2 on the row above the frame.
    - Let Px = c2 - c1, let Py = Px, and let Ox be c1 measured relative to the frame interior's left edge.
    - Stamp the sprite with its bounding-box top-left at every interior coordinate (0 + k*Py, Ox + j*Px) whose full sprite fits inside the frame; do not crop partial copies.
    - If multiple colors want the same pixel, keep the color whose leftmost control pixel c1 is smaller.
    - Preserve the grey frame, erase all control marks and prototype sprites, and keep only the tiled frame interior on black background.

    Key insight:
    - The control pixels do not mark stamp locations directly; their horizontal spacing defines a square tiling period, and their left-to-right order also resolves tile overlaps.
    """
    g = np.asarray(grid, dtype=int)
    bg, frame_bbox, sprites = _parse_scene(g)
    return _render_output(g.shape, bg, frame_bbox, sprites)


def _coverage_set(patch: np.ndarray, positions: list[tuple[int, int]]) -> set[tuple[int, int]]:
    coords = np.argwhere(np.asarray(patch, dtype=bool))
    return {(int(r0 + dr), int(c0 + dc)) for r0, c0 in positions for dr, dc in coords}


def _choose_tiling_specs(inner_h: int, inner_w: int, masks: list[np.ndarray], colors: list[int]) -> list[dict] | None:
    period_choices = [p for p in range(2, inner_w + 1) if inner_w % p == 0 and p <= inner_w // 2]
    if not period_choices:
        return None
    candidates: list[list[dict]] = []
    for mask in masks:
        sh, sw = map(int, mask.shape)
        opts = []
        for period in period_choices:
            for ox in range(period):
                positions = _tile_positions(inner_h, inner_w, mask, period, ox)
                if len(positions) < 2:
                    continue
                rows = sorted({int(r) for r, _ in positions})
                cols = sorted({int(c) for _, c in positions})
                opts.append(
                    {
                        "period": int(period),
                        "ox": int(ox),
                        "positions": positions,
                        "row_count": len(rows),
                        "col_count": len(cols),
                    }
                )
        if not opts:
            return None
        random.shuffle(opts)
        candidates.append(opts)

    order = sorted(range(len(candidates)), key=lambda i: len(candidates[i]))
    chosen: list[dict | None] = [None] * len(candidates)
    used_controls: set[int] = set()

    def valid_final() -> bool:
        specs = []
        for idx, cand in enumerate(chosen):
            assert cand is not None
            specs.append(
                {
                    "color": int(colors[idx]),
                    "patch": np.asarray(masks[idx], dtype=bool),
                    "period": int(cand["period"]),
                    "ox": int(cand["ox"]),
                    "c1": int(cand["ox"]),
                }
            )
        if not any(int(spec["ox"]) > 0 for spec in specs):
            return False
        if not any(len(sorted({r for r, _ in _tile_positions(inner_h, inner_w, spec["patch"], spec["period"], spec["ox"])})) >= 2 for spec in specs):
            return False
        if not any(len(sorted({c for _, c in _tile_positions(inner_h, inner_w, spec["patch"], spec["period"], spec["ox"])})) >= 2 for spec in specs):
            return False

        coverages = []
        for spec in specs:
            positions = _tile_positions(inner_h, inner_w, spec["patch"], spec["period"], spec["ox"])
            coverages.append(_coverage_set(spec["patch"], positions))
        if not any(coverages[i] & coverages[j] for i in range(len(coverages)) for j in range(i + 1, len(coverages))):
            return False

        correct = _render_interior(inner_h, inner_w, specs)
        if len({int(x) for x in correct.flatten()} - {0}) != len(specs):
            return False
        fill = float(np.count_nonzero(correct)) / float(inner_h * inner_w)
        if not (0.12 <= fill <= 0.58):
            return False
        if np.array_equal(correct, _render_interior(inner_h, inner_w, specs, reverse_priority=True)):
            return False
        if np.array_equal(correct, _render_interior(inner_h, inner_w, specs, single_copy=True)):
            return False
        if np.array_equal(correct, _render_interior(inner_h, inner_w, specs, force_zero_offset=True)):
            return False
        return True

    def backtrack(k: int) -> bool:
        if k == len(order):
            return valid_final()
        idx = order[k]
        for cand in candidates[idx]:
            control_cols = {int(cand["ox"]), int(cand["ox"] + cand["period"])}
            if used_controls & control_cols:
                continue
            chosen[idx] = cand
            used_controls.update(control_cols)
            if backtrack(k + 1):
                return True
            used_controls.difference_update(control_cols)
            chosen[idx] = None
        return False

    if not backtrack(0):
        return None

    result = []
    for idx, cand in enumerate(chosen):
        assert cand is not None
        result.append(
            {
                "color": int(colors[idx]),
                "patch": np.asarray(masks[idx], dtype=bool),
                "period": int(cand["period"]),
                "ox": int(cand["ox"]),
                "c1": int(cand["ox"]),
            }
        )
    return result


def _zone_boxes(frame_bbox: tuple[int, int, int, int], grid_shape: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    h, w = map(int, grid_shape)
    return {
        "left": (fr0 + 1, 1, fr1 - 1, fc0 - 2),
        "right": (fr0 + 1, fc1 + 2, fr1 - 1, w - 2),
        "top": (1, fc0 + 1, fr0 - 3, fc1 - 1),
        "bottom": (fr1 + 2, fc0 + 1, h - 2, fc1 - 1),
    }


def _fits_zone(zone: tuple[int, int, int, int], patch: np.ndarray) -> bool:
    r0, c0, r1, c1 = map(int, zone)
    sh, sw = map(int, patch.shape)
    return r0 <= r1 and c0 <= c1 and (r1 - r0 + 1) >= sh and (c1 - c0 + 1) >= sw


def _paint_patch(grid: np.ndarray, patch: np.ndarray, color: int, top: int, left: int) -> None:
    coords = np.argwhere(np.asarray(patch, dtype=bool))
    for dr, dc in coords:
        grid[int(top + dr), int(left + dc)] = int(color)


def _place_prototypes(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], sprites: list[dict]) -> bool:
    zones = _zone_boxes(frame_bbox, grid.shape)
    zone_names = list(zones)
    random.shuffle(zone_names)
    used = set()
    for spec in sprites:
        options = [name for name in zone_names if name not in used and _fits_zone(zones[name], spec["patch"])]
        if not options:
            return False
        zone_name = random.choice(options)
        used.add(zone_name)
        zr0, zc0, zr1, zc1 = zones[zone_name]
        sh, sw = map(int, spec["patch"].shape)
        top = random.randint(int(zr0), int(zr1 - sh + 1))
        left = random.randint(int(zc0), int(zc1 - sw + 1))
        _paint_patch(grid, spec["patch"], int(spec["color"]), top, left)
    return True


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    inner_h = random.randint(10, 16)
    inner_w = random.choice([10, 12, 14, 15, 16])
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    top_margin = random.randint(4, 6)
    bottom_margin = random.randint(4, 7)
    left_margin = random.randint(4, 7)
    right_margin = random.randint(4, 7)
    height = top_margin + outer_h + bottom_margin
    width = left_margin + outer_w + right_margin
    if height > 30 or width > 30:
        return None

    n_sprites = random.choice([2, 3])
    colors = random.sample(SPRITE_COLORS, n_sprites)
    masks = [_random_sprite_mask() for _ in range(n_sprites)]
    specs = _choose_tiling_specs(inner_h, inner_w, masks, colors)
    if specs is None:
        return None

    frame_bbox = (top_margin, left_margin, top_margin + outer_h - 1, left_margin + outer_w - 1)
    grid = np.full((height, width), BG, dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0 : fr1 + 1, fc0] = FRAME_COLOR
    grid[fr0 : fr1 + 1, fc1] = FRAME_COLOR
    grid[fr0, fc0 : fc1 + 1] = FRAME_COLOR
    grid[fr1, fc0 : fc1 + 1] = FRAME_COLOR

    control_row = fr0 - 1
    inner_left = fc0 + 1
    for spec in specs:
        c1 = inner_left + int(spec["ox"])
        c2 = c1 + int(spec["period"])
        grid[control_row, c1] = int(spec["color"])
        grid[control_row, c2] = int(spec["color"])
        spec["c1"] = int(c1)

    if not _place_prototypes(grid, frame_bbox, specs):
        return None

    out = _render_output(grid.shape, BG, frame_bbox, specs)
    solved = main(grid)
    if not np.array_equal(solved, out):
        return None

    bg, parsed_frame, parsed_sprites = _parse_scene(grid)
    if bg != BG or parsed_frame != frame_bbox:
        return None
    if not np.array_equal(_render_output(grid.shape, bg, parsed_frame, parsed_sprites), out):
        return None
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid control-bar sprite tiling input")
