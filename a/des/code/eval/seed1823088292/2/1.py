from __future__ import annotations

import random
from typing import Iterable

import numpy as np


# --------------- solver helpers ---------------

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _find_rectangular_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            expected = set()
            for cc in range(c0, c1 + 1):
                expected.add((r0, cc))
                expected.add((r1, cc))
            for rr in range(r0 + 1, r1):
                expected.add((rr, c0))
                expected.add((rr, c1))
            coord_set = set(coords)
            if coord_set != expected:
                continue
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "coords": coord_set,
                }
            )
    return frames


def _strictly_contains(outer: dict, inner: dict) -> bool:
    or0, oc0, or1, oc1 = outer["bbox"]
    ir0, ic0, ir1, ic1 = inner["bbox"]
    return or0 + 1 <= ir0 and oc0 + 1 <= ic0 and ir1 <= or1 - 1 and ic1 <= oc1 - 1


def _find_seed_component(interior: np.ndarray, bg: int, border: int) -> tuple[str, int, set[tuple[int, int]]]:
    nontrivial = [int(v) for v in np.unique(interior) if int(v) not in (bg, border)]
    candidates: list[tuple[str, int, set[tuple[int, int]]]] = []
    for color in nontrivial:
        pts = set(map(tuple, np.argwhere(interior == color)))
        if len(pts) != 2:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        if r0 == r1 and c1 == c0 + 1:
            candidates.append(("H", color, pts))
        elif c0 == c1 and r1 == r0 + 1:
            candidates.append(("V", color, pts))
    assert len(candidates) == 1, "expected exactly one 2-pixel seed marker"
    return candidates[0]


def _parse_demo_subframe(grid: np.ndarray, frame: dict, bg: int) -> tuple[str, np.ndarray]:
    r0, c0, r1, c1 = frame["bbox"]
    border = int(frame["color"])
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    orientation, _seed_color, seed_pts = _find_seed_component(interior, bg, border)

    paint = np.zeros_like(interior, dtype=bool)
    for rr in range(interior.shape[0]):
        for cc in range(interior.shape[1]):
            if (rr, cc) in seed_pts:
                continue
            value = int(interior[rr, cc])
            if value not in (bg, border):
                paint[rr, cc] = True

    template_pts = set(map(tuple, np.argwhere(paint)))
    assert template_pts, "demo subframe must contain a template"
    tr0, tc0, tr1, tc1 = _bbox(template_pts)
    template = paint[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
    assert template.shape[0] == template.shape[1], "template must be square"
    return orientation, template


def _parse_task_region(grid: np.ndarray, frame: dict, bg: int) -> tuple[str, int, set[tuple[int, int]]]:
    r0, c0, r1, c1 = frame["bbox"]
    border = int(frame["color"])
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    orientation, seed_color, local_pts = _find_seed_component(interior, bg, border)
    seed_pts = {(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in local_pts}
    return orientation, seed_color, seed_pts


def _apply_template(
    out: np.ndarray,
    original: np.ndarray,
    frame: dict,
    bg: int,
    template: np.ndarray,
    seed_color: int,
    seed_pts: set[tuple[int, int]],
) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    th, tw = template.shape
    for rr in range(r0 + 1, r1):
        for cc in range(c0 + 1, c1):
            if (rr, cc) in seed_pts:
                continue
            if int(original[rr, cc]) != bg:
                continue
            if template[(rr - (r0 + 1)) % th, (cc - (c0 + 1)) % tw]:
                out[rr, cc] = seed_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill closed rectangles using the template selected by the seed orientation.

    The reference panel is a corner frame containing two smaller demo frames.
    Each demo frame shows a 2-pixel seed and a nearby square template. Horizontal
    and vertical demo seeds identify which template to use for task regions with
    the same seed orientation. The selected template is tiled from the task
    region's interior top-left, repainting only background cells and preserving
    the original seed pixels.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _find_rectangular_frames(g, bg)
    assert frames, "expected at least one rectangular frame"

    containers = {i: [] for i in range(len(frames))}
    for i, outer in enumerate(frames):
        for j, inner in enumerate(frames):
            if i != j and _strictly_contains(outer, inner):
                containers[i].append(j)

    top_level = [i for i in range(len(frames)) if not any(i in containers[j] for j in range(len(frames)))]
    panel_candidates = [i for i in top_level if containers[i]]
    assert len(panel_candidates) == 1, "expected one reference panel frame"
    panel_idx = panel_candidates[0]

    direct_subframes = []
    for child_idx in containers[panel_idx]:
        if not any(
            child_idx in containers[mid_idx]
            for mid_idx in containers[panel_idx]
            if mid_idx != child_idx
        ):
            direct_subframes.append(child_idx)
    assert len(direct_subframes) == 2, "reference panel must contain two demo frames"

    templates: dict[str, np.ndarray] = {}
    for sub_idx in direct_subframes:
        orientation, template = _parse_demo_subframe(g, frames[sub_idx], bg)
        templates[orientation] = template
    assert set(templates) == {"H", "V"}, "reference panel must define both orientations"

    out = g.copy()
    for idx in top_level:
        if idx == panel_idx:
            continue
        orientation, seed_color, seed_pts = _parse_task_region(g, frames[idx], bg)
        _apply_template(out, g, frames[idx], bg, templates[orientation], seed_color, seed_pts)
    return out


# --------------- generator helpers ---------------

MAX_GEN_ATTEMPTS = 400
GRID_H_RANGE = (22, 31)
GRID_W_RANGE = (24, 35)
TEMPLATE_SIZE_CHOICES = (3, 4)
TASK_COUNT_RANGE = (2, 6)
TASK_INNER_CHOICES = (4, 6, 8, 10)
PLACEMENT_RETRIES = 200


def _random_mask(rng: random.Random, size: int) -> np.ndarray:
    min_area = size + 1
    max_area = size * size - 2
    for _ in range(300):
        density = rng.uniform(0.35, 0.65)
        mask = np.array(
            [[rng.random() < density for _ in range(size)] for _ in range(size)],
            dtype=bool,
        )
        if mask.sum() < min_area or mask.sum() > max_area:
            continue
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        even_present = any(mask[r, c] and (r + c) % 2 == 0 for r in range(size) for c in range(size))
        odd_present = any(mask[r, c] and (r + c) % 2 == 1 for r in range(size) for c in range(size))
        if not (even_present and odd_present):
            continue
        return mask
    raise RuntimeError("failed to generate a valid template mask")


def _choose_templates(rng: random.Random, size: int) -> tuple[np.ndarray, np.ndarray]:
    first = _random_mask(rng, size)
    for _ in range(300):
        second = _random_mask(rng, size)
        if np.array_equal(first, second):
            continue
        if int(np.sum(first != second)) < max(3, size):
            continue
        return first, second
    raise RuntimeError("failed to generate distinct templates")


def _render_template(mask: np.ndarray, color_a: int, color_b: int, bg: int) -> np.ndarray:
    size = mask.shape[0]
    arr = np.full((size, size), bg, dtype=int)
    for r in range(size):
        for c in range(size):
            if mask[r, c]:
                arr[r, c] = color_a if (r + c) % 2 == 0 else color_b
    return arr


def _make_demo_subframe(
    mask: np.ndarray,
    orientation: str,
    seed_color: int,
    border: int,
    color_a: int,
    color_b: int,
    bg: int,
) -> np.ndarray:
    size = mask.shape[0]
    sub = np.full((size + 2, size + 5), bg, dtype=int)
    sub[0, :] = border
    sub[-1, :] = border
    sub[:, 0] = border
    sub[:, -1] = border

    inner = sub[1:-1, 1:-1]
    seed_row = (size - 2) // 2
    if orientation == "H":
        inner[seed_row, 0] = seed_color
        inner[seed_row, 1] = seed_color
    else:
        inner[seed_row, 0] = seed_color
        inner[seed_row + 1, 0] = seed_color

    inner[:, 3:] = _render_template(mask, color_a, color_b, bg)
    return sub


def _make_panel(
    rng: random.Random,
    masks: dict[str, np.ndarray],
    demo_colors: dict[str, int],
    border: int,
    color_a: int,
    color_b: int,
    bg: int,
) -> np.ndarray:
    entries = [
        _make_demo_subframe(masks["H"], "H", demo_colors["H"], border, color_a, color_b, bg),
        _make_demo_subframe(masks["V"], "V", demo_colors["V"], border, color_a, color_b, bg),
    ]
    if rng.random() < 0.5:
        entries.reverse()

    sub_h, sub_w = entries[0].shape
    panel = np.full((sub_h + 4, sub_w * 2 + 5), bg, dtype=int)
    panel[0, :] = border
    panel[-1, :] = border
    panel[:, 0] = border
    panel[:, -1] = border

    panel[2 : 2 + sub_h, 2 : 2 + sub_w] = entries[0]
    panel[2 : 2 + sub_h, 3 + sub_w : 3 + sub_w + sub_w] = entries[1]
    return panel


def _tile_gain(mask: np.ndarray, inner_h: int, inner_w: int, seed_local: set[tuple[int, int]]) -> int:
    size = mask.shape[0]
    gain = 0
    for r in range(inner_h):
        for c in range(inner_w):
            if (r, c) in seed_local:
                continue
            if mask[r % size, c % size]:
                gain += 1
    return gain


def _make_task_patch(
    rng: random.Random,
    orientation: str,
    mask: np.ndarray,
    seed_color: int,
    border: int,
    bg: int,
) -> tuple[np.ndarray, dict]:
    for _ in range(200):
        inner_h = rng.choice(TASK_INNER_CHOICES)
        inner_w = rng.choice(TASK_INNER_CHOICES)
        patch = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
        patch[0, :] = border
        patch[-1, :] = border
        patch[:, 0] = border
        patch[:, -1] = border

        if orientation == "H":
            sr = rng.randrange(inner_h)
            sc = rng.randrange(inner_w - 1)
            seed_local = {(sr, sc), (sr, sc + 1)}
        else:
            sr = rng.randrange(inner_h - 1)
            sc = rng.randrange(inner_w)
            seed_local = {(sr, sc), (sr + 1, sc)}

        if _tile_gain(mask, inner_h, inner_w, seed_local) == 0:
            continue
        for rr, cc in seed_local:
            patch[1 + rr, 1 + cc] = seed_color
        return patch, {"orientation": orientation, "seed_local": seed_local, "shape": patch.shape}
    raise RuntimeError("failed to build a useful task patch")


def _placement_origin(corner: str, grid_h: int, grid_w: int, patch_h: int, patch_w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, grid_w - patch_w
    if corner == "BL":
        return grid_h - patch_h, 0
    return grid_h - patch_h, grid_w - patch_w


def _can_place(occupied: np.ndarray, top: int, left: int, patch_h: int, patch_w: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + patch_h > h or left + patch_w > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + patch_h + pad)
    c1 = min(w, left + patch_w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _mark_occupied(occupied: np.ndarray, top: int, left: int, patch_h: int, patch_w: int, pad: int = 1) -> None:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + patch_h + pad)
    c1 = min(w, left + patch_w + pad)
    occupied[r0:r1, c0:c1] = True


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    grid[top : top + ph, left : left + pw] = patch


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    size = rng.choice(TEMPLATE_SIZE_CHOICES)
    grid_h = rng.randrange(*GRID_H_RANGE)
    grid_w = rng.randrange(*GRID_W_RANGE)
    if grid_h < 18 or grid_w < 18:
        return None

    palette = list(range(10))
    rng.shuffle(palette)
    bg, border, templ_a, templ_b, demo_h_col, demo_v_col = palette[:6]
    task_palette = palette[6:]
    if len(task_palette) < 2:
        return None

    h_mask, v_mask = _choose_templates(rng, size)
    masks = {"H": h_mask, "V": v_mask}
    panel = _make_panel(
        rng,
        masks,
        {"H": demo_h_col, "V": demo_v_col},
        border,
        templ_a,
        templ_b,
        bg,
    )

    panel_corner = rng.choice(["TL", "TR", "BL", "BR"])
    panel_top, panel_left = _placement_origin(panel_corner, grid_h, grid_w, panel.shape[0], panel.shape[1])

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    _place_patch(grid, panel, panel_top, panel_left)
    _mark_occupied(occupied, panel_top, panel_left, panel.shape[0], panel.shape[1], pad=1)

    task_count = rng.randrange(*TASK_COUNT_RANGE)
    if task_count < 2:
        return None
    orientations = ["H", "V"] * (task_count // 2)
    if task_count % 2 == 1:
        orientations.append(rng.choice(["H", "V"]))
    rng.shuffle(orientations)

    task_specs = []
    for orientation in orientations:
        seed_color = rng.choice(task_palette)
        patch, meta = _make_task_patch(rng, orientation, masks[orientation], seed_color, border, bg)
        task_specs.append((patch, meta))
    task_specs.sort(key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True)

    for patch, _meta in task_specs:
        ph, pw = patch.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randrange(0, grid_h - ph + 1)
            left = rng.randrange(0, grid_w - pw + 1)
            if not _can_place(occupied, top, left, ph, pw, pad=1):
                continue
            _place_patch(grid, patch, top, left)
            _mark_occupied(occupied, top, left, ph, pw, pad=1)
            placed = True
            break
        if not placed:
            return None

    if _most_common_color(grid) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    changed_h = False
    changed_v = False
    frames = _find_rectangular_frames(grid, bg)
    containers = {i: [] for i in range(len(frames))}
    for i, outer in enumerate(frames):
        for j, inner in enumerate(frames):
            if i != j and _strictly_contains(outer, inner):
                containers[i].append(j)
    top_level = [i for i in range(len(frames)) if not any(i in containers[j] for j in range(len(frames)))]
    panel_idx = next(i for i in top_level if containers[i])
    for idx in top_level:
        if idx == panel_idx:
            continue
        orientation, _seed_color, seed_pts = _parse_task_region(grid, frames[idx], bg)
        r0, c0, r1, c1 = frames[idx]["bbox"]
        before = grid[r0 + 1 : r1, c0 + 1 : c1]
        after = out[r0 + 1 : r1, c0 + 1 : c1]
        changed = any(after[rr, cc] != before[rr, cc] for rr in range(before.shape[0]) for cc in range(before.shape[1]))
        if orientation == "H":
            changed_h |= changed
        else:
            changed_v |= changed
        assert seed_pts
    if not (changed_h and changed_v):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
