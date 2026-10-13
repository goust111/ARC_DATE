from __future__ import annotations

import random

import numpy as np


_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    if values.size == 0:
        raise AssertionError("grid must be non-empty")
    counts = np.bincount(values)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise AssertionError("mask cannot be empty")
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _hole_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    holes: list[np.ndarray] = []
    for comp in _components(~mask):
        rows = comp[:, 0]
        cols = comp[:, 1]
        if np.any(rows == 0) or np.any(rows == h - 1) or np.any(cols == 0) or np.any(cols == w - 1):
            continue
        holes.append(comp)
    return holes


def _mask_connected(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    pts = int(mask.sum())
    return pts > 0 and len(_components(mask)) == 1


def _is_valid_fragment_mask(mask: np.ndarray) -> bool:
    if not _mask_connected(mask):
        return False
    holes = _hole_components(mask)
    return len(holes) == 1


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    pts: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        for dr, dc in _NEIGHBORS4:
            nr, nc = int(r) + dr, int(c) + dc
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                pts.append((int(r), int(c)))
                break
    return pts


def _placement_options(
    frag_shape: tuple[int, int],
    hole_coords: np.ndarray,
    interior_shape: tuple[int, int],
    anchor_local: tuple[int, int],
) -> list[tuple[int, int]]:
    fh, fw = frag_shape
    ih, iw = interior_shape
    valid: list[tuple[int, int]] = []
    ar, ac = anchor_local
    for hr, hc in hole_coords:
        top = ar - int(hr)
        left = ac - int(hc)
        if 0 <= top <= ih - fh and 0 <= left <= iw - fw:
            valid.append((top, left))
    return valid


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            if len(comp) != 2 * h + 2 * w - 4:
                continue
            rel = comp - np.array([r0, c0], dtype=int)
            patch = np.zeros((h, w), dtype=bool)
            patch[rel[:, 0], rel[:, 1]] = True
            if not (np.all(patch[0, :]) and np.all(patch[-1, :]) and np.all(patch[:, 0]) and np.all(patch[:, -1])):
                continue
            if np.any(patch[1:-1, 1:-1]):
                continue
            anchors = [
                (rr, cc)
                for rr in range(r0 + 1, r1)
                for cc in range(c0 + 1, c1)
                if int(g[rr, cc]) not in (bg, color)
            ]
            if len(anchors) != 1:
                continue
            ar, ac = anchors[0]
            frames.append(
                {
                    "frame_color": color,
                    "bbox": (r0, c0, r1, c1),
                    "anchor": (int(ar), int(ac)),
                    "anchor_color": int(g[ar, ac]),
                }
            )
    anchor_colors = [frame["anchor_color"] for frame in frames]
    assert len(frames) >= 2, "expected at least two template frames"
    assert len(set(anchor_colors)) == len(anchor_colors), "template anchor colors must be unique"
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _extract_fragments(grid: np.ndarray, bg: int, frames: list[dict]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    outside = np.ones(g.shape, dtype=bool)
    frame_colors = {int(frame["frame_color"]) for frame in frames}
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    fragments: list[dict] = []
    for color in map(int, np.unique(g)):
        if color == bg or color in frame_colors:
            continue
        for comp in _components((g == color) & outside):
            r0, c0, r1, c1 = _bbox(comp)
            rel = comp - np.array([r0, c0], dtype=int)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[rel[:, 0], rel[:, 1]] = True
            holes = _hole_components(mask)
            if len(holes) != 1:
                continue
            hole = holes[0]
            fragments.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "coords": comp,
                    "rel_coords": rel,
                    "shape": mask.shape,
                    "hole_rel": hole,
                }
            )
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment back into its matching template by aligning the fragment hole with the template anchor.

    Input structure:
    - The background is black, and the upper part contains 2 or 3 grey rectangular template outlines.
    - Each template has exactly one colored anchor pixel somewhere inside its interior, with anchor colors unique across templates.
    - Outside the templates, monochromatic 4-connected fragments of the same anchor colors are scattered on the background, and each fragment contains exactly one internal background hole.

    Rule:
    - Identify each fragment's destination template by matching the fragment color to the template's anchor color.
    - Translate the fragment without rotating or flipping it so that one cell of its internal hole lands exactly on the template anchor, and keep only the unique placement that fits fully inside the template interior.
    - Remove the fragment from its original location, keep the grey outlines unchanged, and redraw the fragment inside the matched template while leaving the hole as background so the anchor pixel disappears.

    Key insight:
    - Color determines which template a fragment belongs to, but the precise placement comes from aligning the fragment's hole pattern with the anchor position inside that template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)

    frames = _extract_frames(g, bg)
    fragments = _extract_fragments(g, bg, frames)
    frag_by_color: dict[int, dict] = {}
    for fragment in fragments:
        color = int(fragment["color"])
        assert color not in frag_by_color, "expected one fragment per anchor color"
        frag_by_color[color] = fragment

    assert len(frag_by_color) == len(frames), "template and fragment counts must match"

    out = g.copy()
    for fragment in fragments:
        out[fragment["coords"][:, 0], fragment["coords"][:, 1]] = bg

    for frame in frames:
        color = int(frame["anchor_color"])
        assert color in frag_by_color, "each template anchor needs a matching fragment"
        fragment = frag_by_color[color]

        r0, c0, r1, c1 = frame["bbox"]
        interior_shape = (r1 - r0 - 1, c1 - c0 - 1)
        anchor_local = (frame["anchor"][0] - r0 - 1, frame["anchor"][1] - c0 - 1)
        options = _placement_options(fragment["shape"], fragment["hole_rel"], interior_shape, anchor_local)
        assert len(options) == 1, "hole alignment must determine a unique placement"

        top, left = options[0]
        base_r = r0 + 1 + top
        base_c = c0 + 1 + left
        out[frame["anchor"][0], frame["anchor"][1]] = bg
        for dr, dc in fragment["rel_coords"]:
            out[base_r + int(dr), base_c + int(dc)] = color

    return out


_GEN_BG = 0
_GEN_FRAME = 5
_GEN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_GEN_ATTEMPTS = 400
_GEN_SINGLE_HOLES = (
    ((0, 0),),
    ((0, 0),),
)
_GEN_MULTI_HOLES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def _pattern_shape(pattern: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in pattern]
    cols = [c for _, c in pattern]
    return max(rows) + 1, max(cols) + 1


def _gen_fragment_mask(max_h: int, max_w: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        patterns = _GEN_SINGLE_HOLES if rng.random() < 0.5 else _GEN_MULTI_HOLES
        pattern = rng.choice(patterns)
        hh, ww = _pattern_shape(pattern)
        min_h = max(4, hh + 2)
        min_w = max(4, ww + 2)
        if min_h > max_h or min_w > max_w:
            continue

        frag_h = rng.randint(min_h, max_h)
        frag_w = rng.randint(min_w, max_w)
        mask = np.ones((frag_h, frag_w), dtype=bool)
        hole_r = rng.randint(1, frag_h - hh - 1)
        hole_c = rng.randint(1, frag_w - ww - 1)
        for dr, dc in pattern:
            mask[hole_r + dr, hole_c + dc] = False

        carve_target = rng.randint(1, max(1, (frag_h * frag_w) // 6))
        carved = 0
        for _ in range(40):
            candidates = _boundary_cells(mask)
            if len(candidates) <= 8:
                break
            rr, cc = rng.choice(candidates)
            trial = mask.copy()
            trial[rr, cc] = False
            trial = _crop_mask(trial)
            if trial.shape[0] > max_h or trial.shape[1] > max_w:
                continue
            if not _is_valid_fragment_mask(trial):
                continue
            mask = trial
            carved += 1
            if carved >= carve_target:
                break

        mask = _crop_mask(mask)
        if not _is_valid_fragment_mask(mask):
            continue
        if int(mask.sum()) < 8:
            continue
        hole = _hole_components(mask)[0]
        if carved == 0 and len(hole) == 1:
            continue
        return mask, hole
    raise RuntimeError("failed to build a valid fragment mask")


def _pick_unique_alignment(
    frag_shape: tuple[int, int],
    hole_coords: np.ndarray,
    interior_shape: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int, tuple[int, int]]:
    fh, fw = frag_shape
    ih, iw = interior_shape
    choices: list[tuple[int, int, tuple[int, int]]] = []
    for top in range(ih - fh + 1):
        for left in range(iw - fw + 1):
            order = [tuple(map(int, pt)) for pt in hole_coords]
            rng.shuffle(order)
            for hr, hc in order:
                anchor_local = (top + hr, left + hc)
                options = _placement_options(frag_shape, hole_coords, interior_shape, anchor_local)
                if len(options) == 1:
                    choices.append((top, left, anchor_local))
    if not choices:
        raise RuntimeError("no unique anchor placement available")
    return rng.choice(choices)


def _place_fragment_on_grid(
    grid: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    color: int,
) -> None:
    coords = np.argwhere(mask)
    for dr, dc in coords:
        grid[top + int(dr), left + int(dc)] = color


def _can_place_bbox(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _mark_bbox(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    occupied[top : top + h, left : left + w] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(20, 28)
    template_count = 2 if rng.random() < 0.5 else 3
    mid = grid_h // 2
    bottom_start = mid + 1

    frame_specs: list[dict] = []
    max_frame_h = max(4, bottom_start - 3)
    if max_frame_h < 6:
        return None

    for _ in range(100):
        frame_specs.clear()
        widths = []
        for _idx in range(template_count):
            inner_h = rng.randint(4, min(7, max_frame_h - 2))
            inner_w = rng.randint(5, 7 if template_count == 3 else 8)
            frame_specs.append({"inner_h": inner_h, "inner_w": inner_w})
            widths.append(inner_w + 2)

        gaps = [rng.randint(2, 4) for _ in range(template_count - 1)]
        min_total = sum(widths) + sum(gaps) + 2
        if min_total > 30:
            continue
        slack = rng.randint(0, min(4, 30 - min_total))
        slots = [1] * (template_count + 1)
        for _ in range(slack):
            slots[rng.randrange(len(slots))] += 1
        grid_w = sum(widths) + sum(gaps) + sum(slots)
        if grid_w > 30:
            continue

        col = slots[0]
        for idx, spec in enumerate(frame_specs):
            frame_h = spec["inner_h"] + 2
            frame_w = spec["inner_w"] + 2
            max_top = bottom_start - frame_h - 1
            if max_top < 1:
                break
            top = rng.randint(1, max_top)
            spec["top"] = top
            spec["left"] = col
            spec["frame_h"] = frame_h
            spec["frame_w"] = frame_w
            col += frame_w
            if idx < len(gaps):
                col += gaps[idx] + slots[idx + 1] - 1
        else:
            break
    else:
        return None

    grid = np.full((grid_h, grid_w), _GEN_BG, dtype=int)
    colors = rng.sample(_GEN_COLORS, template_count)
    fragments_for_scatter: list[dict] = []

    for spec, color in zip(frame_specs, colors):
        top = spec["top"]
        left = spec["left"]
        frame_h = spec["frame_h"]
        frame_w = spec["frame_w"]
        inner_h = spec["inner_h"]
        inner_w = spec["inner_w"]
        bottom = top + frame_h - 1
        right = left + frame_w - 1

        grid[top, left : right + 1] = _GEN_FRAME
        grid[bottom, left : right + 1] = _GEN_FRAME
        grid[top : bottom + 1, left] = _GEN_FRAME
        grid[top : bottom + 1, right] = _GEN_FRAME

        try:
            frag_mask, hole = _gen_fragment_mask(inner_h, inner_w, rng)
            place_top, place_left, anchor_local = _pick_unique_alignment(frag_mask.shape, hole, (inner_h, inner_w), rng)
        except RuntimeError:
            return None

        anchor_r = top + 1 + anchor_local[0]
        anchor_c = left + 1 + anchor_local[1]
        grid[anchor_r, anchor_c] = color
        fragments_for_scatter.append(
            {
                "color": color,
                "mask": frag_mask,
                "hole": hole,
                "template_pos": (place_top, place_left),
            }
        )

    occupied = np.zeros_like(grid, dtype=bool)
    for spec in frame_specs:
        _mark_bbox(occupied, spec["top"], spec["left"], spec["frame_h"], spec["frame_w"])

    for fragment in fragments_for_scatter:
        mask = fragment["mask"]
        frag_h, frag_w = mask.shape
        placed = False
        for _ in range(200):
            top = rng.randint(bottom_start, grid_h - frag_h)
            left = rng.randint(0, grid_w - frag_w)
            if not _can_place_bbox(occupied, top, left, frag_h, frag_w, pad=1):
                continue
            _place_fragment_on_grid(grid, top, left, mask, int(fragment["color"]))
            _mark_bbox(occupied, top, left, frag_h, frag_w)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fragment-to-template ARC sample")
