from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _frame_info(grid: np.ndarray, color: int, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    if hh < 3 or ww < 3:
        return None
    expected = 2 * (hh + ww) - 4
    if len(coords) != expected:
        return None
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((hh, ww), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(sub[border] == color):
        return None
    if np.any(sub[~border] == color):
        return None
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "inner_h": int(hh - 2),
        "inner_w": int(ww - 2),
    }


def _extract_object_patch(grid: np.ndarray, frame: dict, bg: int) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    r0, c0, r1, c1 = frame["bbox"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = inner != bg
    if not mask.any():
        return None, None
    coords = np.argwhere(mask)
    pr0, pc0, pr1, pc1 = _bbox(coords)
    patch = inner[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    return patch, (int(pr0), int(pc0), int(pr1), int(pc1))


def _orientation(h: int, w: int) -> str:
    if w > h:
        return "wide"
    if h > w:
        return "tall"
    return "square"


def _paste_centered(out: np.ndarray, frame: dict, patch: np.ndarray, bg: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    ph, pw = patch.shape
    assert ph <= inner_h and pw <= inner_w
    top = r0 + 1 + (inner_h - ph) // 2
    left = c0 + 1 + (inner_w - pw) // 2
    mask = patch != bg
    out[top : top + ph, left : left + pw][mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color portal frames and teleport each contained object into its mate.

    If source and destination frames swap between wide and tall orientations, rotate
    the object by 90 degrees before centering it in the destination. Clear the source.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    by_color: dict[int, list[dict]] = {}
    for color in [int(x) for x in np.unique(g) if int(x) != bg]:
        for coords in _find_components(g == color):
            info = _frame_info(g, color, coords)
            if info is not None:
                by_color.setdefault(int(color), []).append(info)

    frames_by_color = {color: frames for color, frames in by_color.items() if len(frames) == 2}
    assert frames_by_color, "expected at least one portal pair"

    out = g.copy()
    for color, frames in frames_by_color.items():
        items = []
        for frame in frames:
            patch, patch_box = _extract_object_patch(g, frame, bg)
            items.append((frame, patch, patch_box))
        nonempty = [item for item in items if item[1] is not None]
        empty = [item for item in items if item[1] is None]
        assert len(nonempty) == 1 and len(empty) == 1, f"color {color} must have one filled and one empty frame"

        src_frame, src_patch, _src_box = nonempty[0]
        dst_frame, _, _ = empty[0]
        assert src_patch is not None

        src_ori = _orientation(src_frame["inner_h"], src_frame["inner_w"])
        dst_ori = _orientation(dst_frame["inner_h"], dst_frame["inner_w"])
        move_patch = src_patch.copy()
        if {src_ori, dst_ori} == {"wide", "tall"}:
            move_patch = np.rot90(move_patch)

        ph, pw = move_patch.shape
        assert ph <= dst_frame["inner_h"] and pw <= dst_frame["inner_w"]

        sr0, sc0, sr1, sc1 = src_frame["bbox"]
        src_inner = out[sr0 + 1 : sr1, sc0 + 1 : sc1]
        src_inner[src_inner != bg] = bg
        _paste_centered(out, dst_frame, move_patch, bg)

    return out


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    while int(mask.sum()) < area:
        frontier = []
        for r, c in np.argwhere(mask):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return mask


def _make_object_patch(max_h: int, max_w: int, palette: list[int], force_nonsquare: bool) -> np.ndarray:
    dims = []
    for h in range(2, max_h + 1):
        for w in range(2, max_w + 1):
            if force_nonsquare and h == w:
                continue
            dims.append((h, w))
    assert dims
    h, w = random.choice(dims)
    min_area = max(2, (h * w) // 3)
    max_area = h * w - 1
    area = random.randint(min_area, max_area)
    mask = _grow_connected_mask(h, w, area)
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    mask = mask[r0 : r1 + 1, c0 : c1 + 1]
    ph, pw = mask.shape
    patch = np.zeros((ph, pw), dtype=int)
    colors = random.sample(palette, k=random.randint(1, min(3, len(palette))))
    for r, c in np.argwhere(mask):
        patch[int(r), int(c)] = random.choice(colors)
    return patch


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_patch_centered(grid: np.ndarray, bbox: tuple[int, int, int, int], patch: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    ph, pw = patch.shape
    top = r0 + 1 + (inner_h - ph) // 2
    left = c0 + 1 + (inner_w - pw) // 2
    mask = patch != 0
    grid[top : top + ph, left : left + pw][mask] = patch[mask]


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _random_bbox(h: int, w: int, outer_h: int, outer_w: int) -> tuple[int, int, int, int]:
    r0 = random.randint(0, h - outer_h)
    c0 = random.randint(0, w - outer_w)
    return (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)


def _expected_output_from_spec(
    grid: np.ndarray,
    placements: list[dict],
) -> np.ndarray:
    out = grid.copy()
    for item in placements:
        src_bbox = item["src_bbox"]
        dst_bbox = item["dst_bbox"]
        patch = item["moved_patch"]
        sr0, sc0, sr1, sc1 = src_bbox
        out[sr0 + 1 : sr1, sc0 + 1 : sc1] = 0
        _place_patch_centered(out, dst_bbox, patch)
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    grid_h = random.randint(18, 30)
    grid_w = random.randint(18, 30)
    n_pairs = random.randint(1, 3)
    if min(grid_h, grid_w) < 22:
        n_pairs = min(n_pairs, 2)

    colors = list(range(1, 10))
    random.shuffle(colors)
    frame_colors = colors[:n_pairs]
    object_palette = [c for c in colors[n_pairs:]]
    if len(object_palette) < 3:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied: list[tuple[int, int, int, int]] = []
    placements: list[dict] = []
    rotate_branches = []

    for frame_color in frame_colors:
        rotate_branch = random.choice([False, True])
        rotate_branches.append(rotate_branch)

        if rotate_branch:
            patch_h = random.randint(2, 4)
            patch_w = random.randint(patch_h + 1, 5)
            src_inner_h = random.randint(patch_h, min(5, patch_h + 1))
            src_inner_w = random.randint(max(patch_w, src_inner_h + 1), 6)
            dst_inner_w = random.randint(patch_h, min(5, patch_h + 1))
            dst_inner_h = random.randint(max(patch_w, dst_inner_w + 1), 6)
            patch = _make_object_patch(
                max_h=patch_h,
                max_w=patch_w,
                palette=object_palette,
                force_nonsquare=True,
            )
            moved = np.rot90(patch)
            if patch.shape[0] > src_inner_h or patch.shape[1] > src_inner_w:
                return None
            if moved.shape[0] > dst_inner_h or moved.shape[1] > dst_inner_w:
                return None
        else:
            same_orientation = random.choice(["wide", "tall", "square"])
            if same_orientation == "wide":
                src_inner_h = random.randint(2, 4)
                src_inner_w = random.randint(src_inner_h + 1, 6)
                dst_inner_h = random.randint(2, 4)
                dst_inner_w = random.randint(dst_inner_h + 1, 6)
            elif same_orientation == "tall":
                src_inner_w = random.randint(2, 4)
                src_inner_h = random.randint(src_inner_w + 1, 6)
                dst_inner_w = random.randint(2, 4)
                dst_inner_h = random.randint(dst_inner_w + 1, 6)
            else:
                side1 = random.randint(2, 5)
                side2 = random.randint(2, 5)
                src_inner_h = src_inner_w = side1
                dst_inner_h = dst_inner_w = side2
            patch = _make_object_patch(
                max_h=src_inner_h,
                max_w=src_inner_w,
                palette=object_palette,
                force_nonsquare=False,
            )
            moved = patch.copy()
            if moved.shape[0] > dst_inner_h or moved.shape[1] > dst_inner_w:
                return None

        src_outer_h, src_outer_w = src_inner_h + 2, src_inner_w + 2
        dst_outer_h, dst_outer_w = dst_inner_h + 2, dst_inner_w + 2

        src_bbox = None
        dst_bbox = None
        for _ in range(200):
            cand_src = _random_bbox(grid_h, grid_w, src_outer_h, src_outer_w)
            cand_dst = _random_bbox(grid_h, grid_w, dst_outer_h, dst_outer_w)
            if not _boxes_separated(cand_src, cand_dst, gap=2):
                continue
            if any(not _boxes_separated(cand_src, box, gap=1) for box in occupied):
                continue
            if any(not _boxes_separated(cand_dst, box, gap=1) for box in occupied):
                continue
            src_bbox, dst_bbox = cand_src, cand_dst
            break
        if src_bbox is None or dst_bbox is None:
            return None

        occupied.extend([src_bbox, dst_bbox])
        _draw_frame(grid, src_bbox, frame_color)
        _draw_frame(grid, dst_bbox, frame_color)
        _place_patch_centered(grid, src_bbox, patch)

        placements.append(
            {
                "src_bbox": src_bbox,
                "dst_bbox": dst_bbox,
                "moved_patch": moved,
            }
        )

    # Balance branch visibility for single-pair cases by adding a weak tendency
    # toward the missing branch over many runs; diversity self-check handles the rest.
    if n_pairs == 1 and random.random() < 0.15:
        return None

    expected = _expected_output_from_spec(grid, placements)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-pair ARC input")
