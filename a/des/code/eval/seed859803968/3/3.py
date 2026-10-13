import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
AZURE = 8

MARKER_COLORS = {BLUE, RED}
FRAME_COLORS = (GREEN, YELLOW)
GAP_COLORS = (AZURE, MAGENTA)

GRID_HEIGHT_RANGE = (22, 31)
GRID_WIDTH_RANGE = (22, 31)
GROUP_COUNT_CHOICES = (1, 2, 3, 4)
GROUP_COUNT_WEIGHTS = (2, 3, 2, 1)
LAYER_RANGE = (2, 4)
INNER_HEIGHT_RANGE = (2, 6)
INNER_WIDTH_RANGE = (2, 7)
GAP_RANGE = (1, 2)
PLACEMENT_PADDING = 2
GENERATE_RETRIES = 400
GROUP_RETRIES = 80
PLACEMENT_RETRIES = 200


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    want: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        want.add((r0, c))
        want.add((r1, c))
    for r in range(r0 + 1, r1):
        want.add((r, c0))
        want.add((r, c1))
    return set(points) == want


def _rotate_cw_coord(r: int, c: int, center_r2: int, center_c2: int) -> tuple[int, int]:
    nr_num = center_r2 - center_c2 + 2 * c
    nc_num = center_r2 + center_c2 - 2 * r
    assert nr_num % 2 == 0 and nc_num % 2 == 0
    return nr_num // 2, nc_num // 2


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    mask = np.zeros((h, w), dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _gap_mask(
    shape: tuple[int, int],
    outer_bbox: tuple[int, int, int, int],
    inner_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    h, w = shape
    rr = np.arange(h)[:, None]
    cc = np.arange(w)[None, :]
    ot, ol, ob, or_ = outer_bbox
    it, il, ib, ir = inner_bbox
    mask = (rr > ot) & (rr < ob) & (cc > ol) & (cc < or_)
    mask &= ~((rr >= it) & (rr <= ib) & (cc >= il) & (cc <= ir))
    return mask


def _extract_groups(grid: np.ndarray) -> list[dict]:
    grey_mask = grid == GREY
    frame_components = _components_mask(grey_mask)
    frames = []
    for idx, pts in enumerate(frame_components):
        if not _is_rect_frame(pts):
            continue
        frames.append(
            {
                "id": idx,
                "bbox": _bbox(pts),
            }
        )

    markers = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(np.isin(grid, list(MARKER_COLORS)))]
    groups = []
    used_frame_ids: set[int] = set()
    for mr, mc, color in markers:
        containing = [
            frame
            for frame in frames
            if frame["bbox"][0] < mr < frame["bbox"][2] and frame["bbox"][1] < mc < frame["bbox"][3]
        ]
        assert 2 <= len(containing) <= 4
        inner_to_outer = sorted(
            containing,
            key=lambda frame: (
                frame["bbox"][2] - frame["bbox"][0] + 1,
                frame["bbox"][3] - frame["bbox"][1] + 1,
            ),
        )
        for inner, outer in zip(inner_to_outer, inner_to_outer[1:]):
            it, il, ib, ir = inner["bbox"]
            ot, ol, ob, or_ = outer["bbox"]
            assert ot < it - 1 and ol < il - 1 and ob > ib + 1 and or_ > ir + 1
        groups.append(
            {
                "marker": (mr, mc),
                "marker_color": color,
                "frames": list(reversed(inner_to_outer)),
            }
        )
        for frame in containing:
            used_frame_ids.add(int(frame["id"]))

    assert groups, "expected at least one nested structure"
    assert len(used_frame_ids) == len(frames), "every grey frame should belong to exactly one marker-defined group"
    return groups


def _build_group_canvas(
    frames_outer_to_inner: list[dict],
    marker_rc: tuple[int, int],
    marker_color: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    outer_bbox = frames_outer_to_inner[0]["bbox"]
    r0, c0, r1, c1 = outer_bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    canvas = np.zeros((h, w), dtype=int)

    rel_bboxes = []
    for depth, frame in enumerate(frames_outer_to_inner):
        fr0, fc0, fr1, fc1 = frame["bbox"]
        rel_bbox = (fr0 - r0, fc0 - c0, fr1 - r0, fc1 - c0)
        rel_bboxes.append(rel_bbox)
        canvas[_frame_mask((h, w), rel_bbox)] = FRAME_COLORS[depth % 2]

    for gap_idx, (outer_bbox_rel, inner_bbox_rel) in enumerate(zip(rel_bboxes, rel_bboxes[1:])):
        canvas[_gap_mask((h, w), outer_bbox_rel, inner_bbox_rel)] = GAP_COLORS[gap_idx % 2]

    mr, mc = marker_rc
    canvas[mr - r0, mc - c0] = marker_color
    return canvas, outer_bbox


def _paint_group(out: np.ndarray, canvas: np.ndarray, outer_bbox: tuple[int, int, int, int], rotate: bool) -> None:
    r0, c0, r1, c1 = outer_bbox
    if rotate:
        assert ((r1 - r0) - (c1 - c0)) % 2 == 0, "rotated bbox height and width must have matching parity"
    center_r2 = r0 + r1
    center_c2 = c0 + c1
    for lr, lc in np.argwhere(canvas != BG):
        color = int(canvas[lr, lc])
        gr = r0 + int(lr)
        gc = c0 + int(lc)
        if rotate:
            nr, nc = _rotate_cw_coord(gr, gc, center_r2, center_c2)
        else:
            nr, nc = gr, gc
        assert 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]
        out[nr, nc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested grey rectangular frames and rotate each structure when its inner marker is red.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    for group in _extract_groups(g):
        canvas, outer_bbox = _build_group_canvas(group["frames"], group["marker"], group["marker_color"])
        _paint_group(out, canvas, outer_bbox, rotate=group["marker_color"] == RED)
    return out


def _reserve_geometry(height: int, width: int) -> tuple[int, int, int, int]:
    assert (height - width) % 2 == 0
    coords = [
        _rotate_cw_coord(r, c, height - 1, width - 1)
        for r in range(height)
        for c in range(width)
    ]
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    min_r = min(0, min(rows))
    max_r = max(height - 1, max(rows))
    min_c = min(0, min(cols))
    max_c = max(width - 1, max(cols))
    return min_r, min_c, max_r, max_c


def _marker_positions(
    inner_h: int,
    inner_w: int,
    inner_bbox: tuple[int, int, int, int],
    outer_h: int,
    outer_w: int,
) -> list[tuple[int, int]]:
    it, il, _ib, _ir = inner_bbox
    cavity_top = it + 1
    cavity_left = il + 1
    positions = [(cavity_top + dr, cavity_left + dc) for dr in range(inner_h) for dc in range(inner_w)]
    center = None
    if outer_h % 2 == 1 and outer_w % 2 == 1:
        center = (outer_h // 2, outer_w // 2)
    if center in positions and len(positions) > 1:
        positions.remove(center)
    return positions


def _make_group_patch(rng: random.Random, marker_color: int) -> dict | None:
    for _ in range(GROUP_RETRIES):
        layers = rng.randint(LAYER_RANGE[0], LAYER_RANGE[1])
        inner_h = rng.randint(INNER_HEIGHT_RANGE[0], INNER_HEIGHT_RANGE[1])
        inner_w = rng.randint(INNER_WIDTH_RANGE[0], INNER_WIDTH_RANGE[1])
        if (inner_h - inner_w) % 2 != 0:
            continue
        gaps_inner_out = [rng.randint(GAP_RANGE[0], GAP_RANGE[1]) for _ in range(layers - 1)]

        frames_inner_out = [(0, 0, inner_h + 1, inner_w + 1)]
        for gap in gaps_inner_out:
            pad = gap + 1
            t, l, b, r = frames_inner_out[-1]
            frames_inner_out.append((t - pad, l - pad, b + pad, r + pad))

        raw_outer = frames_inner_out[-1]
        shift_r = -raw_outer[0]
        shift_c = -raw_outer[1]
        frames_outer_to_inner = []
        for bbox in reversed(frames_inner_out):
            t, l, b, r = bbox
            frames_outer_to_inner.append((t + shift_r, l + shift_c, b + shift_r, r + shift_c))

        outer_h = frames_outer_to_inner[0][2] + 1
        outer_w = frames_outer_to_inner[0][3] + 1
        reserve = _reserve_geometry(outer_h, outer_w)
        reserve_h = reserve[2] - reserve[0] + 1
        reserve_w = reserve[3] - reserve[1] + 1
        if reserve_h > GRID_HEIGHT_RANGE[1] - 4 or reserve_w > GRID_WIDTH_RANGE[1] - 4:
            continue

        inner_bbox = frames_outer_to_inner[-1]
        marker_choices = _marker_positions(inner_h, inner_w, inner_bbox, outer_h, outer_w)
        if not marker_choices:
            continue
        marker_r, marker_c = rng.choice(marker_choices)

        patch = np.zeros((outer_h, outer_w), dtype=int)
        for t, l, b, r in frames_outer_to_inner:
            patch[_frame_mask((outer_h, outer_w), (t, l, b, r))] = GREY
        patch[marker_r, marker_c] = marker_color

        return {
            "patch": patch,
            "reserve": reserve,
            "frames": frames_outer_to_inner,
            "marker": (marker_r, marker_c),
            "marker_color": marker_color,
        }
    return None


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _choose_marker_colors(rng: random.Random, n_groups: int) -> list[int]:
    if n_groups == 1:
        return [RED if rng.random() < 0.5 else BLUE]
    reds = n_groups // 2
    if n_groups % 2 == 1 and rng.random() < 0.5:
        reds += 1
    colors = [RED] * reds + [BLUE] * (n_groups - reds)
    rng.shuffle(colors)
    return colors


def _place_groups(rng: random.Random, canvas_h: int, canvas_w: int, groups: list[dict]) -> np.ndarray | None:
    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    placed_reserves: list[tuple[int, int, int, int]] = []

    for group in groups:
        patch = group["patch"]
        h, w = patch.shape
        min_r, min_c, max_r, max_c = group["reserve"]
        reserve_h = max_r - min_r + 1
        reserve_w = max_c - min_c + 1

        if reserve_h > canvas_h or reserve_w > canvas_w:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            rr0 = rng.randint(0, canvas_h - reserve_h)
            rc0 = rng.randint(0, canvas_w - reserve_w)
            reserve_rect = (rr0, rc0, rr0 + reserve_h - 1, rc0 + reserve_w - 1)
            if any(_rectangles_overlap(reserve_rect, other, PLACEMENT_PADDING) for other in placed_reserves):
                continue

            pr0 = rr0 - min_r
            pc0 = rc0 - min_c
            assert 0 <= pr0 <= canvas_h - h and 0 <= pc0 <= canvas_w - w
            sub = grid[pr0 : pr0 + h, pc0 : pc0 + w]
            if np.any((sub != BG) & (patch != BG)):
                continue
            sub[patch != BG] = patch[patch != BG]
            placed_reserves.append(reserve_rect)
            placed = True
            break
        if not placed:
            return None

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    canvas_h = rng.randint(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1])
    canvas_w = rng.randint(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1])
    n_groups = rng.choices(GROUP_COUNT_CHOICES, weights=GROUP_COUNT_WEIGHTS, k=1)[0]
    marker_colors = _choose_marker_colors(rng, n_groups)

    groups = []
    for marker_color in marker_colors:
        group = _make_group_patch(rng, marker_color)
        if group is None:
            return None
        groups.append(group)

    grid = _place_groups(rng, canvas_h, canvas_w, groups)
    if grid is None:
        return None

    solved = main(grid)
    assert np.any(solved == GREEN) and np.any(solved == YELLOW)
    assert np.any(solved == AZURE)
    if any(color == RED for color in marker_colors):
        assert np.any(solved == RED)
    if any(color == BLUE for color in marker_colors):
        assert np.any(solved == BLUE)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {GENERATE_RETRIES} attempts")
