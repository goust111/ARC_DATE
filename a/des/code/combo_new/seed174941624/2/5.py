from __future__ import annotations

import random
from typing import Iterable

import numpy as np


FRAME_COLOR = 5
ALL_COLORS = tuple(range(10))
CORNER_OFFSETS = ((0, 0), (0, -1), (-1, -1), (-1, 0))  # TL, TR, BR, BL
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _corner_positions(h: int, w: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)]


def _frame_border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _cw_turns_to_ccw_k(cw_turns: int) -> int:
    return (-int(cw_turns)) % 4


def _restore_frame(out: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    out[r0, c0 : c1 + 1] = FRAME_COLOR
    out[r1, c0 : c1 + 1] = FRAME_COLOR
    out[r0 : r1 + 1, c0] = FRAME_COLOR
    out[r0 : r1 + 1, c1] = FRAME_COLOR


def _place_rotated_source(
    out: np.ndarray,
    source_patch: np.ndarray,
    bbox: tuple[int, int, int, int],
    bg: int,
    cw_turns: int,
) -> None:
    rotated = np.rot90(source_patch, _cw_turns_to_ccw_k(cw_turns))
    r0, c0, r1, c1 = bbox
    h, w = r1 - r0 + 1, c1 - c0 + 1
    assert rotated.shape == (h, w), "rotated source must fit its paired frame"
    for rr in range(1, h - 1):
        for cc in range(1, w - 1):
            val = int(rotated[rr, cc])
            if val != int(bg):
                out[r0 + rr, c0 + cc] = val


def _parse_frame_component(g: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    border = _frame_border_mask(h, w)
    if np.any(sub[~border] != int(bg)):
        return None

    non_frame = np.argwhere(border & (sub != FRAME_COLOR))
    if len(non_frame) != 1:
        return None
    mr, mc = map(int, non_frame[0])
    corners = _corner_positions(h, w)
    if (mr, mc) not in corners:
        return None

    marker_color = int(sub[mr, mc])
    if marker_color in {int(bg), FRAME_COLOR}:
        return None

    expected_grey = int(border.sum()) - 1
    if len(coords) != expected_grey:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "shape": (h, w),
        "marker_color": marker_color,
        "corner_index": corners.index((mr, mc)),
    }


def _find_frames(g: np.ndarray, bg: int) -> tuple[dict[int, dict], np.ndarray]:
    reserved = np.zeros_like(g, dtype=bool)
    frames: dict[int, dict] = {}
    for comp in _components(g == FRAME_COLOR):
        frame = _parse_frame_component(g, comp, bg)
        if frame is None:
            continue
        color = int(frame["marker_color"])
        assert color not in frames, "frame marker colors must be unique"
        frames[color] = frame
        r0, c0, r1, c1 = frame["bbox"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True

    assert frames, "expected at least one target frame"
    return frames, reserved


def _find_sources(
    g: np.ndarray, bg: int, reserved: np.ndarray, marker_colors: set[int]
) -> dict[int, dict]:
    mask = (g != int(bg)) & (~reserved)
    sources: dict[int, dict] = {}
    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        corners = _corner_positions(*patch.shape)
        matches = [
            int(patch[rr, cc]) for rr, cc in corners if int(patch[rr, cc]) in marker_colors
        ]
        assert len(matches) == 1, "every source must expose exactly one marker color in a corner"
        marker_color = matches[0]
        assert marker_color not in sources, "source marker colors must be unique"
        sources[marker_color] = {
            "bbox": (r0, c0, r1, c1),
            "coords": comp,
            "patch": patch,
        }

    assert set(sources) == marker_colors, "source and target marker colors must match exactly"
    return sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each source pattern and place it into the matching marked frame.

    Input structure:
    - The background is the dominant color.
    - Several compact multicolor source patterns sit on the canvas; each source has a tight bounding box and exactly one corner pixel whose color is used as a pairing key.
    - Several hollow rectangular target frames are drawn with a one-pixel grey border; each frame has exactly one corner border pixel recolored to a non-grey marker color, and its interior is background.

    Rule:
    - Match every source to the unique target frame whose colored corner uses the same marker color.
    - Read the marked corner of the target frame to choose the clockwise rotation: top-left = 0, top-right = 90, bottom-right = 180, bottom-left = 270.
    - Erase each source from its original location, restore the target's marked corner back to grey, and stamp the rotated source into the paired frame while keeping the frame border intact.

    Key insight:
    - The same colored corner plays two roles at once: its color identifies which source belongs to the frame, and its position specifies the rotation to apply before placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    frames, reserved = _find_frames(g, bg)
    marker_colors = set(frames)
    sources = _find_sources(g, bg, reserved, marker_colors)

    out = g.copy()
    for source in sources.values():
        rows = source["coords"][:, 0]
        cols = source["coords"][:, 1]
        out[rows, cols] = int(bg)

    for marker_color, frame in frames.items():
        _restore_frame(out, frame["bbox"], bg)
        _place_rotated_source(
            out,
            sources[marker_color]["patch"],
            frame["bbox"],
            bg,
            frame["corner_index"],
        )

    return out


def _random_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    path = [(r, c)]
    row_first = bool(random.getrandbits(1))
    axes = (0, 1) if row_first else (1, 0)
    for axis in axes:
        if axis == 0:
            step = 1 if tr > r else -1
            while r != tr:
                r += step
                path.append((r, c))
        else:
            step = 1 if tc > c else -1
            while c != tc:
                c += step
                path.append((r, c))
    return path


def _grow_connected_mask(h: int, w: int, target_area: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    while int(mask.sum()) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in zip(*np.where(mask)):
            for dr, dc in NEIGHBORS4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return mask


def _colorize_mask(mask: np.ndarray, bg: int, colors: list[int]) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    used_count = min(len(colors), 2 if len(coords) < 5 else 3)
    used = random.sample(colors, used_count)
    for color, (r, c) in zip(used, coords):
        patch[r, c] = int(color)
    for r, c in coords[used_count:]:
        patch[r, c] = int(random.choice(used))
    return patch


def _visible_variant(patch: np.ndarray, bg: int) -> np.ndarray:
    vis = patch.copy()
    vis[0, :] = int(bg)
    vis[-1, :] = int(bg)
    vis[:, 0] = int(bg)
    vis[:, -1] = int(bg)
    return vis


def _variant_count(patch: np.ndarray, bg: int) -> int:
    seen: set[bytes] = set()
    for cw in range(4):
        rot = np.rot90(patch, _cw_turns_to_ccw_k(cw))
        vis = _visible_variant(rot, bg)
        seen.add(vis.tobytes())
    return len(seen)


def _source_corner_pos(size: int, corner_index: int) -> tuple[int, int]:
    if corner_index == 0:
        return 0, 0
    if corner_index == 1:
        return 0, size - 1
    if corner_index == 2:
        return size - 1, size - 1
    return size - 1, 0


def _required_edge_targets(size: int, corner_index: int) -> list[tuple[int, int]]:
    row_target_row = size - 1 if corner_index in (0, 1) else 0
    col_target_col = size - 1 if corner_index in (0, 3) else 0
    row_target = (row_target_row, random.randint(1, size - 2))
    col_target = (random.randint(1, size - 2), col_target_col)
    return [row_target, col_target]


def _forbidden_corners(size: int, keep: tuple[int, int]) -> set[tuple[int, int]]:
    return set(_corner_positions(size, size)) - {keep}


def _nearest(points: Iterable[tuple[int, int]], target: tuple[int, int]) -> tuple[int, int]:
    return min(points, key=lambda p: (abs(p[0] - target[0]) + abs(p[1] - target[1]), p[0], p[1]))


def _generate_source_patch(
    size: int,
    bg: int,
    marker_color: int,
    source_corner: int,
    decor_colors: list[int],
) -> np.ndarray | None:
    inner = size - 2
    for _ in range(200):
        area_min = 3
        area_max = inner * inner if inner == 2 else inner * inner - 1
        core_area = random.randint(area_min, area_max)
        core_mask = _grow_connected_mask(inner, inner, core_area)
        if core_mask is None:
            continue

        core = _colorize_mask(core_mask, bg, decor_colors)
        patch = np.full((size, size), int(bg), dtype=int)
        patch[1:-1, 1:-1] = core

        marker_pos = _source_corner_pos(size, source_corner)
        bad_corners = _forbidden_corners(size, marker_pos)
        patch[marker_pos] = int(marker_color)

        occupied = {tuple(map(int, rc)) for rc in np.argwhere(patch != int(bg))}

        def add_path(start: tuple[int, int], end: tuple[int, int]) -> bool:
            nonlocal occupied
            color = int(random.choice(decor_colors))
            for cell in _random_path(start, end):
                if cell in bad_corners:
                    return False
                if cell == marker_pos:
                    continue
                if cell not in occupied:
                    patch[cell] = color
                    occupied.add(cell)
            return True

        core_cells = {tuple(map(int, rc)) for rc in np.argwhere(core != int(bg))}
        core_cells = {(r + 1, c + 1) for r, c in core_cells}
        attach = _nearest(core_cells, marker_pos)
        if not add_path(marker_pos, attach):
            continue

        ok = True
        for target in _required_edge_targets(size, source_corner):
            attach = _nearest(occupied, target)
            if not add_path(target, attach):
                ok = False
                break
        if not ok:
            continue

        extra_steps = random.randint(0, size)
        for _ in range(extra_steps):
            frontier: list[tuple[int, int]] = []
            for r, c in occupied:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < size
                        and 0 <= nc < size
                        and (nr, nc) not in occupied
                        and (nr, nc) not in bad_corners
                    ):
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            patch[nr, nc] = int(random.choice(decor_colors))
            occupied.add((nr, nc))

        coords = np.argwhere(patch != int(bg))
        r0, c0, r1, c1 = _bbox(coords)
        if (r0, c0, r1, c1) != (0, 0, size - 1, size - 1):
            continue
        if any(patch[r, c] != int(bg) for r, c in bad_corners):
            continue
        if _variant_count(patch, bg) < 3:
            continue
        if int((_visible_variant(patch, bg) != int(bg)).sum()) < 3:
            continue
        return patch
    return None


def _make_frame_patch(size: int, bg: int, marker_color: int, corner_index: int) -> np.ndarray:
    patch = np.full((size, size), int(bg), dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR
    rr, cc = _corner_positions(size, size)[corner_index]
    patch[rr, cc] = int(marker_color)
    return patch


def _place_rectangles(
    shape: tuple[int, int], items: list[dict], padding: int = 1
) -> dict[int, tuple[int, int]] | None:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    positions: dict[int, tuple[int, int]] = {}
    order = sorted(range(len(items)), key=lambda i: items[i]["patch"].shape[0], reverse=True)
    for idx in order:
        patch = items[idx]["patch"]
        ph, pw = patch.shape
        candidates = [
            (r0, c0)
            for r0 in range(h - ph + 1)
            for c0 in range(w - pw + 1)
        ]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(h, r0 + ph + padding)
            cc1 = min(w, c0 + pw + padding)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            positions[idx] = (r0, c0)
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True
            placed = True
            break
        if not placed:
            return None
    return positions


def _render_input(
    shape: tuple[int, int], bg: int, items: list[dict], positions: dict[int, tuple[int, int]]
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for idx, item in enumerate(items):
        r0, c0 = positions[idx]
        patch = item["patch"]
        ph, pw = patch.shape
        mask = patch != int(bg)
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        sub[mask] = patch[mask]
    return grid


def _expected_output(
    inp: np.ndarray,
    pairs: list[dict],
    source_positions: dict[int, tuple[int, int]],
    frame_positions: dict[int, tuple[int, int]],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    for pair in pairs:
        sr, sc = source_positions[pair["pair_id"]]
        source_patch = pair["source_patch"]
        source_mask = source_patch != int(bg)
        source_sub = out[sr : sr + source_patch.shape[0], sc : sc + source_patch.shape[1]]
        source_sub[source_mask] = int(bg)

    for pair in pairs:
        fr, fc = frame_positions[pair["pair_id"]]
        size = pair["size"]
        bbox = (fr, fc, fr + size - 1, fc + size - 1)
        _restore_frame(out, bbox, bg)
        _place_rotated_source(out, pair["source_patch"], bbox, bg, pair["rotation"])
    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        bg = random.choice([c for c in ALL_COLORS if c != FRAME_COLOR])
        n_pairs = random.randint(2, 4)
        marker_pool = [c for c in ALL_COLORS if c not in {bg, FRAME_COLOR}]
        marker_colors = random.sample(marker_pool, n_pairs)
        decor_pool = [c for c in marker_pool if c not in marker_colors]
        if len(decor_pool) < 2:
            continue

        rotations = random.sample([0, 1, 2, 3], n_pairs)
        pairs: list[dict] = []
        items: list[dict] = []
        for pair_id, (marker_color, rotation) in enumerate(zip(marker_colors, rotations)):
            size = random.choice([4, 5])
            source_corner = random.randint(0, 3)
            source_patch = _generate_source_patch(
                size=size,
                bg=bg,
                marker_color=marker_color,
                source_corner=source_corner,
                decor_colors=decor_pool,
            )
            if source_patch is None:
                break
            frame_patch = _make_frame_patch(size, bg, marker_color, rotation)
            pairs.append(
                {
                    "pair_id": pair_id,
                    "marker_color": marker_color,
                    "rotation": rotation,
                    "size": size,
                    "source_patch": source_patch,
                    "frame_patch": frame_patch,
                }
            )
            items.append({"kind": "source", "pair_id": pair_id, "patch": source_patch})
            items.append({"kind": "frame", "pair_id": pair_id, "patch": frame_patch})
        else:
            min_dim = max(18, 7 + 4 * n_pairs)
            rows = random.randint(min_dim, 30)
            cols = random.randint(min_dim, 30)
            if random.random() < 0.65:
                cols = min(30, max(min_dim, rows + random.randint(-3, 3)))

            positions = _place_rectangles((rows, cols), items, padding=1)
            if positions is None:
                continue

            inp = _render_input((rows, cols), bg, items, positions)
            source_positions = {}
            frame_positions = {}
            for idx, item in enumerate(items):
                if item["kind"] == "source":
                    source_positions[item["pair_id"]] = positions[idx]
                else:
                    frame_positions[item["pair_id"]] = positions[idx]

            expected = _expected_output(inp, pairs, source_positions, frame_positions, bg)
            try:
                actual = main(inp)
            except Exception:
                continue
            if not np.array_equal(actual, expected):
                continue
            return inp

    raise RuntimeError("failed to generate a valid ARC instance")
