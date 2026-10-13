from __future__ import annotations

import random

import numpy as np


BG = 0
SPRITE_COLOR = 4
FRAME_COLORS = (1, 2, 3)
CORNER_SIGN = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}
COMPONENT_SHAPES = (
    np.array([[1]], dtype=int),
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
)
SPRITE_SHAPES = (
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    nbrs = neighbors4 if connectivity == 4 else neighbors8
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in nbrs(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(comp, dtype=int))
    return out


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    view[mask] = patch[mask]


def _corner_matches_bbox(corner: str, point: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    if corner == "tl":
        return point == (r1, c1)
    if corner == "tr":
        return point == (r1, c0)
    if corner == "bl":
        return point == (r0, c1)
    if corner == "br":
        return point == (r0, c0)
    raise AssertionError(f"unknown corner: {corner}")


def _orient_sprite(sprite: np.ndarray, source_corner: str, target_corner: str, bg: int) -> np.ndarray:
    out = sprite.copy()
    src_sign = CORNER_SIGN[source_corner]
    dst_sign = CORNER_SIGN[target_corner]
    if src_sign[0] != dst_sign[0]:
        out = out[::-1, :]
    if src_sign[1] != dst_sign[1]:
        out = out[:, ::-1]
    return np.where(out != bg, SPRITE_COLOR, bg)


def _frame_corners(r0: int, c0: int, side: int) -> dict[str, tuple[int, int]]:
    span = side - 1
    return {
        "tl": (r0, c0),
        "tr": (r0, c0 + span),
        "bl": (r0 + span, c0),
        "br": (r0 + span, c0 + span),
    }


def _three_corner_frames(grid: np.ndarray, color: int) -> list[dict]:
    singles = {
        tuple(map(int, comp[0]))
        for comp in components_mask(grid == color, connectivity=8)
        if len(comp) == 1
    }
    rows = sorted({r for r, _ in singles})
    cols = sorted({c for _, c in singles})
    frames = []
    seen = set()
    for r0 in rows:
        for span in (2, 3):
            r1 = r0 + span
            if r1 not in rows:
                continue
            for c0 in cols:
                c1 = c0 + span
                if c1 not in cols:
                    continue
                corners = {
                    "tl": (r0, c0),
                    "tr": (r0, c1),
                    "bl": (r1, c0),
                    "br": (r1, c1),
                }
                present = [name for name, pt in corners.items() if pt in singles]
                if len(present) != 3:
                    continue
                markers = tuple(sorted(corners[name] for name in present))
                if markers in seen:
                    continue
                seen.add(markers)
                missing = next(name for name in corners if name not in present)
                frames.append(
                    {
                        "color": int(color),
                        "side": int(span + 1),
                        "missing_kind": missing,
                        "missing": corners[missing],
                        "markers": markers,
                    }
                )
    return frames


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    sprite_comps = components_mask(g == SPRITE_COLOR, connectivity=4)
    assert len(sprite_comps) == 1, "expected exactly one yellow sprite component"
    sprite_box = bbox_coords(sprite_comps[0])
    sprite = np.where(crop_bbox(g, sprite_box) == SPRITE_COLOR, SPRITE_COLOR, bg)

    frame_candidates = []
    for color in FRAME_COLORS:
        frame_candidates.extend(_three_corner_frames(g, color))
    assert 2 <= len(frame_candidates) <= 3, "expected 2-3 control frames"
    assert len({frame["color"] for frame in frame_candidates}) == len(frame_candidates), "frame colors must be unique"

    source_frames = [
        frame
        for frame in frame_candidates
        if _corner_matches_bbox(frame["missing_kind"], frame["missing"], sprite_box)
    ]
    assert len(source_frames) == 1, "expected exactly one source frame"
    source = source_frames[0]

    counts: dict[int, int] = {}
    for frame in frame_candidates:
        temp = (g == frame["color"]).copy()
        for r, c in frame["markers"]:
            temp[r, c] = False
        counts[frame["color"]] = len(components_mask(temp, connectivity=4))
        assert 1 <= counts[frame["color"]] <= 3, "component count must be 1..3"

    return {
        "bg": int(bg),
        "frames": sorted(frame_candidates, key=lambda d: d["color"]),
        "source": source,
        "sprite": sprite,
        "sprite_box": sprite_box,
        "counts": counts,
    }


def _stamp_stack(
    out: np.ndarray,
    sprite: np.ndarray,
    missing_corner: tuple[int, int],
    corner_kind: str,
    count: int,
    bg: int,
) -> None:
    sign = CORNER_SIGN[corner_kind]
    h, w = sprite.shape
    for idx in range(count):
        if sign[0] < 0:
            r0 = missing_corner[0] - (idx + 1) * h + 1
        else:
            r0 = missing_corner[0] + idx * h
        if sign[1] < 0:
            c0 = missing_corner[1] - (idx + 1) * w + 1
        else:
            c0 = missing_corner[1] + idx * w
        assert 0 <= r0 and 0 <= c0
        assert r0 + h <= out.shape[0] and c0 + w <= out.shape[1]
        paste(out, sprite, r0, c0, transparent=bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count left-side same-color components and stamp the yellow sprite repeatedly at each frame's missing corner.

    Input structure:
    - Black background contains 2-3 colored three-corner square frames, each using a distinct control color.
    - Exactly one frame has a yellow 4-connected sprite attached at its missing corner; that frame defines the source sprite and source missing-corner direction.
    - Elsewhere, small 4-connected components in the frame colors act as counting tokens; their geometry is irrelevant except for color and component count.

    Rule:
    - Detect every three-corner frame, its color, and its missing corner.
    - Identify the unique source frame whose missing corner matches the yellow sprite bbox anchor.
    - For each frame color K, count the number of 4-connected components of color K after excluding that frame's three marker pixels.
    - Keep the source sprite unchanged, but for every non-source frame, flip the source sprite horizontally and/or vertically so its anchor corner matches that frame's missing corner, then stamp that oriented sprite outward in that corner direction exactly n(K) times with no gaps.

    Key insight:
    - The frame color has a dual role: it selects which scattered components to count, and that count becomes the repetition number for the yellow stamp placed at the frame whose missing corner also sets the flip and stacking direction.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    out = np.array(grid, copy=True)
    bg = scene["bg"]
    source_corner = scene["source"]["missing_kind"]
    for frame in scene["frames"]:
        if frame["color"] == scene["source"]["color"]:
            continue
        oriented = _orient_sprite(scene["sprite"], source_corner, frame["missing_kind"], bg)
        _stamp_stack(out, oriented, frame["missing"], frame["missing_kind"], scene["counts"][frame["color"]], bg)
    return out


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = out[::-1, :]
    if random.random() < 0.5:
        out = out[:, ::-1]
    return out


def _sprite_patch() -> np.ndarray:
    while True:
        mask = _random_transform(random.choice(SPRITE_SHAPES))
        if np.array_equal(mask, mask[::-1, :]) or np.array_equal(mask, mask[:, ::-1]):
            continue
        return np.where(mask > 0, SPRITE_COLOR, BG).astype(int)


def _component_patch(color: int) -> np.ndarray:
    mask = _random_transform(random.choice(COMPONENT_SHAPES))
    if random.random() < 0.25:
        mask = mask.T
    return np.where(mask > 0, color, BG).astype(int)


def _group_cells(
    frame_top: int,
    frame_left: int,
    side: int,
    missing_kind: str,
    sprite: np.ndarray,
    copies: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    corners = _frame_corners(frame_top, frame_left, side)
    marker_cells = {pt for name, pt in corners.items() if name != missing_kind}
    stack_cells = set()
    sign = CORNER_SIGN[missing_kind]
    h, w = sprite.shape
    anchor = corners[missing_kind]
    mask = sprite != BG
    for idx in range(copies):
        if sign[0] < 0:
            r0 = anchor[0] - (idx + 1) * h + 1
        else:
            r0 = anchor[0] + idx * h
        if sign[1] < 0:
            c0 = anchor[1] - (idx + 1) * w + 1
        else:
            c0 = anchor[1] + idx * w
        for rr, cc in np.argwhere(mask):
            stack_cells.add((int(r0 + rr), int(c0 + cc)))
    return marker_cells, stack_cells


def _dilate_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _random_frame_position(
    h: int,
    w: int,
    right_start: int,
    side: int,
    missing_kind: str,
    sprite: np.ndarray,
    copies: int,
    forbidden: set[tuple[int, int]],
) -> tuple[int, int] | None:
    sh, sw = sprite.shape
    span = side - 1
    candidates = []
    for r0 in range(h - span):
        for c0 in range(right_start, w - span):
            corners = _frame_corners(r0, c0, side)
            anchor_r, anchor_c = corners[missing_kind]
            min_r = anchor_r - copies * sh + 1 if CORNER_SIGN[missing_kind][0] < 0 else anchor_r
            max_r = anchor_r if CORNER_SIGN[missing_kind][0] < 0 else anchor_r + copies * sh - 1
            min_c = anchor_c - copies * sw + 1 if CORNER_SIGN[missing_kind][1] < 0 else anchor_c
            max_c = anchor_c if CORNER_SIGN[missing_kind][1] < 0 else anchor_c + copies * sw - 1
            if min_r < 0 or min_c < right_start or max_r >= h or max_c >= w:
                continue
            marker_cells, stack_cells = _group_cells(r0, c0, side, missing_kind, sprite, copies)
            occupied = marker_cells | stack_cells
            if occupied & forbidden:
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _place_count_components(
    grid: np.ndarray,
    left_width: int,
    counts: dict[int, int],
) -> bool:
    h = grid.shape[0]
    forbidden: set[tuple[int, int]] = set()
    items = []
    for color, n in counts.items():
        for _ in range(n):
            items.append(color)
    random.shuffle(items)
    for color in items:
        patch = _component_patch(color)
        ph, pw = patch.shape
        mask = patch != BG
        candidates = []
        for r0 in range(1, h - ph - 1):
            for c0 in range(1, left_width - pw - 1):
                cells = {
                    (r0 + int(rr), c0 + int(cc))
                    for rr, cc in np.argwhere(mask)
                }
                if cells & forbidden:
                    continue
                candidates.append((r0, c0, cells))
        if not candidates:
            return False
        r0, c0, cells = random.choice(candidates)
        paste(grid, patch, r0, c0, transparent=BG)
        forbidden |= _dilate_cells(cells, h, left_width)
    return True


def _render_expected(
    grid: np.ndarray,
    frames: list[dict],
    source_color: int,
    source_corner: str,
    sprite: np.ndarray,
    counts: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        if frame["color"] == source_color:
            continue
        oriented = _orient_sprite(sprite, source_corner, frame["missing_kind"], BG)
        _stamp_stack(out, oriented, frame["missing"], frame["missing_kind"], counts[frame["color"]], BG)
    return out


def _scene_changes_with_counts(
    counts: dict[int, int],
    colors: list[int],
    source_color: int,
) -> bool:
    non_source = [color for color in colors if color != source_color]
    return any(counts[color] > 1 for color in non_source)


def _scene_changes_with_flip(
    sprite: np.ndarray,
    source_corner: str,
    frames: list[dict],
    source_color: int,
) -> bool:
    for frame in frames:
        if frame["color"] == source_color:
            continue
        oriented = _orient_sprite(sprite, source_corner, frame["missing_kind"], BG)
        if not np.array_equal(oriented, sprite):
            return True
    return False


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(20, 28)
    w = random.randint(24, 30)
    frame_count = random.choice((2, 3, 3))
    colors = list(FRAME_COLORS[:frame_count])
    random.shuffle(colors)
    source_color = random.choice(colors)
    source_corner = random.choice(tuple(CORNER_SIGN))
    other_corners = []
    while len(other_corners) < frame_count - 1:
        corner = random.choice(tuple(CORNER_SIGN))
        if not other_corners and corner == source_corner:
            continue
        other_corners.append(corner)
    corner_by_color = {source_color: source_corner}
    for color, corner in zip([color for color in colors if color != source_color], other_corners):
        corner_by_color[color] = corner

    counts = {color: random.randint(1, 3) for color in colors}
    if len(set(counts.values())) == 1:
        first = colors[0]
        counts[first] = 1 + (counts[first] % 3)
    non_source = [color for color in colors if color != source_color]
    if not any(counts[color] > 1 for color in non_source):
        counts[random.choice(non_source)] = random.choice((2, 3))
    if len(non_source) == 2 and counts[non_source[0]] == counts[non_source[1]]:
        counts[non_source[1]] = 1 + (counts[non_source[1]] % 3)

    sprite = _sprite_patch()
    side = random.choice((3, 4))
    if sprite.shape[0] > side or sprite.shape[1] > side:
        return None

    separator = random.randint(2, 4)
    left_width = random.randint(8, 11)
    right_start = left_width + separator
    if right_start + 9 > w:
        return None

    frame_specs = []
    for idx, color in enumerate(colors):
        frame_specs.append(
            {
                "color": color,
                "side": side,
                "missing_kind": corner_by_color[color],
                "copies": 1 if color == source_color else counts[color],
            }
        )
    if not _scene_changes_with_counts(counts, colors, source_color):
        return None

    canonical = _orient_sprite(sprite, source_corner, source_corner, BG)
    if not _scene_changes_with_flip(canonical, source_corner, frame_specs, source_color):
        return None

    order = sorted(frame_specs, key=lambda d: (-d["copies"], d["color"]))
    forbidden: set[tuple[int, int]] = set()
    placements: dict[int, tuple[int, int]] = {}
    for spec in order:
        pos = _random_frame_position(
            h,
            w,
            right_start,
            spec["side"],
            spec["missing_kind"],
            canonical,
            spec["copies"],
            forbidden,
        )
        if pos is None:
            return None
        placements[spec["color"]] = pos
        marker_cells, stack_cells = _group_cells(
            pos[0], pos[1], spec["side"], spec["missing_kind"], canonical, spec["copies"]
        )
        forbidden |= _dilate_cells(marker_cells | stack_cells, h, w)

    grid = np.zeros((h, w), dtype=int)
    frames = []
    for spec in frame_specs:
        r0, c0 = placements[spec["color"]]
        corners_map = _frame_corners(r0, c0, spec["side"])
        for name, pt in corners_map.items():
            if name != spec["missing_kind"]:
                grid[pt] = spec["color"]
        frame = {
            "color": spec["color"],
            "side": spec["side"],
            "missing_kind": spec["missing_kind"],
            "missing": corners_map[spec["missing_kind"]],
            "markers": tuple(sorted(pt for name, pt in corners_map.items() if name != spec["missing_kind"])),
        }
        frames.append(frame)

    source_frame = next(frame for frame in frames if frame["color"] == source_color)
    source_sprite = _orient_sprite(canonical, source_corner, source_frame["missing_kind"], BG)
    _stamp_stack(grid, source_sprite, source_frame["missing"], source_frame["missing_kind"], 1, BG)

    if not _place_count_components(grid, left_width, counts):
        return None

    parsed = _extract_scene(grid)
    if {frame["color"] for frame in parsed["frames"]} != set(colors):
        return None
    if parsed["source"]["color"] != source_color:
        return None
    if parsed["counts"] != counts:
        return None

    expected = _render_expected(grid, frames, source_color, source_frame["missing_kind"], source_sprite, counts)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    if not _scene_changes_with_flip(source_sprite, source_frame["missing_kind"], frames, source_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
