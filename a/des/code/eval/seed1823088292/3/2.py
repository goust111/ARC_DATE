import math
import random
from collections import Counter, deque

import numpy as np


BG = 0
GEN_RETRIES = 400
CONTAINER_COUNT_CHOICES = (2, 3)
INTERIOR_SIDE_MIN = 4
INTERIOR_SIDE_MAX = 9
PIECE_SIDE_MAX = 5
PADDING = 1
FRAME_COLOR_POOL = tuple(range(1, 10))
CORNERS = ("tl", "tr", "bl", "br")
VALID_CORNER_SETS = (
    ("tl", "br"),
    ("tr", "bl"),
    ("tl", "tr", "bl"),
    ("tl", "tr", "br"),
    ("tl", "bl", "br"),
    ("tr", "bl", "br"),
    ("tl", "tr", "bl", "br"),
)
RNG = random.Random()


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    q = deque([tuple(map(int, pts[0]))])
    seen[q[0]] = True
    count = 0
    while q:
        r, c = q.popleft()
        count += 1
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return count == int(mask.sum())


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for pts in components_mask(grid == color):
            r0, c0, r1, c1 = bbox_coords(pts)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            if len(pts) != 2 * (h + w) - 4:
                continue
            pset = {tuple(map(int, pt)) for pt in pts}
            ok = True
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    on_perim = rr in (r0, r1) or cc in (c0, c1)
                    if ((rr, cc) in pset) != on_perim:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
                continue
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return frames


def _component_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch, (r0, c0, r1, c1)


def _anchor_candidates(
    patch: np.ndarray, frame_colors: set[int], bg: int
) -> list[tuple[int, str, int, int, int]]:
    h, w = patch.shape
    color_counts = Counter(int(v) for v in patch.flat if int(v) != bg)
    corners = {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "bl": (h - 1, 0),
        "br": (h - 1, w - 1),
    }
    out = {}
    for name, (rr, cc) in corners.items():
        value = int(patch[rr, cc])
        if value != bg and value in frame_colors:
            out[(rr, cc)] = (color_counts[value], name, value, rr, cc)
    return sorted(out.values())


def _place_piece_in_frame(out: np.ndarray, frame: dict, patch: np.ndarray, corner: str, bg: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    ir0, ic0 = r0 + 1, c0 + 1
    ir1, ic1 = r1 - 1, c1 - 1
    h, w = patch.shape
    top = ir0 if corner[0] == "t" else ir1 - h + 1
    left = ic0 if corner[1] == "l" else ic1 - w + 1
    assert ir0 <= top <= ir1 - h + 1
    assert ic0 <= left <= ic1 - w + 1
    mask = patch != bg
    out[top : top + h, left : left + w][mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frames = _find_frames(g, bg)
    assert frames, "expected at least one rectangular frame"
    frame_by_color: dict[int, dict] = {}
    for frame in frames:
        color = int(frame["color"])
        assert color not in frame_by_color, "frame colors must be unique"
        frame_by_color[color] = frame

    frame_mask = np.zeros_like(g, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        frame_mask[r0 : r1 + 1, c0] = True
        frame_mask[r0 : r1 + 1, c1] = True
        frame_mask[r0, c0 : c1 + 1] = True
        frame_mask[r1, c0 : c1 + 1] = True

    out = g.copy()
    fragments = components_mask((g != bg) & ~frame_mask)
    for coords in fragments:
        patch, _ = _component_patch(g, coords, bg)
        candidates = _anchor_candidates(patch, set(frame_by_color), bg)
        assert candidates, "fragment must have a corner marker"
        best_count, corner, frame_color, ar, ac = candidates[0]
        if len(candidates) > 1:
            assert best_count < candidates[1][0], "ambiguous marker color in fragment"

        body_counts = Counter()
        for rr, cc in np.argwhere(patch != bg):
            rr = int(rr)
            cc = int(cc)
            if rr == ar and cc == ac:
                continue
            body_counts[int(patch[rr, cc])] += 1
        assert body_counts, "fragment body must contain at least one non-marker pixel"
        body_color = body_counts.most_common(1)[0][0]

        for r, c in coords:
            out[int(r), int(c)] = bg

        restored = patch.copy()
        restored[ar, ac] = body_color
        _place_piece_in_frame(out, frame_by_color[int(frame_color)], restored, corner, bg)

    return out


def _corner_limits(height: int, width: int, split_r: int, split_c: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (split_r, split_c),
        "tr": (split_r, width - split_c),
        "bl": (height - split_r, split_c),
        "br": (height - split_r, width - split_c),
    }


def _required_subset_has_all_sides(corners: tuple[str, ...]) -> bool:
    return (
        any(c in corners for c in ("tl", "tr"))
        and any(c in corners for c in ("bl", "br"))
        and any(c in corners for c in ("tl", "bl"))
        and any(c in corners for c in ("tr", "br"))
    )


def _make_piece_mask(height: int, width: int, corner: str, rng: random.Random) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)

    anchor = {
        "tl": (0, 0),
        "tr": (0, width - 1),
        "bl": (height - 1, 0),
        "br": (height - 1, width - 1),
    }[corner]
    protected = {anchor}

    if corner == "tl":
        if width > 1:
            protected.add((0, 1))
        elif height > 1:
            protected.add((1, 0))
        protected.add((height - 1, rng.randrange(width)))
        protected.add((rng.randrange(height), width - 1))
    elif corner == "tr":
        if width > 1:
            protected.add((0, width - 2))
        elif height > 1:
            protected.add((1, width - 1))
        protected.add((height - 1, rng.randrange(width)))
        protected.add((rng.randrange(height), 0))
    elif corner == "bl":
        if width > 1:
            protected.add((height - 1, 1))
        elif height > 1:
            protected.add((height - 2, 0))
        protected.add((0, rng.randrange(width)))
        protected.add((rng.randrange(height), width - 1))
    else:
        if width > 1:
            protected.add((height - 1, width - 2))
        elif height > 1:
            protected.add((height - 2, width - 1))
        protected.add((0, rng.randrange(width)))
        protected.add((rng.randrange(height), 0))

    min_area = max(2, (height * width + 1) // 2)
    target_area = rng.randint(min_area, height * width)
    candidates = [(r, c) for r in range(height) for c in range(width) if (r, c) not in protected]
    rng.shuffle(candidates)

    for r, c in candidates:
        if int(mask.sum()) <= target_area:
            break
        trial = mask.copy()
        trial[r, c] = False
        if not is_connected(trial):
            continue
        rows = np.any(trial, axis=1)
        cols = np.any(trial, axis=0)
        if not (rows[0] and rows[-1] and cols[0] and cols[-1]):
            continue
        mask = trial

    assert is_connected(mask)
    assert mask[anchor]
    assert np.any(mask[0]) and np.any(mask[-1]) and np.any(mask[:, 0]) and np.any(mask[:, -1])
    assert int(mask.sum()) >= 2
    return mask


def _paint_piece_patch(mask: np.ndarray, body_color: int, frame_color: int, corner: str) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = body_color
    anchor = {
        "tl": (0, 0),
        "tr": (0, mask.shape[1] - 1),
        "bl": (mask.shape[0] - 1, 0),
        "br": (mask.shape[0] - 1, mask.shape[1] - 1),
    }[corner]
    patch[anchor] = frame_color
    return patch


def _make_frame_patch(height: int, width: int, color: int) -> np.ndarray:
    patch = np.full((height + 2, width + 2), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _try_place_patch(
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
    pad: int = PADDING,
) -> tuple[int, int] | None:
    h, w = patch.shape
    gh, gw = grid.shape
    if h > gh or w > gw:
        return None
    positions = [(top, left) for top in range(gh - h + 1) for left in range(gw - w + 1)]
    rng.shuffle(positions)
    for top, left in positions:
        r0 = max(0, top - pad)
        c0 = max(0, left - pad)
        r1 = min(gh, top + h + pad)
        c1 = min(gw, left + w + pad)
        if reserved[r0:r1, c0:c1].any():
            continue
        mask = patch != BG
        grid[top : top + h, left : left + w][mask] = patch[mask]
        reserved[r0:r1, c0:c1] = True
        return top, left
    return None


def _paint_expected(
    input_grid: np.ndarray, placed_containers: list[dict], placed_pieces: list[dict]
) -> np.ndarray:
    out = input_grid.copy()
    for spec in placed_pieces:
        top = spec["top"]
        left = spec["left"]
        patch = spec["patch"]
        mask = patch != BG
        out[top : top + patch.shape[0], left : left + patch.shape[1]][mask] = BG
    frame_lookup = {int(spec["frame_color"]): spec for spec in placed_containers}
    for spec in placed_pieces:
        frame = frame_lookup[int(spec["frame_color"])]
        restored = spec["patch"].copy()
        anchor = {
            "tl": (0, 0),
            "tr": (0, restored.shape[1] - 1),
            "bl": (restored.shape[0] - 1, 0),
            "br": (restored.shape[0] - 1, restored.shape[1] - 1),
        }[spec["corner"]]
        restored[anchor] = int(spec["body_color"])
        placed_frame = {
            "bbox": (
                int(frame["top"]),
                int(frame["left"]),
                int(frame["top"] + frame["patch"].shape[0] - 1),
                int(frame["left"] + frame["patch"].shape[1] - 1),
            )
        }
        _place_piece_in_frame(out, placed_frame, restored, spec["corner"], BG)
    return out


def _sample_piece_dims(limit_h: int, limit_w: int, rng: random.Random) -> tuple[int, int]:
    dims = []
    for h in range(2, min(PIECE_SIDE_MAX, limit_h) + 1):
        for w in range(2, min(PIECE_SIDE_MAX, limit_w) + 1):
            if h * w >= 4:
                dims.append((h, w))
    assert dims
    return dims[rng.randrange(len(dims))]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    container_count = rng.choice(CONTAINER_COUNT_CHOICES)
    frame_colors = list(rng.sample(FRAME_COLOR_POOL, container_count))
    body_colors_pool = [c for c in FRAME_COLOR_POOL if c not in frame_colors]

    containers = []
    all_corners: set[str] = set()
    for frame_color in frame_colors:
        inner_h = rng.randint(INTERIOR_SIDE_MIN, INTERIOR_SIDE_MAX)
        inner_w = rng.randint(INTERIOR_SIDE_MIN, INTERIOR_SIDE_MAX)
        split_r = rng.randint(2, inner_h - 2)
        split_c = rng.randint(2, inner_w - 2)
        active_corners = tuple(rng.choice(VALID_CORNER_SETS))
        assert _required_subset_has_all_sides(active_corners)
        all_corners.update(active_corners)

        limits = _corner_limits(inner_h, inner_w, split_r, split_c)
        pieces = []
        for corner in active_corners:
            limit_h, limit_w = limits[corner]
            piece_h, piece_w = _sample_piece_dims(limit_h, limit_w, rng)
            mask = _make_piece_mask(piece_h, piece_w, corner, rng)
            body_color = rng.choice(body_colors_pool)
            patch = _paint_piece_patch(mask, body_color, frame_color, corner)
            pieces.append(
                {
                    "corner": corner,
                    "body_color": body_color,
                    "frame_color": frame_color,
                    "patch": patch,
                }
            )
        containers.append(
            {
                "frame_color": frame_color,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "patch": _make_frame_patch(inner_h, inner_w, frame_color),
                "pieces": pieces,
            }
        )

    if all_corners != set(CORNERS):
        return None

    object_shapes = [spec["patch"].shape for spec in containers]
    for spec in containers:
        object_shapes.extend(piece["patch"].shape for piece in spec["pieces"])
    reserved_area = sum((h + 2 * PADDING) * (w + 2 * PADDING) for h, w in object_shapes)
    side = max(
        max(h for h, _ in object_shapes),
        max(w for _, w in object_shapes),
        int(math.ceil(math.sqrt(reserved_area * 2.0))),
    )
    side = max(side, 24)

    for extra in range(8):
        grid_h = side + rng.randint(1, 6) + extra
        grid_w = side + rng.randint(1, 6) + extra
        inp = np.full((grid_h, grid_w), BG, dtype=int)
        reserved = np.zeros((grid_h, grid_w), dtype=bool)
        placed_containers = []

        ordered_containers = sorted(containers, key=lambda spec: spec["patch"].shape[0] * spec["patch"].shape[1], reverse=True)
        ok = True
        for spec in ordered_containers:
            pos = _try_place_patch(inp, reserved, spec["patch"], rng)
            if pos is None:
                ok = False
                break
            top, left = pos
            placed = dict(spec)
            placed["top"] = top
            placed["left"] = left
            placed_containers.append(placed)
        if not ok:
            continue

        placed_pieces = []
        piece_specs = []
        for container in placed_containers:
            piece_specs.extend(container["pieces"])
        piece_specs.sort(key=lambda spec: spec["patch"].shape[0] * spec["patch"].shape[1], reverse=True)
        for spec in piece_specs:
            pos = _try_place_patch(inp, reserved, spec["patch"], rng)
            if pos is None:
                ok = False
                break
            top, left = pos
            placed = dict(spec)
            placed["top"] = top
            placed["left"] = left
            placed_pieces.append(placed)
        if not ok:
            continue

        expected = _paint_expected(inp, placed_containers, placed_pieces)
        if np.array_equal(inp, expected):
            continue
        try:
            solved = main(inp)
        except AssertionError:
            continue
        if np.array_equal(solved, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
