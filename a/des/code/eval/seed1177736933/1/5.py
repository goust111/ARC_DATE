import numpy as np


DIRECTIONS = ("up", "right", "down", "left")
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
KEY_MASKS = {
    # The opening direction is an explicit convention shared by the solver and generator.
    "up": np.array([[1, 1], [0, 1]], dtype=bool),
    "right": np.array([[1, 1], [1, 0]], dtype=bool),
    "down": np.array([[1, 0], [1, 1]], dtype=bool),
    "left": np.array([[0, 1], [1, 1]], dtype=bool),
}

MIN_LOCKS = 2
MAX_LOCKS = 4
FRAME_H_RANGE = (5, 9)  # [lo, hi)
FRAME_W_RANGE = (5, 10)  # [lo, hi)
GRID_H_RANGE = (22, 33)  # [lo, hi)
GRID_W_RANGE = (22, 33)  # [lo, hi)
KEY_REACH = 4
BASE_MARGIN = 3
PLACEMENT_PAD = 1
PATCH_ATTEMPTS = 60
SCENE_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
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
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in components_mask(grid == color):
            bbox = bbox_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": [(int(r), int(c)) for r, c in coords],
                    "bbox": bbox,
                    "area": int(len(coords)),
                }
            )
    return comps


def _classify_key(comp: dict) -> dict | None:
    if int(comp["area"]) != 3:
        return None
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 + 1 != 2 or c1 - c0 + 1 != 2:
        return None
    mask = np.zeros((2, 2), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    opening = None
    for direction, pattern in KEY_MASKS.items():
        if np.array_equal(mask, pattern):
            opening = direction
            break
    if opening is None:
        return None
    return {
        "color": int(comp["color"]),
        "coords": list(comp["coords"]),
        "bbox": comp["bbox"],
        "opening": opening,
    }


def _classify_frame(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    coords = {(int(r), int(c)) for r, c in comp["coords"]}
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    if coords - border:
        return None
    missing = sorted(border - coords)
    if len(missing) != 1:
        return None
    gap = missing[0]
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        gap_dir = "up"
    elif gr == r1 and c0 < gc < c1:
        gap_dir = "down"
    elif gc == c0 and r0 < gr < r1:
        gap_dir = "left"
    elif gc == c1 and r0 < gr < r1:
        gap_dir = "right"
    else:
        return None
    if len(coords) != len(border) - 1:
        return None
    return {
        "color": int(comp["color"]),
        "coords": list(comp["coords"]),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap": (int(gr), int(gc)),
        "gap_dir": gap_dir,
    }


def _content_info(grid: np.ndarray, bg: int, frame: dict) -> dict:
    r0, c0, r1, c1 = frame["bbox"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    pts = np.argwhere(inner != bg)
    assert len(pts) > 0, "lock interior must contain hidden content"
    content_colors = np.unique(inner[inner != bg])
    assert len(content_colors) == 1, "hidden content must use a single color"
    color = int(content_colors[0])
    coords = [(int(r0 + 1 + dr), int(c0 + 1 + dc)) for dr, dc in pts]
    gap_r, gap_c = frame["gap"]
    if frame["gap_dir"] in ("left", "right"):
        rows = {r for r, _c in coords}
        assert len(rows) == 1 and gap_r in rows, "horizontal content must align with side gap row"
        cols = sorted(c for _r, c in coords)
        assert cols == list(range(cols[0], cols[-1] + 1)), "content row must be contiguous"
    else:
        cols = {c for _r, c in coords}
        assert len(cols) == 1 and gap_c in cols, "vertical content must align with side gap column"
        rows = sorted(r for r, _c in coords)
        assert rows == list(range(rows[0], rows[-1] + 1)), "content column must be contiguous"
    return {"color": color, "coords": coords, "length": int(len(coords))}


def _released_coords(frame_bbox: tuple[int, int, int, int], gap_dir: str, gap: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    gr, gc = gap
    if gap_dir == "left":
        return [(int(gr), int(c)) for c in range(c0 - length, c0)]
    if gap_dir == "right":
        return [(int(gr), int(c)) for c in range(c1 + 1, c1 + length + 1)]
    if gap_dir == "up":
        return [(int(r), int(gc)) for r in range(r0 - length, r0)]
    if gap_dir == "down":
        return [(int(r), int(gc)) for r in range(r1 + 1, r1 + length + 1)]
    raise ValueError(f"unknown direction {gap_dir}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each key to the same-colored lock, use the key opening to identify the
    gap side, then release the hidden line through that gap.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    keys_by_color: dict[int, list[dict]] = {}
    frames_by_color: dict[int, list[dict]] = {}
    for comp in _monochrome_components(g, bg):
        key = _classify_key(comp)
        if key is not None:
            keys_by_color.setdefault(int(key["color"]), []).append(key)
            continue
        frame = _classify_frame(comp)
        if frame is not None:
            frames_by_color.setdefault(int(frame["color"]), []).append(frame)

    assert frames_by_color, "expected at least one lock"
    for color, frames in frames_by_color.items():
        assert len(frames) == 1, "each lock color should identify one frame"
        assert color in keys_by_color and len(keys_by_color[color]) == 1, "each lock must have one matching key"
        frame = frames[0]
        key = keys_by_color[color][0]
        assert frame["gap_dir"] == OPPOSITE[key["opening"]], "key opening must point at the lock gap"

        content = _content_info(g, bg, frame)
        for r, c in key["coords"]:
            out[r, c] = bg
        for r, c in content["coords"]:
            out[r, c] = bg

        gr, gc = frame["gap"]
        out[int(gr), int(gc)] = int(color)
        for rr, cc in _released_coords(frame["bbox"], frame["gap_dir"], frame["gap"], int(content["length"])):
            out[int(rr), int(cc)] = int(content["color"])

    return out


def _draw_frame_with_gap(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, gap: tuple[int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[gap] = 0  # temporary marker; overwritten with bg by caller


def _key_candidates(
    patch_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    gap_dir: str,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    ph, pw = patch_shape
    r0, c0, r1, c1 = frame_bbox
    candidates: list[tuple[int, int]] = []
    for kr in range(ph - 1):
        for kc in range(pw - 1):
            if gap_dir == "left":
                if not (max(0, c0 - KEY_REACH) <= kc <= c0 - 2):
                    continue
            elif gap_dir == "right":
                if not (c1 + 1 <= kc <= min(pw - 2, c1 + KEY_REACH - 1)):
                    continue
            elif gap_dir == "up":
                if not (max(0, r0 - KEY_REACH) <= kr <= r0 - 2):
                    continue
            else:
                if not (r1 + 1 <= kr <= min(ph - 2, r1 + KEY_REACH - 1)):
                    continue
            cells = {(kr + dr, kc + dc) for dr in (0, 1) for dc in (0, 1)}
            if cells & forbidden:
                continue
            candidates.append((int(kr), int(kc)))
    return candidates


def _make_patch(
    rng: np.random.Generator,
    bg: int,
    frame_color: int,
    content_color: int,
    gap_dir: str,
) -> tuple[np.ndarray, dict] | None:
    h = int(rng.integers(*FRAME_H_RANGE))
    w = int(rng.integers(*FRAME_W_RANGE))
    max_len = (w - 2) if gap_dir in ("left", "right") else (h - 2)
    if max_len < 2:
        return None
    length = int(rng.integers(2, max_len + 1))

    margins = {"up": BASE_MARGIN, "right": BASE_MARGIN, "down": BASE_MARGIN, "left": BASE_MARGIN}
    margins[gap_dir] = max(margins[gap_dir], length + 1, KEY_REACH + 1)
    if gap_dir in ("left", "right"):
        margins["up"] = max(margins["up"], 2)
        margins["down"] = max(margins["down"], 2)
    else:
        margins["left"] = max(margins["left"], 2)
        margins["right"] = max(margins["right"], 2)

    ph = margins["up"] + h + margins["down"]
    pw = margins["left"] + w + margins["right"]
    patch = np.full((ph, pw), bg, dtype=int)
    r0 = margins["up"]
    c0 = margins["left"]
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    frame_bbox = (int(r0), int(c0), int(r1), int(c1))

    if gap_dir == "left":
        gr = int(rng.integers(r0 + 1, r1))
        gc = c0
        content_coords = [(gr, int(c0 + i)) for i in range(1, length + 1)]
    elif gap_dir == "right":
        gr = int(rng.integers(r0 + 1, r1))
        gc = c1
        content_coords = [(gr, int(c1 - i)) for i in range(length, 0, -1)]
    elif gap_dir == "up":
        gr = r0
        gc = int(rng.integers(c0 + 1, c1))
        content_coords = [(int(r0 + i), gc) for i in range(1, length + 1)]
    else:
        gr = r1
        gc = int(rng.integers(c0 + 1, c1))
        content_coords = [(int(r1 - i), gc) for i in range(length, 0, -1)]

    gap = (int(gr), int(gc))
    _draw_frame_with_gap(patch, frame_bbox, frame_color, gap)
    patch[gap] = bg
    for r, c in content_coords:
        patch[int(r), int(c)] = content_color

    reserve = set(_released_coords(frame_bbox, gap_dir, gap, length))
    forbidden = {(int(r), int(c)) for r, c in content_coords}
    forbidden |= {(int(r), int(c)) for r, c in reserve}
    forbidden |= {
        (r0, c) for c in range(c0, c1 + 1)
    } | {
        (r1, c) for c in range(c0, c1 + 1)
    } | {
        (r, c0) for r in range(r0, r1 + 1)
    } | {
        (r, c1) for r in range(r0, r1 + 1)
    }

    opening = OPPOSITE[gap_dir]
    key_mask = KEY_MASKS[opening]
    candidates = _key_candidates((ph, pw), frame_bbox, gap_dir, forbidden)
    if not candidates:
        return None
    kr, kc = candidates[int(rng.integers(0, len(candidates)))]
    key_coords: list[tuple[int, int]] = []
    for dr in range(2):
        for dc in range(2):
            if key_mask[dr, dc]:
                rr = int(kr + dr)
                cc = int(kc + dc)
                patch[rr, cc] = frame_color
                key_coords.append((rr, cc))

    spec = {
        "frame_color": int(frame_color),
        "content_color": int(content_color),
        "frame_bbox": frame_bbox,
        "gap": gap,
        "gap_dir": gap_dir,
        "content_coords": [(int(r), int(c)) for r, c in content_coords],
        "line_length": int(length),
        "key_coords": key_coords,
        "opening": opening,
        "patch_shape": (int(ph), int(pw)),
    }
    return patch, spec


def _translate_spec(spec: dict, dr: int, dc: int) -> dict:
    r0, c0, r1, c1 = spec["frame_bbox"]
    gap_r, gap_c = spec["gap"]
    out = dict(spec)
    out["frame_bbox"] = (int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc))
    out["gap"] = (int(gap_r + dr), int(gap_c + dc))
    out["content_coords"] = [(int(r + dr), int(c + dc)) for r, c in spec["content_coords"]]
    out["key_coords"] = [(int(r + dr), int(c + dc)) for r, c in spec["key_coords"]]
    return out


def _apply_specs(grid: np.ndarray, specs: list[dict], bg: int) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        for r, c in spec["key_coords"]:
            out[int(r), int(c)] = bg
        for r, c in spec["content_coords"]:
            out[int(r), int(c)] = bg
        gr, gc = spec["gap"]
        out[int(gr), int(gc)] = int(spec["frame_color"])
        for rr, cc in _released_coords(spec["frame_bbox"], spec["gap_dir"], spec["gap"], int(spec["line_length"])):
            out[int(rr), int(cc)] = int(spec["content_color"])
    return out


def _scene_rect_fits(rect: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = rect
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + PLACEMENT_PAD < pr0 or pr1 + PLACEMENT_PAD < r0 or c1 + PLACEMENT_PAD < pc0 or pc1 + PLACEMENT_PAD < c0):
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_locks = int(rng.integers(MIN_LOCKS, MAX_LOCKS + 1))
    bg = int(rng.integers(0, 10))
    palette = np.array([c for c in range(10) if c != bg], dtype=int)
    rng.shuffle(palette)
    frame_colors = [int(c) for c in palette[:n_locks]]
    content_colors = [int(c) for c in palette[n_locks : 2 * n_locks]]

    patches: list[tuple[np.ndarray, dict]] = []
    for frame_color, content_color in zip(frame_colors, content_colors):
        patch = None
        for _ in range(PATCH_ATTEMPTS):
            gap_dir = DIRECTIONS[int(rng.integers(0, len(DIRECTIONS)))]
            candidate = _make_patch(rng, bg, frame_color, content_color, gap_dir)
            if candidate is not None:
                patch = candidate
                break
        if patch is None:
            return None
        patches.append(patch)

    rng.shuffle(patches)
    grid_h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1]))
    grid_w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1]))
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    specs: list[dict] = []
    placed_rects: list[tuple[int, int, int, int]] = []
    for patch, spec in sorted(patches, key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True):
        ph, pw = patch.shape
        if ph > grid_h or pw > grid_w:
            return None
        placed = False
        for _ in range(PATCH_ATTEMPTS * 2):
            sr = int(rng.integers(0, grid_h - ph + 1))
            sc = int(rng.integers(0, grid_w - pw + 1))
            rect = (sr, sc, sr + ph - 1, sc + pw - 1)
            if not _scene_rect_fits(rect, placed_rects):
                continue
            target = grid[sr : sr + ph, sc : sc + pw]
            if np.any((patch != bg) & (target != bg)):
                continue
            mask = patch != bg
            target[mask] = patch[mask]
            specs.append(_translate_spec(spec, sr, sc))
            placed_rects.append(rect)
            placed = True
            break
        if not placed:
            return None

    if get_bg(grid) != bg:
        return None
    expected = _apply_specs(grid, specs, bg)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(SCENE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-and-key scene")
