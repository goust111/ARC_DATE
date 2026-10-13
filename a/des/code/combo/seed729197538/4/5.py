import random
from collections import deque

import numpy as np


TOP_TEMPLATE_STRINGS = [
    [
        "101",
        "111",
    ],
    [
        "1001",
        "1111",
    ],
    [
        "101",
        "111",
        "011",
    ],
    [
        "101",
        "111",
        "110",
    ],
    [
        "101",
        "101",
        "111",
    ],
    [
        "1001",
        "1111",
        "1100",
    ],
]

SIDES = ("top", "right", "bottom", "left")
SIDE_TO_TURNS = {"top": 0, "right": 1, "bottom": 2, "left": 3}
OUTER_RETRIES = 400
FRAME_COUNT_CHOICES = (3, 4, 5)
FRAME_HEIGHT_RANGE = (7, 9)
FRAME_WIDTH_RANGE = (8, 11)
GRID_MAX = 30
EDGE_MARGIN = 4
FRAME_PADDING = 2
FRAME_PLACEMENT_RETRIES = 200
KEY_PLACEMENT_RETRIES = 200


def _template_from_strings(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


TEMPLATES = [_template_from_strings(rows) for rows in TOP_TEMPLATE_STRINGS]


def _canonical_mask(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = []
    for turns in range(4):
        rot = np.rot90(mask, turns)
        coords = np.argwhere(rot)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        crop = rot[r0 : r1 + 1, c0 : c1 + 1]
        variants.append((tuple(int(v) for v in crop.shape), crop.astype(np.uint8).tobytes()))
    return min(variants)


_template_signatures = {_canonical_mask(mask[1:, :]) for mask in TEMPLATES}
assert len(_template_signatures) == len(TEMPLATES), "template shapes must be rotation-distinct"


def _rotate_mask(mask: np.ndarray, side: str) -> np.ndarray:
    turns = SIDE_TO_TURNS[side]
    return np.rot90(mask, -turns)


def _component_bfs(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            touches_boundary = False

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_boundary = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "touches_boundary": touches_boundary,
                }
            )
    return comps


def _coords_to_signature(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], bytes]:
    arr = np.array(coords, dtype=int)
    arr -= arr.min(axis=0)
    h = int(arr[:, 0].max()) + 1
    w = int(arr[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[arr[:, 0], arr[:, 1]] = True
    return _canonical_mask(mask)


def _frame_signature(comp: dict) -> tuple[tuple[int, int], bytes]:
    r0, c0, r1, c1 = comp["bbox"]
    inner = [(r, c) for r, c in comp["coords"] if r0 < r < r1 and c0 < c < c1]
    assert inner, "frame must contain a non-empty notch contour"
    return _coords_to_signature(inner)


def _key_signature(comp: dict, h: int, w: int) -> tuple[tuple[int, int], bytes]:
    inner = [(r, c) for r, c in comp["coords"] if 0 < r < h - 1 and 0 < c < w - 1]
    assert inner, "key must protrude inward from the border"
    return _coords_to_signature(inner)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame's inward notch contour to a border key shape under rotation.

    Frames that find a matching key are recolored to the key's color.
    Frames without a match are erased to black.
    Border keys remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    key_map: dict[tuple[tuple[int, int], bytes], int] = {}
    frames = []
    for comp in _component_bfs(g):
        if comp["touches_boundary"]:
            sig = _key_signature(comp, h, w)
            assert sig not in key_map, "key shapes must be unique"
            key_map[sig] = int(comp["color"])
        else:
            frames.append(comp)

    out = g.copy()
    for comp in frames:
        sig = _frame_signature(comp)
        rows = np.array([r for r, _ in comp["coords"]], dtype=int)
        cols = np.array([c for _, c in comp["coords"]], dtype=int)
        if sig in key_map:
            out[rows, cols] = key_map[sig]
        else:
            out[rows, cols] = 0
    return out


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _place_frames(n_frames: int, fh: int, fw: int, h: int, w: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(FRAME_PLACEMENT_RETRIES):
        positions.clear()
        boxes.clear()
        for _idx in range(n_frames):
            for _ in range(FRAME_PLACEMENT_RETRIES):
                r0 = random.randint(EDGE_MARGIN, h - EDGE_MARGIN - fh)
                c0 = random.randint(EDGE_MARGIN, w - EDGE_MARGIN - fw)
                box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
                if any(_rectangles_overlap(box, old, pad=FRAME_PADDING) for old in boxes):
                    continue
                boxes.append(box)
                positions.append((r0, c0))
                break
            else:
                break
        if len(positions) == n_frames:
            return positions
    raise AssertionError("failed to place frames")


def _frame_notch_start(side: str, fh: int, fw: int, mask: np.ndarray) -> int:
    span = mask.shape[1] if side in ("top", "bottom") else mask.shape[0]
    limit = (fw if side in ("top", "bottom") else fh) - span - 2
    assert limit >= 1, "frame must be large enough for the notch"
    return random.randint(1, limit)


def _frame_cells(r0: int, c0: int, fh: int, fw: int, side: str, mask: np.ndarray) -> list[tuple[int, int]]:
    cells = set()
    for rr in range(r0, r0 + fh):
        cells.add((rr, c0))
        cells.add((rr, c0 + fw - 1))
    for cc in range(c0, c0 + fw):
        cells.add((r0, cc))
        cells.add((r0 + fh - 1, cc))

    start = _frame_notch_start(side, fh, fw, mask)
    ph, pw = mask.shape
    if side == "top":
        for dx in range(pw):
            cells.discard((r0, c0 + start + dx))
        base_r, base_c = r0, c0 + start
    elif side == "bottom":
        for dx in range(pw):
            cells.discard((r0 + fh - 1, c0 + start + dx))
        base_r, base_c = r0 + fh - ph, c0 + start
    elif side == "left":
        for dy in range(ph):
            cells.discard((r0 + start + dy, c0))
        base_r, base_c = r0 + start, c0
    else:
        for dy in range(ph):
            cells.discard((r0 + start + dy, c0 + fw - 1))
        base_r, base_c = r0 + start, c0 + fw - pw

    for rr, cc in np.argwhere(mask):
        cells.add((base_r + int(rr), base_c + int(cc)))

    return sorted(cells)


def _try_place_key(
    grid: np.ndarray,
    occupied_boxes: list[tuple[int, int, int, int]],
    color: int,
    mask: np.ndarray,
) -> bool:
    h, w = grid.shape
    sides = list(SIDES)
    random.shuffle(sides)
    for side in sides:
        ph, pw = mask.shape
        if side in ("top", "bottom"):
            if w - pw - 2 < 1:
                continue
            for _ in range(KEY_PLACEMENT_RETRIES):
                start = random.randint(1, w - pw - 2)
                r0 = 0 if side == "top" else h - ph
                c0 = start
                box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if any(_rectangles_overlap(box, old, pad=1) for old in occupied_boxes):
                    continue
                occupied_boxes.append(box)
                for rr, cc in np.argwhere(mask):
                    grid[r0 + int(rr), c0 + int(cc)] = color
                return True
        else:
            if h - ph - 2 < 1:
                continue
            for _ in range(KEY_PLACEMENT_RETRIES):
                start = random.randint(1, h - ph - 2)
                r0 = start
                c0 = 0 if side == "left" else w - pw
                box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if any(_rectangles_overlap(box, old, pad=1) for old in occupied_boxes):
                    continue
                occupied_boxes.append(box)
                for rr, cc in np.argwhere(mask):
                    grid[r0 + int(rr), c0 + int(cc)] = color
                return True
    return False


def _build_expected(
    grid: np.ndarray,
    frame_infos: list[dict],
    match_color: dict[tuple[tuple[int, int], bytes], int],
) -> np.ndarray:
    out = grid.copy()
    for info in frame_infos:
        rows = np.array([r for r, _ in info["coords"]], dtype=int)
        cols = np.array([c for _, c in info["coords"]], dtype=int)
        sig = info["signature"]
        if sig in match_color:
            out[rows, cols] = match_color[sig]
        else:
            out[rows, cols] = 0
    return out


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(FRAME_COUNT_CHOICES)
    fh = random.randint(*FRAME_HEIGHT_RANGE)
    fw = random.randint(*FRAME_WIDTH_RANGE)
    n_cols = random.choice([2, 3] if n_frames >= 4 else [2])
    n_rows = (n_frames + n_cols - 1) // n_cols

    min_h = max(EDGE_MARGIN * 2 + n_rows * fh + (n_rows - 1) * FRAME_PADDING, 16)
    min_w = max(EDGE_MARGIN * 2 + n_cols * fw + (n_cols - 1) * FRAME_PADDING, 16)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = random.randint(min_h, GRID_MAX)
    w = random.randint(min_w, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)

    try:
        positions = _place_frames(n_frames, fh, fw, h, w)
    except AssertionError:
        return None

    frame_colors = random.sample(range(1, 10), n_frames)
    max_matches = min(4, n_frames, 9 - n_frames)
    matched_count = random.randint(2, max_matches)
    key_colors = random.sample([c for c in range(1, 10) if c not in frame_colors], matched_count)
    template_indices = random.sample(range(len(TEMPLATES)), n_frames)
    matched_indices = set(random.sample(range(n_frames), matched_count))

    frame_infos = []
    key_targets = []
    for idx, (r0, c0) in enumerate(positions):
        side = random.choice(SIDES)
        template = TEMPLATES[template_indices[idx]]
        oriented = _rotate_mask(template, side)
        coords = _frame_cells(r0, c0, fh, fw, side, oriented)
        color = frame_colors[idx]
        for rr, cc in coords:
            grid[rr, cc] = color

        sig = _coords_to_signature(
            [(r, c) for r, c in coords if r0 < r < r0 + fh - 1 and c0 < c < c0 + fw - 1]
        )
        frame_infos.append({"coords": coords, "signature": sig})
        if idx in matched_indices:
            key_targets.append(sig)

    occupied_key_boxes: list[tuple[int, int, int, int]] = []
    match_color: dict[tuple[tuple[int, int], bytes], int] = {}
    for sig, key_color in zip(key_targets, key_colors):
        match_color[sig] = key_color

    items = list(match_color.items())
    random.shuffle(items)
    for sig, key_color in items:
        template = next(mask for mask in TEMPLATES if _canonical_mask(mask[1:, :]) == sig)
        side = random.choice(SIDES)
        oriented = _rotate_mask(template, side)
        if not _try_place_key(grid, occupied_key_boxes, key_color, oriented):
            return None

    expected = _build_expected(grid, frame_infos, match_color)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
