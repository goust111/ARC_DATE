import random
from collections import deque
from math import gcd

import numpy as np

BG = 8
GREY = 5
COLOR_POOL = (0, 1, 2, 3, 4, 6, 7, 9)
MARKER_POOL = (1, 2, 3, 4, 6, 7, 9)
CORNER_NAMES = ("TL", "TR", "BL", "BR")
STAMP_DIM_OPTIONS = (
    (2, 2),
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 4),
    (4, 3),
    (2, 5),
    (5, 2),
    (4, 5),
    (5, 4),
)
GEN_ATTEMPTS = 300
GRID_MIN_BY_COUNT = {2: 20, 3: 22, 4: 24}
GRID_MAX = 30
PLACEMENT_GAP = 1
PLACEMENT_TRIES = 250


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each outside stamp to the envelope sharing its marker color, then scale the stamp into that envelope.

    Input structure:
    - The background is azure.
    - Several grey hollow rectangles act as envelopes; each envelope interior is otherwise azure except for one colored marker pixel in one interior corner.
    - Elsewhere on the grid are solid multicolor rectangular stamps with no border.
    - Each stamp contains exactly one pixel whose color matches exactly one envelope marker color.

    Rule:
    - Detect every grey frame, read its single marker color, and note which interior corner contains that marker.
    - Detect every outside solid stamp rectangle and pair it to the unique envelope whose marker color appears inside the stamp.
    - Resize the full stamp by nearest-neighbor scaling so that its aspect ratio-preserving size matches the paired envelope interior.
    - Clear the original stamp locations to azure, keep the grey borders unchanged, and paste each scaled stamp into the paired envelope interior aligned to the marker corner.

    Key insight:
    - Three visible cues must be combined: the shared color links each stamp to one envelope, the rectangle dimensions determine the scale, and the marker corner specifies the anchor for the paste.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    envelopes = _find_envelopes(g)
    stamps = _find_stamps(g, envelopes)
    pairs = _pair_stamps(envelopes, stamps)

    out = g.copy()
    for _env, stamp in pairs:
        r0, c0, r1, c1 = stamp["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for env, stamp in pairs:
        inner_r0, inner_c0, inner_r1, inner_c1 = env["inner_bbox"]
        target_h = inner_r1 - inner_r0 + 1
        target_w = inner_c1 - inner_c0 + 1
        resized = _nearest_resize(stamp["sub"], target_h, target_w)

        dst_r0, dst_c0 = _corner_aligned_origin(env["corner"], env["inner_bbox"], resized.shape)
        dst_r1 = dst_r0 + resized.shape[0]
        dst_c1 = dst_c0 + resized.shape[1]
        out[dst_r0:dst_r1, dst_c0:dst_c1] = resized

    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _corner_name(pos: tuple[int, int], inner_bbox: tuple[int, int, int, int]) -> str | None:
    r0, c0, r1, c1 = inner_bbox
    mapping = {
        (r0, c0): "TL",
        (r0, c1): "TR",
        (r1, c0): "BL",
        (r1, c1): "BR",
    }
    return mapping.get(pos)


def _corner_aligned_origin(
    corner: str, inner_bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = inner_bbox
    h, w = shape
    dst_r = r0 if corner.startswith("T") else r1 - h + 1
    dst_c = c0 if corner.endswith("L") else c1 - w + 1
    return dst_r, dst_c


def _find_envelopes(grid: np.ndarray) -> list[dict]:
    envelopes = []
    for pts in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border_mask = np.zeros(sub.shape, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        if not np.all(sub[border_mask] == GREY):
            continue
        if np.any(sub[1:-1, 1:-1] == GREY):
            continue

        inner_bbox = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_points = [
            (r, c)
            for r in range(inner_bbox[0], inner_bbox[2] + 1)
            for c in range(inner_bbox[1], inner_bbox[3] + 1)
            if grid[r, c] not in (BG, GREY)
        ]
        if len(marker_points) != 1:
            continue
        marker_pos = marker_points[0]
        corner = _corner_name(marker_pos, inner_bbox)
        if corner is None:
            continue
        marker_color = int(grid[marker_pos])

        envelopes.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_bbox": inner_bbox,
                "corner": corner,
                "marker_color": marker_color,
                "inner_shape": inner.shape,
            }
        )

    envelopes.sort(key=lambda item: item["bbox"])
    assert envelopes, "expected at least one envelope"
    marker_colors = [env["marker_color"] for env in envelopes]
    assert len(marker_colors) == len(set(marker_colors)), "marker colors must be unique"
    return envelopes


def _find_stamps(grid: np.ndarray, envelopes: list[dict]) -> list[dict]:
    allowed = (grid != BG) & (grid != GREY)
    for env in envelopes:
        r0, c0, r1, c1 = env["bbox"]
        allowed[r0 : r1 + 1, c0 : c1 + 1] = False

    stamps = []
    for pts in _components(allowed):
        r0, c0, r1, c1 = _bbox(pts)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == BG) or np.any(sub == GREY):
            continue
        if len(pts) != sub.size:
            continue
        stamps.append(
            {
                "bbox": (r0, c0, r1, c1),
                "sub": sub.copy(),
                "colors": set(map(int, np.unique(sub))),
            }
        )

    stamps.sort(key=lambda item: item["bbox"])
    assert stamps, "expected at least one stamp"
    return stamps


def _pair_stamps(envelopes: list[dict], stamps: list[dict]) -> list[tuple[dict, dict]]:
    marker_map = {env["marker_color"]: env for env in envelopes}
    marker_colors = set(marker_map)
    pairs = []
    used_markers = set()
    for stamp in stamps:
        shared = stamp["colors"] & marker_colors
        assert len(shared) == 1, "each stamp must contain exactly one marker color"
        marker = next(iter(shared))
        assert marker not in used_markers, "marker colors must pair stamps one-to-one"
        env = marker_map[marker]
        inner_h, inner_w = env["inner_shape"]
        stamp_h, stamp_w = stamp["sub"].shape
        assert inner_h * stamp_w == inner_w * stamp_h, "stamp and envelope must share aspect ratio"
        pairs.append((env, stamp))
        used_markers.add(marker)

    assert len(pairs) == len(envelopes) == len(stamps), "expected a one-to-one envelope/stamp pairing"
    return pairs


def _nearest_resize(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w = arr.shape
    row_idx = np.minimum((np.arange(target_h) * src_h) // target_h, src_h - 1)
    col_idx = np.minimum((np.arange(target_w) * src_w) // target_w, src_w - 1)
    return arr[row_idx][:, col_idx]


def _ratio_key(h: int, w: int) -> tuple[int, int]:
    d = gcd(h, w)
    return h // d, w // d


def _sample_scale(rng: random.Random, pair_count: int, stamp_h: int, stamp_w: int) -> int:
    max_scale = 4 if pair_count == 2 else 3
    if pair_count == 4 and max(stamp_h, stamp_w) >= 4:
        max_scale = 2
    if max(stamp_h, stamp_w) >= 5 or stamp_h * stamp_w >= 16:
        max_scale = min(max_scale, 2)
    return rng.randint(2, max_scale)


def _stamp_pattern(
    rng: random.Random, stamp_h: int, stamp_w: int, marker_color: int, filler_pool: list[int]
) -> np.ndarray:
    filler_count = rng.randint(2, min(3, len(filler_pool)))
    fillers = rng.sample(filler_pool, filler_count)
    mode = rng.choice(("checker", "rows", "cols", "diag", "blocks"))
    stamp = np.empty((stamp_h, stamp_w), dtype=int)

    for r in range(stamp_h):
        for c in range(stamp_w):
            if mode == "checker":
                idx = (r + c) % filler_count
            elif mode == "rows":
                idx = r % filler_count
            elif mode == "cols":
                idx = c % filler_count
            elif mode == "diag":
                idx = (r + 2 * c) % filler_count
            else:
                idx = ((r >= stamp_h // 2) + 2 * (c >= stamp_w // 2)) % filler_count
            stamp[r, c] = fillers[idx]

    if len(set(map(int, np.unique(stamp)))) < 2:
        stamp[:, ::2] = fillers[0]
        stamp[:, 1::2] = fillers[1]

    special_r = rng.randrange(stamp_h)
    special_c = rng.randrange(stamp_w)
    stamp[special_r, special_c] = marker_color
    return stamp


def _place_rectangles(
    rng: random.Random, grid_h: int, grid_w: int, items: list[dict]
) -> dict[tuple[str, int], tuple[int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placed: dict[tuple[str, int], tuple[int, int]] = {}

    for item in sorted(items, key=lambda rec: rec["h"] * rec["w"], reverse=True):
        h = item["h"]
        w = item["w"]
        if h > grid_h or w > grid_w:
            return None
        found = False
        for _ in range(PLACEMENT_TRIES):
            r = rng.randint(0, grid_h - h)
            c = rng.randint(0, grid_w - w)
            pad_r0 = max(0, r - PLACEMENT_GAP)
            pad_c0 = max(0, c - PLACEMENT_GAP)
            pad_r1 = min(grid_h, r + h + PLACEMENT_GAP)
            pad_c1 = min(grid_w, c + w + PLACEMENT_GAP)
            if occupied[pad_r0:pad_r1, pad_c0:pad_c1].any():
                continue
            occupied[r : r + h, c : c + w] = True
            placed[(item["kind"], item["idx"])] = (r, c)
            found = True
            break
        if not found:
            return None

    return placed


def _draw_frame(grid: np.ndarray, top: int, left: int, outer_h: int, outer_w: int) -> None:
    grid[top, left : left + outer_w] = GREY
    grid[top + outer_h - 1, left : left + outer_w] = GREY
    grid[top : top + outer_h, left] = GREY
    grid[top : top + outer_h, left + outer_w - 1] = GREY


def _marker_position(top: int, left: int, outer_h: int, outer_w: int, corner: str) -> tuple[int, int]:
    mapping = {
        "TL": (top + 1, left + 1),
        "TR": (top + 1, left + outer_w - 2),
        "BL": (top + outer_h - 2, left + 1),
        "BR": (top + outer_h - 2, left + outer_w - 2),
    }
    return mapping[corner]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(2, 4)
    marker_colors = rng.sample(list(MARKER_POOL), pair_count)
    dim_specs = rng.sample(list(STAMP_DIM_OPTIONS), pair_count)
    corners = [rng.choice(CORNER_NAMES) for _ in range(pair_count)]

    pairs = []
    used_ratios = set()
    for idx in range(pair_count):
        stamp_h, stamp_w = dim_specs[idx]
        ratio = _ratio_key(stamp_h, stamp_w)
        if ratio in used_ratios:
            return None
        used_ratios.add(ratio)

        scale = _sample_scale(rng, pair_count, stamp_h, stamp_w)
        inner_h = stamp_h * scale
        inner_w = stamp_w * scale
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        filler_pool = [color for color in COLOR_POOL if color not in marker_colors]
        stamp = _stamp_pattern(rng, stamp_h, stamp_w, marker_colors[idx], filler_pool)
        pairs.append(
            {
                "marker_color": marker_colors[idx],
                "corner": corners[idx],
                "stamp": stamp,
                "stamp_shape": (stamp_h, stamp_w),
                "inner_shape": (inner_h, inner_w),
                "outer_shape": (outer_h, outer_w),
            }
        )

    grid_min = GRID_MIN_BY_COUNT[pair_count]
    grid_h = rng.randint(grid_min, GRID_MAX)
    grid_w = rng.randint(grid_min, GRID_MAX)

    items = []
    for idx, pair in enumerate(pairs):
        outer_h, outer_w = pair["outer_shape"]
        stamp_h, stamp_w = pair["stamp_shape"]
        items.append({"kind": "env", "idx": idx, "h": outer_h, "w": outer_w})
        items.append({"kind": "stamp", "idx": idx, "h": stamp_h, "w": stamp_w})

    placements = _place_rectangles(rng, grid_h, grid_w, items)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for idx, pair in enumerate(pairs):
        env_r, env_c = placements[("env", idx)]
        outer_h, outer_w = pair["outer_shape"]
        _draw_frame(grid, env_r, env_c, outer_h, outer_w)
        mr, mc = _marker_position(env_r, env_c, outer_h, outer_w, pair["corner"])
        grid[mr, mc] = pair["marker_color"]

        stamp_r, stamp_c = placements[("stamp", idx)]
        stamp_h, stamp_w = pair["stamp_shape"]
        grid[stamp_r : stamp_r + stamp_h, stamp_c : stamp_c + stamp_w] = pair["stamp"]

    expected = grid.copy()
    for idx, pair in enumerate(pairs):
        stamp_r, stamp_c = placements[("stamp", idx)]
        stamp_h, stamp_w = pair["stamp_shape"]
        expected[stamp_r : stamp_r + stamp_h, stamp_c : stamp_c + stamp_w] = BG

        env_r, env_c = placements[("env", idx)]
        inner_h, inner_w = pair["inner_shape"]
        resized = _nearest_resize(pair["stamp"], inner_h, inner_w)
        expected[env_r + 1 : env_r + 1 + inner_h, env_c + 1 : env_c + 1 + inner_w] = resized

    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-into-envelope puzzle")
